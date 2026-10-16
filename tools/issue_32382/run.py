# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Validate a prebuilt runtime and run bounded MatMulNBits experiments locally."""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import site
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

FOCUSED = ["DirectScaleExperimentSelection", "Fp16_Int4_DirectScaleExperiment", "Cuda_DirectScaleExperimentFallbacks"]
REGRESSIONS = [
    "Float32_4b_Accuracy0",
    "Float16_4b_Accuracy0",
    "LegacyShape_4b",
    "Fp16_Int4_Int4ZeroPoint",
    "Fp16_Int4_Fp16ZeroPoint",
    "BFloat16_Int4_Int4ZeroPoint",
    "BFloat16_Int4_BFloat16ZeroPoint",
    "Fp16_Int4_NoZeroPoint",
    "BFloat16_Int4_NoZeroPoint",
    "Fp16_Int4_BlockSize32_FpAIntB",
    "BFloat16_Int4_BlockSize32_FpAIntB",
    "Fp16_Int4_NoZeroPoint_Bias",
    "BFloat16_Int4_NoZeroPoint_Bias",
    "Fp16_Int4_SmallMBatchedTiles",
    "BFloat16_Int4_SmallMBatchedTiles",
    "Fp16_Int4_GptOssRouterShapeNoZeroPoint",
]
ROUTE_SHAPES = "3072,1024;4096,256;8192,256;8192,512;4096,512;4096,1024;8192,1024"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=2700)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--shapes", default="3072,1024;4096,256;3072,3072;3072,8192;8192,3072;3072,32000")
    parser.add_argument("--local-smoke", action="store_true", help="Permit the SM120 laptop validation bundle")
    parser.add_argument("--skip-install", action="store_true", help="Verify the already installed provider instead")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.flags.optimize:
        raise RuntimeError("Use Python 3.12 without -O/PYTHONOPTIMIZE")
    if not 1 <= args.deadline_seconds <= 2700 or not 1 <= args.rounds <= 100:
        raise ValueError("Execution deadline must be <=45 minutes and rounds must be in [1,100]")
    root = args.artifact_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True).strip()
    if commit != args.expected_commit or manifest["source_commit"] != commit:
        raise RuntimeError("Git checkout, approved commit and compiled artifacts must identify the same commit")
    expected_architecture = "120" if args.local_smoke else "90"
    if manifest["architecture"] != expected_architecture:
        raise RuntimeError("Wrong artifact architecture for this experiment")
    for relative, digest in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise RuntimeError("Missing or mismatched artifact: " + relative)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    state = {
        "source_commit": commit,
        "architecture": expected_architecture,
        "provider_sha256": manifest["provider_sha256"],
        "phases": [],
        "exit_code": None,
    }
    started = time.monotonic()
    env = dict(os.environ)
    library_dirs = [str(root / "bin")]
    for directory in site.getsitepackages():
        nvidia = Path(directory) / "nvidia"
        if nvidia.exists():
            library_dirs.extend(str(path) for path in sorted(nvidia.rglob("lib")) if path.is_dir())
    env["LD_LIBRARY_PATH"] = ":".join([*library_dirs, env.get("LD_LIBRARY_PATH", "")])
    env["ORT_FPA_INTB_GEMM"] = "0"
    scripts = Path(__file__).resolve().parent

    def run_phase(name, command, mode=None):
        remaining = args.deadline_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Experiment execution deadline reached")
        phase_env = dict(env)
        if mode is not None:
            phase_env["ORT_MATMUL4BITS_M1_EXPERIMENT"] = str(mode)
        begin = time.monotonic()
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=root / "bin", env=phase_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError("Experiment execution deadline reached") from None
        record = {"name": name, "exit_code": code, "elapsed_seconds": time.monotonic() - begin}
        state["phases"].append(record)
        (output / "status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(record), flush=True)
        if code:
            raise RuntimeError("Phase failed; inspect its log on the execution host: " + name)

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
            env=env,
            text=True,
            timeout=20,
        ).strip()
        state["gpu_summary"] = gpu
        if not args.local_smoke and ("H100" not in gpu or len(gpu.splitlines()) != 1 or ", 9.0," not in gpu):
            raise RuntimeError("Expected one visible H100 with compute capability 9.0")
        if not args.skip_install:
            run_phase(
                "install",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    str(root / manifest["wheel"]),
                ],
            )
        spec = importlib.util.find_spec("onnxruntime")
        if spec is None:
            raise RuntimeError("Custom runtime is not installed")
        provider = Path(spec.origin).parent / "capi/libonnxruntime_providers_cuda.so"
        if sha256(provider) != manifest["provider_sha256"]:
            raise RuntimeError("Installed provider does not match the compiled manifest")
        for name, mode, selected in [
            ("focused-0", 0, FOCUSED),
            ("focused-1", 1, FOCUSED),
            ("focused-2", 2, FOCUSED),
            ("regressions", 2, REGRESSIONS),
        ]:
            xml_path = output / (name + ".xml")
            run_phase(
                name,
                [
                    str(root / "bin/onnxruntime_provider_test"),
                    "--gtest_filter=" + ":".join("MatMulNBits." + test for test in selected),
                    "--gtest_output=xml:" + str(xml_path),
                ],
                mode,
            )
            xml = ET.parse(xml_path).getroot()
            if int(xml.attrib["tests"]) != len(selected) or any(
                test.find("skipped") is not None or test.find("failure") is not None for test in xml.iter("testcase")
            ):
                raise RuntimeError("Missing, skipped or failed required operator tests")
        for mode in [0, 1, 2]:
            directory = output / f"routes-{mode}"
            directory.mkdir()
            run_phase(
                f"routes-{mode}",
                [
                    sys.executable,
                    str(scripts / "probe.py"),
                    "--profile",
                    "--shapes",
                    ROUTE_SHAPES,
                    "--output",
                    str(directory / "profile.json"),
                ],
                mode,
            )
            profile = json.loads((directory / "profile.json").read_text())
            if len(profile["rows"]) != 7 or not all(row["kernel_route_observed"] for row in profile["rows"]):
                raise RuntimeError("Required exact kernel route was not observed")
        run_phase(
            "paired",
            [
                sys.executable,
                str(scripts / "benchmark.py"),
                "--rounds",
                str(args.rounds),
                "--shapes",
                args.shapes,
                "--output",
                str(output / "paired/results.json"),
            ],
        )
        paired = json.loads((output / "paired/results.json").read_text())
        for row in paired["rows"]:
            if not all(value["sufficient_stable_brackets"] for value in row["summary"].values()):
                raise RuntimeError("Insufficient stable timing brackets; retain this as inconclusive")
        state["exit_code"] = 0
    except BaseException:
        state["exit_code"] = 1
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - started
        (output / "status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps({"exit_code": state["exit_code"], "elapsed_seconds": state["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
