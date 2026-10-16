# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Stage a locally built runtime for independent user publication; never upload it."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True, help="Release directory, not its parent")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside the source checkout")
    parser.add_argument("--architecture", choices=["90", "120"], default="90")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    architecture = re.search(r"^CMAKE_CUDA_ARCHITECTURES:[^=]*=(.*)$", cache, re.MULTILINE)
    if architecture is None or architecture[1].strip() != args.architecture:
        raise RuntimeError("Build architecture does not match the requested artifact label")
    home = re.search(r"^CMAKE_HOME_DIRECTORY:[^=]*=(.*)$", cache, re.MULTILINE)
    source = Path(home[1].strip()).parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True).strip():
        raise RuntimeError("Refusing to attribute artifacts to a dirty source checkout")
    if args.output_dir.resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Write binary artifacts outside the source checkout")
    for artifact_name, source_name in [
        ("libonnxruntime_providers_cuda.so", "onnxruntime/contrib_ops/cuda/quantization/matmul_4bits_m1_impl.cuh"),
        ("libonnxruntime_providers_cuda.so", "onnxruntime/contrib_ops/cuda/quantization/matmul_4bits_m1_direct.h"),
        ("onnxruntime_provider_test", "onnxruntime/test/contrib_ops/matmul_4bits_test.cc"),
    ]:
        if (build / artifact_name).stat().st_mtime_ns < (source / source_name).stat().st_mtime_ns:
            raise RuntimeError("Artifact predates the experiment source: " + artifact_name)
    # Explicitly restage after a provider relink; the Python post-build copy can be stale.
    for name in ["libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"]:
        shutil.copy2(build / name, build / "onnxruntime/capi" / name)
    subprocess.run(
        [sys.executable, str(source / "setup.py"), "bdist_wheel", "--wheel_name_suffix=gpu"], cwd=build, check=True
    )
    wheels = sorted((build / "dist").glob("onnxruntime_gpu-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError("Build the Python wheel before packaging")
    provider = build / "libonnxruntime_providers_cuda.so"
    with zipfile.ZipFile(wheels[-1]) as archive:
        packaged_hash = hashlib.sha256(archive.read("onnxruntime/capi/libonnxruntime_providers_cuda.so")).hexdigest()
    if packaged_hash != sha256(provider):
        raise RuntimeError("Stale wheel: rerun setup.py bdist_wheel --wheel_name_suffix=gpu from Release")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "bin").mkdir()
    shutil.copy2(wheels[-1], output / wheels[-1].name)
    for name in ["onnxruntime_provider_test", "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"]:
        shutil.copy2(build / name, output / "bin" / name)
    files = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()}
    manifest = {
        "source_commit": commit,
        "architecture": args.architecture,
        "python_tag": "cp312",
        "wheel": wheels[-1].name,
        "provider_sha256": sha256(provider),
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "source_commit": commit,
                "architecture": args.architecture,
                "bundle_files": len(files),
                "output_dir": str(output),
            }
        )
    )
    if args.archive:
        target = output.with_suffix(".tar.gz")
        if target.exists():
            raise FileExistsError(target)
        with tarfile.open(target, "w:gz") as archive:
            archive.add(output, arcname=output.name)
        print(json.dumps({"archive": str(target), "bytes": target.stat().st_size, "sha256": sha256(target)}))


if __name__ == "__main__":
    main()
