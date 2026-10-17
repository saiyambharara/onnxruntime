# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Persistent isolated mode workers, closely interleaved real-ORT ABBA comparisons.

Different workers are required because the experiment option is process-cached.
Output hashes and the earlier CUPTI traces provide numerical and routing evidence.
These are repeated-weight CUDA graph event means, not full-model measurements.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from ctypes.util import find_library
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--worker", action="store_true")
parser.add_argument("--shape", default="3072,1024")
parser.add_argument("--output", type=Path)
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--intra-op-threads", type=int, default=0, help="ORT host threads; 0 preserves automatic selection")
parser.add_argument("--shapes", default="3072,1024;4096,256;3072,3072;3072,8192;8192,3072;3072,32000")
args = parser.parse_args()
if args.intra_op_threads < 0:
    parser.error("--intra-op-threads must be nonnegative")
PREFIX = "ORT_WORKER:"


def emit(value):
    print(PREFIX + json.dumps(value), flush=True)


def worker():
    k, n = map(int, args.shape.split(","))
    rt = C.CDLL(find_library("cudart") or "libcudart.so.13")
    ptr = C.c_void_p
    for name, signature in {
        "cudaStreamCreate": [C.POINTER(ptr)],
        "cudaStreamSynchronize": [ptr],
        "cudaStreamBeginCapture": [ptr, C.c_int],
        "cudaStreamEndCapture": [ptr, C.POINTER(ptr)],
        "cudaGraphInstantiateWithFlags": [C.POINTER(ptr), ptr, C.c_ulonglong],
        "cudaGraphLaunch": [ptr, ptr],
        "cudaGraphExecDestroy": [ptr],
        "cudaGraphDestroy": [ptr],
        "cudaEventCreate": [C.POINTER(ptr)],
        "cudaEventRecord": [ptr, ptr],
        "cudaEventSynchronize": [ptr],
        "cudaEventElapsedTime": [C.POINTER(C.c_float), ptr, ptr],
        "cudaEventDestroy": [ptr],
        "cudaStreamDestroy": [ptr],
    }.items():
        fn = getattr(rt, name)
        fn.argtypes, fn.restype = signature, C.c_int
    rt.cudaGetErrorString.argtypes, rt.cudaGetErrorString.restype = [C.c_int], C.c_char_p

    def call(name, *values):
        code = getattr(rt, name)(*values)
        if code:
            raise RuntimeError(f"{name}: {code}: {rt.cudaGetErrorString(code).decode()}")

    stream = ptr()
    call("cudaStreamCreate", C.byref(stream))
    rng = np.random.default_rng(k * 100003 + n)
    a = rng.normal(0, 0.15, (1, k)).astype(np.float16)
    blocks = (k + 31) // 32
    b = rng.integers(0, 256, (n, blocks, 16), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.03, (n, blocks)).astype(np.float16)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "MatMulNBits",
                    ["A", "B", "scales"],
                    ["Y"],
                    domain="com.microsoft",
                    K=k,
                    N=n,
                    bits=4,
                    block_size=32,
                    accuracy_level=0,
                )
            ],
            f"bracketed-{k}-{n}",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, k])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, n])],
            [numpy_helper.from_array(b, "B"), numpy_helper.from_array(scales, "scales")],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    if args.intra_op_threads:
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("ep.cuda.fpa_intb_gemm", "0")
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"user_compute_stream": str(stream.value), "enable_cuda_graph": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    av = ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0)
    yv = ort.OrtValue.ortvalue_from_shape_and_type([1, n], np.float16, "cuda", 0)
    io = session.io_binding()
    io.bind_ortvalue_input("A", av)
    io.bind_ortvalue_output("Y", yv)
    run_options = ort.RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    for _ in range(10):
        session.run_with_iobinding(io, run_options)
    call("cudaStreamSynchronize", stream)
    output = yv.numpy()
    assert np.isfinite(output).all()
    graph, executable = ptr(), ptr()
    nodes_per_graph = 500
    call("cudaStreamBeginCapture", stream, 2)
    for _ in range(nodes_per_graph):
        session.run_with_iobinding(io, run_options)
    call("cudaStreamEndCapture", stream, C.byref(graph))
    call("cudaGraphInstantiateWithFlags", C.byref(executable), graph, 0)
    begin, end = ptr(), ptr()
    call("cudaEventCreate", C.byref(begin))
    call("cudaEventCreate", C.byref(end))
    emit(
        {
            "ready": True,
            "K": k,
            "N": n,
            "ort_version": ort.__version__,
            "mode": os.environ["ORT_MATMUL4BITS_M1_EXPERIMENT"],
            "nodes_per_graph": nodes_per_graph,
            "intra_op_threads": args.intra_op_threads,
            "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        }
    )
    try:
        for line in sys.stdin:
            command = json.loads(line)
            action = command["action"]
            if action == "close":
                break
            if action == "warm":
                start = time.perf_counter()
                while time.perf_counter() - start < command["seconds"]:
                    call("cudaGraphLaunch", executable, stream)
                    call("cudaStreamSynchronize", stream)
                emit({"warm_seconds": time.perf_counter() - start})
                continue
            assert action == "measure"
            samples = []
            for _ in range(command.get("trials", 5)):
                call("cudaEventRecord", begin, stream)
                for _ in range(2):
                    call("cudaGraphLaunch", executable, stream)
                call("cudaEventRecord", end, stream)
                call("cudaEventSynchronize", end)
                milliseconds = C.c_float()
                call("cudaEventElapsedTime", C.byref(milliseconds), begin, end)
                samples.append(milliseconds.value * 1000 / (2 * nodes_per_graph))
            emit({"median_us": statistics.median(samples), "samples_us": samples})
    finally:
        call("cudaEventDestroy", begin)
        call("cudaEventDestroy", end)
        call("cudaGraphExecDestroy", executable)
        call("cudaGraphDestroy", graph)
        del io, session, av, yv
        call("cudaStreamDestroy", stream)


def master():
    assert args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(32382)
    shapes = [tuple(map(int, shape.split(","))) for shape in args.shapes.split(";")]
    document = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds": args.rounds,
        "graph_nodes": 500,
        "intra_op_threads": args.intra_op_threads,
        "harness_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip(),
        "rows": [],
    }
    with (Path(ort.__file__).parent / "capi/libonnxruntime_providers_cuda.so").open("rb") as stream:
        document["provider_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    started = time.perf_counter()

    def receive(process):
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"Worker exited unexpectedly: {process.poll()}")
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX) :])

    def command(process, value):
        process.stdin.write(json.dumps(value) + "\n")
        process.stdin.flush()
        return receive(process)

    for k, n in shapes:
        processes, logs, ready = {}, [], {}
        row = {"K": k, "N": n, "comparisons": []}
        try:
            for mode in [0, 1, 2]:
                log = (args.output.parent / f"worker-{k}-{n}-mode-{mode}.log").open("w")
                logs.append(log)
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--shape",
                        f"{k},{n}",
                        "--intra-op-threads",
                        str(args.intra_op_threads),
                    ],
                    env=dict(os.environ, ORT_FPA_INTB_GEMM="0", ORT_MATMUL4BITS_M1_EXPERIMENT=str(mode)),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                )
                processes[mode] = process
                ready[mode] = receive(process)
            assert len({value["output_sha256"] for value in ready.values()}) == 1
            row["worker_metadata"] = ready
            for mode in [0, 1, 2]:
                command(processes[mode], {"action": "warm", "seconds": 0.5})
            for repetition in range(args.rounds):
                order = [1, 2]
                randomizer.shuffle(order)
                for mode in order:
                    sequence = [0, mode, mode, 0]
                    if randomizer.choice([False, True]):
                        sequence = [mode, 0, 0, mode]
                    values = []
                    for selected in sequence:
                        value = command(processes[selected], {"action": "measure"})
                        values.append({"mode": selected, **value})
                    baseline = [value["median_us"] for value in values if value["mode"] == 0]
                    candidate = [value["median_us"] for value in values if value["mode"] == mode]
                    drift = max(baseline) / min(baseline)
                    ratio = statistics.mean(baseline) / statistics.mean(candidate)
                    row["comparisons"].append(
                        {
                            "round": repetition + 1,
                            "mode": mode,
                            "sequence": values,
                            "baseline_drift_ratio": drift,
                            "accepted": drift <= 1.15,
                            "speedup": ratio,
                            "latency_reduction_percent": 100 * (1 - 1 / ratio),
                        }
                    )
            row["summary"] = {}
            for mode in [1, 2]:
                accepted = [
                    value["speedup"] for value in row["comparisons"] if value["mode"] == mode and value["accepted"]
                ]
                row["summary"][str(mode)] = {
                    "accepted_brackets": len(accepted),
                    "total_brackets": args.rounds,
                    "sufficient_stable_brackets": len(accepted) >= 0.6 * args.rounds,
                    "median_speedup": statistics.median(accepted) if accepted else None,
                    "speedup_p10_p90": list(map(float, np.quantile(accepted, [0.1, 0.9]))) if accepted else None,
                    "median_latency_reduction_percent": 100 * (1 - 1 / statistics.median(accepted))
                    if accepted
                    else None,
                }
            print(k, n, row["summary"], flush=True)
            document["rows"].append(row)
            document["elapsed_seconds"] = time.perf_counter() - started
            args.output.write_text(json.dumps(document, indent=2))
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.stdin.write(json.dumps({"action": "close"}) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=30)
                assert process.returncode == 0
            for log in logs:
                log.close()


if __name__ == "__main__":
    worker() if args.worker else master()
