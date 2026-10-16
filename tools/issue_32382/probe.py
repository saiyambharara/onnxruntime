# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Real ONNX MatMulNBits: route traces, output hashes, GPU graph and API timings.

Run each experiment mode in a fresh process; the dispatch option is cached by ORT.
External CUDA graphs contain repeated real ORT inferences. Event means include device
launch gaps and are not a per-kernel Nsight duration or full-model decode measurement.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
import subprocess
import time
from ctypes.util import find_library
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--fpa-intb", choices=["0", "1"], default="0")
parser.add_argument("--shapes")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
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


def telemetry():
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,pstate,clocks.gr,clocks.mem,temperature.gpu,power.draw,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"unavailable": str(error)}


stream = ptr()
call("cudaStreamCreate", C.byref(stream))
rows = []
shapes = [
    (3072, 1024),
    (4096, 256),
    (3072, 3072),
    (3072, 8192),
    (8192, 3072),
    (3072, 32000),
    (4088, 256),
    (1032, 256),
    (264, 24),
    (8, 8),
]
if args.shapes:
    shapes = [tuple(map(int, value.split(","))) for value in args.shapes.split(";")]
elif args.quick:
    shapes = shapes[:2]
started = time.time()
for k, n in shapes:
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
            f"issue32382-{k}-{n}",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, k])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, n])],
            [numpy_helper.from_array(b, "B"), numpy_helper.from_array(scales, "scales")],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("ep.cuda.fpa_intb_gemm", args.fpa_intb)
    options.enable_profiling = args.profile
    options.profile_file_prefix = str(args.output.parent / f"profile-{k}-{n}")
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
    row = {
        "K": k,
        "N": n,
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "output_finite": bool(np.isfinite(output).all()),
        "providers": session.get_providers(),
    }
    np.save(args.output.parent / f"output-{k}-{n}.npy", output)
    if args.profile:
        path = session.end_profiling()
        events = json.loads(Path(path).read_text())
        gpu_events = [e for e in events if "MatMulFloat4BitsKernelM1" in json.dumps(e)]
        row["profile_path"] = path
        row["candidate_kernel_events"] = gpu_events
        row["cuda_node_events"] = [e for e in events if e.get("args", {}).get("provider") == "CUDAExecutionProvider"]
        row["gpu_kernel_names"] = sorted(
            {e.get("name", "") for e in events if e.get("args", {}).get("grid_x", "-1") != "-1"}
        )
        # Keep failures visible; a CUDA node event alone does not identify the kernel variant.
        mode = int(os.environ.get("ORT_MATMUL4BITS_M1_EXPERIMENT", "0"))
        unroll = 4 if mode == 2 and k < 4096 else 16
        direct = mode in [1, 2] and not (mode == 2 and k >= 4096 and n < 1024)
        if not direct:
            unroll = 16
        marker = f",32,false,{'true' if direct else 'false'},{unroll}>"
        row["expected_kernel_template_suffix"] = marker
        row["kernel_route_observed"] = any(marker in e.get("name", "").replace(" ", "") for e in gpu_events)
    else:
        # Host API latency is measured separately; it must not be mistaken for kernel duration.
        wall_started = time.perf_counter()
        for _ in range(200):
            session.run_with_iobinding(io, run_options)
        call("cudaStreamSynchronize", stream)
        row["api_mean_us"] = (time.perf_counter() - wall_started) * 1e6 / 200
        graph, executable = ptr(), ptr()
        nodes_per_graph, graphs_per_trial = 50, 20
        call("cudaStreamBeginCapture", stream, 2)
        for _ in range(nodes_per_graph):
            session.run_with_iobinding(io, run_options)
        call("cudaStreamEndCapture", stream, C.byref(graph))
        call("cudaGraphInstantiateWithFlags", C.byref(executable), graph, 0)
        begin, end = ptr(), ptr()
        call("cudaEventCreate", C.byref(begin))
        call("cudaEventCreate", C.byref(end))
        # Warm clock/power state as well as allocator/launch state before collecting samples.
        warm_started = time.perf_counter()
        while time.perf_counter() - warm_started < 1.0:
            for _ in range(graphs_per_trial):
                call("cudaGraphLaunch", executable, stream)
            call("cudaStreamSynchronize", stream)
        row["graph_warmup_seconds"] = time.perf_counter() - warm_started
        row["telemetry_before_samples"] = telemetry()
        samples = []
        for _ in range(21):
            call("cudaEventRecord", begin, stream)
            for _ in range(graphs_per_trial):
                call("cudaGraphLaunch", executable, stream)
            call("cudaEventRecord", end, stream)
            call("cudaEventSynchronize", end)
            ms = C.c_float()
            call("cudaEventElapsedTime", C.byref(ms), begin, end)
            samples.append(ms.value * 1000 / (nodes_per_graph * graphs_per_trial))
        row["graph_event_mean_per_node_us"] = samples
        row["median_graph_us"] = float(np.median(samples))
        row["p10_graph_us"], row["p90_graph_us"] = map(float, np.quantile(samples, [0.1, 0.9]))
        row["graph_inferences_per_trial"] = nodes_per_graph * graphs_per_trial
        row["telemetry_after_samples"] = telemetry()
        call("cudaEventDestroy", begin)
        call("cudaEventDestroy", end)
        call("cudaGraphExecDestroy", executable)
        call("cudaGraphDestroy", graph)
    assert row["output_finite"]
    rows.append(row)
    del io, session, av, yv
    print(k, n, row.get("median_graph_us"), row.get("kernel_route_observed"), flush=True)
call("cudaStreamDestroy", stream)
args.output.write_text(
    json.dumps(
        {
            "ort_version": ort.__version__,
            "elapsed_seconds": time.time() - started,
            "fpa_intb_option": args.fpa_intb,
            "rows": rows,
        },
        indent=2,
    )
)
