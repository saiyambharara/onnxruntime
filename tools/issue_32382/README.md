# MatMulNBits issue 32382: portable H100 experiment

This directory runs the opt-in FP16/group-32/M=1 direct-scale experiment against
the original staged-scale kernel. It is an experiment harness, not automatic
repository CI coverage or a default-dispatch performance fix. Results describe
resident tensors with repeated weights and CUDA-event graph replay means;
they do not establish full-model decode gains.

## Destination and build strategy

Use Linux x86-64, Ubuntu 24.04, Python 3.12, CUDA 13.1 and the pinned runtime
dependencies in `requirements.txt`. The local compiler is NVCC 13.1.115 with
GCC 13.3. Explicitly generate native SM90 code. H100 PCIe, SXM and NVL can run
this build, but are different performance baselines. Record the actual GPU.
The initial issue comparison ultimately needs H100 PCIe.

A registry-verified, suitable Vast base image is:

```
vastai/base-image@sha256:c68a7d64287792a266fcbc6a0ad5321129327a712daf6c84dae2366ff383da33
```

This amd64 image corresponds to `cuda-13.1.1-auto`; its metadata identifies
Ubuntu 24.04, Python 3.12 and CUDA 13.1.1. It includes cuDNN 9.17; install the
pinned 9.20 runtime into the experiment's isolated venv, whose library paths
the runner places first. Prefer a host advertising CUDA >=13.1, such as driver
595.71.05, to avoid dependence on older-driver minor-compatibility exceptions.
The host's OS label does not replace the container userspace requirement.

No H100 is needed to compile. Build locally before paying for GPU execution.
Keep the existing SM120 build; changing architectures requires a separate tree.
The focused build disables several attention implementations and is suitable
for this operator experiment, not production attention/full-model conclusions.

Run in a Python 3.12 build venv with the existing CUDA/cuDNN SDKs available.
`ORT_CUDA` and `ORT_CUDNN` below are SDK roots chosen on the build machine;
do not publish or forward source-device configuration files to the VM.

```bash
python tools/ci_build/build.py \
  --build_dir "$ORT_BUILD" --config Release --use_cuda \
  --cuda_home "$ORT_CUDA" --cudnn_home "$ORT_CUDNN" \
  --cmake_generator Ninja --parallel 3 --nvcc_threads 1 \
  --enable_pybind --build_wheel --enable_cuda_profiling \
  --update --build --skip_tests \
  --targets onnxruntime_pybind11_state onnxruntime_providers_cuda onnxruntime_provider_test \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 \
  FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER onnxruntime_QUICK_BUILD=ON \
  onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_LEAN_ATTENTION=OFF \
  onnxruntime_USE_TRT_FUSED_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
  onnxruntime_USE_FPA_INTB_GEMM_FULL=OFF

python tools/issue_32382/package.py \
  --build-dir "$ORT_BUILD/Release" --architecture 90 \
  --output-dir "$ORT_ARTIFACTS/h100-runtime" --archive
```

The packager restages the linked provider, rebuilds the wheel, verifies its
provider hash, and includes only the wheel, focused test executable, required
provider libraries and a relative-path checksum/source manifest. It never
uploads anything. Use a new output directory. Keep compiler intermediates,
build logs, credentials and device configuration outside the public bundle.
Inspect native SM90 MatMul kernel coverage with CUDA binary tools before the
GPU run; architecture-specific code cannot be executed on the SM120 laptop.

## Approved remote workflow

Provisioning requires separate explicit approval. This repository contains no
provisioning/API credentials, instance creation or file-upload implementation.
Code must be cloned from the user's public fork at the exact approved commit.
Binary artifacts are independently published by the user at a public Dropbox
URL and fetched by the approved VM from that URL, with the supplied SHA256.
Do not use local-to-VM uploads, source-device environments or SSH agent forwarding.

After cloning `perf/32382-h100-experiment`, checking out and verifying the
approved SHA, execute the committed setup script on the destination:

```bash
bash tools/issue_32382/setup.sh /workspace/ort-32382-env
```

Fetch and checksum the user-provided archive on the VM; extract it under
`/workspace/artifacts`. Install the custom wheel and execute the suite through
the runner in a detached tmux session, using an unused output directory:

```bash
/workspace/ort-32382-env/bin/python tools/issue_32382/run.py \
  --artifact-dir /workspace/artifacts/h100-runtime \
  --expected-commit APPROVED_FULL_COMMIT_SHA \
  --output /workspace/results/initial --deadline-seconds 2700
```

The runner checks Git/artifact identity and checksums, validates the installed
provider, runs 25 non-skipped GoogleTest invocations, verifies 21 exact kernel
routes in CUPTI traces, and runs six-shape randomized paired ABBA/BAAB timing.
Each experiment mode uses a fresh process because the launcher option is cached.
It rejects a result with insufficient stable brackets and preserves failure
status; numerical or CUDA-provider success alone is not kernel-route proof.

`--local-smoke` permits an explicitly labelled SM120 bundle for testing the
portable harness on the laptop. It does not validate SM90 execution. `--rounds`
and `--shapes` can reduce only the local smoke benchmark workload.

## One-hour early-testing limit

Budget at most ten minutes for setup/driver checks, up to 45 minutes for the
bounded suite and a small optional profiling pass, and five minutes for cleanup.
Treat setup failure or missing kernel traces as a failed/inconclusive gate;
do not spend the reservation rebuilding. Full-model validation is outside this
first session. Native GPU profiling permissions still need to be checked.

The runner kills its process group on its execution deadline. This does not
end Vast billing. The separately approved control-plane session must enforce
an absolute one-hour creation-to-destruction deadline, check bounded summaries
over SSH, and destroy the exact instance. Do not create replacements or extend
the run without approval. Retain no paid storage by default. Keep outputs/logs
on the VM; only bounded status/summary metrics may be viewed through SSH.
No VM-to-device artifact or log downloads are part of this workflow.
