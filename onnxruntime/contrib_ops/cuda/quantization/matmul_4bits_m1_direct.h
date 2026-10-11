// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Internal experiment: 0 keeps the existing kernel, 1 uses direct scales, and 2 also limits
// short-K unrolling. The caller restricts this to FP16 and retains its shared-memory eligibility
// check, so the experiment does not expand the existing kernel's supported shapes.
constexpr int SelectMatMul4BitsM1DirectUnroll(int mode, int n, int k, int block_size, bool has_zero_point) {
  if ((mode != 1 && mode != 2) || block_size != 32 || has_zero_point ||
      n <= 0 || n % 8 != 0 || k <= 0 || k % 8 != 0) {
    return 0;
  }
  // With CUDA 13.1, paired real-operator tests regress on long reductions with a narrow
  // output grid. Keep the staged kernel for that region in the conservative mode. This
  // deliberately forgoes a small K=4096,N=512 gain; mode 1 remains an unrestricted control.
  if (mode == 2 && k >= 4096 && n < 1024) {
    return 0;
  }
  // The original 16-way loop only runs from K=4096. Retain it for longer reductions, where
  // globally switching to four-way unrolling regressed the measured K=4096 control.
  return mode == 2 && k < 4096 ? 4 : 16;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
