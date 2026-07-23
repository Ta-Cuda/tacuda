/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <indicators/MININDEX.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

// Exact TA-Lib state machine (ta_MININDEX.c): recompute picks the OLDEST
// tie (`tmp < lowest`), an incoming bar replaces on equality
// (`tmp <= lowest`). Path-dependent — walked sequentially by one thread.
__global__ void minIndexKernel(const float *__restrict__ input,
                               float *__restrict__ output, int period,
                               int size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int lowestIdx = -1;
  float lowest = 0.0f;
  int trailingIdx = 0;
  for (int today = period - 1; today < size; ++today, ++trailingIdx) {
    float tmp = input[today];
    if (lowestIdx < trailingIdx) {
      lowestIdx = trailingIdx;
      lowest = input[lowestIdx];
      for (int i = lowestIdx + 1; i <= today; ++i) {
        float v = input[i];
        if (v < lowest) {
          lowestIdx = i;
          lowest = v;
        }
      }
    } else if (tmp <= lowest) {
      lowestIdx = today;
      lowest = tmp;
    }
    // TA-Lib MININDEX returns the absolute index in the input series.
    output[today] = static_cast<float>(lowestIdx);
  }
}

tacuda::MININDEX::MININDEX(int period) : period(period) {}

void tacuda::MININDEX::calculate(const float *input, float *output,
                         int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("MININDEX: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  // Sequential state machine: single thread reproduces TA-Lib exactly.
  minIndexKernel<<<1, 1, 0, stream>>>(input, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}
