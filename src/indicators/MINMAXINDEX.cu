/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <indicators/MINMAXINDEX.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

// TA-Lib 0.8.3 names the newest tie, including a stale-window recompute.
__global__ void minmaxIndexKernel(const float *__restrict__ input,
                                  float *__restrict__ minOut,
                                  float *__restrict__ maxOut, int period,
                                  int size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int lowestIdx = -1, highestIdx = -1;
  float lowest = 0.0f, highest = 0.0f;
  int trailingIdx = 0;
  for (int today = period - 1; today < size; ++today, ++trailingIdx) {
    const float tmp = input[today];
    if (lowestIdx < trailingIdx) {
      lowestIdx = today;
      lowest = input[lowestIdx];
      for (int i = today - 1; i >= trailingIdx; --i) {
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
    if (highestIdx < trailingIdx) {
      highestIdx = today;
      highest = input[highestIdx];
      for (int i = today - 1; i >= trailingIdx; --i) {
        float v = input[i];
        if (v > highest) {
          highestIdx = i;
          highest = v;
        }
      }
    } else if (tmp >= highest) {
      highestIdx = today;
      highest = tmp;
    }
    // TA-Lib MINMAXINDEX returns absolute indices in the input series.
    minOut[today] = static_cast<float>(lowestIdx);
    maxOut[today] = static_cast<float>(highestIdx);
  }
}

tacuda::MINMAXINDEX::MINMAXINDEX(int period) : period(period) {}

void tacuda::MINMAXINDEX::calculate(const float *input, float *output,
                            int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("MINMAXINDEX: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * size * sizeof(float), stream));
  float *minOut = output;
  float *maxOut = output + size;
  // Sequential state machine: single thread reproduces TA-Lib exactly.
  minmaxIndexKernel<<<1, 1, 0, stream>>>(input, minOut, maxOut, period, size);
  CUDA_CHECK(cudaGetLastError());
}
