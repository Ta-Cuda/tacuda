/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <indicators/MINMAXINDEX.h>
#include <stdexcept>
#include <utils/BatchRows.h>
#include <utils/CudaUtils.h>

// TA-Lib 0.8.3 names the newest tie, including a stale-window recompute.
__device__ __forceinline__ void min_max_index_row(const float *__restrict__ input,
                                                  float *__restrict__ minOut,
                                                  float *__restrict__ maxOut,
                                                  int period, int size) {
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

__global__ void minmaxIndexKernel(const float *__restrict__ input,
                                  float *__restrict__ minOut,
                                  float *__restrict__ maxOut, int period,
                                  int size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  min_max_index_row(input, minOut, maxOut, period, size);
}

// Batch path: one GPU thread per row; indices stay relative to the row.
__global__ void minmaxIndexRowsKernel(const float *__restrict__ input,
                                      float *__restrict__ minOut,
                                      float *__restrict__ maxOut, int period,
                                      int length, int series, int spread) {
  size_t r = 0, step = 0;
  if (!tacuda::detail::row_slot(spread, r, step)) return;
  for (; r < static_cast<size_t>(series); r += step) {
    const size_t off = r * static_cast<size_t>(length);
    min_max_index_row(input + off, minOut + off, maxOut + off, period, length);
  }
}

void tacuda::detail::min_max_index_rows(const float *in, float *minOut,
                                        float *maxOut, int length, int series,
                                        int period, cudaStream_t stream) {
  if (period <= 0) {
    throw std::invalid_argument("MINMAXINDEX: invalid period");
  }
  const size_t bytes = static_cast<size_t>(length) * series * sizeof(float);
  CUDA_CHECK(cudaMemsetAsync(minOut, 0xFF, bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(maxOut, 0xFF, bytes, stream));
  if (period > length) return;  // every row is warm-up
  const RowLaunch launch = divergent_row_launch(series, /*fp64=*/false);
  minmaxIndexRowsKernel<<<launch.blocks, launch.threads, 0, stream>>>(
      in, minOut, maxOut, period, length, series, launch.spread);
  CUDA_CHECK(cudaGetLastError());
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
