/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <indicators/MININDEX.h>
#include <stdexcept>
#include <utils/BatchRows.h>
#include <utils/CudaUtils.h>

// TA-Lib 0.8.3 names the newest tie, including a stale-window recompute.
__device__ __forceinline__ void min_index_row(const float *__restrict__ input,
                                              float *__restrict__ output,
                                              int period, int size) {
  int lowestIdx = -1;
  float lowest = 0.0f;
  int trailingIdx = 0;
  for (int today = period - 1; today < size; ++today, ++trailingIdx) {
    float tmp = input[today];
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
    // TA-Lib MININDEX returns the absolute index in the input series.
    output[today] = static_cast<float>(lowestIdx);
  }
}

__global__ void minIndexKernel(const float *__restrict__ input,
                               float *__restrict__ output, int period,
                               int size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  min_index_row(input, output, period, size);
}

// Batch path: one GPU thread per row; indices stay relative to the row.
__global__ void minIndexRowsKernel(const float *__restrict__ input,
                                   float *__restrict__ output, int period,
                                   int length, int series, int spread) {
  size_t r = 0, step = 0;
  if (!tacuda::detail::row_slot(spread, r, step)) return;
  for (; r < static_cast<size_t>(series); r += step) {
    const size_t off = r * static_cast<size_t>(length);
    min_index_row(input + off, output + off, period, length);
  }
}

void tacuda::detail::min_index_rows(const float *in, float *out, int length,
                                    int series, int period, cudaStream_t stream) {
  if (period <= 0) {
    throw std::invalid_argument("MININDEX: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(out, 0xFF, static_cast<size_t>(length) * series * sizeof(float), stream));
  if (period > length) return;  // every row is warm-up
  const RowLaunch launch = divergent_row_launch(series, /*fp64=*/false);
  minIndexRowsKernel<<<launch.blocks, launch.threads, 0, stream>>>(in, out, period, length, series, launch.spread);
  CUDA_CHECK(cudaGetLastError());
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
