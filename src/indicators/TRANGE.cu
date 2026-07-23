/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/TRANGE.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void trangeKernel(const float *__restrict__ high,
                             const float *__restrict__ low,
                             const float *__restrict__ close,
                             float *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (idx == 0) {
      // TA-Lib TRANGE has a lookback of 1: the first bar has no previous
      // close, so the output is NaN (not high-low).
      output[idx] = nanf("");
    } else {
      float tr1 = high[idx] - low[idx];
      float tr2 = fabsf(high[idx] - close[idx - 1]);
      float tr3 = fabsf(low[idx] - close[idx - 1]);
      float m = tr1 > tr2 ? tr1 : tr2;
      output[idx] = m > tr3 ? m : tr3;
    }
  }
}

void tacuda::TRANGE::calculate(const float *high, const float *low, const float *close,
                       float *output, int size, cudaStream_t stream) noexcept(false) {
  if (size <= 0) {
    throw std::invalid_argument("TRANGE: invalid size");
  }
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  trangeKernel<<<grid, block, 0, stream>>>(high, low, close, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void tacuda::TRANGE::calculate(const float *input, float *output,
                       int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  const float *close = input + 2 * size;
  calculate(high, low, close, output, size, stream);
}
