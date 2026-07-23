/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <indicators/MINMAX.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

// TA-Lib convention: the window [end-period+1, end] is reported at `end`,
// with a leading warm-up of period-1 NaNs.
__global__ void minmaxKernel(const float *__restrict__ input,
                             float *__restrict__ minOut,
                             float *__restrict__ maxOut, int period, int size) {
  int end = blockIdx.x * blockDim.x + threadIdx.x;
  if (end >= size || end < period - 1) return;
  int start = end - period + 1;
  float minVal = input[start];
  float maxVal = input[start];
  for (int k = 1; k < period; ++k) {
    float v = input[start + k];
    minVal = fminf(minVal, v);
    maxVal = fmaxf(maxVal, v);
  }
  minOut[end] = minVal;
  maxOut[end] = maxVal;
}

tacuda::MINMAX::MINMAX(int period) : period(period) {}

void tacuda::MINMAX::calculate(const float *input, float *output,
                       int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("MINMAX: invalid period");
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
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  minmaxKernel<<<grid, block, 0, stream>>>(input, minOut, maxOut, period, size);
  CUDA_CHECK(cudaGetLastError());
}
