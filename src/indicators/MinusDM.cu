/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MinusDM.h>
#include <indicators/detail/WilderCommon.cuh>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

namespace {

// Compute raw -DM values in parallel
__global__ void minusDMValuesKernel(const float *__restrict__ high,
                                    const float *__restrict__ low,
                                    float *__restrict__ dmMinus,
                                    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    dmMinus[0] = 0.0f;
  } else if (idx < size) {
    float diffP = high[idx] - high[idx - 1];
    float diffM = low[idx - 1] - low[idx];
    dmMinus[idx] = (diffM > 0.0f && diffM > diffP) ? diffM : 0.0f;
  }
}

} // namespace

tacuda::MinusDM::MinusDM(int period) : period(period) {}

void tacuda::MinusDM::calculate(const float *high, const float *low, float *output,
                        int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("MinusDM: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  // 1. Compute -DM values in parallel
  auto dmMinusBuffer = acquireDeviceBufferAsync<float>(size, stream);
  {
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    minusDMValuesKernel<<<grid, block, 0, stream>>>(high, low, dmMinusBuffer.get(), size);
    CUDA_CHECK(cudaGetLastError());
  }

  // 2. Apply TA-Lib-compatible DM smoothing (SUM seed, no alpha scaling)
  tacuda::detail::computeDMSmoothing(dmMinusBuffer.get(), output, size, period, stream);
}

void tacuda::MinusDM::calculate(const float *input, float *output,
                        int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  calculate(high, low, output, size, stream);
}
