/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/PlusDM.h>
#include <indicators/detail/WilderCommon.cuh>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

namespace {

// Compute raw +DM values in parallel
__global__ void plusDMValuesKernel(const float *__restrict__ high,
                                   const float *__restrict__ low,
                                   float *__restrict__ dmPlus,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    dmPlus[0] = 0.0f;
  } else if (idx < size) {
    float diffP = high[idx] - high[idx - 1];
    float diffM = low[idx - 1] - low[idx];
    dmPlus[idx] = (diffP > 0.0f && diffP > diffM) ? diffP : 0.0f;
  }
}

} // namespace

tacuda::PlusDM::PlusDM(int period) : period(period) {}

void tacuda::PlusDM::calculate(const float *high, const float *low, float *output,
                       int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("PlusDM: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  // 1. Compute +DM values in parallel
  auto dmPlusBuffer = acquireDeviceBufferAsync<float>(size, stream);
  {
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    plusDMValuesKernel<<<grid, block, 0, stream>>>(high, low, dmPlusBuffer.get(), size);
    CUDA_CHECK(cudaGetLastError());
  }

  // 2. Apply TA-Lib-compatible DM smoothing (SUM seed, no alpha scaling)
  tacuda::detail::computeDMSmoothing(dmPlusBuffer.get(), output, size, period, stream);
}

void tacuda::PlusDM::calculate(const float *input, float *output,
                       int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  calculate(high, low, output, size, stream);
}
