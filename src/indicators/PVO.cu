/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <algorithm>
#include <stdexcept>
#include <indicators/PVO.h>
#include <utils/CudaUtils.h>

static __device__ float ema_at(const float* __restrict__ x, int idx, int period) {
  const float k = 2.0f / (period + 1.0f);
  float weight = 1.0f;
  float weightedSum = x[idx];
  float weightSum = 1.0f;
  int steps = min(period, idx);
#pragma unroll
  for (int i = 1; i <= steps; ++i) {
    weight *= (1.0f - k);
    weightedSum += x[idx - i] * weight;
    weightSum += weight;
  }
  return weightedSum / weightSum;
}

__global__ void pvoKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int fastP, int slowP, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= slowP && idx < size) {
    float emaFast = ema_at(input, idx, fastP);
    float emaSlow = ema_at(input, idx, slowP);
    output[idx] = emaSlow == 0.0f ? 0.0f : 100.0f * (emaFast - emaSlow) / emaSlow;
  }
}

tacuda::PVO::PVO(int fastPeriod, int slowPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod) {}

void tacuda::PVO::calculate(const float* input, float* output,
                    int size, cudaStream_t stream) noexcept(false) {
  if (fastPeriod <= 0 || slowPeriod <= 0) {
    throw std::invalid_argument("PVO: invalid periods");
  }
  if (fastPeriod >= slowPeriod) {
    throw std::invalid_argument("PVO: fastPeriod must be < slowPeriod");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  pvoKernel<<<grid, block, 0, stream>>>(input, output, fastPeriod, slowPeriod, size);
  CUDA_CHECK(cudaGetLastError());
}
