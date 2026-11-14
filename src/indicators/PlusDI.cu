/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/PlusDI.h>
#include <utils/AffineScan.cuh>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/DeviceSeed.cuh>

namespace {

using AffinePair = tacuda::detail::AffineTransform1<double>;
using AffineCompose = tacuda::detail::AffineCompose1<double>;

__global__ void buildPlusDmAndTrKernel(const float* __restrict__ high,
                                       const float* __restrict__ low,
                                       const float* __restrict__ close,
                                       float* __restrict__ dm,
                                       float* __restrict__ tr,
                                       int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  if (i == 0) {
    dm[0] = 0.0f;
    tr[0] = 0.0f;
    return;
  }

  float upMove = high[i] - high[i - 1];
  float downMove = low[i - 1] - low[i];
  dm[i] = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;

  float a = high[i] - low[i];
  float b = fabsf(high[i] - close[i - 1]);
  float c = fabsf(low[i] - close[i - 1]);
  tr[i] = fmaxf(a, fmaxf(b, c));
}

__global__ void fillPairsKernel(const float* __restrict__ x,
                                AffinePair* __restrict__ pairs,
                                int offset, int nPairs,
                                double a) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nPairs) return;
  int i = offset + k;
  pairs[k] = AffinePair{a, static_cast<double>(x[i])};
}

__global__ void finalizeSmoothKernel(const AffinePair* __restrict__ pref,
                                     float* __restrict__ y,
                                     const double* __restrict__ seedp,
                                     int seedIdx,
                                     int nPairs) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nPairs) return;
  AffinePair p = pref[k];
  y[seedIdx + 1 + k] = static_cast<float>(p.A * (*seedp) + p.b);
}

__global__ void computeDiKernel(const float* __restrict__ smDm,
                                const float* __restrict__ smTr,
                                float* __restrict__ output,
                                int size,
                                int period) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  if (i < period) {
    output[i] = nanf("");
    return;
  }
  float trv = smTr[i];
  output[i] = (trv == 0.0f) ? 0.0f : (100.0f * smDm[i] / trv);
}

static void smoothFromPeriodMinusOneSeed(const float* x, float* y, int size, int period,
                                         cudaStream_t stream) {
  // Seed at index=period-1 with SUM(x[1..period-1]), then apply Wilder update from i=period.
  // The seed is computed and kept on the device (after the NaN fill it overwrites).
  CUDA_CHECK(cudaMemsetAsync(y, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
  const int seedIdx = period - 1;
  auto seed = acquireDeviceBufferAsync<double>(1, stream);
  tacuda::detail::launch_seed_sum(x, 1, period - 1, 1.0, seed.get(), y, seedIdx, stream);

  const int firstIdx = period;
  const int nPairs = size - firstIdx;
  if (nPairs <= 0) return;

  const double a = (static_cast<double>(period) - 1.0) / static_cast<double>(period);
  auto pairs = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);
  auto pref = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);

  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(nPairs);
  fillPairsKernel<<<grid, block, 0, stream>>>(x, pairs.get(), firstIdx, nPairs, a);
  CUDA_CHECK(cudaGetLastError());

  tacuda::detail::affine_inclusive_scan<AffinePair, AffineCompose>(
      pairs.get(), pref.get(), nPairs, stream);

  finalizeSmoothKernel<<<grid, block, 0, stream>>>(pref.get(), y, seed.get(), seedIdx, nPairs);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace

tacuda::PlusDI::PlusDI(int period) : period(period) {}

void tacuda::PlusDI::calculate(const float *high, const float *low, const float *close,
                       float *output, int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("PlusDI: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }

  auto dmRaw = acquireDeviceBufferAsync<float>(size, stream);
  auto trRaw = acquireDeviceBufferAsync<float>(size, stream);
  auto dmSmoothed = acquireDeviceBufferAsync<float>(size, stream);
  auto trSmoothed = acquireDeviceBufferAsync<float>(size, stream);

  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  buildPlusDmAndTrKernel<<<grid, block, 0, stream>>>(high, low, close, dmRaw.get(), trRaw.get(), size);
  CUDA_CHECK(cudaGetLastError());

  smoothFromPeriodMinusOneSeed(dmRaw.get(), dmSmoothed.get(), size, period, stream);
  smoothFromPeriodMinusOneSeed(trRaw.get(), trSmoothed.get(), size, period, stream);

  computeDiKernel<<<grid, block, 0, stream>>>(dmSmoothed.get(), trSmoothed.get(), output, size, period);
  CUDA_CHECK(cudaGetLastError());
}

void tacuda::PlusDI::calculate(const float *input, float *output,
                       int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  const float *close = input + 2 * size;
  calculate(high, low, close, output, size, stream);
}
