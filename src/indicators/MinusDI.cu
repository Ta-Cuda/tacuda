/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MinusDI.h>
#include <utils/AffineScan.cuh>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace {

using AffinePair = tacuda::detail::AffineTransform1<double>;
using AffineCompose = tacuda::detail::AffineCompose1<double>;

struct ToDouble {
  __host__ __device__ double operator()(float v) const {
    return static_cast<double>(v);
  }
};

__global__ void buildMinusDmAndTrKernel(const float* __restrict__ high,
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
  dm[i] = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;

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
                                     double seed,
                                     int seedIdx,
                                     int nPairs) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nPairs) return;
  AffinePair p = pref[k];
  y[seedIdx + 1 + k] = static_cast<float>(p.A * seed + p.b);
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
  double seed = thrust::transform_reduce(
      thrust::cuda::par.on(stream),
      thrust::device_pointer_cast(x) + 1,
      thrust::device_pointer_cast(x) + period,
      ToDouble{}, 0.0, thrust::plus<double>{});

  CUDA_CHECK(cudaMemsetAsync(y, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
  const int seedIdx = period - 1;
  {
    float seedVal = static_cast<float>(seed);
    CUDA_CHECK(cudaMemcpyAsync(y + seedIdx, &seedVal, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  }

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

  finalizeSmoothKernel<<<grid, block, 0, stream>>>(pref.get(), y, seed, seedIdx, nPairs);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace

tacuda::MinusDI::MinusDI(int period) : period(period) {}

void tacuda::MinusDI::calculate(const float *high, const float *low, const float *close,
                        float *output, int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("MinusDI: invalid period");
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
  buildMinusDmAndTrKernel<<<grid, block, 0, stream>>>(high, low, close, dmRaw.get(), trRaw.get(), size);
  CUDA_CHECK(cudaGetLastError());

  smoothFromPeriodMinusOneSeed(dmRaw.get(), dmSmoothed.get(), size, period, stream);
  smoothFromPeriodMinusOneSeed(trRaw.get(), trSmoothed.get(), size, period, stream);

  computeDiKernel<<<grid, block, 0, stream>>>(dmSmoothed.get(), trSmoothed.get(), output, size, period);
  CUDA_CHECK(cudaGetLastError());
}

void tacuda::MinusDI::calculate(const float *input, float *output,
                        int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  const float *close = input + 2 * size;
  calculate(high, low, close, output, size, stream);
}
