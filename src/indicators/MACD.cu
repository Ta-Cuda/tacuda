/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// MACD.cu — accelerated MACD with segmented scan for EMA (series-major)
// Public API unchanged.

#include <stdexcept>
#include <indicators/MACD.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <utils/TacudaConfig.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>

// ===== batched cfg, compatible in style with MAMA/RSI =====
namespace {

struct MACDBatchCfg {
  int  batchCount = 1;
  int  seriesLen  = 0;
  bool timeMajor  = false;   // for time-major we keep the sequential path
};

static MACDBatchCfg getMACDCfg(int size) {
  auto& gcfg = tacuda::Config::instance();
  MACDBatchCfg c;
  c.batchCount = gcfg.batch.count;
  c.seriesLen  = gcfg.batch.seriesLen;
  c.timeMajor  = gcfg.batch.timeMajor;
  if (c.seriesLen <= 0) c.seriesLen = size / c.batchCount;
  if (c.batchCount * c.seriesLen != size) { c.batchCount = 1; c.seriesLen = size; }
  return c;
}

// ===== affine EMA model: y_t = a * y_{t-1} + (alpha * x_t) =====
using Affine1 = tacuda::detail::AffineTransform1<float>;
using AffineCompose = tacuda::detail::AffineCompose1<float>;

// --- initial SMA for EMA (TA-Lib parity): over window [0..period-1], at index (period-1)
__global__ void ema_initial_sma_kernel(const float* __restrict__ in,
                                       float* __restrict__ y0,
                                       int seriesLen, int batchCount, int period)
{
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= batchCount) return;

#ifdef TACUDA_MACD_TIME_MAJOR
  // time-major fallback does not use this path
  y0[s] = 0.f;
#else
  const float* base = in + (size_t)s * seriesLen;
  float sum = 0.f;
  int last = min(seriesLen, period);
  for (int i = 0; i < last; ++i) sum += base[i];
  y0[s] = sum / (float)period;
#endif
}

// --- build transforms and keys for segmented scan (series-major)
__global__ void ema_build_transforms_kernel(const float* __restrict__ in,
                                            Affine1* __restrict__ trans,
                                            int* __restrict__ keys,
                                            int seriesLen, int batchCount, int period,
                                            float a, float alpha)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N   = seriesLen * batchCount;
  if (idx >= N) return;

  int s = idx / seriesLen;
  int t = idx - s * seriesLen;
  keys[idx] = s;

  Affine1 v;
  if (t <= period - 1) {
    v.A = 1.f; v.b = 0.f;  // identity up to and including (period-1): y0 is stored here
  } else {
    const float* base = in + (size_t)s * seriesLen;
    v.A = a;
    v.b = alpha * base[t];
  }
  trans[idx] = v;
}

// --- finalize EMA from transform prefixes
__global__ void ema_finalize_kernel(const Affine1* __restrict__ pref,
                                    const float*  __restrict__ y0,
                                    float* __restrict__ emaOut,
                                    int seriesLen, int batchCount, int period)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N   = seriesLen * batchCount;
  if (idx >= N) return;

  int s = idx / seriesLen;
  int t = idx - s * seriesLen;

  if (t < period - 1) {
    emaOut[idx] = nanf("");
  } else if (t == period - 1) {
    emaOut[idx] = y0[s];
  } else {
    Affine1 a = pref[idx];
    emaOut[idx] = a.A * y0[s] + a.b;
  }
}

// --- diff EMAfast - EMAslow; write only where slowEMA is valid
__global__ void macd_diff_batched_kernel(const float* __restrict__ emaFast,
                                         const float* __restrict__ emaSlow,
                                         float* __restrict__ macd,
                                         int seriesLen, int batchCount, int slowPeriod)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N   = seriesLen * batchCount;
  if (idx >= N) return;

  int t = idx % seriesLen;
  // output pre-filled with NaN; write only valid slowEMA
  if (t >= slowPeriod - 1) {
    macd[idx] = emaFast[idx] - emaSlow[idx];
  }
}

// --- fallback EMA: "one series = one thread" (for time-major or just in case)
__device__ __forceinline__
void ema_seq_one_series(const float* __restrict__ in,
                        float* __restrict__ out, int n, int period, int stride)
{
  // NaN up to period-1
  for (int i = 0; i < period - 1 && i < n; ++i) out[(size_t)i * stride] = nanf("");
  if (n <= 0) return;

  const float alpha = 2.f / (period + 1.f);
  const float a = 1.f - alpha;

  // SMA at index (period-1)
  float sum = 0.f;
  int last = min(n, period);
  for (int i = 0; i < last; ++i) sum += in[(size_t)i * stride];
  if (period - 1 < n) out[(size_t)(period - 1) * stride] = sum / (float)period;

  // then EMA recursion
  float prev = (period - 1 < n) ? out[(size_t)(period - 1) * stride] : 0.f;
  for (int t = period; t < n; ++t) {
    float x = in[(size_t)t * stride];
    float y = a * prev + alpha * x;
    out[(size_t)t * stride] = y;
    prev = y;
  }
}

__global__ void ema_seq_many_series_kernel(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int seriesLen, int batchCount, int period)
{
  for (int s = blockIdx.x * blockDim.x + threadIdx.x;
       s < batchCount;
       s += blockDim.x * gridDim.x)
  {
#ifdef TACUDA_MACD_TIME_MAJOR
    const int   stride = batchCount;              // [t][batch]
    const float* pin   = in  + s;
    float*       pout  = out + s;
#else
    const int   stride = 1;                       // [series][t]
    const float* pin   = in  + (size_t)s * seriesLen;
    float*       pout  = out + (size_t)s * seriesLen;
#endif
    ema_seq_one_series(pin, pout, seriesLen, period, stride);
  }
}

} // namespace

// ===== public API — unchanged =====
namespace tacuda {

MACD::MACD(int fastPeriod, int slowPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod) {}

void MACD::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
  if (fastPeriod <= 0 || slowPeriod <= 0) {
    throw std::invalid_argument("MACD: invalid periods");
  }
  // TA-Lib normalizes reversed periods (slow = max) instead of rejecting
  // them; equal periods are legal (the MACD line is zero).
  if (slowPeriod < fastPeriod) {
    int t = fastPeriod;
    fastPeriod = slowPeriod;
    slowPeriod = t;
  }

  // Initialize with NaN (as in TA-Lib; valid values start at slowPeriod-1 per series)
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  // Batch settings
  MACDBatchCfg cfg = getMACDCfg(size);
  const int batchCount = cfg.batchCount;
  const int seriesLen  = cfg.seriesLen;

  // Fallback path (time-major): sequential EMA per-series
  if (cfg.timeMajor) {
    auto dEmaF = acquireDeviceBufferAsync<float>(size, stream);
    auto dEmaS = acquireDeviceBufferAsync<float>(size, stream);

    int minGrid=0, blockSize=0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, ema_seq_many_series_kernel, 0, 0));
    blockSize = std::min(std::max(128, ((blockSize + 31)/32)*32), 256);
    int gridSize = (batchCount + blockSize - 1) / blockSize;

    int dev=0, sms=0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    gridSize = std::max(1, std::min(gridSize, sms * 4));

    // fast EMA
    ema_seq_many_series_kernel<<<gridSize, blockSize, 0, stream>>>(input, dEmaF.get(), seriesLen, batchCount, fastPeriod);
    CUDA_CHECK(cudaGetLastError());
    // slow EMA
    ema_seq_many_series_kernel<<<gridSize, blockSize, 0, stream>>>(input, dEmaS.get(), seriesLen, batchCount, slowPeriod);
    CUDA_CHECK(cudaGetLastError());

    // diff (write only from index slowPeriod-1 in each series)
    dim3 block = defaultBlock();
    dim3 grid  = defaultGrid(size);
    macd_diff_batched_kernel<<<grid, block, 0, stream>>>(dEmaF.get(), dEmaS.get(), output, seriesLen, batchCount, slowPeriod);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // ===== Fast path: series-major + segmented scan for each EMA =====
  auto dEmaF  = acquireDeviceBufferAsync<float>(size, stream);
  auto dEmaS  = acquireDeviceBufferAsync<float>(size, stream);

  // Buffers reused for both periods
  auto dTrans = acquireDeviceBufferAsync<Affine1>(size, stream);
  auto dKeys  = acquireDeviceBufferAsync<int>(size, stream);
  auto dPref  = acquireDeviceBufferAsync<Affine1>(size, stream);

  // -------- EMA(fast) --------
  {
    auto dY0 = acquireDeviceBufferAsync<float>(batchCount, stream);

    // y0 = SMA(0..P-1)
    {
      int block = 128;
      int grid  = (batchCount + block - 1) / block;
      ema_initial_sma_kernel<<<grid, block, 0, stream>>>(input, dY0.get(), seriesLen, batchCount, fastPeriod);
      CUDA_CHECK(cudaGetLastError());
    }
    // build transforms
    {
      const float alpha = 2.f / (fastPeriod + 1.f);
      const float a     = 1.f - alpha;
      dim3 block = defaultBlock();
      dim3 grid  = defaultGrid(size);
      ema_build_transforms_kernel<<<grid, block, 0, stream>>>(
          input, dTrans.get(), dKeys.get(), seriesLen, batchCount, fastPeriod, a, alpha);
      CUDA_CHECK(cudaGetLastError());
    }
    // segmented inclusive_scan_by_key
    {
      tacuda::detail::affine_inclusive_scan_by_key<Affine1, AffineCompose>(
          dKeys.get(), dTrans.get(), dPref.get(), size, stream);
    }
    // finalize EMA
    {
      dim3 block = defaultBlock();
      dim3 grid  = defaultGrid(size);
      ema_finalize_kernel<<<grid, block, 0, stream>>>(dPref.get(), dY0.get(), dEmaF.get(),
                                                      seriesLen, batchCount, fastPeriod);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // -------- EMA(slow) --------
  {
    auto dY0 = acquireDeviceBufferAsync<float>(batchCount, stream);
    {
      int block = 128;
      int grid  = (batchCount + block - 1) / block;
      ema_initial_sma_kernel<<<grid, block, 0, stream>>>(input, dY0.get(), seriesLen, batchCount, slowPeriod);
      CUDA_CHECK(cudaGetLastError());
    }
    {
      const float alpha = 2.f / (slowPeriod + 1.f);
      const float a     = 1.f - alpha;
      dim3 block = defaultBlock();
      dim3 grid  = defaultGrid(size);
      ema_build_transforms_kernel<<<grid, block, 0, stream>>>(
          input, dTrans.get(), dKeys.get(), seriesLen, batchCount, slowPeriod, a, alpha);
      CUDA_CHECK(cudaGetLastError());
    }
    {
      tacuda::detail::affine_inclusive_scan_by_key<Affine1, AffineCompose>(
          dKeys.get(), dTrans.get(), dPref.get(), size, stream);
    }
    {
      dim3 block = defaultBlock();
      dim3 grid  = defaultGrid(size);
      ema_finalize_kernel<<<grid, block, 0, stream>>>(dPref.get(), dY0.get(), dEmaS.get(),
                                                      seriesLen, batchCount, slowPeriod);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // MACD = EMAfast - EMAslow, valid from (slowPeriod-1) per series
  {
    dim3 block = defaultBlock();
    dim3 grid  = defaultGrid(size);
    macd_diff_batched_kernel<<<grid, block, 0, stream>>>(dEmaF.get(), dEmaS.get(), output,
                                                         seriesLen, batchCount, slowPeriod);
    CUDA_CHECK(cudaGetLastError());
  }
}

} // namespace tacuda
