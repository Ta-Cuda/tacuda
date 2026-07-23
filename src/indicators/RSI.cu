/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// RSI.cu — accelerated version with segmented scan (series-major), API unchanged
#include <indicators/RSI.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <utils/TacudaConfig.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>

namespace {

// --------- batch cfg from unified Config ----------
struct RSIBatchCfg {
  int  batchCount = 1;
  int  seriesLen  = 0;
  bool timeMajor  = false;
};

static RSIBatchCfg getRSICfg(int size) {
  auto& gcfg = tacuda::Config::instance();
  RSIBatchCfg c;
  c.batchCount = gcfg.batch.count;
  c.seriesLen  = gcfg.batch.seriesLen;
  c.timeMajor  = gcfg.batch.timeMajor;
  if (c.seriesLen <= 0) c.seriesLen = size / c.batchCount;
  if (c.batchCount * c.seriesLen != size) { c.batchCount = 1; c.seriesLen = size; }
  return c;
}

// ===================== Templated kernels =====================

// Compute differences -> gains/losses
template<typename T>
__global__ void diffs_gain_loss_kernel(const T* __restrict__ in,
                                       T* __restrict__ gain,
                                       T* __restrict__ loss,
                                       int seriesLen, int batchCount)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N   = seriesLen * batchCount;
  if (idx >= N) return;

  int s = idx / seriesLen;
  int t = idx - s * seriesLen;

  if (t == 0) { gain[idx] = T(0); loss[idx] = T(0); return; }

  const T* base = in + (size_t)s * seriesLen;
  T cur  = base[t];
  T prev = base[t - 1];
  T d = cur - prev;
  if (isnan(d)) {
    // NaN inputs must poison the recursion (TA-Lib computes on the raw
    // values, so a NaN propagates), not be silently treated as zero change.
    gain[idx] = d;
    loss[idx] = d;
    return;
  }
  gain[idx] = d > T(0) ? d : T(0);
  loss[idx] = d < T(0) ? -d : T(0);
}

// Initial SMA for gains/losses per series
template<typename T>
__global__ void initial_avg_kernel(const T* __restrict__ gain,
                                   const T* __restrict__ loss,
                                   T* __restrict__ avgG0,
                                   T* __restrict__ avgL0,
                                   int seriesLen, int batchCount, int period)
{
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= batchCount) return;

  const T* g = gain + (size_t)s * seriesLen;
  const T* l = loss + (size_t)s * seriesLen;

  double sumG = 0.0, sumL = 0.0;
  int last = min(seriesLen - 1, period);
  for (int i = 1; i <= last; ++i) { sumG += static_cast<double>(g[i]); sumL += static_cast<double>(l[i]); }

  avgG0[s] = static_cast<T>(sumG / static_cast<double>(period));
  avgL0[s] = static_cast<T>(sumL / static_cast<double>(period));
}

// Build affine transforms and keys for segmented scan
template<typename T>
__global__ void build_transforms_kernel(const T* __restrict__ gain,
                                        const T* __restrict__ loss,
                                        tacuda::detail::AffineTransform2<T>* __restrict__ trans,
                                        int* __restrict__ keys,
                                        int seriesLen, int batchCount, int period,
                                        T a, T invP)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N   = seriesLen * batchCount;
  if (idx >= N) return;

  int s = idx / seriesLen;
  int t = idx - s * seriesLen;
  keys[idx] = s;

  tacuda::detail::AffineTransform2<T> v;
  if (t <= period) {
    v.A = T(1); v.bG = T(0); v.bL = T(0);
  } else {
    v.A  = a;
    v.bG = gain[idx] * invP;
    v.bL = loss[idx] * invP;
  }
  trans[idx] = v;
}

// Finalize RSI from prefix scans
template<typename T>
__global__ void finalize_rsi_kernel(const tacuda::detail::AffineTransform2<T>* __restrict__ pref,
                                    const T* __restrict__ avgG0,
                                    const T* __restrict__ avgL0,
                                    T* __restrict__ out,
                                    int seriesLen, int batchCount, int period)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N   = seriesLen * batchCount;
  if (idx >= N) return;

  int s = idx / seriesLen;
  int t = idx - s * seriesLen;

  if (t < period) { out[idx] = T(nanf("")); return; }

  T avgG, avgL;
  if (t == period) {
    avgG = avgG0[s];
    avgL = avgL0[s];
  } else {
    auto a = pref[idx];
    avgG = a.A * avgG0[s] + a.bG;
    avgL = a.A * avgL0[s] + a.bL;
  }

  T rsi;
  // TA-Lib: rsi = 100 * avgGain / (avgGain + avgLoss); a fully flat window
  // (both averages zero) yields 0, not 50.
  if (avgL == T(0)) rsi = (avgG == T(0)) ? T(0) : T(100);
  else if (avgG == T(0)) rsi = T(0);
  else { T rs = avgG / avgL; rsi = T(100) - T(100) / (T(1) + rs); }

  out[idx] = rsi;
}

// Fallback kernel: one thread = one series (for time-major layout)
template<typename T>
__device__ __forceinline__
void rsi_wilder_one_series(const T* __restrict__ in,
                           T* __restrict__ out,
                           int n, int period, int stride)
{
  for (int i = 0; i < period && i < n; ++i) out[(size_t)i * stride] = T(nanf(""));
  if (n <= period) return;

  double sumG = 0.0, sumL = 0.0;
  for (int i = 1; i <= period; ++i) {
    double d = static_cast<double>(in[(size_t)i * stride]) - static_cast<double>(in[(size_t)(i-1) * stride]);
    if (isnan(d)) { sumG += d; sumL += d; }  // propagate NaN inputs
    else {
      sumG += d > 0.0 ? d : 0.0;
      sumL += d < 0.0 ? -d : 0.0;
    }
  }
  double avgG = sumG / period, avgL = sumL / period;
  {
    T rsi;
    // TA-Lib: flat window (both averages zero) yields 0, not 50.
    if (avgL == 0.0) rsi = (avgG == 0.0) ? T(0) : T(100);
    else if (avgG == 0.0) rsi = T(0);
    else { double rs = avgG / avgL; rsi = static_cast<T>(100.0 - 100.0 / (1.0 + rs)); }
    out[(size_t)period * stride] = rsi;
  }
  const double invP = 1.0 / (double)period;
  for (int t = period + 1; t < n; ++t) {
    double d = static_cast<double>(in[(size_t)t * stride]) - static_cast<double>(in[(size_t)(t-1) * stride]);
    double g, l;
    if (isnan(d)) { g = d; l = d; }
    else {
      g = d > 0.0 ? d : 0.0;
      l = d < 0.0 ? -d : 0.0;
    }
    avgG = (avgG * (period - 1) + g) * invP;
    avgL = (avgL * (period - 1) + l) * invP;
    T rsi;
    if (avgL == 0.0) rsi = (avgG == 0.0) ? T(0) : T(100);
    else if (avgG == 0.0) rsi = T(0);
    else { double rs = avgG / avgL; rsi = static_cast<T>(100.0 - 100.0 / (1.0 + rs)); }
    out[(size_t)t * stride] = rsi;
  }
}

template<typename T>
__global__ void rsiWilderManySeriesKernel(const T* __restrict__ input,
                                          T* __restrict__ output,
                                          int seriesLen, int batchCount, int period)
{
  for (int s = blockIdx.x * blockDim.x + threadIdx.x;
       s < batchCount;
       s += blockDim.x * gridDim.x)
  {
    const int   stride = 1;
    const T* in    = input  + (size_t)s * seriesLen;
    T*       out   = output + (size_t)s * seriesLen;
    rsi_wilder_one_series<T>(in, out, seriesLen, period, stride);
  }
}

// ===================== Templated implementation =====================

template<typename T>
void rsi_calculate_impl(int period, const T* input, T* output, int size, cudaStream_t stream) {
  if (period <= 0) throw std::invalid_argument("RSI: invalid period");
  if (period >= size) {
    // TA-Lib succeeds with zero output elements when the input is shorter
    // than the lookback; the whole line is warm-up NaNs.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(T), stream));
    return;
  }

  // Initialize NaN
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(T), stream));

  // Batch settings
  RSIBatchCfg cfg = getRSICfg(size);
  const int batchCount = cfg.batchCount;
  const int seriesLen  = cfg.seriesLen;

  // time-major path: per-series sequential (scan not applicable to interleaved keys)
  if (cfg.timeMajor) {
    int minGrid=0, blockSize=0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, rsiWilderManySeriesKernel<T>, 0, 0));
    blockSize = std::min(std::max(128, ((blockSize + 31)/32)*32), 256);
    int gridSize = (batchCount + blockSize - 1) / blockSize;
    int dev=0, sms=0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    gridSize = std::max(1, std::min(gridSize, sms * 4));

    rsiWilderManySeriesKernel<T><<<gridSize, blockSize, 0, stream>>>(input, output, seriesLen, batchCount, period);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  using Affine2 = tacuda::detail::AffineTransform2<T>;
  using AffineCompose = tacuda::detail::AffineCompose2<T>;

  // ------ Fast path: series-major + segmented scan ------
  auto dGain  = acquireDeviceBufferAsync<T>(size, stream);
  auto dLoss  = acquireDeviceBufferAsync<T>(size, stream);
  auto dAvgG0 = acquireDeviceBufferAsync<T>(batchCount, stream);
  auto dAvgL0 = acquireDeviceBufferAsync<T>(batchCount, stream);
  auto dTrans = acquireDeviceBufferAsync<Affine2>(size, stream);
  auto dKeys  = acquireDeviceBufferAsync<int>(size, stream);
  auto dPref  = acquireDeviceBufferAsync<Affine2>(size, stream);

  // 1) differences -> gains/losses
  {
    dim3 block = defaultBlock();
    dim3 grid  = defaultGrid(size);
    diffs_gain_loss_kernel<T><<<grid, block, 0, stream>>>(input, dGain.get(), dLoss.get(), seriesLen, batchCount);
    CUDA_CHECK(cudaGetLastError());
  }

  // 2) initial SMA for each series
  {
    int block = 128;
    int grid  = (batchCount + block - 1) / block;
    initial_avg_kernel<T><<<grid, block, 0, stream>>>(dGain.get(), dLoss.get(), dAvgG0.get(), dAvgL0.get(),
                                                   seriesLen, batchCount, period);
    CUDA_CHECK(cudaGetLastError());
  }

  // 3) build transforms and keys
  {
    T a    = static_cast<T>((double)(period - 1) / (double)period);
    T invP = static_cast<T>(1.0 / (double)period);
    dim3 block = defaultBlock();
    dim3 grid  = defaultGrid(size);
    build_transforms_kernel<T><<<grid, block, 0, stream>>>(dGain.get(), dLoss.get(), dTrans.get(), dKeys.get(),
                                                        seriesLen, batchCount, period, a, invP);
    CUDA_CHECK(cudaGetLastError());
  }

  // 4) segmented inclusive_scan by keys (per series)
  {
    tacuda::detail::affine_inclusive_scan_by_key<Affine2, AffineCompose>(
        dKeys.get(), dTrans.get(), dPref.get(), size, stream);
  }

  // 5) finalize RSI from prefix scans
  {
    dim3 block = defaultBlock();
    dim3 grid  = defaultGrid(size);
    finalize_rsi_kernel<T><<<grid, block, 0, stream>>>(dPref.get(), dAvgG0.get(), dAvgL0.get(),
                                                    output, seriesLen, batchCount, period);
    CUDA_CHECK(cudaGetLastError());
  }
}

} // namespace

// ===================== Explicit instantiations =====================

namespace tacuda {

// --- RSIT<float> ---
template<>
void RSIT<float>::calculate(const float* input, float* output, int size, cudaStream_t stream) {
    rsi_calculate_impl<float>(period, input, output, size, stream);
}

// --- RSIT<double> ---
template<>
void RSIT<double>::calculate(const double* input, double* output, int size, cudaStream_t stream) {
    rsi_calculate_impl<double>(period, input, output, size, stream);
}

// --- Backward-compatible RSI (float, inherits from Indicator) ---
RSI::RSI(int period) : period(period) {}

void RSI::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    rsi_calculate_impl<float>(period, input, output, size, stream);
}

template class RSIT<float>;
template class RSIT<double>;

} // namespace tacuda
