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
#include <utils/LeadingNaN.cuh>
#include <utils/TacudaConfig.h>
#include <cstdint>
#include <stdexcept>
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

// A float RSI is consumed by strict pivot comparisons. Accumulate in double
// and round at the output, as TA-Lib does when its result is stored as float.
using Transform = tacuda::detail::AffineTransform2<double>;
constexpr int kTile = 16;

__device__ double rsi_value(double gain, double loss) {
  const double total = gain + loss;
  return total == 0.0 ? 50.0 : 100.0 * (gain / total);
}

template<typename T>
__device__ double price_change(const T* input, size_t index, int stride) {
  return static_cast<double>(input[index]) - static_cast<double>(input[index - stride]);
}

// Both terms propagate NaN, so an interior NaN poisons the recursion.
__device__ double gain_of(double d) { return d < 0.0 ? 0.0 : d; }
__device__ double loss_of(double d) { return d > 0.0 ? 0.0 : -d; }

template<typename T>
__global__ void initial_avg_kernel(const T* __restrict__ input,
                                   double* __restrict__ avgG0,
                                   double* __restrict__ avgL0,
                                   int* __restrict__ seedIndex,
                                   int seriesLen, int batchCount, int period)
{
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= batchCount) return;
  const T* row = input + static_cast<size_t>(s) * seriesLen;
  const int first = tacuda::detail::leading_nan_count(row, seriesLen);
  const int seed = period < seriesLen - first ? first + period : seriesLen;
  seedIndex[s] = seed;

  double gain = 0.0, loss = 0.0;
  if (seed < seriesLen) {
    for (int t = first + 1; t <= seed; ++t) {
      const double d = price_change(row, t, 1);
      gain += gain_of(d);
      loss += loss_of(d);
    }
  }
  const double invP = 1.0 / period;
  avgG0[s] = gain * invP;
  avgL0[s] = loss * invP;
}

template<typename T>
__global__ void build_transforms_kernel(const T* __restrict__ input,
                                        Transform* __restrict__ trans,
                                        int* __restrict__ keys,
                                        int seriesLen, int tilesPerSeries, int tiles, const int* seedIndex,
                                        double a, double invP)
{
  const int tile = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile >= tiles) return;
  const int s = tile / tilesPerSeries;
  const int begin = (tile % tilesPerSeries) * kTile;
  const int end = begin + min(kTile, seriesLen - begin);
  const T* row = input + static_cast<size_t>(s) * seriesLen;
  Transform v{1.0, 0.0, 0.0};
  for (int t = begin; t < end; ++t) {
    if (t > seedIndex[s]) {
      const double d = price_change(row, t, 1);
      v.A *= a;
      v.bG = a * v.bG + gain_of(d) * invP;
      v.bL = a * v.bL + loss_of(d) * invP;
    }
  }
  if (keys) keys[tile] = s;
  trans[tile] = v;
}

template<typename T>
__global__ void finalize_rsi_kernel(const T* __restrict__ input,
                                    const Transform* __restrict__ pref,
                                    const double* __restrict__ avgG0,
                                    const double* __restrict__ avgL0,
                                    T* __restrict__ out,
                                    int seriesLen, int tilesPerSeries, int tiles,
                                    const int* seedIndex, int period, double invP)
{
  const int tile = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile >= tiles) return;
  const int s = tile / tilesPerSeries;
  const int begin = (tile % tilesPerSeries) * kTile;
  const int end = begin + min(kTile, seriesLen - begin);
  const size_t base = static_cast<size_t>(s) * seriesLen;
  double gain = avgG0[s], loss = avgL0[s];
  if (begin > seedIndex[s]) {
    const Transform carry = pref[tile - 1];
    gain = carry.A * gain + carry.bG;
    loss = carry.A * loss + carry.bL;
  }
  for (int t = begin; t < end; ++t) {
    if (t < seedIndex[s]) { out[base + t] = T(nanf("")); continue; }
    if (t > seedIndex[s]) {
      const double d = price_change(input + base, t, 1);
      gain = (gain * (period - 1) + gain_of(d)) * invP;
      loss = (loss * (period - 1) + loss_of(d)) * invP;
    }
    out[base + t] = static_cast<T>(rsi_value(gain, loss));
  }
}

// Fallback kernel: one thread = one series (for time-major layout)
template<typename T>
__device__ __forceinline__
void rsi_wilder_one_series(const T* __restrict__ in,
                           T* __restrict__ out,
                           int n, int period, int stride)
{
  const int first = tacuda::detail::leading_nan_count(in, n, stride);
  for (int i = 0; i < first; ++i) out[(size_t)i * stride] = T(nanf(""));
  if (first == n) return;
  in += (size_t)first * stride;
  out += (size_t)first * stride;
  n -= first;
  for (int i = 0; i < period && i < n; ++i) out[(size_t)i * stride] = T(nanf(""));
  if (n <= period) return;

  double sumG = 0.0, sumL = 0.0;
  for (int i = 1; i <= period; ++i) {
    const double d = price_change(in, static_cast<size_t>(i) * stride, stride);
    sumG += gain_of(d);
    sumL += loss_of(d);
  }
  const double invP = 1.0 / period;
  double avgG = sumG * invP, avgL = sumL * invP;
  out[static_cast<size_t>(period) * stride] = static_cast<T>(rsi_value(avgG, avgL));
  for (int t = period + 1; t < n; ++t) {
    const size_t idx = static_cast<size_t>(t) * stride;
    const double d = price_change(in, idx, stride);
    avgG = (avgG * (period - 1) + gain_of(d)) * invP;
    avgL = (avgL * (period - 1) + loss_of(d)) * invP;
    out[idx] = static_cast<T>(rsi_value(avgG, avgL));
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

  // Tile replay reads prices while other tiles write their output. Preserve
  // the prices when the caller supplies overlapping device buffers.
  const auto src = reinterpret_cast<std::uintptr_t>(input);
  const auto dst = reinterpret_cast<std::uintptr_t>(output);
  const size_t bytes = static_cast<size_t>(size) * sizeof(T);
  DeviceBufferPtrAsync<T> inputCopy;
  if ((src < dst ? dst - src : src - dst) < bytes) {
    inputCopy = acquireDeviceBufferAsync<T>(size, stream);
    CUDA_CHECK(cudaMemcpyAsync(inputCopy.get(), input, bytes, cudaMemcpyDeviceToDevice, stream));
    input = inputCopy.get();
  }

  // No NaN pre-fill: both paths below write every element, warm-up NaNs
  // included (rsi_wilder_one_series / finalize_rsi_kernel).

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

  // Scan one transform per tile. Replaying the local Wilder steps costs
  // less bandwidth than passing three doubles per bar through the scan.
  const int tilesPerSeries = 1 + (seriesLen - 1) / kTile;
  const int tiles = tilesPerSeries * batchCount;
  auto dAvgG0 = acquireDeviceBufferAsync<double>(batchCount, stream);
  auto dAvgL0 = acquireDeviceBufferAsync<double>(batchCount, stream);
  auto dSeedIndex = acquireDeviceBufferAsync<int>(batchCount, stream);
  auto dTrans = acquireDeviceBufferAsync<Transform>(tiles, stream);
  DeviceBufferPtrAsync<int> dKeys;
  if (batchCount > 1) dKeys = acquireDeviceBufferAsync<int>(tiles, stream);

  const int seedBlock = 128;
  initial_avg_kernel<<<(batchCount + seedBlock - 1) / seedBlock, seedBlock, 0, stream>>>(
      input, dAvgG0.get(), dAvgL0.get(), dSeedIndex.get(), seriesLen, batchCount, period);
  CUDA_CHECK(cudaGetLastError());

  const dim3 grid = defaultGrid(tiles), block = defaultBlock();
  const double invP = 1.0 / period;
  build_transforms_kernel<<<grid, block, 0, stream>>>(
      input, dTrans.get(), dKeys.get(), seriesLen, tilesPerSeries, tiles, dSeedIndex.get(),
      (period - 1) * invP, invP);
  CUDA_CHECK(cudaGetLastError());

  using Compose = tacuda::detail::AffineCompose2<double>;
  if (batchCount == 1) {
    tacuda::detail::affine_inclusive_scan<Transform, Compose>(
        dTrans.get(), dTrans.get(), tiles, stream);
  } else {
    tacuda::detail::affine_inclusive_scan_by_key<Transform, Compose>(
        dKeys.get(), dTrans.get(), dTrans.get(), tiles, stream);
  }

  finalize_rsi_kernel<<<grid, block, 0, stream>>>(
      input, dTrans.get(), dAvgG0.get(), dAvgL0.get(), output,
      seriesLen, tilesPerSeries, tiles, dSeedIndex.get(), period, invP);
  CUDA_CHECK(cudaGetLastError());
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
