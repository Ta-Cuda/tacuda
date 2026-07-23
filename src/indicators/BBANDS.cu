/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// BBANDS.cu — accelerated Bollinger Bands (batched, seg-scan)
// Public API unchanged.

#include <indicators/BBANDS.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/TacudaConfig.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <algorithm>
#include <cstdlib>

// ===== batched cfg (as in MAMA/RSI/MACD) =====
namespace {

struct BBatchCfg {
  int  batchCount = 1;
  int  seriesLen  = 0;
  bool timeMajor  = false;   // fallback — sequential per-series calculation
};

static BBatchCfg getCfg(int sizeTotal) {
  auto& gcfg = tacuda::Config::instance();
  BBatchCfg c;
  c.batchCount = gcfg.batch.count;
  c.seriesLen  = gcfg.batch.seriesLen;
  c.timeMajor  = gcfg.batch.timeMajor;
  if (c.seriesLen <= 0) c.seriesLen = sizeTotal / c.batchCount;
  if (c.batchCount * c.seriesLen != sizeTotal) { c.batchCount = 1; c.seriesLen = sizeTotal; }
  return c;
}

// ===== KEYS/TRANSFORMS moved to file scope (fixes CUB compilation errors) =====

struct WelfordData {
  double mean;
  double m2;
  double count;
};

__host__ __device__ inline WelfordData makeMoment(double v) {
  return WelfordData{v, 0.0, 1.0};
}

__host__ __device__ inline WelfordData combineMoments(const WelfordData& a, const WelfordData& b) {
  if (a.count == 0.0) return b;
  if (b.count == 0.0) return a;
  WelfordData out;
  out.count = a.count + b.count;
  double delta = b.mean - a.mean;
  out.mean = (a.mean * a.count + b.mean * b.count) / out.count;
  out.m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / out.count);
  return out;
}

__host__ __device__ inline WelfordData subtractMoments(const WelfordData& total,
                                                       const WelfordData& part) {
  double remaining = total.count - part.count;
  if (remaining <= 0.0) return WelfordData{0.0, 0.0, 0.0};
  if (part.count == 0.0) return total;
  double meanRem = (total.mean * total.count - part.mean * part.count) / remaining;
  double delta = meanRem - part.mean;
  double m2Rem = total.m2 - part.m2 - delta * delta * (part.count * remaining / total.count);
  if (m2Rem < 0.0) m2Rem = 0.0;
  return WelfordData{meanRem, m2Rem, remaining};
}

__global__ void initMomentsAndKeysKernel(const float* __restrict__ in,
                                         WelfordData* __restrict__ moments,
                                         int* __restrict__ keys,
                                         int totalSize, int seriesLen) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < totalSize;
       idx += blockDim.x * gridDim.x) {
    moments[idx] = makeMoment(static_cast<double>(in[idx]));
    keys[idx] = idx / seriesLen;
  }
}

__global__ void bbands_finalize_from_moments_kernel(const WelfordData* __restrict__ prefix,
                                                    float* __restrict__ upper,
                                                    float* __restrict__ middle,
                                                    float* __restrict__ lower,
                                                    int totalSize, int seriesLen, int period,
                                                    float upMult, float dnMult) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < totalSize;
       idx += blockDim.x * gridDim.x) {
    int t = idx % seriesLen;
    if (t < period - 1) continue;
    WelfordData total = prefix[idx];
    WelfordData prev = (t >= period) ? prefix[idx - period] : WelfordData{0.0, 0.0, 0.0};
    WelfordData win = subtractMoments(total, prev);
    if (win.count <= 0.0) continue;

    double mean = win.mean;
    double var  = win.m2 / win.count;
    if (var < 0.0) var = 0.0;
    double sd = sqrt(var);

    middle[idx] = static_cast<float>(mean);
    upper [idx] = static_cast<float>(mean + static_cast<double>(upMult) * sd);
    lower [idx] = static_cast<float>(mean - static_cast<double>(dnMult) * sd);
  }
}

// time-major fallback: one thread = one series, sliding window via (sum,sum2)
__global__ void bbands_time_major_kernel(const float* __restrict__ in,
                                         float* __restrict__ upper,
                                         float* __restrict__ middle,
                                         float* __restrict__ lower,
                                         int seriesLen, int batchCount, int period,
                                         float upMult, float dnMult)
{
  for (int s = blockIdx.x * blockDim.x + threadIdx.x;
       s < batchCount;
       s += blockDim.x * gridDim.x)
  {
    const int stride = batchCount;           // [t][batch]
    const float* base = in;
    double sum = 0.0, sum2 = 0.0;

    // accumulate first period values
    int last = min(seriesLen, period);
    for (int i = 0; i < last; ++i) {
      double v = static_cast<double>(base[(size_t)i * stride + s]);
      sum  += v;
      sum2 += v * v;
    }
    if (seriesLen >= period) {
      int t = period - 1;
      double mean = sum / static_cast<double>(period);
      double var  = sum2 / static_cast<double>(period) - mean * mean;
      if (var < 0.0) var = 0.0;
      double sd = sqrt(var);
      const int outIdx = (size_t)t * stride + s;
      middle[outIdx] = static_cast<float>(mean);
      upper [outIdx] = static_cast<float>(mean + static_cast<double>(upMult) * sd);
      lower [outIdx] = static_cast<float>(mean - static_cast<double>(dnMult) * sd);

      // slide window
      for (t = period; t < seriesLen; ++t) {
        double enter = static_cast<double>(base[(size_t)t * stride + s]);
        double leave = static_cast<double>(base[(size_t)(t - period) * stride + s]);
        sum  += enter - leave;
        sum2 += enter * enter - leave * leave;

        mean = sum / static_cast<double>(period);
        var  = sum2 / static_cast<double>(period) - mean * mean;
        if (var < 0.0) var = 0.0;
        sd = sqrt(var);

        const int o = (size_t)t * stride + s;
        middle[o] = static_cast<float>(mean);
        upper [o] = static_cast<float>(mean + static_cast<double>(upMult) * sd);
        lower [o] = static_cast<float>(mean - static_cast<double>(dnMult) * sd);
      }
    }
  }
}

} // namespace

// ===== public API — unchanged =====
namespace tacuda {

BBANDS::BBANDS(int period, float upperMultiplier, float lowerMultiplier)
    : period(period), upperMultiplier(upperMultiplier), lowerMultiplier(lowerMultiplier) {}

void BBANDS::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
  calculate(input, output, output + size, output + 2 * size, size, stream);
}

void BBANDS::calculate(const float* input, float* upper, float* middle, float* lower,
                       int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("BBANDS: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; all three bands are warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(upper, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      CUDA_CHECK(cudaMemsetAsync(middle, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      CUDA_CHECK(cudaMemsetAsync(lower, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }

  // Unpack batch settings; if inconsistent with size — fall back to single series
  BBatchCfg cfg = getCfg(size);
  const int batchCount = cfg.batchCount;
  const int seriesLen  = cfg.seriesLen;

  // Initialize with NaN (as in TA-Lib: first period-1 values are NaN)
  CUDA_CHECK(cudaMemsetAsync(upper, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
  CUDA_CHECK(cudaMemsetAsync(middle, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
  CUDA_CHECK(cudaMemsetAsync(lower, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));

  // ===== Fallback for time-major: sequential per-series path =====
  if (cfg.timeMajor) {
    int minGrid=0, blockSize=0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, bbands_time_major_kernel, 0, 0));
    blockSize = std::min(std::max(128, ((blockSize + 31)/32)*32), 256);
    int gridSize = (batchCount + blockSize - 1) / blockSize;

    int dev=0, sms=0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    gridSize = std::max(1, std::min(gridSize, sms * 4));

    bbands_time_major_kernel<<<gridSize, blockSize, 0, stream>>>(
        input, upper, middle, lower, seriesLen, batchCount, period, upperMultiplier, lowerMultiplier);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // ===== Fast path: series-major + segmented scan by series keys =====
  auto dMoments = acquireDeviceBufferAsync<WelfordData>(size, stream);
  auto dKeys = acquireDeviceBufferAsync<int>(size, stream);

  // build initial moments and series keys
  {
    dim3 block = defaultBlock();
    dim3 grid  = defaultGrid(size);
    initMomentsAndKeysKernel<<<grid, block, 0, stream>>>(
        input, dMoments.get(), dKeys.get(), size, seriesLen);
    CUDA_CHECK(cudaGetLastError());
  }

  // segmented inclusive scans by series (Welford combine)
  {
    thrust::device_ptr<WelfordData> momentsBegin(dMoments.get());
    thrust::device_ptr<int> keysBegin(dKeys.get());
    thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream),
                                  keysBegin, keysBegin + size,
                                  momentsBegin,
                                  momentsBegin,
                                  thrust::equal_to<int>(),
                                  [] __host__ __device__(const WelfordData& a, const WelfordData& b) {
                                    return combineMoments(a, b);
                                  });
  }

  // finalize from prefix sums
  {
    dim3 block = defaultBlock();
    dim3 grid  = defaultGrid(size);
    bbands_finalize_from_moments_kernel<<<grid, block, 0, stream>>>(
        dMoments.get(), upper, middle, lower,
        size, seriesLen, period, upperMultiplier, lowerMultiplier);
    CUDA_CHECK(cudaGetLastError());
  }
}

} // namespace tacuda
