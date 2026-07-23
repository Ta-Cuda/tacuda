/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// src/indicators/ATR.cu
#include <indicators/ATR.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <utils/TacudaConfig.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>

// ---- batch config from environment (series consecutive, series-major) ----
namespace {
struct ATRBatchCfg {
    int batchCount = 1;
    int seriesLen  = 0;   // 0 => single series of 'size'
};
static ATRBatchCfg getBatchCfg(int size) {
    auto& gcfg = tacuda::Config::instance();
    ATRBatchCfg cfg;
    cfg.batchCount = gcfg.batch.count;
    cfg.seriesLen  = gcfg.batch.seriesLen;
    if (cfg.batchCount > 1 && cfg.seriesLen > 0 && cfg.batchCount * cfg.seriesLen == size) return cfg;
    cfg.batchCount = 1; cfg.seriesLen = size;
    return cfg;
}

// ------------------ TR (True Range) for batches ------------------
__global__ void trueRangeBatchedKernel(const float* __restrict__ high,
                                       const float* __restrict__ low,
                                       const float* __restrict__ close,
                                       float* __restrict__ tr,
                                       int seriesLen, int batchCount)
{
    const int size = seriesLen * batchCount;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const int s = i / seriesLen;     // series id
    const int t = i - s * seriesLen; // index within series

    if (t == 0) {
        tr[i] = high[i] - low[i];
    } else {
        const int prev = i - 1;
        // important: prev belongs to same series, since series are laid consecutively
        const float pc = close[prev];
        const float a = high[i] - low[i];
        const float b = fabsf(high[i] - pc);
        const float c = fabsf(low[i]  - pc);
        tr[i] = fmaxf(a, fmaxf(b, c));
    }
}

// --------- initial ATR₀ (SMA TR[1..period]) one thread per series ---------
__global__ void initialAtrKernel(const float* __restrict__ tr,
                                 float* __restrict__ atr0,
                                 int seriesLen, int batchCount, int period)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= batchCount) return;
    const int base = s * seriesLen;
    if (seriesLen <= period) { atr0[s] = nanf(""); return; }

    double sum = 0.0;
    // TA-Lib style for ATR: sum TR[1..period] (period values, starting from index 1)
    #pragma unroll
    for (int k = 1; k <= period; ++k) sum += static_cast<double>(tr[base + k]);
    atr0[s] = static_cast<float>(sum / static_cast<double>(period));
}

// --------- build keys (series id) and affine coefficients ---------
using AffinePair = tacuda::detail::AffineTransform1<float>;
using AffineCompose = tacuda::detail::AffineCompose1<float>;

__global__ void buildKeysAndTransformsKernel(const float* __restrict__ tr,
                                             int* __restrict__ keys,
                                             AffinePair* __restrict__ xform,
                                             int seriesLen, int batchCount, int period)
{
    const int size = seriesLen * batchCount;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const int s = i / seriesLen;
    const int t = i - s * seriesLen;
    keys[i] = s;

    // Default: identity transform (before and at start point)
    AffinePair v{1.0f, 0.0f};
    const int startIdx = period;
    if (t > startIdx) {
        const float a = (static_cast<float>(period) - 1.0f) / static_cast<float>(period);
        const float b = tr[i] / static_cast<float>(period);
        v = {a, b};
    }
    xform[i] = v;
}

// --------- apply prefix to ATR₀ and write result ---------
__global__ void applyAffineKernel(const AffinePair* __restrict__ pref,
                                  const float* __restrict__ atr0,
                                  float* __restrict__ out, // ATR
                                  int seriesLen, int batchCount, int period)
{
    const int size = seriesLen * batchCount;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const int s = i / seriesLen;
    const int t = i - s * seriesLen;
    const int startIdx = period;

    if (t < startIdx) {
        out[i] = nanf("");
    } else if (t == startIdx) {
        out[i] = atr0[s];
    } else {
        const AffinePair p = pref[i];
        out[i] = p.A * atr0[s] + p.b;
    }
}

} // namespace

// ================== Public API (unchanged) ==================
namespace tacuda {

ATR::ATR(int period, float initial) : period(period), initial(initial) {}

void ATR::calculate(const float* high, const float* low, const float* close,
                    float* output, int size, cudaStream_t stream) noexcept(false)
{
    if (period <= 0) {
        throw std::invalid_argument("ATR: invalid period/size");
    }
    if (period >= size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    const ATRBatchCfg cfg = getBatchCfg(size);
    const int batchCount  = cfg.batchCount;
    const int seriesLen   = cfg.seriesLen; // when batchCount==1 equals size

    if (seriesLen <= period) {
        throw std::invalid_argument("ATR: series length must be > period");
    }

    // TR buffer
    auto trBuf   = acquireDeviceBufferAsync<float>(size, stream);
    auto atr0Buf = acquireDeviceBufferAsync<float>(batchCount, stream);

    // 1) True Range at scale
    {
        dim3 block = defaultBlock();
        dim3 grid  = defaultGrid(size);
        trueRangeBatchedKernel<<<grid, block, 0, stream>>>(high, low, close, trBuf.get(),
                                                           seriesLen, batchCount);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2) Initial ATR₀ one thread per series (cheap, period ~ 14)
    {
        const int threads = 256;
        const int blocks  = (batchCount + threads - 1) / threads;
        initialAtrKernel<<<blocks, threads, 0, stream>>>(trBuf.get(), atr0Buf.get(),
                                                         seriesLen, batchCount, period);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3) Prepare keys (series id) and affine coefficients (A,B)
    auto keysBuf   = acquireDeviceBufferAsync<int>(size, stream);
    auto xformBuf  = acquireDeviceBufferAsync<AffinePair>(size, stream);
    auto prefBuf   = acquireDeviceBufferAsync<AffinePair>(size, stream);

    {
        dim3 block = defaultBlock();
        dim3 grid  = defaultGrid(size);
        buildKeysAndTransformsKernel<<<grid, block, 0, stream>>>(trBuf.get(),
                                                                  keysBuf.get(),
                                                                  xformBuf.get(),
                                                                  seriesLen, batchCount, period);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4) segmented inclusive_scan by keys (series) with affine function composition
    {
        tacuda::detail::affine_inclusive_scan_by_key<AffinePair, AffineCompose>(
            keysBuf.get(), xformBuf.get(), prefBuf.get(), size, stream);
    }

    // 5) apply prefix to ATR₀ and write result
    {
        dim3 block = defaultBlock();
        dim3 grid  = defaultGrid(size);
        applyAffineKernel<<<grid, block, 0, stream>>>(prefBuf.get(), atr0Buf.get(),
                                                      output, seriesLen, batchCount, period);
        CUDA_CHECK(cudaGetLastError());
    }
}

// compatible shortcut: input = [high | low | close]
void ATR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false)
{
    const float* high  = input;
    const float* low   = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}

} // namespace tacuda
