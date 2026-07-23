/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/Stochastic.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/TacudaConfig.h>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace {

// --------- Environment-based batch configuration ----------
struct StochBatchCfg {
    int batchCount = 1;
    int seriesLen  = 0;
    bool timeMajor = false;
};

static StochBatchCfg getStochCfg(int size) {
    auto& gcfg = tacuda::Config::instance();
    StochBatchCfg c;
    c.batchCount = gcfg.batch.count;
    c.seriesLen  = gcfg.batch.seriesLen;
    c.timeMajor  = gcfg.batch.timeMajor;
    if (c.seriesLen <= 0) {
        c.seriesLen = size / c.batchCount;
    }
    if (c.batchCount * c.seriesLen != size) { 
        c.batchCount = 1; 
        c.seriesLen = size; 
    }
    return c;
}

// --------- Stochastic calculation for one series ----------
// Implements TA-Lib STOCH algorithm:
// 1. Calculate FASTK = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
// 2. Smooth FASTK with SMA(d_period) -> SLOWK (%K output)
// 3. Smooth SLOWK with SMA(d_period) -> SLOWD (%D output)
__device__ __forceinline__
void stochastic_one_series(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          float* __restrict__ kOut,
                          float* __restrict__ dOut,
                          float* __restrict__ tempFastK,
                          int n,
                          int kPeriod,
                          int dPeriod,
                          int stride)
{
    const float qnan = nanf("");
    
    // Lookback period for STOCH = kPeriod + 2*(dPeriod - 1)
    // First kPeriod-1 values of FASTK are NaN
    // Then dPeriod-1 more for first SLOWK
    // Then dPeriod-1 more for first SLOWD
    const int lookbackFastK = kPeriod - 1;
    const int lookbackSlowK = lookbackFastK + dPeriod - 1;
    const int lookbackSlowD = lookbackSlowK + dPeriod - 1;
    
    // Initialize outputs to NaN
    for (int i = 0; i < n && i <= lookbackSlowD; ++i) {
        kOut[(size_t)i * stride] = qnan;
        dOut[(size_t)i * stride] = qnan;
    }
    
    if (n <= lookbackSlowD) return;
    
    // Step 1: Calculate FASTK into temporary buffer
    // FASTK starts at index lookbackFastK
    for (int i = lookbackFastK; i < n; ++i) {
        // Find highest high and lowest low in the last kPeriod bars
        float highest = high[(size_t)i * stride];
        float lowest = low[(size_t)i * stride];
        
        for (int j = 1; j < kPeriod; ++j) {
            float h = high[(size_t)(i - j) * stride];
            float l = low[(size_t)(i - j) * stride];
            if (h > highest) highest = h;
            if (l < lowest) lowest = l;
        }
        
        // Calculate FASTK (TA-Lib emits 0.0 when the range is zero,
        // see ta_STOCHF.c: "if(diff!=0.0) ... else 0.0")
        float denom = highest - lowest;
        float fastK;
        if (denom == 0.0f) {
            fastK = 0.0f;
        } else {
            fastK = 100.0f * (close[(size_t)i * stride] - lowest) / denom;
        }
        
        // Store FASTK in temporary buffer
        tempFastK[(size_t)i * stride] = fastK;
    }
    
    // Step 2: Smooth FASTK with SMA(dPeriod) -> SLOWK (%K output)
    // SLOWK starts at index lookbackSlowK
    for (int i = lookbackSlowK; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < dPeriod; ++j) {
            sum += tempFastK[(size_t)(i - j) * stride];
        }
        kOut[(size_t)i * stride] = sum / dPeriod;
    }
    
    // Step 3: Smooth SLOWK with SMA(dPeriod) -> SLOWD (%D output)
    // SLOWD starts at index lookbackSlowD
    for (int i = lookbackSlowD; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < dPeriod; ++j) {
            sum += kOut[(size_t)(i - j) * stride];
        }
        dOut[(size_t)i * stride] = sum / dPeriod;
    }
}

// --------- Batch kernel for many series ----------
__global__ void stochasticManySeriesKernel(const float* __restrict__ high,
                                          const float* __restrict__ low,
                                          const float* __restrict__ close,
                                          float* __restrict__ kOut,
                                          float* __restrict__ dOut,
                                          float* __restrict__ tempFastK,
                                          int seriesLen,
                                          int batchCount,
                                          int kPeriod,
                                          int dPeriod)
{
    // Grid-stride loop for batch processing
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < batchCount;
         s += blockDim.x * gridDim.x)
    {
#ifdef TACUDA_STOCH_TIME_MAJOR
        // Time-major layout: [time][batch]
        const int stride = batchCount;
        const float* h_in  = high  + s;
        const float* l_in  = low   + s;
        const float* c_in  = close + s;
        float* k_out = kOut + s;
        float* d_out = dOut + s;
        float* temp_fk = tempFastK + s;
#else
        // Series-major layout: [series][time] (default)
        const int stride = 1;
        const float* h_in  = high  + (size_t)s * seriesLen;
        const float* l_in  = low   + (size_t)s * seriesLen;
        const float* c_in  = close + (size_t)s * seriesLen;
        float* k_out = kOut + (size_t)s * seriesLen;
        float* d_out = dOut + (size_t)s * seriesLen;
        float* temp_fk = tempFastK + (size_t)s * seriesLen;
#endif
        stochastic_one_series(h_in, l_in, c_in, k_out, d_out, temp_fk,
                            seriesLen, kPeriod, dPeriod, stride);
    }
}

} // namespace

namespace tacuda {

Stochastic::Stochastic(int kPeriod, int dPeriod) 
    : kPeriod(kPeriod), dPeriod(dPeriod) {}

void Stochastic::calculate(const float* high, 
                          const float* low, 
                          const float* close,
                          float* output, 
                          int size, 
                          cudaStream_t stream) noexcept(false) 
{
    // Always use fused parallel implementation
    calculateFused(high, low, close, output, size, stream);
}

void Stochastic::calculate(const float* high,
                          const float* low,
                          const float* close,
                          float* slowK,
                          float* slowD,
                          int size,
                          cudaStream_t stream) noexcept(false) {
    calculateFused(high, low, close, slowK, slowD, size, stream);
}

void Stochastic::calculate(const float* input, 
                          float* output, 
                          int size, 
                          cudaStream_t stream) noexcept(false) 
{
    // Input layout: [high][low][close]
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}

} // namespace tacuda
