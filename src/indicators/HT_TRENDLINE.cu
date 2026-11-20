/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/HT_TRENDLINE.h>
#include <utils/BatchRows.h>
#include <utils/CudaUtils.h>
#include <utils/HilbertCore.h>
#include <utils/LeadingNaN.cuh>
#include <cuda_runtime.h>
#include <cmath>

using namespace tacuda::hilbert;

// Single-threaded GPU kernel
__device__ __forceinline__ void ht_trendline_row(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    const int lookbackTotal = 63;
    if (size <= lookbackTotal) return;
    const int first = tacuda::detail::leading_nan_count(input, size);
    if (size - first <= lookbackTotal) return;
    input += first;
    output += first;
    size -= first;
    
    // Initialize state
    HilbertState hs;
    WMAState wma;
    hs.init();
    wma.init();
    
    // Trend line state
    double iTrend1 = 0.0, iTrend2 = 0.0, iTrend3 = 0.0;
    
    int startIdx = lookbackTotal;
    int endIdx = size - 1;
    wma.trailingWMAIdx = startIdx - lookbackTotal;
    int today = wma.trailingWMAIdx;
    
    // Initialize WMA
    initWMA(wma, input, today);
    
    // Warm-up WMA (34 iterations)
    double smoothedValue;
    for (int i = 0; i < 34; ++i) {
        double tempReal = static_cast<double>(input[today++]);
        smoothedValue = doWMAStep(wma, input, tempReal);
    }
    
    // Fill lookback with NaN
    for (int i = 0; i < lookbackTotal; ++i) {
        output[i] = nanf("");
    }
    
    // Main loop
    while (today <= endIdx) {
        double todayValue = static_cast<double>(input[today]);
        smoothedValue = doWMAStep(wma, input, todayValue);
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        
        // Compute trendline (no clamps: TA-Lib sums DCPeriodInt bars as-is;
        // smoothPeriod ramps slowly enough that idx never underruns).
        double DCPeriod = hs.smoothPeriod + 0.5;
        int DCPeriodInt = static_cast<int>(DCPeriod);
        
        double avgPrice = 0.0;
        int idx = today;
        for (int i = 0; i < DCPeriodInt; ++i) {
            avgPrice += static_cast<double>(input[idx--]);
        }
        if (DCPeriodInt > 0)
            avgPrice /= static_cast<double>(DCPeriodInt);
        
        double trendline = (4.0 * avgPrice + 3.0 * iTrend1 + 2.0 * iTrend2 + iTrend3) / 10.0;
        iTrend3 = iTrend2;
        iTrend2 = iTrend1;
        iTrend1 = avgPrice;
        
        if (today >= startIdx) {
            output[today] = static_cast<float>(trendline);
        }
        
        today++;
    }
}

__global__ void htTrendlineKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    ht_trendline_row(input, output, size);
}

// Batch path: the same single-series state machine, one GPU thread per row.
__global__ void htTrendlineRowsKernel(const float* __restrict__ input, float* __restrict__ output, int length, int series) {
    for (size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         r < static_cast<size_t>(series);
         r += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t off = r * static_cast<size_t>(length);
        ht_trendline_row(input + off, output + off, length);
    }
}

void tacuda::detail::ht_trendline_rows(const float* in, float* out, int length, int series,
        cudaStream_t stream) {
    if (length <= 0 || series <= 0) return;
    const size_t bytes = static_cast<size_t>(length) * series * sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(out, 0xFF, bytes, stream));
    htTrendlineRowsKernel<<<row_blocks(series), kRowThreads, 0, stream>>>(in, out, length, series);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::HT_TRENDLINE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) return;
    // Whole-line NaN fill: short inputs (size <= lookback) must come back as
    // NaN, not as untouched memory.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
    
    htTrendlineKernel<<<1, 1, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
