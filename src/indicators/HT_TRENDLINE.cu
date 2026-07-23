/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/HT_TRENDLINE.h>
#include <utils/CudaUtils.h>
#include <utils/HilbertCore.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace tacuda::hilbert;

// Single-threaded GPU kernel
__global__ void htTrendlineKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    const int lookbackTotal = 63;
    if (size <= lookbackTotal) return;
    
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

void tacuda::HT_TRENDLINE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) return;
    // Whole-line NaN fill: short inputs (size <= lookback) must come back as
    // NaN, not as untouched memory.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
    
    htTrendlineKernel<<<1, 1, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
