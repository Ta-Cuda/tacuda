/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/HT_TRENDMODE.h>
#include <utils/CudaUtils.h>
#include <utils/HilbertCore.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace tacuda::hilbert;

// Single-threaded GPU kernel
// Output: integer trend mode (0 or 1) as float
__global__ void htTrendModeKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    const int lookbackTotal = 63;
    if (size <= lookbackTotal) return;
    
    constexpr double deg2Rad = 0.017453292519943295;
    
    // Initialize state
    HilbertState hs;
    WMAState wma;
    hs.init();
    wma.init();
    
    // Smooth price circular buffer
    double smoothPrice[50];
    for (int i = 0; i < 50; ++i) smoothPrice[i] = 0.0;
    int smoothPrice_Idx = 0;
    
    // Trend line state (TA-Lib inits: everything zero, sine/leadSine too)
    double iTrend1 = 0.0, iTrend2 = 0.0, iTrend3 = 0.0;
    int daysInTrend = 0, trend = 0;
    double DCPhase = 0.0, prevDCPhase = 0.0;
    double sine = 0.0, leadSine = 0.0;
    double prevSine = 0.0, prevLeadSine = 0.0;
    
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
    
    // Warm-up region is 0 (TA-Lib int-output convention; the host wrapper
    // zero-fills the whole line before this kernel runs).
    
    // Main loop — literal ta_HT_TRENDMODE.c ordering: prevDCPhase/prevSine
    // snapshots taken BEFORE recomputing, DC phase updated on every bar.
    while (today <= endIdx) {
        double todayValue = static_cast<double>(input[today]);
        smoothedValue = doWMAStep(wma, input, todayValue);
        smoothPrice[smoothPrice_Idx] = smoothedValue;
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        
        prevDCPhase = DCPhase;
        updateDCPhase(smoothPrice, smoothPrice_Idx, hs.smoothPeriod, DCPhase);
        
        prevSine = sine;
        prevLeadSine = leadSine;
        sine = sin(DCPhase * deg2Rad);
        leadSine = sin((DCPhase + 45.0) * deg2Rad);
        
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
        
        // Determine trend mode
        trend = 1;
        if (((sine > leadSine) && (prevSine <= prevLeadSine)) ||
            ((sine < leadSine) && (prevSine >= prevLeadSine))) {
            daysInTrend = 0;
            trend = 0;
        }
        daysInTrend++;
        if (daysInTrend < (0.5 * hs.smoothPeriod))
            trend = 0;
        
        double tempReal = DCPhase - prevDCPhase;
        if ((hs.smoothPeriod != 0.0) &&
            (tempReal > (0.67 * 360.0 / hs.smoothPeriod)) &&
            (tempReal < (1.5 * 360.0 / hs.smoothPeriod)))
            trend = 0;
        
        tempReal = smoothPrice[smoothPrice_Idx];
        if ((trendline != 0.0) && (fabs((tempReal - trendline) / trendline) >= 0.015))
            trend = 1;
        
        if (today >= startIdx) {
            output[today] = static_cast<float>(trend);
        }
        
        if (++smoothPrice_Idx == 50)
            smoothPrice_Idx = 0;
        today++;
    }
}

void tacuda::HT_TRENDMODE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) return;
    // TA-Lib int-output convention: 0-filled warm-up (and 0 for inputs
    // shorter than the lookback), consistent across host/device/batch.
    CUDA_CHECK(cudaMemsetAsync(output, 0x00, static_cast<size_t>(size) * sizeof(float), stream));
    
    htTrendModeKernel<<<1, 1, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
