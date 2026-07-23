/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/HT_SINE.h>
#include <utils/CudaUtils.h>
#include <utils/HilbertCore.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace tacuda::hilbert;

// Single-threaded GPU kernel
// Output: [sine[0..size-1], leadSine[0..size-1]]
__global__ void htSineKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    const int lookbackTotal = 63;
    if (size <= lookbackTotal) return;
    
    constexpr double deg2Rad = 0.017453292519943295; // PI / 180
    
    float* outSine = output;
    float* outLeadSine = output + size;
    
    // Initialize state
    HilbertState hs;
    WMAState wma;
    hs.init();
    wma.init();
    
    // Smooth price circular buffer
    double smoothPrice[50];
    for (int i = 0; i < 50; ++i) smoothPrice[i] = 0.0;
    int smoothPrice_Idx = 0;
    
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
        outSine[i] = nanf("");
        outLeadSine[i] = nanf("");
    }
    
    // Main loop. The DC phase is stateful and updated on EVERY bar
    // (including warm-in bars before the first output), like TA-Lib.
    double DCPhase = 0.0;
    while (today <= endIdx) {
        double todayValue = static_cast<double>(input[today]);
        smoothedValue = doWMAStep(wma, input, todayValue);
        smoothPrice[smoothPrice_Idx] = smoothedValue;
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        updateDCPhase(smoothPrice, smoothPrice_Idx, hs.smoothPeriod, DCPhase);
        
        if (today >= startIdx) {
            outSine[today] = static_cast<float>(sin(DCPhase * deg2Rad));
            outLeadSine[today] = static_cast<float>(sin((DCPhase + 45.0) * deg2Rad));
        }
        
        if (++smoothPrice_Idx == 50)
            smoothPrice_Idx = 0;
        today++;
    }
}

void tacuda::HT_SINE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) return;
    // Both stacked lines NaN-filled: short inputs must not return garbage.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * static_cast<size_t>(size) * sizeof(float), stream));
    
    htSineKernel<<<1, 1, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
