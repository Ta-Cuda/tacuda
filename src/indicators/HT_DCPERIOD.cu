/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/HT_DCPERIOD.h>
#include <utils/CudaUtils.h>
#include <utils/HilbertCore.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace tacuda::hilbert;

// Single-threaded GPU kernel - avoids CPU<->GPU data transfer overhead
__global__ void htDCPeriodKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    const int lookbackTotal = 32;
    if (size <= lookbackTotal) return;
    
    // Initialize state
    HilbertState hs;
    WMAState wma;
    hs.init();
    wma.init();
    
    int startIdx = lookbackTotal;
    int endIdx = size - 1;
    wma.trailingWMAIdx = startIdx - lookbackTotal;
    int today = wma.trailingWMAIdx;
    
    // Initialize WMA
    initWMA(wma, input, today);
    
    // Warm-up WMA (9 iterations for DCPERIOD)
    double smoothedValue;
    for (int i = 0; i < 9; ++i) {
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
        
        if (today >= startIdx) {
            output[today] = static_cast<float>(hs.smoothPeriod);
        }
        
        today++;
    }
}

void tacuda::HT_DCPERIOD::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) return;
    // Whole-line NaN fill: short inputs (size <= lookback) must come back as
    // NaN, not as untouched memory.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
    
    // Launch single-threaded kernel on GPU (data stays on device)
    htDCPeriodKernel<<<1, 1, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
