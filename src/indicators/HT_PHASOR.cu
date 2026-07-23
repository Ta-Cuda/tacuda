/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/HT_PHASOR.h>
#include <utils/CudaUtils.h>
#include <utils/HilbertCore.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace tacuda::hilbert;

// Single-threaded GPU kernel - avoids CPU<->GPU data transfer overhead
// Output: [inPhase[0..size-1], quadrature[0..size-1]]
__global__ void htPhasorKernel(
    const float* __restrict__ input,
    float* __restrict__ output,  // [inPhase, quadrature] interleaved by size
    int size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    const int lookbackTotal = 32;
    if (size <= lookbackTotal) return;
    
    float* outInPhase = output;
    float* outQuadrature = output + size;
    
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
    
    // Warm-up WMA (9 iterations for PHASOR)
    double smoothedValue;
    for (int i = 0; i < 9; ++i) {
        double tempReal = static_cast<double>(input[today++]);
        smoothedValue = doWMAStep(wma, input, tempReal);
    }
    
    // Fill lookback with NaN
    for (int i = 0; i < lookbackTotal; ++i) {
        outInPhase[i] = nanf("");
        outQuadrature[i] = nanf("");
    }
    
    // Main loop
    while (today <= endIdx) {
        double todayValue = static_cast<double>(input[today]);
        smoothedValue = doWMAStep(wma, input, todayValue);
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        
        if (today >= startIdx) {
            outInPhase[today] = static_cast<float>(outI1);
            outQuadrature[today] = static_cast<float>(outQ1);
        }
        
        today++;
    }
}

void tacuda::HT_PHASOR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) return;
    // Both stacked lines NaN-filled: short inputs must not return garbage.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * static_cast<size_t>(size) * sizeof(float), stream));
    
    htPhasorKernel<<<1, 1, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
