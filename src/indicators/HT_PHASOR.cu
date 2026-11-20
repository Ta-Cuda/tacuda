/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/HT_PHASOR.h>
#include <utils/BatchRows.h>
#include <utils/CudaUtils.h>
#include <utils/HilbertCore.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace tacuda::hilbert;

// Single-threaded GPU kernel - avoids CPU<->GPU data transfer overhead
// Output: [inPhase[0..size-1], quadrature[0..size-1]]
__device__ __forceinline__ void ht_phasor_row(
    const float* __restrict__ input,
    float* __restrict__ outInPhase,
    float* __restrict__ outQuadrature,
    int size
) {
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

__global__ void htPhasorKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    ht_phasor_row(input, output, output + size, size);
}

// Batch path: the same single-series state machine, one GPU thread per row.
__global__ void htPhasorRowsKernel(const float* __restrict__ input, float* __restrict__ outInPhase, float* __restrict__ outQuadrature, int length, int series) {
    for (size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         r < static_cast<size_t>(series);
         r += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t off = r * static_cast<size_t>(length);
        ht_phasor_row(input + off, outInPhase + off, outQuadrature + off, length);
    }
}

void tacuda::detail::ht_phasor_rows(const float* in, float* inPhase, float* quadrature, int length, int series,
        cudaStream_t stream) {
    if (length <= 0 || series <= 0) return;
    const size_t bytes = static_cast<size_t>(length) * series * sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(inPhase, 0xFF, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(quadrature, 0xFF, bytes, stream));
    htPhasorRowsKernel<<<row_blocks(series), kRowThreads, 0, stream>>>(in, inPhase, quadrature, length, series);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::HT_PHASOR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) return;
    // Both stacked lines NaN-filled: short inputs must not return garbage.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * static_cast<size_t>(size) * sizeof(float), stream));
    
    htPhasorKernel<<<1, 1, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
