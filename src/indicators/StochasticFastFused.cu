/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/StochasticFast.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>
#include <math.h>
#include <cuda_runtime.h>

using namespace tacuda;

// Pass 1: Calculate FASTK for all points in parallel
__global__ void stochFastCalculateFastKKernel(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    float* __restrict__ fastK,
    int size,
    int kPeriod)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        if (idx < kPeriod - 1) {
            fastK[idx] = nanf("");
            return;
        }

        // Find highest high and lowest low over kPeriod
        float highest = high[idx];
        float lowest = low[idx];
        
        for (int j = 1; j < kPeriod; ++j) {
            float h = high[idx - j];
            float l = low[idx - j];
            if (h > highest) highest = h;
            if (l < lowest) lowest = l;
        }

        float denom = highest - lowest;
        fastK[idx] = (denom == 0.0f) ? 0.0f : ((close[idx] - lowest) / denom * 100.0f);
    }
}

// TA-Lib returns FASTK and FASTD with a common outBegIdx of
// (kPeriod-1)+(dPeriod-1): the published K line loses the D warm-up too.
__global__ void stochFastMaskKKernel(float* __restrict__ kOut, int firstValid, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx < firstValid) kOut[idx] = nanf("");
}

// Pass 2: Calculate FASTD (SMA of FASTK) for all points in parallel
__global__ void stochFastCalculateFastDKernel(
    const float* __restrict__ fastK,
    float* __restrict__ fastD,
    int size,
    int kPeriod,
    int dPeriod)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        int lookbackFastD = kPeriod + dPeriod - 2;
        
        if (idx < lookbackFastD) {
            fastD[idx] = nanf("");
            return;
        }

        // Calculate SMA of FASTK over dPeriod
        float sum = 0.0f;
        for (int j = 0; j < dPeriod; ++j) {
            sum += fastK[idx - j];
        }
        
        fastD[idx] = sum / static_cast<float>(dPeriod);
    }
}

void StochasticFast::calculateFused(
    const float* high,
    const float* low,
    const float* close,
    float* output,
    int size,
    cudaStream_t stream) noexcept(false)
{
    // Validate parameters
    if (kPeriod <= 0 || dPeriod <= 0) {
        throw std::invalid_argument("StochasticFast: periods must be positive");
    }
    if (size <= 0) {
        throw std::invalid_argument("StochasticFast: size must be positive");
    }
    if (kPeriod + dPeriod - 2 >= size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the shared K/D lookback; both lines are NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    // Output layout: [FASTK (size), FASTD (size)]
    float* kOut = output;
    float* dOut = output + size;

    // Allocate temporary buffer for FASTK
    auto tempFastK = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);

    // Determine optimal launch configuration
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // PASS 1: Calculate FASTK
    stochFastCalculateFastKKernel<<<gridSize, blockSize, 0, stream>>>(
        high, low, close, tempFastK.get(), size, kPeriod);
    CUDA_CHECK(cudaGetLastError());

    // Copy FASTK to output, masking the shared TA-Lib warm-up
    // (the temp buffer keeps the earlier K values needed for D).
    CUDA_CHECK(cudaMemcpyAsync(kOut, tempFastK.get(), size * sizeof(float), 
                               cudaMemcpyDeviceToDevice, stream));
    stochFastMaskKKernel<<<gridSize, blockSize, 0, stream>>>(
        kOut, kPeriod + dPeriod - 2, size);
    CUDA_CHECK(cudaGetLastError());

    // PASS 2: Calculate FASTD
    stochFastCalculateFastDKernel<<<gridSize, blockSize, 0, stream>>>(
        tempFastK.get(), dOut, size, kPeriod, dPeriod);
    CUDA_CHECK(cudaGetLastError());
}

