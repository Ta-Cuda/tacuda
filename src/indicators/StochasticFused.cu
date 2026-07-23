/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/Stochastic.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

namespace {

// =====================================================================
// FUSED STOCHASTIC KERNEL - 2-pass parallel algorithm
// =====================================================================
// Replaces sequential single-thread algorithm with parallel processing
// Expected improvement: 50-100x from 0.01x to 0.5-1.0x
//
// Algorithm (TA-Lib compatible):
// Pass 1: Calculate FASTK for all points in parallel
// Pass 2: Calculate SLOWK and SLOWD in parallel
//
// Key optimizations:
// - Each thread processes one time point independently
// - No synchronization between threads except at kernel boundaries
// - Minimal global memory access (coalesced reads)
// - Register-only computation (no shared memory overhead for small periods)
// =====================================================================

// ========== PASS 1: Parallel FASTK calculation ==========
__global__ void stochasticCalculateFastKKernel(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    float* __restrict__ fastK,
    int size,
    int kPeriod)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lookbackFastK = kPeriod - 1;
    
    if (idx >= size) return;
    
    // Initialize NaN for lookback period
    if (idx < lookbackFastK) {
        fastK[idx] = nanf("");
        return;
    }
    
    // Find highest high and lowest low in last kPeriod bars (parallel per thread)
    float highest = high[idx];
    float lowest = low[idx];
    
    #pragma unroll 8
    for (int j = 1; j < kPeriod; ++j) {
        float h = high[idx - j];
        float l = low[idx - j];
        highest = fmaxf(highest, h);
        lowest = fminf(lowest, l);
    }
    
    // Calculate FASTK (TA-Lib emits 0.0 when the range is zero,
    // see ta_STOCHF.c: "if(diff!=0.0) ... else 0.0")
    float range = highest - lowest;
    if (range == 0.0f) {
        fastK[idx] = 0.0f;
    } else {
        fastK[idx] = 100.0f * (close[idx] - lowest) / range;
    }
}

// ========== PASS 2: Parallel SLOWK calculation ==========
__global__ void stochasticCalculateSlowKKernel(
    const float* __restrict__ fastK,
    float* __restrict__ slowK,
    int size,
    int kPeriod,
    int slowKPeriod,
    int slowDPeriod)  // Need slowDPeriod to sync with SLOWD lookback
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float qnan = nanf("");
    
    const int lookbackFastK = kPeriod - 1;
    const int lookbackSlowK = lookbackFastK + slowKPeriod - 1;
    const int lookbackSlowD = lookbackSlowK + slowDPeriod - 1;  // TA-Lib syncs both outputs
    
    if (idx >= size) return;
    
    // Calculate SLOWK = SMA(FASTK, slowKPeriod)
    // BUT output NaN until SLOWD lookback for TA-Lib compatibility
    if (idx >= lookbackSlowK) {
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int j = 0; j < slowKPeriod; ++j) {
            sum += fastK[idx - j];
        }
        
        float slowk_val = sum / slowKPeriod;
        
        // Only output if we're past SLOWD lookback (TA-Lib syncs outputs)
        if (idx >= lookbackSlowD) {
            slowK[idx] = slowk_val;
        } else {
            slowK[idx] = qnan;  // NaN for output sync, but value is calculated
        }
    } else {
        slowK[idx] = qnan;
    }
}

// ========== PASS 3: Parallel SLOWD calculation ==========
__global__ void stochasticCalculateSlowDKernel(
    const float* __restrict__ fastK,  // Read from fastK instead of slowK
    const float* __restrict__ slowK,
    float* __restrict__ slowD,
    int size,
    int kPeriod,
    int slowKPeriod,
    int slowDPeriod)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float qnan = nanf("");
    
    const int lookbackFastK = kPeriod - 1;
    const int lookbackSlowK = lookbackFastK + slowKPeriod - 1;
    const int lookbackSlowD = lookbackSlowK + slowDPeriod - 1;
    
    if (idx >= size) return;
    
    // Calculate SLOWD = SMA(SLOWK, slowDPeriod)
    // But SLOWK values at idx-j might be NaN in output, so recalculate from fastK
    if (idx >= lookbackSlowD) {
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int j = 0; j < slowDPeriod; ++j) {
            int slowk_idx = idx - j;
            
            // Recalculate SLOWK = SMA(FASTK, slowKPeriod) inline
            float slowk_sum = 0.0f;
            for (int k = 0; k < slowKPeriod; ++k) {
                slowk_sum += fastK[slowk_idx - k];
            }
            float slowk_val = slowk_sum / slowKPeriod;
            
            sum += slowk_val;
        }
        
        slowD[idx] = sum / slowDPeriod;
    } else {
        slowD[idx] = qnan;
    }
}

} // namespace

namespace tacuda {

// =====================================================================
// PUBLIC API: Fused Stochastic calculation
// =====================================================================
void Stochastic::calculateFused(
    const float* high,
    const float* low,
    const float* close,
    float* output,
    int size,
    cudaStream_t stream) noexcept(false)
{
    calculateFused(high, low, close, output, output + size, size, stream);
}

void Stochastic::calculateFused(
    const float* high,
    const float* low,
    const float* close,
    float* kOut,
    float* dOut,
    int size,
    cudaStream_t stream) noexcept(false)
{
    if (kPeriod <= 0 || dPeriod <= 0) {
        throw std::invalid_argument("Stochastic: kPeriod and dPeriod must be > 0");
    }
    if (size <= 0) {
        throw std::invalid_argument("Stochastic: size must be positive");
    }
    
    const int minLookback = kPeriod + 2 * (dPeriod - 1);
    if (size <= minLookback) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the shared K/D lookback; both lines are NaNs.
        CUDA_CHECK(cudaMemsetAsync(kOut, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(dOut, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    
    // Allocate temporary buffer for FASTK
    auto tempFastK = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);
    
    // Determine optimal launch configuration
    int blockSize = 256;  // Good balance for most GPUs
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Don't limit grid size - we need to cover ALL elements!
    // (Previous limit of sms*4 was causing only partial processing)
    
    // ========== PASS 1: Calculate FASTK in parallel ==========
    stochasticCalculateFastKKernel<<<gridSize, blockSize, 0, stream>>>(
        high, low, close, tempFastK.get(), size, kPeriod);
    CUDA_CHECK(cudaGetLastError());
    
    // ========== PASS 2: Calculate SLOWK in parallel ==========
    // Note: using dPeriod for both slowKPeriod and slowDPeriod (TA-Lib convention in benchmark)
    stochasticCalculateSlowKKernel<<<gridSize, blockSize, 0, stream>>>(
        tempFastK.get(), kOut, size, kPeriod, dPeriod, dPeriod);
    CUDA_CHECK(cudaGetLastError());
    
    // ========== PASS 3: Calculate SLOWD in parallel ==========
    // Pass fastK buffer so we can recalculate SLOWK inline
    stochasticCalculateSlowDKernel<<<gridSize, blockSize, 0, stream>>>(
        tempFastK.get(), kOut, dOut, size, kPeriod, dPeriod, dPeriod);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda

