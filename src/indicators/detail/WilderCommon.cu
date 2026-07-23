/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/detail/WilderCommon.cuh>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <algorithm>

namespace tacuda::detail {
namespace {

using WilderPair = tacuda::detail::AffineTransform1<float>;
using WilderCompose = tacuda::detail::AffineCompose1<float>;

// Prepare transformation for Wilder's smoothing
__global__ void wildersPrepKernel(const float* __restrict__ input,
                                  WilderPair* __restrict__ trans,
                                  float alpha,
                                  float k,
                                  int period,
                                  int size) {
    // Start from index 'period' to align with TA-Lib (seed = SMA of first 'period' elements)
    int idx = blockIdx.x * blockDim.x + threadIdx.x + period;
    if (idx < size) {
        trans[idx - period] = {k, alpha * input[idx]};
    }
}

// Compute SMA seed for Wilder's smoothing (same as EMA)
__global__ void computeWildersSMASeedKernel(const float* __restrict__ input,
                                             float* __restrict__ seed,
                                             int period) {
    // Compute SMA of first 'period' elements as seed (TA-Lib compatible)
    float sum = 0.0f;
    for (int i = 0; i < period; ++i) {
        sum += input[i];
    }
    *seed = sum / period;
}

// Finalize Wilder's smoothing output
__global__ void wildersFinalizeKernel(const float* __restrict__ input,
                                      const WilderPair* __restrict__ trans,
                                      float* __restrict__ output,
                                      const float* __restrict__ seed,
                                      int period,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float smaSeed = *seed;
    if (idx < period) {
        // First period values are NaN (warmup) - already set by caller
    } else if (idx == period) {
        output[idx] = smaSeed;
    } else if (idx < size) {
        auto t = trans[idx - period];
        output[idx] = t.A * smaSeed + t.b;
    }
}

} // namespace

void computeWildersSmoothing(const float* input,
                             float* output,
                             int size,
                             int period,
                             cudaStream_t stream) {
    if (size <= 0 || period <= 0 || period > size) {
        return;
    }

    // Wilder's smoothing: alpha = 1/period, k = (period-1)/period
    float alpha = 1.0f / period;
    float k = 1.0f - alpha;
    
    // Allocate buffer for SMA seed
    auto seedBuffer = acquireDeviceBufferAsync<float>(1, stream);
    
    // Compute SMA seed (TA-Lib compatible: SMA of first 'period' values)
    computeWildersSMASeedKernel<<<1, 1, 0, stream>>>(input, seedBuffer.get(), period);
    CUDA_CHECK(cudaGetLastError());
    
    // Allocate trans buffer for elements from index 'period' onwards
    int transSize = std::max(0, size - period);
    auto trans = acquireDeviceBufferAsync<WilderPair>(static_cast<size_t>(transSize), stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    
    // Prepare transformation (PARALLEL)
    wildersPrepKernel<<<grid, block, 0, stream>>>(input, trans.get(), alpha, k, period, size);
    CUDA_CHECK(cudaGetLastError());

    // Perform parallel scan with affine composition (PARALLEL)
    if (transSize > 0) {
        tacuda::detail::affine_inclusive_scan<WilderPair, WilderCompose>(
            trans.get(), trans.get(), transSize, stream);
    }

    // Finalize output (PARALLEL)
    wildersFinalizeKernel<<<grid, block, 0, stream>>>(input, trans.get(), output, seedBuffer.get(), period, size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// DM-specific smoothing (TA-Lib PLUS_DM / MINUS_DM compatible)
// ============================================================================
namespace {

// Compute SUM seed for DM smoothing: SUM(input[1..period-1])
__global__ void computeDMSumSeedKernel(const float* __restrict__ input,
                                        float* __restrict__ seed,
                                        int period) {
    float sum = 0.0f;
    for (int i = 1; i < period; ++i) {
        sum += input[i];
    }
    *seed = sum;
}

// Prepare transforms for DM smoothing starting from index 'period'.
// The affine recursion is: y[i] = k * y[i-1] + input[i]   (NO alpha scaling!)
__global__ void dmPrepKernel(const float* __restrict__ input,
                              WilderPair* __restrict__ trans,
                              float k,
                              int period,
                              int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + period;
    if (idx < size) {
        trans[idx - period] = {k, input[idx]};   // b = input[idx], NOT alpha*input[idx]
    }
}

// Finalize DM smoothing output.
// seed is at index (period-1), transforms start from index period.
__global__ void dmFinalizeKernel(const WilderPair* __restrict__ trans,
                                  float* __restrict__ output,
                                  const float* __restrict__ seed,
                                  int period,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float seedVal = *seed;
    if (idx < period - 1) {
        // NaN (already set by caller)
    } else if (idx == period - 1) {
        output[idx] = seedVal;
    } else if (idx < size) {
        auto t = trans[idx - period];
        output[idx] = t.A * seedVal + t.b;
    }
}

} // namespace

void computeDMSmoothing(const float* input,
                        float* output,
                        int size,
                        int period,
                        cudaStream_t stream) {
    if (size <= 0 || period <= 0 || period > size) {
        return;
    }

    float k = static_cast<float>(period - 1) / static_cast<float>(period);

    // Compute SUM seed
    auto seedBuffer = acquireDeviceBufferAsync<float>(1, stream);
    computeDMSumSeedKernel<<<1, 1, 0, stream>>>(input, seedBuffer.get(), period);
    CUDA_CHECK(cudaGetLastError());

    // Build transforms for indices [period, size)
    int transSize = std::max(0, size - period);
    auto trans = acquireDeviceBufferAsync<WilderPair>(static_cast<size_t>(transSize), stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);

    dmPrepKernel<<<grid, block, 0, stream>>>(input, trans.get(), k, period, size);
    CUDA_CHECK(cudaGetLastError());

    // Parallel prefix scan
    if (transSize > 0) {
        tacuda::detail::affine_inclusive_scan<WilderPair, WilderCompose>(
            trans.get(), trans.get(), transSize, stream);
    }

    // Finalize output
    dmFinalizeKernel<<<grid, block, 0, stream>>>(trans.get(), output, seedBuffer.get(), period, size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda::detail

