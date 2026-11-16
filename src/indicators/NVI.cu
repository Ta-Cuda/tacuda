/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/NVI.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/DeterministicScan.cuh>
#include <thrust/functional.h>
#include <stdexcept>

namespace {

// Match NumPy TA-Lib's leading-NaN input alignment while retaining the
// deterministic parallel product scan for long histories.
__global__ void nviPrefixKernel(const float* close, const float* volume, int* prefix, int size) {
    if (blockIdx.x || threadIdx.x) return;
    int first = 0;
    while (first < size - 1 && (isnan(close[first]) || isnan(volume[first]))) ++first;
    *prefix = first;
}

// Build per-step multiplicative factors for NVI
// factor[0] = 1000 (initial NVI value)
// factor[i] = close[i]/close[i-1] if volume[i] < volume[i-1], else 1.0
__global__ void nviBuildFactorsKernel(const float* __restrict__ close,
                                      const float* __restrict__ volume,
                                      double* __restrict__ factors, const int* prefix,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (idx < *prefix) {
        factors[idx] = 1.0;
    } else if (idx == *prefix) {
        factors[idx] = 1000.0;
    } else {
        if (volume[idx] < volume[idx - 1] && close[idx - 1] != 0.0f) {
            factors[idx] = static_cast<double>(close[idx]) / static_cast<double>(close[idx - 1]);
        } else {
            factors[idx] = 1.0;
        }
    }
}

__global__ void nviRoundKernel(const double* values, float* output, int size, const int* prefix) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) output[i] = i < *prefix ? nanf("") : static_cast<float>(values[i]);
}

} // namespace

void tacuda::NVI::calculate(const float* close, const float* volume, float* output,
                            int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("NVI: invalid size");
    }

    // 1. Build multiplicative factors in parallel
    auto factors = acquireDeviceBufferAsync<double>(size, stream);
    auto prefix = acquireDeviceBufferAsync<int>(1, stream);
    nviPrefixKernel<<<1, 1, 0, stream>>>(close, volume, prefix.get(), size);
    CUDA_CHECK(cudaGetLastError());
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        nviBuildFactorsKernel<<<grid, block, 0, stream>>>(close, volume, factors.get(), prefix.get(), size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Keep both the ratios and the cumulative state in double. Rounding
    // near-one ratios to float before multiplication accumulates a bias.
    tacuda::detail::deterministic_inclusive_scan<double>(factors.get(), factors.get(),
        size, stream, thrust::multiplies<double>{}, 1.0);
    nviRoundKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(factors.get(), output, size, prefix.get());
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::NVI::calculate(const float* input, float* output, int size,
                            cudaStream_t stream) noexcept(false) {
    const float* close = input;
    const float* volume = input + size;
    calculate(close, volume, output, size, stream);
}
