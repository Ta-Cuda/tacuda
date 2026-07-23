/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/PVI.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <stdexcept>

namespace {

// Build per-step multiplicative factors for PVI
// factor[0] = 1000 (initial PVI value)
// factor[i] = close[i]/close[i-1] if volume[i] > volume[i-1], else 1.0
__global__ void pviBuildFactorsKernel(const float* __restrict__ close,
                                      const float* __restrict__ volume,
                                      float* __restrict__ factors,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (idx == 0) {
        factors[0] = 1000.0f;
    } else {
        if (volume[idx] > volume[idx - 1] && close[idx - 1] != 0.0f) {
            factors[idx] = close[idx] / close[idx - 1];
        } else {
            factors[idx] = 1.0f;
        }
    }
}

} // namespace

void tacuda::PVI::calculate(const float* close, const float* volume, float* output,
                            int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("PVI: invalid size");
    }

    // 1. Build multiplicative factors in parallel
    auto factors = acquireDeviceBufferAsync<float>(size, stream);
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        pviBuildFactorsKernel<<<grid, block, 0, stream>>>(close, volume, factors.get(), size);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. Parallel prefix product (inclusive scan with multiplication)
    // PVI[i] = factors[0] * factors[1] * ... * factors[i]
    {
        thrust::device_ptr<float> in(factors.get());
        thrust::device_ptr<float> out(output);
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                               in, in + size, out,
                               thrust::multiplies<float>());
    }
}

void tacuda::PVI::calculate(const float* input, float* output, int size,
                            cudaStream_t stream) noexcept(false) {
    const float* close = input;
    const float* volume = input + size;
    calculate(close, volume, output, size, stream);
}
