/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/OBV.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <stdexcept>

// Compute OBV contributions in double: contrib[0]=volume[0];
// for i>=1: sign = +1 if price[i]>price[i-1], -1 if <, 0 otherwise; contrib[i]=sign*volume[i]
__global__ void obvContribKernel(const float* __restrict__ price,
                                 const float* __restrict__ volume,
                                 double* __restrict__ contrib,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx == 0) {
        contrib[0] = static_cast<double>(volume[0]);
        return;
    }
    const double p = static_cast<double>(price[idx]);
    const double prev = static_cast<double>(price[idx - 1]);
    const double v = static_cast<double>(volume[idx]);
    double s = 0.0;
    if (p > prev) s = 1.0;
    else if (p < prev) s = -1.0;
    contrib[idx] = s * v;
}

__global__ void castDoubleToFloatKernel(const double* __restrict__ in,
                                        float* __restrict__ out,
                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

void tacuda::OBV::calculate(const float* price, const float* volume,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("OBV: invalid size");
    }
    // Allocate temporary double buffer for contributions / prefix sum
    auto contrib = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);

    // 1) Compute contributions in parallel
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        obvContribKernel<<<grid, block, 0, stream>>>(price, volume, contrib.get(), size);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2) Inclusive scan in double for numerical stability
    {
        thrust::device_ptr<double> ptr(contrib.get());
        thrust::inclusive_scan(thrust::cuda::par.on(stream), ptr, ptr + size, ptr);
    }

    // 3) Cast to float output
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        castDoubleToFloatKernel<<<grid, block, 0, stream>>>(contrib.get(), output, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

void tacuda::OBV::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* price = input;
    const float* volume = input + size;
    calculate(price, volume, output, size, stream);
}

