/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MedPrice.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void medPriceKernel(const float* __restrict__ high,
                               const float* __restrict__ low,
                               float* __restrict__ output,
                               int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 0.5f * (high[idx] + low[idx]);
    }
}

void tacuda::MedPrice::calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("MedPrice: invalid size");
    }
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    medPriceKernel<<<grid, block, 0, stream>>>(high, low, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::MedPrice::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}
