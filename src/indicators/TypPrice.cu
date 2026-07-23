/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/TypPrice.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void typPriceKernel(const float* __restrict__ high,
                               const float* __restrict__ low,
                               const float* __restrict__ close,
                               float* __restrict__ output,
                               int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (high[idx] + low[idx] + close[idx]) / 3.0f;
    }
}

void tacuda::TypPrice::calculate(const float* high, const float* low, const float* close,
                         float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("TypPrice: invalid size");
    }
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    typPriceKernel<<<grid, block, 0, stream>>>(high, low, close, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::TypPrice::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
