/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/AvgPrice.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void avgPriceKernel(const float* __restrict__ open,
                               const float* __restrict__ high,
                               const float* __restrict__ low,
                               const float* __restrict__ close,
                               float* __restrict__ output,
                               int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (open[idx] + high[idx] + low[idx] + close[idx]) * 0.25f;
    }
}

void tacuda::AvgPrice::calculate(const float* open, const float* high, const float* low,
                         const float* close, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("AvgPrice: invalid size");
    }
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    avgPriceKernel<<<grid, block, 0, stream>>>(open, high, low, close, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::AvgPrice::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* open = input;
    const float* high = input + size;
    const float* low = input + 2 * size;
    const float* close = input + 3 * size;
    calculate(open, high, low, close, output, size, stream);
}
