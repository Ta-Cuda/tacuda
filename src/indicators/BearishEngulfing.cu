/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/BearishEngulfing.h>
#include <utils/CandleUtils.h>
#include <utils/CudaUtils.h>

__global__ void bearishEngulfingKernel(const float* __restrict__ open,
                                       const float* __restrict__ high,
                                       const float* __restrict__ low,
                                       const float* __restrict__ close,
                                       float* __restrict__ output,
                                       int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx > 0) {
        output[idx] = engulfing(open, close, idx);
    } else {
        output[idx] = 0.0f;
    }
}

void tacuda::BearishEngulfing::calculate(const float* open, const float* high, const float* low,
                                 const float* close, float* output, int size, cudaStream_t stream) noexcept(false) {
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    bearishEngulfingKernel<<<grid, block, 0, stream>>>(open, high, low, close, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::BearishEngulfing::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* open = input;
    const float* high = input + size;
    const float* low = input + 2 * size;
    const float* close = input + 3 * size;
    calculate(open, high, low, close, output, size, stream);
}
