/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/WILLR.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

// Parallel WILLR: each thread computes one output using sliding window min/max
__global__ void willrParallelKernel(const float *__restrict__ high,
                                     const float *__restrict__ low,
                                     const float *__restrict__ close,
                                     float *__restrict__ output,
                                     int period, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + (period - 1);
    if (i >= size) return;

    float highest = high[i];
    float lowest  = low[i];
    for (int j = 1; j < period; ++j) {
        float h = high[i - j];
        float l = low[i - j];
        if (h > highest) highest = h;
        if (l < lowest)  lowest = l;
    }
    float denom = highest - lowest;
    output[i] = denom == 0.0f ? 0.0f : ((highest - close[i]) / denom) * -100.0f;
}

tacuda::WILLR::WILLR(int period) : period(period) {}

void tacuda::WILLR::calculate(const float *high, const float *low, const float *close,
                      float *output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("WILLR: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    int validCount = size - (period - 1);
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(validCount);
    willrParallelKernel<<<grid, block, 0, stream>>>(high, low, close, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::WILLR::calculate(const float *input, float *output,
                      int size, cudaStream_t stream) noexcept(false) {
    const float *high = input;
    const float *low = input + size;
    const float *close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
