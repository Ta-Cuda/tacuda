/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/AroonOscillator.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

// Parallel AroonOscillator: each thread computes one output
__global__ void aroonOscParallelKernel(const float* __restrict__ high,
                                        const float* __restrict__ low,
                                        float* __restrict__ output,
                                        int period, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + period;
    if (i >= size) return;

    int sinceHigh = 0;
    int sinceLow  = 0;
    float maxVal = high[i];
    float minVal = low[i];
    for (int j = 1; j <= period; ++j) {
        float h = high[i - j];
        float l = low[i - j];
        // Strict comparisons: on ties the most recent bar keeps the
        // extremum, matching TA-Lib's sliding-update semantics.
        if (h > maxVal) {
            maxVal = h;
            sinceHigh = j;
        }
        if (l < minVal) {
            minVal = l;
            sinceLow = j;
        }
    }
    float up   = 100.0f * (period - sinceHigh) / static_cast<float>(period);
    float down = 100.0f * (period - sinceLow)  / static_cast<float>(period);
    output[i] = up - down;
}

tacuda::AroonOscillator::AroonOscillator(int period) : period(period) {}

void tacuda::AroonOscillator::calculate(const float* high, const float* low,
                                float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("AroonOscillator: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    int validCount = size - period;
    if (validCount > 0) {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(validCount);
        aroonOscParallelKernel<<<grid, block, 0, stream>>>(high, low, output, period, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

void tacuda::AroonOscillator::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}
