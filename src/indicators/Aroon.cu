/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/Aroon.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

// Parallel Aroon: each thread computes up/down/osc for one index
__global__ void aroonParallelKernel(const float* __restrict__ high,
                                     const float* __restrict__ low,
                                     float* __restrict__ upOut,
                                     float* __restrict__ downOut,
                                     float* __restrict__ oscOut,
                                     int upPeriod,
                                     int downPeriod,
                                     int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    int maxPeriod = upPeriod > downPeriod ? upPeriod : downPeriod;

    if (i >= upPeriod) {
        int sinceHigh = 0;
        float maxVal = high[i];
        for (int j = 1; j <= upPeriod; ++j) {
            float val = high[i - j];
            // Strictly greater: on ties the most recent bar keeps the
            // extremum, matching TA-Lib's sliding-update semantics.
            if (val > maxVal) {
                maxVal = val;
                sinceHigh = j;
            }
        }
        upOut[i] = 100.0f * (upPeriod - sinceHigh) / static_cast<float>(upPeriod);
    }

    if (i >= downPeriod) {
        int sinceLow = 0;
        float minVal = low[i];
        for (int j = 1; j <= downPeriod; ++j) {
            float val = low[i - j];
            if (val < minVal) {
                minVal = val;
                sinceLow = j;
            }
        }
        downOut[i] = 100.0f * (downPeriod - sinceLow) / static_cast<float>(downPeriod);
    }

    if (i >= maxPeriod) {
        oscOut[i] = upOut[i] - downOut[i];
    }
}

tacuda::Aroon::Aroon(int upPeriod, int downPeriod) : upPeriod(upPeriod), downPeriod(downPeriod) {}

void tacuda::Aroon::calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (upPeriod <= 0 || downPeriod <= 0) {
        throw std::invalid_argument("Aroon: invalid period");
    }
    if (upPeriod > size || downPeriod > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; all three lines are warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 3 * static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 3 * size * sizeof(float), stream));
    float* upOut   = output;
    float* downOut = output + size;
    float* oscOut  = output + 2 * size;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    aroonParallelKernel<<<grid, block, 0, stream>>>(high, low, upOut, downOut, oscOut,
                                                      upPeriod, downPeriod, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::Aroon::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}
