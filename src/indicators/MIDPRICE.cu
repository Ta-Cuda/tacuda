/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MIDPRICE.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void midpriceKernel(const float* __restrict__ high,
                               const float* __restrict__ low,
                               float* __restrict__ output,
                               int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float maxH = high[idx];
        float minL = low[idx];
        for (int k = 1; k < period; ++k) {
            maxH = fmaxf(maxH, high[idx + k]);
            minL = fminf(minL, low[idx + k]);
        }
        // Write result at idx + period - 1 to match TA-Lib behavior
        // (first period-1 values remain NaN as warmup)
        output[idx + period - 1] = 0.5f * (maxH + minL);
    }
}

tacuda::MIDPRICE::MIDPRICE(int period) : period(period) {}

void tacuda::MIDPRICE::calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("MIDPRICE: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    midpriceKernel<<<grid, block, 0, stream>>>(high, low, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::MIDPRICE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}

