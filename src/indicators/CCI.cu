/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/CCI.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void cciKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= period - 1 && idx < size) {
        // Double accumulation mirrors TA-Lib's double math: on constant
        // windows the mean deviation must come out exactly zero, otherwise
        // (tp-sma)/(0.015*md) turns rounding noise into +-66.67.
        double sum = 0.0;
        for (int j = 0; j < period; ++j) {
            int cur = idx - j;
            sum += (static_cast<double>(high[cur]) + low[cur] + close[cur]) / 3.0;
        }
        double sma = sum / period;
        double dev = 0.0;
        for (int j = 0; j < period; ++j) {
            int cur = idx - j;
            double tp = (static_cast<double>(high[cur]) + low[cur] + close[cur]) / 3.0;
            dev += fabs(tp - sma);
        }
        double md = dev / period;
        double tp_cur = (static_cast<double>(high[idx]) + low[idx] + close[idx]) / 3.0;
        double num = tp_cur - sma;
        // TA-Lib: outputs 0 unless BOTH the numerator and mean deviation are nonzero.
        output[idx] = (num != 0.0 && md != 0.0)
                          ? static_cast<float>(num / (0.015 * md))
                          : 0.0f;
    }
}

tacuda::CCI::CCI(int period) : period(period) {}

void tacuda::CCI::calculate(const float* high, const float* low, const float* close,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("CCI: invalid period");
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
    cciKernel<<<grid, block, 0, stream>>>(high, low, close, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::CCI::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
