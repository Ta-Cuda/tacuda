/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MIDPOINT.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void midpointKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float mn = input[idx];
        float mx = input[idx];
        for (int k = 1; k < period; ++k) {
            float v = input[idx + k];
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        // Write result at idx + period - 1 to match TA-Lib behavior
        // (first period-1 values remain NaN as warmup)
        output[idx + period - 1] = 0.5f * (mx + mn);
    }
}

tacuda::MIDPOINT::MIDPOINT(int period) : period(period) {}

void tacuda::MIDPOINT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("MIDPOINT: invalid period");
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
    midpointKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

