/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MAX.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void maxKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Warm-up: the first period-1 outputs are NaN, written here instead of a
    // separate memset (same 0xFFFFFFFF pattern the memset used to produce).
    if (idx < period - 1 && idx < size) {
        output[idx] = __int_as_float(0xFFFFFFFF);
    }
    if (idx <= size - period) {
        float m = input[idx];
        for (int k = 1; k < period; ++k)
            m = fmaxf(m, input[idx + k]);
        // Write result at idx + period - 1 to match TA-Lib behavior
        output[idx + period - 1] = m;
    }
}

tacuda::MAX::MAX(int period) : period(period) {}

void tacuda::MAX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("MAX: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    maxKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
