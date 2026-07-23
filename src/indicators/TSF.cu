/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/TSF.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>
#include <indicators/detail/LinearRegCommon.cuh>

__global__ void tsfKernel(const float* __restrict__ in,
                          float* __restrict__ out,
                          int period, int size) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int windows = size - period + 1;
    if (start < windows) {
        double intercept, slope;
        tacuda::detail::linearRegCoeffs(in, start, period, intercept, slope);
        int outIdx = start + period - 1;
        out[outIdx] = static_cast<float>(intercept + slope * static_cast<double>(period));
    }
}

tacuda::TSF::TSF(int period) : period(period) {}

void tacuda::TSF::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("TSF: invalid period");
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
    tsfKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
