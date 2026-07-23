/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/LINEARREG_ANGLE.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>
#include <indicators/detail/LinearRegCommon.cuh>

__global__ void linearregAngleKernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int period, int size) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int windows = size - period + 1;
    if (start < windows) {
        double intercept, slope;
        tacuda::detail::linearRegCoeffs(in, start, period, intercept, slope);
        int outIdx = start + period - 1;
        const double rad2deg = 180.0 / 3.14159265358979323846;
        out[outIdx] = static_cast<float>(atan(slope) * rad2deg);
    }
}

tacuda::LINEARREG_ANGLE::LINEARREG_ANGLE(int period) : period(period) {}

void tacuda::LINEARREG_ANGLE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("LINEARREG_ANGLE: invalid period");
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
    linearregAngleKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

