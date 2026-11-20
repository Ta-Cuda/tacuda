/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MIDPOINT.h>
#include <utils/BatchRows.h>
#include <utils/LeadingNaN.cuh>
#include <stdexcept>

namespace {

__global__ void midpointKernel(const float* input, float* output, int period,
                               int length, size_t total) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int t = static_cast<int>(i % length);
        if (t < period - 1) {
            output[i] = __int_as_float(0xFFFFFFFF);
            continue;
        }
        const float* row = input + i - t;
        const int start = t - (period - 1);
        float mn = row[start], mx = row[start];
        for (int k = start + 1; k <= t; ++k) {
            mn = fminf(mn, row[k]);
            mx = fmaxf(mx, row[k]);
        }
        if (!isnan(mx) && isnan(row[start]) &&
            tacuda::detail::in_leading_nan_prefix(row, start)) {
            output[i] = nanf("");
            continue;
        }
        output[i] = 0.5f * (mx + mn);
    }
}

}  // namespace

void tacuda::detail::midpoint_rows(const float* input, float* output, int length,
                                   int series, int period, cudaStream_t stream) {
    if (length <= 0 || series <= 0) return;
    const size_t total = static_cast<size_t>(length) * series;
    const dim3 block = defaultBlock();
    const int grid = static_cast<int>(std::min<size_t>((total + block.x - 1) / block.x, 65535));
    midpointKernel<<<grid, block, 0, stream>>>(input, output, period, length, total);
    CUDA_CHECK(cudaGetLastError());
}

tacuda::MIDPOINT::MIDPOINT(int period) : period(period) {}

void tacuda::MIDPOINT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) throw std::invalid_argument("MIDPOINT: invalid period");
    detail::midpoint_rows(input, output, size, 1, period, stream);
}

