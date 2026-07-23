/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/VAR.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

// Compute rolling population variance with Welford for better numerical stability.
__global__ void varWelfordKernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int period,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx < period - 1) return; // keep NaN warmup

    const int start = idx - period + 1;
    double mean = 0.0;
    double m2 = 0.0;

    for (int i = 0; i < period; ++i) {
        double x = static_cast<double>(input[start + i]);
        double delta = x - mean;
        mean += delta / static_cast<double>(i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance = m2 / static_cast<double>(period); // population variance
    output[idx] = static_cast<float>(variance);
}

tacuda::VAR::VAR(int period) : period(period) {}

void tacuda::VAR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("VAR: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    // Initialize warmup region as NaN
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // Compute rolling variance directly (Welford per window)
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        varWelfordKernel<<<grid, block, 0, stream>>>(input, output, period, size);
        CUDA_CHECK(cudaGetLastError());
    }
}
