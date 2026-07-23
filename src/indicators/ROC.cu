/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/ROC.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void rocKernel(const float* __restrict__ input, float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period) {
        float prev = input[idx];
        float curr = input[idx + period];
        // Write result at idx + period to match TA-Lib behavior
        // (first period values remain NaN as warmup). TA-Lib emits 0 when
        // the denominator is zero, not NaN.
        output[idx + period] =
            (prev != 0.0f) ? ((curr - prev) / prev) * 100.0f : 0.0f;
    }
}

tacuda::ROC::ROC(int period) : period(period) {}

void tacuda::ROC::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("ROC: invalid period");
    }
    if (period >= size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    rocKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
