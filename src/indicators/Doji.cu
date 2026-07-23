/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/Doji.h>
#include <utils/CandleUtils.h>
#include <utils/CudaUtils.h>

__global__ void dojiKernel(const float* __restrict__ open,
                           const float* __restrict__ high,
                           const float* __restrict__ low,
                           const float* __restrict__ close,
                           float* __restrict__ output, int size,
                           float factorScale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    (void)factorScale;

    // TA-Lib lookback: BodyDoji = 10. Warm-up bars emit 0 like TA-Lib.
    const int lookback = 10;
    output[idx] = (idx >= lookback)
        ? static_cast<float>(cdl_doji_talib(open, high, low, close, idx))
        : 0.0f;
}

tacuda::Doji::Doji(float threshold) : threshold(threshold) {}

void tacuda::Doji::calculate(const float* open, const float* high, const float* low,
                     const float* close, float* output, int size, cudaStream_t stream) noexcept(false) {
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    const float defaultFactor = candle_setting_factor(CandleSettingType::BodyDoji);
    float factorScale = (defaultFactor > 0.0f) ? threshold / defaultFactor : 1.0f;
    dojiKernel<<<grid, block, 0, stream>>>(open, high, low, close, output, size,
                                           factorScale);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::Doji::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* open = input;
    const float* high = input + size;
    const float* low = input + 2 * size;
    const float* close = input + 3 * size;
    calculate(open, high, low, close, output, size, stream);
}
