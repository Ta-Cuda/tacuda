/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/TRIX.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

__global__ void trixRocKernel(const float* __restrict__ ema3,
                              float* __restrict__ output,
                              int size3, int period, int outOffset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx + period; // first ROC at i=period (prev at i-1 is the first valid EMA3)
    if (i < size3) {
        float prev = ema3[i - 1];
        float curr = ema3[i];
        output[outOffset + i] = (prev != 0.0f) ? ((curr - prev) / prev) * 100.0f : 0.0f;
    }
}

tacuda::TRIX::TRIX(int period) : period(period) {}

void tacuda::TRIX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size <= 0) {
        throw std::invalid_argument("TRIX: invalid period");
    }
    if (size < 3 * period - 1) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto ema1 = acquireDeviceBufferAsync<float>(size, stream);
    const int offset = period - 1; // warmup per EMA
    const int size2 = size - offset;
    const int size3 = size2 - offset;
    auto ema2 = acquireDeviceBufferAsync<float>(size2, stream);
    auto ema3 = acquireDeviceBufferAsync<float>(size3, stream);

    tacuda::EMA ema(period);
    ema.calculate(input, ema1.get(), size, stream);
    // Important: feed only the valid EMA segment (skip NaN warmup) into the next EMA,
    // matching TA-Lib's internal chaining.
    ema.calculate(ema1.get() + offset, ema2.get(), size2, stream);
    ema.calculate(ema2.get() + offset, ema3.get(), size3, stream);

    // ema3 corresponds to a series starting at original index 2*(period-1)
    // First ROC output is at original index: 2*(period-1) + period = 3*(period-1) + 1 (TA-Lib lookback)
    const int outOffset = 2 * offset;
    const int valid = size3 - period;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    trixRocKernel<<<grid, block, 0, stream>>>(ema3.get(), output, size3, period, outOffset);
    CUDA_CHECK(cudaGetLastError());
}
