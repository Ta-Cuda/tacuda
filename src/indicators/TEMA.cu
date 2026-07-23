/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/TEMA.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

__global__ void temaComposeKernel(const float* __restrict__ ema1,
                                  const float* __restrict__ ema2,
                                  const float* __restrict__ ema3,
                                  float* __restrict__ output,
                                  int period,
                                  int valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= valid) return;
    int outIdx = 3 * period - 3 + i;
    float e1 = ema1[outIdx];
    float e2 = ema2[2 * (period - 1) + i];
    float e3 = ema3[(period - 1) + i];
    output[outIdx] = 3.0f * e1 - 3.0f * e2 + e3;
}

namespace tacuda {

TEMA::TEMA(int period) : period(period) {}

void TEMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size <= 0) {
        throw std::invalid_argument("TEMA: invalid period");
    }
    if (size < 3 * period - 2) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
        return;
    }

    // Initialize output with NaNs
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // Buffers: ema1 size=N, ema2 size=N-(p-1), ema3 size=N-2*(p-1)
    auto ema1 = acquireDeviceBufferAsync<float>(size, stream);
    auto ema2 = acquireDeviceBufferAsync<float>(size - (period - 1), stream);
    auto ema3 = acquireDeviceBufferAsync<float>(size - 2 * (period - 1), stream);

    tacuda::EMA ema(period);

    // EMA1 over input
    ema.calculate(input, ema1.get(), size, stream);

    // EMA2 over EMA1 valid tail
    const float* ema1Valid = ema1.get() + (period - 1);
    int size2 = size - (period - 1);
    ema.calculate(ema1Valid, ema2.get(), size2, stream);

    // EMA3 over EMA2 valid tail
    const float* ema2Valid = ema2.get() + (period - 1);
    int size3 = size2 - (period - 1);
    ema.calculate(ema2Valid, ema3.get(), size3, stream);

    // Compose TEMA: valid outputs count
    int valid = size - (3 * period - 3);
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    temaComposeKernel<<<grid, block, 0, stream>>>(ema1.get(), ema2.get(), ema3.get(), output, period, valid);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda
