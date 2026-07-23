/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/DEMA.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

__global__ void demaComposeKernel(const float* __restrict__ ema1,
                                  const float* __restrict__ ema2,
                                  float* __restrict__ output,
                                  int period,
                                  int valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= valid) return;
    int outIdx = (period << 1) - 2 + i; // 2*period - 2 + i
    float e1 = ema1[outIdx];
    float e2 = ema2[(period - 1) + i]; // skip warmup of second EMA
    output[outIdx] = 2.0f * e1 - e2;
}

namespace tacuda {

DEMA::DEMA(int period) : period(period) {}

void DEMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size <= 0) {
        throw std::invalid_argument("DEMA: invalid period");
    }
    if (size < (2 * period - 1)) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    // Initialize output with NaNs (warmup region remains NaN)
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // Buffers for EMA1 (length size) and EMA2 (length size - period + 1)
    auto ema1 = acquireDeviceBufferAsync<float>(size, stream);
    auto ema2 = acquireDeviceBufferAsync<float>(size - period + 1, stream);

    // First EMA on input
    tacuda::EMA ema(period);
    ema.calculate(input, ema1.get(), size, stream);

    // Second EMA on EMA1 valid tail starting at (period-1)
    const float* ema1Valid = ema1.get() + (period - 1);
    int size2 = size - period + 1; // length of valid EMA1 tail
    ema.calculate(ema1Valid, ema2.get(), size2, stream);

    // Compose DEMA where valid count = size - (2*period - 2)
    int valid = size - (2 * period - 2);
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    demaComposeKernel<<<grid, block, 0, stream>>>(ema1.get(), ema2.get(), output, period, valid);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda
