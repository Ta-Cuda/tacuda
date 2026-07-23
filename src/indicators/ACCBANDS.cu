/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/ACCBANDS.h>
#include <indicators/SMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

namespace {
__global__ void accbandsPrepareKernel(const float* __restrict__ high,
                                      const float* __restrict__ low,
                                      float* __restrict__ upperRaw,
                                      float* __restrict__ lowerRaw,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = high[idx];
        float l = low[idx];
        float sum = h + l;
        if (sum != 0.0f) {
            float factor = 4.0f * (h - l) / sum;
            upperRaw[idx] = h * (1.0f + factor);
            lowerRaw[idx] = l * (1.0f - factor);
        } else {
            upperRaw[idx] = h;
            lowerRaw[idx] = l;
        }
    }
}
} // namespace

tacuda::ACCBANDS::ACCBANDS(int period) : period(period) {}

void tacuda::ACCBANDS::calculate(const float* high, const float* low, const float* close,
                                 float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("ACCBANDS: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 3 * static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 3 * size * sizeof(float), stream));
    auto upperRaw = acquireDeviceBufferAsync<float>(size, stream);
    auto lowerRaw = acquireDeviceBufferAsync<float>(size, stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    accbandsPrepareKernel<<<grid, block, 0, stream>>>(high, low, upperRaw.get(), lowerRaw.get(), size);
    CUDA_CHECK(cudaGetLastError());

    float* upperOut = output;
    float* middleOut = output + size;
    float* lowerOut = output + 2 * size;

    tacuda::SMA sma(period);
    sma.calculate(close, middleOut, size, stream);
    sma.calculate(upperRaw.get(), upperOut, size, stream);
    sma.calculate(lowerRaw.get(), lowerOut, size, stream);
}

void tacuda::ACCBANDS::calculate(const float* input, float* output, int size,
                                 cudaStream_t stream) noexcept(false) {
    const float* high = input + size;
    const float* low = input + 2 * size;
    const float* close = input + 3 * size;
    calculate(high, low, close, output, size, stream);
}
