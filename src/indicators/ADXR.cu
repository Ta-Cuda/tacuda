/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/ADXR.h>
#include <indicators/ADX.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

// TA-Lib compatible ADXR: (ADX[i] + ADX[i-period+1]) / 2
__global__ void adxrKernel(const float* __restrict__ adx,
                           float* __restrict__ output,
                           int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TA-Lib ADXR formula: (ADX[i] + ADX[i-period+1]) / 2
    // ADX starts at 2*period-1
    // ADXR needs ADX[i-period+1] to be valid, so i-period+1 >= 2*period-1
    // Therefore i >= 3*period-2
    // For period=14: start = 3*14-2 = 40
    int start = 3 * period - 2;
    if (idx >= start && idx < size) {
        output[idx] = 0.5f * (adx[idx] + adx[idx - period + 1]);
    }
}

tacuda::ADXR::ADXR(int period) : period(period) {}

void tacuda::ADXR::calculate(const float* high, const float* low, const float* close,
                     float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("ADXR: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    auto adx = acquireDeviceBufferAsync<float>(size, stream);

    tacuda::ADX adxInd(period);
    adxInd.calculate(high, low, close, adx.get(), size, stream);

    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    adxrKernel<<<grid, block, 0, stream>>>(adx.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::ADXR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
