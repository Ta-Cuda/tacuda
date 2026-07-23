/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MA.h>
#include <indicators/SMA.h>
#include <indicators/EMA.h>
#include <indicators/WMA.h>
#include <indicators/DEMA.h>
#include <indicators/TEMA.h>
#include <indicators/TRIMA.h>
#include <indicators/KAMA.h>
#include <indicators/MAMA.h>
#include <indicators/T3.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

tacuda::MA::MA(int period, tacuda::MAType type) : period(period), type(type) {}

void tacuda::MA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("MA: invalid period");
    }
    // TA-Lib succeeds with zero output elements when the input is shorter
    // than the lookback of the selected MA type; the line is warm-up NaNs.
    const bool tooShort =
        (period > size) ||
        (type == tacuda::MAType::TEMA && size < 3 * period - 2) ||
        (type == tacuda::MAType::T3 && size < 6 * period - 5) ||
        (type == tacuda::MAType::DEMA && size < 2 * period - 1) ||
        (type == tacuda::MAType::TRIMA && size < period);
    if (tooShort) {
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    
    switch (type) {
    case tacuda::MAType::SMA: {
        tacuda::SMA sma(period);
        sma.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::EMA: {
        tacuda::EMA ema(period);
        ema.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::WMA: {
        tacuda::WMA wma(period);
        wma.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::DEMA: {
        tacuda::DEMA dema(period);
        dema.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::TEMA: {
        tacuda::TEMA tema(period);
        tema.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::TRIMA: {
        tacuda::TRIMA trima(period);
        trima.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::KAMA: {
        tacuda::KAMA kama(period, 2, 30);
        kama.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::MAMA: {
        // MAMA doesn't use period parameter - uses default fastLimit/slowLimit instead
        tacuda::MAMA mama(0.5f, 0.05f);
        // MAMA outputs two values - allocate temp device buffer for both
        float* d_temp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_temp, 2 * size * sizeof(float)));
        mama.calculate(input, d_temp, size, stream);
        // Copy only MAMA line (first half) to output
        CUDA_CHECK(cudaMemcpyAsync(output, d_temp, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        // Wait for async copy to complete before freeing temp buffer
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_temp));
        break;
    }
    case tacuda::MAType::T3: {
        tacuda::T3 t3(period, 0.7f);
        t3.calculate(input, output, size, stream);
        break;
    }
    default:
        throw std::invalid_argument("MA: unsupported type");
    }
}
