/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/SAR.h>
#include <utils/BatchRows.h>
#include <utils/CudaUtils.h>
#include <utils/SarTaLib.h>
#include <stdexcept>

// SAR is inherently sequential (state carried across bars), so the device
// path runs the shared TA-Lib port on a single GPU thread. The host API
// (ct_sar) uses the same algorithm directly on the CPU and never reaches
// this kernel.
__global__ void sarKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          float* __restrict__ output,
                          float acceleration,
                          float maximum,
                          int size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    tacuda::detail::sar_talib_run(high, low, size,
                                  static_cast<double>(acceleration),
                                  static_cast<double>(maximum), output);
}

// Batch path: the same port, one GPU thread per row of a series-major matrix.
__global__ void sarRowsKernel(const float* __restrict__ high,
                              const float* __restrict__ low,
                              float* __restrict__ output,
                              float acceleration,
                              float maximum,
                              int length, int series, int spread) {
    size_t r = 0, step = 0;
    if (!tacuda::detail::row_slot(spread, r, step)) return;
    for (; r < static_cast<size_t>(series); r += step) {
        const size_t off = r * static_cast<size_t>(length);
        tacuda::detail::sar_talib_run(high + off, low + off, length,
                                      static_cast<double>(acceleration),
                                      static_cast<double>(maximum), output + off);
    }
}

void tacuda::detail::sar_rows(const float* high, const float* low, float* out,
                              int length, int series, float acceleration,
                              float maximum, cudaStream_t stream) {
    if (length <= 0 || series <= 0) {
        throw std::invalid_argument("SAR: invalid size");
    }
    CUDA_CHECK(cudaMemsetAsync(out, 0xFF, static_cast<size_t>(length) * series * sizeof(float), stream));
    const RowLaunch launch = divergent_row_launch(series, /*fp64=*/true);
    sarRowsKernel<<<launch.blocks, launch.threads, 0, stream>>>(
        high, low, out, acceleration, maximum, length, series, launch.spread);
    CUDA_CHECK(cudaGetLastError());
}

tacuda::SAR::SAR(float step, float maxAcceleration)
    : step(step), maxAcceleration(maxAcceleration) {}

void tacuda::SAR::calculate(const float* high, const float* low,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("SAR: invalid size");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    sarKernel<<<1, 1, 0, stream>>>(high, low, output, step, maxAcceleration, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::SAR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}
