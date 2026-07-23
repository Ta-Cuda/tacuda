/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/SAREXT.h>
#include <utils/CudaUtils.h>
#include <utils/SarTaLib.h>
#include <stdexcept>

// SAREXT is inherently sequential, so the device path runs the shared TA-Lib
// port on a single GPU thread. The host API (ct_sarext) uses the same
// algorithm directly on the CPU and never reaches this kernel.
__global__ void sarextKernel(const float* __restrict__ high,
                             const float* __restrict__ low,
                             float* __restrict__ output,
                             float startValue, float offset,
                             float accInitLong, float accLong, float accMaxLong,
                             float accInitShort, float accShort, float accMaxShort,
                             int size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    tacuda::detail::sarext_talib_run(
        high, low, size, static_cast<double>(startValue),
        static_cast<double>(offset), static_cast<double>(accInitLong),
        static_cast<double>(accLong), static_cast<double>(accMaxLong),
        static_cast<double>(accInitShort), static_cast<double>(accShort),
        static_cast<double>(accMaxShort), output);
}

tacuda::SAREXT::SAREXT(float startValue, float offsetOnReverse,
               float accInitLong, float accLong, float accMaxLong,
               float accInitShort, float accShort, float accMaxShort)
    : startValue(startValue), offsetOnReverse(offsetOnReverse),
      accInitLong(accInitLong), accLong(accLong), accMaxLong(accMaxLong),
      accInitShort(accInitShort), accShort(accShort), accMaxShort(accMaxShort) {}

void tacuda::SAREXT::calculate(const float* high, const float* low,
                       float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("SAREXT: invalid size");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    sarextKernel<<<1, 1, 0, stream>>>(high, low, output, startValue, offsetOnReverse,
                          accInitLong, accLong, accMaxLong,
                          accInitShort, accShort, accMaxShort, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::SAREXT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}
