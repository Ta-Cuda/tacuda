/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/StochasticFast.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

tacuda::StochasticFast::StochasticFast(int kPeriod, int dPeriod) : kPeriod(kPeriod), dPeriod(dPeriod) {}

void tacuda::StochasticFast::calculate(const float* high, const float* low, const float* close,
                               float* output, int size, cudaStream_t stream) noexcept(false) {
    // Always use fused parallel implementation
    calculateFused(high, low, close, output, size, stream);
}

void tacuda::StochasticFast::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
