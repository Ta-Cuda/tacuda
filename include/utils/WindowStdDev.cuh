/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <cuda_runtime.h>

namespace tacuda::detail {

constexpr int kDirectStdDevPeriod = 64;

__device__ inline float direct_window_stddev(const float* window, int period) {
    const double origin = window[0];
    double sum = 0.0, squares = 0.0;
    for (int i = 0; i < period; ++i) {
        const double delta = static_cast<double>(window[i]) - origin;
        sum += delta;
        squares += delta * delta;
    }
    const double m2 = squares - sum * (sum / period);
    return static_cast<float>(sqrt((m2 < 0.0 ? 0.0 : m2) / period));
}

}  // namespace tacuda::detail
