/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <cuda_runtime.h>

namespace tacuda::detail {

// Shared device helper for LINEARREG/TSF family.
// Computes least-squares coefficients for y = intercept + slope * x,
// where x = 0..period-1 and y = in[start + x].
__device__ __forceinline__
void linearRegCoeffs(const float* __restrict__ in,
                     int start,
                     int period,
                     double& intercept,
                     double& slope) {
    // Precomputed sums for x = 0..period-1
    const double n = static_cast<double>(period);
    const double sumX  = (n - 1.0) * n / 2.0;
    const double sumX2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;

    double sumY = 0.0;
    double sumXY = 0.0;
    for (int i = 0; i < period; ++i) {
        const double x = static_cast<double>(i);
        const double y = static_cast<double>(in[start + i]);
        sumY += y;
        sumXY += x * y;
    }

    const double denom = n * sumX2 - sumX * sumX;
    if (denom == 0.0) {
        slope = 0.0;
        intercept = sumY / n;
        return;
    }

    slope = (n * sumXY - sumX * sumY) / denom;
    intercept = (sumY - slope * sumX) / n;
}

} // namespace tacuda::detail

