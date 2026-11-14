/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <utils/CudaUtils.h>
#include <cmath>

namespace tacuda { namespace detail {

// ATR/NATR period one is unnormalized TRANGE, independent at every bar.
// Share the warm-up/NaN policy between both full-series dispatchers.
static __global__ void period_one_true_range_kernel(const float* high, const float* low,
                                                    const float* close, float* out,
                                                    int size, int seriesLen) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(size)) return;
    if (i % seriesLen == 0) { out[i] = nanf(""); return; }
    const double hi = high[i], lo = low[i], prev = close[i - 1];
    double tr = hi - lo;
    const double a = fabs(hi - prev), b = fabs(lo - prev);
    if (a > tr) tr = a;
    if (b > tr) tr = b;
    out[i] = static_cast<float>(tr);
}

inline void true_range_period_one(const float* high, const float* low, const float* close,
                                  float* out, int size, int seriesLen, cudaStream_t stream) {
    period_one_true_range_kernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        high, low, close, out, size, seriesLen);
    CUDA_CHECK(cudaGetLastError());
}

}}  // namespace tacuda::detail
