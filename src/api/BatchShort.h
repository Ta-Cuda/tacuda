/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <cuda_runtime.h>

namespace tacuda::detail {

enum class ShortFamily {
    Dema, Tema, T3, Kama, Trix, Cmo, Adx, Adxr, Dx, PlusDi, MinusDi,
    PlusDm, MinusDm, Apo, Ppo, Pvo, Ma, MacdLine, StochRsi, Imi,
    Nvi, Pvi, Correl, Beta, Asi, Obtr, Wobv, Hpi, Xaverage
};

constexpr int kShortWindow = 64;
constexpr int kShortLength = 1024;
bool short_batch_shape(int length, int series);

// These legacy scans keep float coefficients and state. If the coefficient
// can decay into subnormals across the row, keep the original scan order.
inline bool short_float_wilder(int length, int period) {
    return period >= 2 && length <= 80LL * (period - 1);
}

struct ShortOptions {
    int p = 0, q = 0, r = 0;
    float factor = 0.0f;
};

void launch_short_rows(ShortFamily family, const float* const* inputs, int nIn,
                       float* const* outputs, int nOut, int length, int series,
                       ShortOptions options, cudaStream_t stream);

}  // namespace tacuda::detail
