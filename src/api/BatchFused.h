/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <cuda_runtime.h>
#include <tacuda.h>

namespace tacuda { namespace detail {

enum class FusedFamily {
    None, Sma, Ema, Rsi, Atr, Macd, Bbands, Stochastic,
    Wma, Natr, Stddev, Sum, Min, Max, Obv, Mfi, Ad, Adosc
};

// No allocations, configuration mutation or host/device transfers. The
// descriptor is passed by value and every device row owns its entire state.
struct FusedCall {
    FusedFamily family = FusedFamily::None;
    int p = 0, q = 0, r = 0, maType = 1;
    float up = 2, down = 2;

    bool enabled(int seriesLen, int numSeries) const;
    ctStatus_t launch(const float* const* in, float* const* out,
                      int seriesLen, int numSeries, cudaStream_t stream) const;
};

inline FusedCall fused_call(FusedFamily f, int p = 0, int q = 0, int r = 0,
                            int maType = 1, float up = 2, float down = 2) {
    return {f, p, q, r, maType, up, down};
}

}}  // namespace tacuda::detail
