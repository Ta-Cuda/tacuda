/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_DATA_SYNTHETIC_H
#define TACUDA_OPT_DATA_SYNTHETIC_H

#include <cstdint>

#include "core/Types.h"

namespace topt {

// Deterministic synthetic OHLCV series: geometric random walk with a mild
// trend-cycle component so that MA/RSI/BBANDS strategies produce trades.
// Bars are daily (weekends skipped) starting 2019-01-01, which lets
// weekday-aware strategies see a realistic Mon..Fri calendar.
// The same (bars, seed) pair always yields the same series.
Series generate_synthetic(int bars, std::uint64_t seed);

} // namespace topt

#endif // TACUDA_OPT_DATA_SYNTHETIC_H
