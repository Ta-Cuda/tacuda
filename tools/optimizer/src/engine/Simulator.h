/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_ENGINE_SIMULATOR_H
#define TACUDA_OPT_ENGINE_SIMULATOR_H

#include <vector>

#include "core/Types.h"

namespace topt {

// Execution model: costs and sizing. Signals are target positions in
// {-1, 0, +1}; the actual position is signal * qty units.
struct ExecConfig {
  double commissionBps = 0.0;   // per order side, of traded notional
  double commissionFixed = 0.0; // per order side, currency
  double slippageTicks = 0.0;   // fill worsens by ticks * tickSize per side
  double tickSize = 0.01;
  double qty = 1.0;             // position size, units
  double capital = 10000.0;     // starting cash (equity baseline)
  int barsPerYear = 252;        // Sharpe/Sortino annualization
};

// One round trip. Costs (commission + the slippage already embedded in fill
// prices) are included in pnl; on a stop-and-reverse fill the order cost is
// split per unit between the closing and the opening trade.
struct Trade {
  int entryBar = 0;
  int exitBar = 0;
  int dir = 0; // +1 long, -1 short
  double entryPrice = 0.0;
  double exitPrice = 0.0;
  double pnl = 0.0;
};

struct SimResult {
  std::vector<Trade> trades;
  std::vector<double> equity; // marked at each bar close, PnL in double
  double initialCapital = 0.0;
  double finalEquity = 0.0;
  int barsInPosition = 0;
};

// Half-open bar range [begin, end) to simulate; end < 0 means bars.size().
// Used by walk-forward runs to trade only inside an IS or OOS segment while
// signals keep their full-series history context.
struct SimRange {
  int begin = 0;
  int end = -1;
};

// Single pass over the series. desired[i] is the position decided at the
// close of bar i and is filled at the OPEN of bar i+1 — look-ahead is
// excluded by construction. A position still open after the last bar is
// force-closed at the last close so every trade completes.
//
// With a range, trading starts flat at range.begin; when range.begin > 0 the
// decision from bar range.begin-1 is filled at open[range.begin] (a signal
// formed on the last bar before the segment executes on its first bar).
// equity covers [begin, end) (local indices); Trade bar numbers stay
// absolute series indices.
SimResult simulate(const Series& bars, const Signal* desired, const ExecConfig& cfg,
                   SimRange range = {});

} // namespace topt

#endif // TACUDA_OPT_ENGINE_SIMULATOR_H
