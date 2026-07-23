/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_ENGINE_MONTECARLO_H
#define TACUDA_OPT_ENGINE_MONTECARLO_H

#include <cstdint>
#include <string>
#include <vector>

#include "engine/Simulator.h"

namespace topt {

// Trade-order randomization mode.
//  - Permute: each iteration is a random permutation of the realized trade
//    PnLs. Net profit is order-invariant; the drawdown path is not — this
//    answers "how bad could the same trades have felt in another order".
//  - Resample: each iteration draws numTrades PnLs with replacement
//    (bootstrap), so both net-profit and drawdown distributions vary.
enum class McMode { Permute, Resample };

const char* to_string(McMode m);
McMode parse_mc_mode(const std::string& text); // "permute" | "resample"

struct McConfig {
  int iterations = 0; // 0 = disabled (run_monte_carlo requires > 0)
  McMode mode = McMode::Permute;
  std::uint64_t seed = 42;
};

// Distribution summary of one statistic across iterations.
// `worst` is the pessimistic tail: min for net profit, max for drawdown.
struct McPercentiles {
  double p5 = 0.0;
  double p25 = 0.0;
  double p50 = 0.0;
  double p75 = 0.0;
  double p95 = 0.0;
  double worst = 0.0;
};

struct McResult {
  int iterations = 0;
  McMode mode = McMode::Permute;
  int numTrades = 0;

  // Statistics of the original (chronological) trade order, computed on the
  // same trade-level equity path the iterations use — not on the bar-level
  // curve, so open-position excursions between entry and exit are not seen.
  double originalNetProfit = 0.0;
  double originalMaxDrawdown = 0.0;

  McPercentiles netProfit;   // degenerate (all equal) under Permute
  McPercentiles maxDrawdown; // absolute currency drawdown

  double probNetLoss = 0.0;        // share of iterations with net profit < 0
  double probDdAboveOriginal = 0.0; // share with maxDD > original order's
};

// Runs cfg.iterations randomizations of the closed-trade PnL sequence and
// summarizes the resulting equity-path statistics. Deterministic for a given
// (trades, cfg) on a given platform. Trades must be in chronological order
// (as produced by simulate / run_portfolio). With no trades returns a
// zero-filled result with numTrades == 0.
// Throws std::invalid_argument when cfg.iterations <= 0.
McResult run_monte_carlo(const std::vector<Trade>& trades, double capital,
                         const McConfig& cfg);

} // namespace topt

#endif // TACUDA_OPT_ENGINE_MONTECARLO_H
