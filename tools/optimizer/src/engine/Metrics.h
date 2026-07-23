/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_ENGINE_METRICS_H
#define TACUDA_OPT_ENGINE_METRICS_H

#include <map>
#include <string>
#include <vector>

#include "engine/Simulator.h"

namespace topt {

// Performance metrics of one simulation. All money values are in account
// currency; ratios are unitless. profitFactor is capped at 1e9 when there
// are no losing trades.
struct MetricsResult {
  double netProfit = 0.0;
  double grossProfit = 0.0;
  double grossLoss = 0.0;    // positive magnitude
  double profitFactor = 0.0;
  double maxDrawdown = 0.0;  // absolute, from the equity curve
  double maxDrawdownPct = 0.0;
  double sharpe = 0.0;       // annualized, rf = 0, from per-bar equity diffs
  double sortino = 0.0;
  double winRate = 0.0;      // 0..1
  double avgTrade = 0.0;
  double exposure = 0.0;     // fraction of bars with an open position
  int numTrades = 0;
};

MetricsResult compute_metrics(const SimResult& sim, int barsPerYear);

// Variable names usable in --fitness expressions, and the corresponding
// values for a computed metrics set.
const std::vector<std::string>& fitness_variable_names();
std::map<std::string, double> fitness_variables(const MetricsResult& m);

} // namespace topt

#endif // TACUDA_OPT_ENGINE_METRICS_H
