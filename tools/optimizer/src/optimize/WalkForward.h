/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_OPTIMIZE_WALKFORWARD_H
#define TACUDA_OPT_OPTIMIZE_WALKFORWARD_H

#include <string>
#include <vector>

#include "optimize/Genetic.h"
#include "optimize/GridSearch.h"

namespace topt {

// One rolling window: in-sample [isBegin, isEnd), out-of-sample [isEnd, oosEnd).
struct WfoWindow {
  int isBegin = 0;
  int isEnd = 0;
  int oosEnd = 0;

  int is_bars() const { return isEnd - isBegin; }
  int oos_bars() const { return oosEnd - isEnd; }
};

// Rolling split: every window has the same IS/OOS length; window w starts at
// w * L_oos, so consecutive OOS segments tile [L_is, L_is + K*L_oos)
// contiguously and the stitched OOS equity has no gaps or overlaps. Up to
// (L_is + L_oos) leftover bars at the tail stay unused. Throws when the
// series is too short for the requested split.
std::vector<WfoWindow> make_wfo_windows(int n, int windows, double isFraction);

// Walk-forward efficiency: per-bar OOS net profit relative to per-bar IS net
// profit of the chosen variants. NaN when IS profit is not positive (the
// ratio is meaningless then).
double compute_wfe(double isNetProfit, double isBars, double oosNetProfit,
                   double oosBars);

enum class WfoOptimizer { Grid, Genetic };

struct WfoConfig {
  GridConfig grid;        // exec model, threads, cache budget; grid optimizer
  GeneticConfig genetic;  // used when optimizer == Genetic (its eval.range is set per window)
  WfoOptimizer optimizer = WfoOptimizer::Grid;
  int windows = 6;
  double isFraction = 0.75; // in-sample share of each window
};

struct WfoWindowResult {
  WfoWindow window;
  bool optimized = false;   // a valid best was found on IS
  ParamSet params;
  double isFitness = 0.0;
  MetricsResult isMetrics;
  double oosFitness = 0.0;
  MetricsResult oosMetrics;
  std::vector<Trade> oosTrades;   // absolute bar indices
  std::vector<double> oosEquity;  // local to the OOS range, starts at capital
  std::string note;               // why the window was skipped, if it was
};

struct WfoResult {
  std::vector<WfoWindowResult> windows;
  std::vector<double> stitchedEquity; // OOS segments chained by equity deltas
  MetricsResult oosOverall;           // metrics over the stitched OOS curve
  double wfe = 0.0;                   // NaN when undefined
  std::size_t isEvaluations = 0;      // variants simulated during IS optimization
  int tailUnused = 0;                 // bars after the last OOS segment
};

// Runs the full walk-forward loop: per window optimizes on IS (grid or
// genetic), evaluates the chosen params on OOS, stitches OOS equity and
// aggregates overall metrics + WFE. Indicators are computed once over the
// full series (they are causal, so this leaks nothing) and shared between
// windows through the cache. Skipped windows (no valid IS variant)
// contribute flat equity and are excluded from the WFE sums.
WfoResult run_wfo(const Series& bars, const IStrategy& strategy,
                  const std::vector<GridAxis>& axes,
                  const std::vector<std::pair<std::string, double>>& sets,
                  IndicatorCache& cache, const WfoConfig& cfg,
                  const FitnessExpr& fitness);

} // namespace topt

#endif // TACUDA_OPT_OPTIMIZE_WALKFORWARD_H
