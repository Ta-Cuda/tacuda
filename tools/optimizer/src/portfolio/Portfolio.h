/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_PORTFOLIO_PORTFOLIO_H
#define TACUDA_OPT_PORTFOLIO_PORTFOLIO_H

#include <string>
#include <vector>

#include "backend/BatchBackend.h"
#include "core/Fitness.h"
#include "engine/Metrics.h"
#include "engine/Simulator.h"
#include "strategy/IStrategy.h"

namespace topt {

// Brings a multi-symbol data set to one common bar grid, in place.
// - When every series carries usable timestamps (non-zero, strictly
//   increasing), each series is filtered to the timestamp intersection.
// - Otherwise all series must already have equal lengths.
// `log` receives one line describing what happened. Throws
// std::invalid_argument when alignment is impossible (no overlap, or
// unequal lengths without timestamps).
void align_portfolio(std::vector<Series>& series, std::string& log);

struct PortfolioConfig {
  // exec.capital is the TOTAL portfolio capital; each symbol trades an
  // equal share (capital / numSymbols) with the same qty and costs.
  ExecConfig exec;
};

struct SymbolResult {
  std::string name;
  bool valid = false;
  std::string error;   // why the symbol was skipped
  MetricsResult metrics;
  SimResult sim;       // per-symbol trades + equity (capital-share base)
};

struct PortfolioResult {
  std::vector<SymbolResult> symbols;
  int validSymbols = 0;
  int bars = 0;

  // Aggregated view: equity[t] is the sum of per-symbol equities (skipped
  // symbols contribute their flat capital share); trades are all symbols'
  // trades merged in chronological order (by exit bar, then entry bar).
  std::vector<double> equity;
  std::vector<Trade> trades;

  // Metrics over the aggregated curve + merged trade list. exposure is the
  // mean of per-symbol exposures (share of bars each sub-account holds a
  // position, averaged over symbols).
  MetricsResult metrics;
  double fitness = 0.0;

  // Phase timing: the indicator phase covers every backend.compute() call
  // (uploads + kernels + downloads for the GPU paths); the signal phase
  // covers per-symbol signals() + simulation + metrics. Lets backends be
  // compared on the part they actually own.
  double indicatorSec = 0.0;
  double signalSec = 0.0;

  bool has_valid() const { return validSymbols > 0; }
};

// Multi-symbol backtest with shared parameters: every indicator request is
// computed for all symbols in one batch-backend call, then each symbol runs
// the usual signals -> warm-up mask -> next-bar-open simulation on its own
// capital share. Series must be aligned (equal lengths) beforehand.
// `fitness` is optional (nullptr -> portfolio fitness stays 0).
// Throws std::invalid_argument on an invalid parameter set or input shape
// mismatches; per-symbol data problems mark that symbol invalid instead.
PortfolioResult run_portfolio(const std::vector<Series>& aligned,
                              const std::vector<std::string>& names,
                              const IStrategy& strategy, const ParamSet& ps,
                              IBatchIndicatorBackend& backend,
                              const PortfolioConfig& cfg,
                              const FitnessExpr* fitness);

} // namespace topt

#endif // TACUDA_OPT_PORTFOLIO_PORTFOLIO_H
