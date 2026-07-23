/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_OPTIMIZE_GRIDSEARCH_H
#define TACUDA_OPT_OPTIMIZE_GRIDSEARCH_H

#include <cstddef>
#include <string>
#include <vector>

#include "core/Fitness.h"
#include "engine/IndicatorCache.h"
#include "engine/Metrics.h"
#include "engine/Simulator.h"
#include "strategy/IStrategy.h"

namespace topt {

struct VariantResult {
  ParamSet params;
  MetricsResult metrics;
  double fitness = 0.0;
  bool valid = false;
  std::string error; // set when the variant was skipped or failed
};

struct GridConfig {
  ExecConfig exec;
  int threads = 0;                          // 0 = hardware concurrency
  std::size_t cacheBytes = 2ull << 30;      // indicator cache budget per chunk
  bool quiet = false;                       // suppress stderr progress
  SimRange range{};                         // trade only inside this bar range
};

// Runs one variant end-to-end: indicators (through the cache), signals with
// the warm-up mask applied, next-bar-open simulation, metrics. `fitness`
// is optional (nullptr -> fitness stays 0). `simOut` is optional and
// receives the full simulation result (trades + equity) when non-null.
// Signals are computed over the whole series (full history context);
// `range` restricts where trading and metrics happen.
VariantResult run_variant(const Series& bars, const IStrategy& strategy,
                          const ParamSet& ps, IndicatorCache& cache,
                          const ExecConfig& exec, const FitnessExpr* fitness,
                          SimResult* simOut = nullptr, SimRange range = {});

// Exhaustive grid run. Variants are processed in deterministic order in
// memory-bounded chunks: per chunk, unique indicator requests are computed
// once through the cache, then simulations run in parallel on CPU threads.
// The cache is cleared between chunks to respect cfg.cacheBytes.
std::vector<VariantResult> run_grid(const Series& bars, const IStrategy& strategy,
                                    const std::vector<ParamSet>& variants,
                                    IndicatorCache& cache, const GridConfig& cfg,
                                    const FitnessExpr& fitness);

} // namespace topt

#endif // TACUDA_OPT_OPTIMIZE_GRIDSEARCH_H
