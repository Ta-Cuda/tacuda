/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_OPTIMIZE_GENETIC_H
#define TACUDA_OPT_OPTIMIZE_GENETIC_H

#include <cstdint>
#include <vector>

#include "core/ParamSpace.h"
#include "optimize/GridSearch.h"

namespace topt {

// Genetic search over the same discrete space a grid run would enumerate:
// a genome is one value-index per strategy parameter (grid axes vary, --set
// and defaults stay fixed), so results are directly comparable with the
// exhaustive grid on the same axes.
struct GeneticConfig {
  GridConfig eval;               // exec model, threads, cache budget, range
  int population = 128;
  int maxGenerations = 60;
  int elite = 8;                 // best individuals copied unchanged
  int tournament = 4;            // tournament selection size
  double crossoverProb = 0.9;    // per child: uniform crossover vs clone
  double mutationProb = 0.15;    // per gene
  double mutationSigmaSteps = 2.0; // gaussian sigma, in units of axis steps
  int stagnationLimit = 12;      // early stop: generations without improvement
  std::uint64_t seed = 42;
  bool quiet = false;            // per-generation stderr progress
};

struct GeneticResult {
  // Every unique genome evaluated, in evaluation order (metrics on
  // eval.range). Reports reuse the grid writers on this list.
  std::vector<VariantResult> evaluated;
  std::size_t bestIndex = static_cast<std::size_t>(-1); // into evaluated
  int generations = 0;
  std::size_t evaluations = 0;   // == evaluated.size(), unique genomes
  std::vector<double> bestHistory; // best fitness after each generation

  bool has_best() const { return bestIndex != static_cast<std::size_t>(-1); }
};

// Deterministic for a fixed (config, seed, data): duplicate genomes are
// never re-simulated, invalid parameter combinations get -inf fitness,
// elitism makes the best fitness non-decreasing between generations.
GeneticResult run_genetic(const Series& bars, const IStrategy& strategy,
                          const std::vector<GridAxis>& axes,
                          const std::vector<std::pair<std::string, double>>& sets,
                          IndicatorCache& cache, const GeneticConfig& cfg,
                          const FitnessExpr& fitness);

} // namespace topt

#endif // TACUDA_OPT_OPTIMIZE_GENETIC_H
