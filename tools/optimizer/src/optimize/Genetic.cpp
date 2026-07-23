/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "optimize/Genetic.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace topt {

namespace {

using Genome = std::vector<std::size_t>;

std::string genome_key(const Genome& g) {
  std::string key;
  key.reserve(g.size() * 4);
  for (std::size_t v : g) {
    key += std::to_string(v);
    key += '|';
  }
  return key;
}

void validate_config(const GeneticConfig& cfg) {
  if (cfg.population < 2) throw std::invalid_argument("genetic: population must be >= 2");
  if (cfg.maxGenerations < 1) throw std::invalid_argument("genetic: generations must be >= 1");
  if (cfg.elite < 0 || cfg.elite >= cfg.population)
    throw std::invalid_argument("genetic: elite must be in [0, population)");
  if (cfg.tournament < 1) throw std::invalid_argument("genetic: tournament must be >= 1");
  if (cfg.crossoverProb < 0.0 || cfg.crossoverProb > 1.0)
    throw std::invalid_argument("genetic: crossover probability must be in [0, 1]");
  if (cfg.mutationProb < 0.0 || cfg.mutationProb > 1.0)
    throw std::invalid_argument("genetic: mutation probability must be in [0, 1]");
  if (!(cfg.mutationSigmaSteps > 0.0))
    throw std::invalid_argument("genetic: mutation sigma must be positive");
  if (cfg.stagnationLimit < 1) throw std::invalid_argument("genetic: stagnation must be >= 1");
}

// Runs the whole GA loop; split out so run_genetic stays a thin wrapper.
class GeneticRun {
public:
  GeneticRun(const Series& bars, const IStrategy& strategy,
             const std::vector<std::vector<double>>& valueLists,
             IndicatorCache& cache, const GeneticConfig& cfg,
             const FitnessExpr& fitness)
      : bars_(bars), strategy_(strategy), defs_(strategy.params()),
        valueLists_(valueLists), cache_(cache), cfg_(cfg), fitness_(fitness),
        rng_(cfg.seed) {
    evalCfg_ = cfg_.eval;
    evalCfg_.quiet = true; // GA prints its own per-generation progress
  }

  GeneticResult run() {
    init_population();
    double bestFit = -std::numeric_limits<double>::infinity();
    int stagnation = 0;
    int gen = 0;
    while (gen < cfg_.maxGenerations) {
      ++gen;
      evaluate_population();
      const std::size_t genBest = best_of_population();
      const double genBestFit = fitness_of(genBest);
      if (genBestFit > bestFit) {
        bestFit = genBestFit;
        stagnation = 0;
      } else {
        ++stagnation;
      }
      result_.bestHistory.push_back(bestFit);
      if (!cfg_.quiet) {
        std::fprintf(stderr,
                     "[ga] gen %d/%d: best %.6g, evals %zu, stagnation %d/%d\n",
                     gen, cfg_.maxGenerations, bestFit, result_.evaluated.size(),
                     stagnation, cfg_.stagnationLimit);
      }
      if (stagnation >= cfg_.stagnationLimit) break;
      if (gen < cfg_.maxGenerations) next_generation();
      if (cache_.bytes() > evalCfg_.cacheBytes) cache_.clear();
    }
    result_.generations = gen;
    result_.evaluations = result_.evaluated.size();
    finalize_best();
    return std::move(result_);
  }

private:
  double fitness_of(std::size_t individual) const {
    const VariantResult& v = result_.evaluated[popEval_[individual]];
    return v.valid ? v.fitness : -std::numeric_limits<double>::infinity();
  }

  void init_population() {
    pop_.resize(static_cast<std::size_t>(cfg_.population));
    for (auto& genome : pop_) {
      genome.resize(valueLists_.size());
      for (std::size_t i = 0; i < valueLists_.size(); ++i) {
        std::uniform_int_distribution<std::size_t> pick(0, valueLists_[i].size() - 1);
        genome[i] = pick(rng_);
      }
    }
    popEval_.assign(pop_.size(), 0);
  }

  // Batch-simulates genomes not seen before (shared chunked/parallel path
  // with the grid runner); known genomes reuse their recorded result.
  void evaluate_population() {
    std::vector<ParamSet> pending;
    std::vector<std::string> pendingKeys;
    for (const auto& genome : pop_) {
      std::string key = genome_key(genome);
      if (evalMap_.count(key) || pendingSet_.count(key)) continue;
      pendingSet_.insert(key);
      pendingKeys.push_back(std::move(key));
      pending.push_back(materialize(defs_, valueLists_, genome));
    }
    if (!pending.empty()) {
      std::vector<VariantResult> batch =
          run_grid(bars_, strategy_, pending, cache_, evalCfg_, fitness_);
      for (std::size_t i = 0; i < batch.size(); ++i) {
        evalMap_[pendingKeys[i]] = result_.evaluated.size();
        result_.evaluated.push_back(std::move(batch[i]));
      }
    }
    pendingSet_.clear();
    for (std::size_t i = 0; i < pop_.size(); ++i) {
      popEval_[i] = evalMap_.at(genome_key(pop_[i]));
    }
  }

  std::size_t best_of_population() const {
    std::size_t best = 0;
    for (std::size_t i = 1; i < pop_.size(); ++i) {
      if (fitness_of(i) > fitness_of(best)) best = i;
    }
    return best;
  }

  std::size_t tournament_pick() {
    std::uniform_int_distribution<std::size_t> pick(0, pop_.size() - 1);
    std::size_t best = pick(rng_);
    for (int k = 1; k < cfg_.tournament; ++k) {
      std::size_t challenger = pick(rng_);
      if (fitness_of(challenger) > fitness_of(best)) best = challenger;
    }
    return best;
  }

  Genome crossover(const Genome& a, const Genome& b) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (uni(rng_) >= cfg_.crossoverProb) return a;
    Genome child(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      child[i] = uni(rng_) < 0.5 ? a[i] : b[i];
    }
    return child;
  }

  // Moves a gene by a gaussian number of axis steps (sigma from config),
  // clamped to the axis; a draw that rounds to zero still moves +/-1 step
  // so a selected mutation always changes the genome.
  void mutate(Genome& g) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, cfg_.mutationSigmaSteps);
    for (std::size_t i = 0; i < g.size(); ++i) {
      const std::size_t size = valueLists_[i].size();
      if (size < 2) continue; // fixed parameter, nothing to mutate
      if (uni(rng_) >= cfg_.mutationProb) continue;
      long long delta = std::llround(gauss(rng_));
      if (delta == 0) delta = uni(rng_) < 0.5 ? -1 : 1;
      long long idx = static_cast<long long>(g[i]) + delta;
      idx = std::max(0ll, std::min(idx, static_cast<long long>(size) - 1));
      g[i] = static_cast<std::size_t>(idx);
    }
  }

  void next_generation() {
    // Stable fitness ranking; ties keep lower population index (determinism).
    std::vector<std::size_t> order(pop_.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return fitness_of(a) > fitness_of(b);
    });

    std::vector<Genome> next;
    next.reserve(pop_.size());
    for (int e = 0; e < cfg_.elite; ++e) {
      next.push_back(pop_[order[static_cast<std::size_t>(e)]]);
    }
    while (next.size() < pop_.size()) {
      const Genome& a = pop_[tournament_pick()];
      const Genome& b = pop_[tournament_pick()];
      Genome child = crossover(a, b);
      mutate(child);
      next.push_back(std::move(child));
    }
    pop_ = std::move(next);
  }

  void finalize_best() {
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < result_.evaluated.size(); ++i) {
      const VariantResult& v = result_.evaluated[i];
      if (v.valid && (v.fitness > best || !result_.has_best())) {
        best = v.fitness;
        result_.bestIndex = i;
      }
    }
  }

  const Series& bars_;
  const IStrategy& strategy_;
  std::vector<ParamDef> defs_;
  const std::vector<std::vector<double>>& valueLists_;
  IndicatorCache& cache_;
  const GeneticConfig& cfg_;
  const FitnessExpr& fitness_;
  GridConfig evalCfg_;
  std::mt19937_64 rng_;

  std::vector<Genome> pop_;
  std::vector<std::size_t> popEval_; // individual -> index into evaluated
  std::unordered_map<std::string, std::size_t> evalMap_;
  std::unordered_set<std::string> pendingSet_;
  GeneticResult result_;
};

} // namespace

GeneticResult run_genetic(const Series& bars, const IStrategy& strategy,
                          const std::vector<GridAxis>& axes,
                          const std::vector<std::pair<std::string, double>>& sets,
                          IndicatorCache& cache, const GeneticConfig& cfg,
                          const FitnessExpr& fitness) {
  validate_config(cfg);
  const std::vector<ParamDef> defs = strategy.params();
  const std::vector<std::vector<double>> valueLists = build_value_lists(defs, axes, sets);
  bool anyAxis = false;
  for (const auto& list : valueLists) {
    if (list.size() > 1) anyAxis = true;
  }
  if (!anyAxis) {
    throw std::invalid_argument("genetic: search space is a single point; add --grid axes");
  }
  GeneticRun run(bars, strategy, valueLists, cache, cfg, fitness);
  return run.run();
}

} // namespace topt
