/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <stdexcept>

#include "TestBackend.h"
#include "core/Fitness.h"
#include "core/ParamSpace.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"
#include "optimize/GridSearch.h"
#include "strategy/IStrategy.h"

using namespace topt;
using topt::testing::TestBackend;

namespace {

std::vector<ParamSet> ma_cross_variants(const IStrategy& s,
                                        const std::string& fastSpec,
                                        const std::string& slowSpec) {
  return enumerate_variants(s.params(),
                            {parse_grid_spec(fastSpec), parse_grid_spec(slowSpec)}, {});
}

GridConfig quiet_config() {
  GridConfig cfg;
  cfg.quiet = true;
  cfg.threads = 4;
  return cfg;
}

} // namespace

TEST(IndicatorCache, SharedRequestsComputeOnce) {
  Series bars = generate_synthetic(1000, 1);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);

  IndicatorRequest ema20{Family::EMA, Column::Close, {20.0}};
  const auto& a = cache.get(ema20);
  const auto& b = cache.get(ema20);
  EXPECT_EQ(&a, &b);
  EXPECT_EQ(backend.computeCalls, 1);
  EXPECT_EQ(cache.compute_count(), 1u);

  // Params quantization: 20.0000000001 aliases to the same entry.
  IndicatorRequest emaAlias{Family::EMA, Column::Close, {20.0000000001}};
  cache.get(emaAlias);
  EXPECT_EQ(backend.computeCalls, 1);

  IndicatorRequest ema21{Family::EMA, Column::Close, {21.0}};
  cache.get(ema21);
  EXPECT_EQ(backend.computeCalls, 2);

  cache.clear();
  cache.get(ema20);
  EXPECT_EQ(backend.computeCalls, 3);
}

TEST(IndicatorCache, FirstValidTracksWarmup) {
  Series bars = generate_synthetic(100, 2);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  const auto& entry = cache.get({Family::EMA, Column::Close, {20.0}});
  ASSERT_EQ(entry.firstValid.size(), 1u);
  EXPECT_EQ(entry.firstValid[0], 19); // EMA warm-up: period-1
}

TEST(GridSearch, CacheDeduplicatesAcrossVariants) {
  Series bars = generate_synthetic(2000, 4);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  // fast in {5,10}, slow=20: unique EMAs are 5, 10, 20 -> 3 computes, not 4.
  auto variants = ma_cross_variants(*s, "fast=5,10", "slow=20");
  ASSERT_EQ(variants.size(), 2u);

  auto fitness = FitnessExpr::parse("netprofit", fitness_variable_names());
  auto results = run_grid(bars, *s, variants, cache, quiet_config(), fitness);
  ASSERT_EQ(results.size(), 2u);
  EXPECT_TRUE(results[0].valid);
  EXPECT_TRUE(results[1].valid);
  EXPECT_EQ(backend.computeCalls, 3);
}

TEST(GridSearch, DeterministicAcrossRunsAndThreadCounts) {
  Series bars = generate_synthetic(5000, 9);
  auto s = make_strategy("ma_cross");
  auto variants = ma_cross_variants(*s, "fast=5..25:5", "slow=30..90:20");
  auto fitness = FitnessExpr::parse("netprofit - 2*maxdd", fitness_variable_names());

  auto run_once = [&](int threads) {
    TestBackend backend(bars);
    IndicatorCache cache(bars, backend);
    GridConfig cfg = quiet_config();
    cfg.threads = threads;
    return run_grid(bars, *s, variants, cache, cfg, fitness);
  };

  auto r1 = run_once(1);
  auto r8 = run_once(8);
  ASSERT_EQ(r1.size(), r8.size());
  for (std::size_t i = 0; i < r1.size(); ++i) {
    EXPECT_EQ(r1[i].valid, r8[i].valid);
    EXPECT_DOUBLE_EQ(r1[i].fitness, r8[i].fitness);
    EXPECT_DOUBLE_EQ(r1[i].metrics.netProfit, r8[i].metrics.netProfit);
    EXPECT_EQ(r1[i].metrics.numTrades, r8[i].metrics.numTrades);
  }
}

TEST(GridSearch, ChunkedRunMatchesUnchunked) {
  Series bars = generate_synthetic(3000, 13);
  auto s = make_strategy("ma_cross");
  auto variants = ma_cross_variants(*s, "fast=5..20:5", "slow=30..60:10");
  auto fitness = FitnessExpr::parse("sharpe", fitness_variable_names());

  TestBackend big(bars);
  IndicatorCache bigCache(bars, big);
  auto reference = run_grid(bars, *s, variants, bigCache, quiet_config(), fitness);

  TestBackend small(bars);
  IndicatorCache smallCache(bars, small);
  GridConfig tiny = quiet_config();
  tiny.cacheBytes = 2 * static_cast<std::size_t>(bars.size()) * sizeof(float) + 1;
  auto chunked = run_grid(bars, *s, variants, smallCache, tiny, fitness);

  ASSERT_EQ(reference.size(), chunked.size());
  for (std::size_t i = 0; i < reference.size(); ++i) {
    EXPECT_DOUBLE_EQ(reference[i].fitness, chunked[i].fitness) << "variant " << i;
  }
  // Chunking recomputes overlapping requests: never fewer backend calls.
  EXPECT_GE(small.computeCalls, big.computeCalls);
}

TEST(GridSearch, InvalidVariantsAreSkippedNotFatal) {
  Series bars = generate_synthetic(1000, 17);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  // fast=40 > slow=20 is invalid; fast=10 < slow=20 is fine.
  auto variants = ma_cross_variants(*s, "fast=10,40", "slow=20");
  ASSERT_EQ(variants.size(), 2u);
  auto fitness = FitnessExpr::parse("netprofit", fitness_variable_names());
  auto results = run_grid(bars, *s, variants, cache, quiet_config(), fitness);

  EXPECT_TRUE(results[0].valid);
  EXPECT_FALSE(results[1].valid);
  EXPECT_FALSE(results[1].error.empty());
}

TEST(GridSearch, FitnessOrdersResults) {
  Series bars = generate_synthetic(4000, 23);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("rsi_reversion");
  auto variants = enumerate_variants(
      s->params(), {parse_grid_spec("period=7..28:7"), parse_grid_spec("low=20..35:5")},
      {{"high", 70.0}});
  auto fitness = FitnessExpr::parse("netprofit", fitness_variable_names());
  auto results = run_grid(bars, *s, variants, cache, quiet_config(), fitness);

  for (const auto& r : results) {
    ASSERT_TRUE(r.valid) << r.error;
    EXPECT_DOUBLE_EQ(r.fitness, r.metrics.netProfit);
  }
}

TEST(GridSearch, MinTradesMarksSparseVariantsSkipped) {
  Series bars = generate_synthetic(2000, 31);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  auto variants = ma_cross_variants(*s, "fast=5", "slow=20");
  auto fitness = FitnessExpr::parse("netprofit", fitness_variable_names());

  auto plain = run_grid(bars, *s, variants, cache, quiet_config(), fitness);
  ASSERT_EQ(plain.size(), 1u);
  ASSERT_TRUE(plain[0].valid);
  ASSERT_GT(plain[0].metrics.numTrades, 0);

  GridConfig strict = quiet_config();
  strict.minTrades = plain[0].metrics.numTrades + 1;
  auto filtered = run_grid(bars, *s, variants, cache, strict, fitness);
  ASSERT_EQ(filtered.size(), 1u);
  EXPECT_FALSE(filtered[0].valid);
  EXPECT_NE(filtered[0].error.find("trades"), std::string::npos) << filtered[0].error;
  // The metrics stay available for the CSV even though the variant is skipped.
  EXPECT_EQ(filtered[0].metrics.numTrades, plain[0].metrics.numTrades);

  GridConfig loose = quiet_config();
  loose.minTrades = plain[0].metrics.numTrades;
  auto kept = run_grid(bars, *s, variants, cache, loose, fitness);
  EXPECT_TRUE(kept[0].valid);
}

TEST(GridSearch, RobustFitnessAveragesValidAxisNeighbours) {
  // 3x3 grid in enumerate_variants order (second axis fastest):
  // fitness(i, j) = 10*i + j; the centre's lower neighbour (2, 1) is invalid.
  std::vector<VariantResult> results(9);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      VariantResult& v = results[i * 3 + j];
      v.valid = true;
      v.fitness = 10.0 * static_cast<double>(i) + static_cast<double>(j);
    }
  }
  results[7].valid = false; // (2, 1)
  compute_robust_fitness(results, {3, 3});

  // Corner (0, 0): itself, (0, 1) and (1, 0).
  EXPECT_EQ(results[0].neighborhood, 3);
  EXPECT_NEAR(results[0].robustFitness, (0.0 + 1.0 + 10.0) / 3.0, 1e-12);
  // Centre (1, 1): itself, (0, 1), (1, 0), (1, 2); (2, 1) is invalid.
  EXPECT_EQ(results[4].neighborhood, 4);
  EXPECT_NEAR(results[4].robustFitness, (11.0 + 1.0 + 10.0 + 12.0) / 4.0, 1e-12);
  // Edge (2, 2): itself, (1, 2); (2, 1) is invalid.
  EXPECT_EQ(results[8].neighborhood, 2);
  EXPECT_NEAR(results[8].robustFitness, (22.0 + 12.0) / 2.0, 1e-12);
  // The invalid variant stays out of the ranking.
  EXPECT_EQ(results[7].neighborhood, 0);
  EXPECT_DOUBLE_EQ(results[7].robustFitness, 0.0);

  // A fixed (single-value) axis contributes no neighbours.
  std::vector<VariantResult> line(3);
  for (std::size_t k = 0; k < 3; ++k) {
    line[k].valid = true;
    line[k].fitness = static_cast<double>(k);
  }
  compute_robust_fitness(line, {1, 3});
  EXPECT_EQ(line[1].neighborhood, 3);
  EXPECT_NEAR(line[1].robustFitness, 1.0, 1e-12);

  // Shape mismatch is an error, not a silent misalignment.
  EXPECT_THROW(compute_robust_fitness(results, {2, 2}), std::invalid_argument);
}

TEST(GridSearch, RobustRankingPrefersPlateausOverSpikes) {
  // One axis, seven steps: an isolated spike (9 next to 0) against a plateau
  // of sixes. Raw fitness picks the spike, the neighbourhood mean the plateau.
  const double fitness[] = {9.0, 0.0, 5.0, 6.0, 6.0, 6.0, 5.0};
  std::vector<VariantResult> results(7);
  for (std::size_t k = 0; k < 7; ++k) {
    results[k].valid = true;
    results[k].fitness = fitness[k];
  }
  compute_robust_fitness(results, {7});

  std::size_t bestRaw = 0, bestRobust = 0;
  for (std::size_t k = 1; k < results.size(); ++k) {
    if (results[k].fitness > results[bestRaw].fitness) bestRaw = k;
    if (results[k].robustFitness > results[bestRobust].robustFitness) bestRobust = k;
  }
  EXPECT_EQ(bestRaw, 0u);
  EXPECT_EQ(bestRobust, 4u);
  EXPECT_NEAR(results[0].robustFitness, 4.5, 1e-12);
  EXPECT_NEAR(results[4].robustFitness, 6.0, 1e-12);
}

TEST(GridSearch, RobustFitnessFollowsEnumerationOrderOnARealGrid) {
  // The neighbour lookup relies on the enumerate_variants() layout; check it
  // against neighbours found by parameter value on a real run.
  Series bars = generate_synthetic(4000, 41);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  std::vector<GridAxis> axes = {parse_grid_spec("fast=3..15:3"),
                                parse_grid_spec("slow=20..60:10")};
  auto variants = enumerate_variants(s->params(), axes, {});
  auto fitness = FitnessExpr::parse("netprofit", fitness_variable_names());
  auto results = run_grid(bars, *s, variants, cache, quiet_config(), fitness);
  ASSERT_EQ(results.size(), 25u);
  compute_robust_fitness(results,
                         value_list_sizes(build_value_lists(s->params(), axes, {})));

  auto find = [&](double fast, double slow) -> const VariantResult& {
    for (const auto& r : results) {
      if (r.params.get("fast") == fast && r.params.get("slow") == slow) return r;
    }
    throw std::logic_error("variant not found");
  };
  for (const auto& r : results) ASSERT_TRUE(r.valid) << r.params.to_string();

  const VariantResult& centre = find(9, 40);
  EXPECT_EQ(centre.neighborhood, 5);
  EXPECT_NEAR(centre.robustFitness,
              (centre.fitness + find(6, 40).fitness + find(12, 40).fitness +
               find(9, 30).fitness + find(9, 50).fitness) / 5.0,
              1e-9);

  const VariantResult& corner = find(3, 20);
  EXPECT_EQ(corner.neighborhood, 3);
  EXPECT_NEAR(corner.robustFitness,
              (corner.fitness + find(6, 20).fitness + find(3, 30).fitness) / 3.0, 1e-9);

  const VariantResult& edge = find(15, 40);
  EXPECT_EQ(edge.neighborhood, 4);
  EXPECT_NEAR(edge.robustFitness,
              (edge.fitness + find(12, 40).fitness + find(15, 30).fitness +
               find(15, 50).fitness) / 4.0,
              1e-9);
}

TEST(GridSearch, NonFiniteFitnessIsAnErrorNotARankedResult) {
  Series bars = generate_synthetic(600, 5);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  auto variants = ma_cross_variants(*s, "fast=5", "slow=20");
  auto fitness = FitnessExpr::parse("1e308*1e308", fitness_variable_names());
  auto results = run_grid(bars, *s, variants, cache, quiet_config(), fitness);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FALSE(results[0].valid);
  EXPECT_NE(results[0].error.find("non-finite"), std::string::npos) << results[0].error;
}
