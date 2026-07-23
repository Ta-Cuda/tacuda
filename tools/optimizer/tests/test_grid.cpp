/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

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
