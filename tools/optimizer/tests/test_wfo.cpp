/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "TestBackend.h"
#include "core/Fitness.h"
#include "core/ParamSpace.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"
#include "optimize/WalkForward.h"
#include "strategy/IStrategy.h"

using namespace topt;
using topt::testing::TestBackend;

namespace {

WfoConfig quiet_wfo(int windows, double isFraction) {
  WfoConfig cfg;
  cfg.grid.quiet = true;
  cfg.grid.threads = 4;
  cfg.windows = windows;
  cfg.isFraction = isFraction;
  return cfg;
}

WfoResult run_wfo_on(const Series& bars, const WfoConfig& cfg,
                     const std::vector<GridAxis>& axes,
                     const std::string& fitnessExpr = "netprofit") {
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  auto fitness = FitnessExpr::parse(fitnessExpr, fitness_variable_names());
  return run_wfo(bars, *s, axes, {}, cache, cfg, fitness);
}

} // namespace

TEST(WfoWindows, RollingSplitTilesTheSeries) {
  // n=1000, K=4, is=0.75: L_oos = floor(1000/(4+3)) = 142, L_is = 426.
  auto w = make_wfo_windows(1000, 4, 0.75);
  ASSERT_EQ(w.size(), 4u);
  EXPECT_EQ(w[0].isBegin, 0);
  EXPECT_EQ(w[0].isEnd, 426);
  EXPECT_EQ(w[0].oosEnd, 568);
  for (const auto& win : w) {
    EXPECT_EQ(win.is_bars(), 426);
    EXPECT_EQ(win.oos_bars(), 142);
  }
  // OOS segments are contiguous: next window's IS ends where this OOS ends.
  for (std::size_t i = 1; i < w.size(); ++i) {
    EXPECT_EQ(w[i].isEnd, w[i - 1].oosEnd);
    EXPECT_EQ(w[i].isBegin, w[i - 1].isBegin + 142);
  }
  EXPECT_LE(w.back().oosEnd, 1000);
}

TEST(WfoWindows, SingleWindowIsPlainIsOosSplit) {
  auto w = make_wfo_windows(1000, 1, 0.8);
  ASSERT_EQ(w.size(), 1u);
  EXPECT_EQ(w[0].isBegin, 0);
  EXPECT_EQ(w[0].oosEnd, w[0].isEnd + w[0].oos_bars());
  EXPECT_NEAR(static_cast<double>(w[0].is_bars()) / w[0].oos_bars(), 4.0, 0.05);
}

TEST(WfoWindows, RejectsBadSplits) {
  EXPECT_THROW(make_wfo_windows(1000, 0, 0.75), std::invalid_argument);
  EXPECT_THROW(make_wfo_windows(1000, 4, 0.0), std::invalid_argument);
  EXPECT_THROW(make_wfo_windows(1000, 4, 1.0), std::invalid_argument);
  EXPECT_THROW(make_wfo_windows(10, 50, 0.75), std::invalid_argument);
  EXPECT_THROW(make_wfo_windows(1, 1, 0.5), std::invalid_argument);
}

TEST(Wfe, RatioOfPerBarProfits) {
  // IS: 100 over 1000 bars; OOS: 20 over 250 bars -> (20/250)/(100/1000) = 0.8.
  EXPECT_NEAR(compute_wfe(100.0, 1000.0, 20.0, 250.0), 0.8, 1e-12);
  EXPECT_TRUE(std::isnan(compute_wfe(-5.0, 1000.0, 20.0, 250.0)));
  EXPECT_TRUE(std::isnan(compute_wfe(0.0, 1000.0, 20.0, 250.0)));
  EXPECT_TRUE(std::isnan(compute_wfe(100.0, 0.0, 20.0, 250.0)));
  EXPECT_TRUE(std::isnan(compute_wfe(100.0, 1000.0, 20.0, 0.0)));
}

TEST(WalkForward, StitchedEquityAggregatesWindowResults) {
  Series bars = generate_synthetic(8000, 42);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=5,10"),
                                parse_grid_spec("slow=30,60")};
  WfoConfig cfg = quiet_wfo(4, 0.7);
  WfoResult res = run_wfo_on(bars, cfg, axes);

  ASSERT_EQ(res.windows.size(), 4u);
  std::size_t oosBars = 0;
  double sumNp = 0.0;
  int sumTrades = 0;
  for (const auto& wr : res.windows) {
    ASSERT_TRUE(wr.optimized) << wr.note;
    oosBars += static_cast<std::size_t>(wr.window.oos_bars());
    sumNp += wr.oosMetrics.netProfit;
    sumTrades += wr.oosMetrics.numTrades;
    // Every OOS trade stays inside its window's OOS segment.
    for (const Trade& t : wr.oosTrades) {
      EXPECT_GE(t.entryBar, wr.window.isEnd);
      EXPECT_LT(t.exitBar, wr.window.oosEnd);
    }
  }
  EXPECT_EQ(res.stitchedEquity.size(), oosBars);
  EXPECT_NEAR(res.oosOverall.netProfit, sumNp, 1e-6);
  EXPECT_EQ(res.oosOverall.numTrades, sumTrades);
  // 4 variants x 4 windows, exhaustive per window.
  EXPECT_EQ(res.isEvaluations, 16u);
}

TEST(WalkForward, ChosenParamsDependOnlyOnInSampleData) {
  const int n = 6000;
  Series a = generate_synthetic(n, 3);
  Series b = a;
  auto windows = make_wfo_windows(n, 3, 0.75);
  const int modStart = windows[1].isEnd; // window 1's OOS and everything after
  for (int i = modStart; i < n; ++i) {
    auto idx = static_cast<std::size_t>(i);
    b.open[idx] *= 1.5f;
    b.high[idx] *= 1.5f;
    b.low[idx] *= 1.5f;
    b.close[idx] *= 1.5f;
  }

  std::vector<GridAxis> axes = {parse_grid_spec("fast=3..15:3"),
                                parse_grid_spec("slow=20..80:15")};
  WfoConfig cfg = quiet_wfo(3, 0.75);
  WfoResult ra = run_wfo_on(a, cfg, axes);
  WfoResult rb = run_wfo_on(b, cfg, axes);

  // Windows 0 and 1 optimize on bars < modStart only: identical choices.
  for (std::size_t w = 0; w < 2; ++w) {
    ASSERT_TRUE(ra.windows[w].optimized);
    ASSERT_TRUE(rb.windows[w].optimized);
    EXPECT_EQ(ra.windows[w].params.to_string(), rb.windows[w].params.to_string())
        << "window " << w << " looked past its in-sample end";
    EXPECT_DOUBLE_EQ(ra.windows[w].isFitness, rb.windows[w].isFitness);
  }
}

TEST(WalkForward, DeterministicAcrossRuns) {
  Series bars = generate_synthetic(6000, 15);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=4..12:4"),
                                parse_grid_spec("slow=24..72:24")};
  WfoConfig cfg = quiet_wfo(3, 0.75);
  WfoResult r1 = run_wfo_on(bars, cfg, axes);
  WfoResult r2 = run_wfo_on(bars, cfg, axes);

  ASSERT_EQ(r1.windows.size(), r2.windows.size());
  for (std::size_t w = 0; w < r1.windows.size(); ++w) {
    EXPECT_EQ(r1.windows[w].optimized, r2.windows[w].optimized);
    if (r1.windows[w].optimized) {
      EXPECT_EQ(r1.windows[w].params.to_string(), r2.windows[w].params.to_string());
      EXPECT_DOUBLE_EQ(r1.windows[w].oosFitness, r2.windows[w].oosFitness);
    }
  }
  EXPECT_DOUBLE_EQ(r1.oosOverall.netProfit, r2.oosOverall.netProfit);
  if (std::isnan(r1.wfe)) {
    EXPECT_TRUE(std::isnan(r2.wfe));
  } else {
    EXPECT_DOUBLE_EQ(r1.wfe, r2.wfe);
  }
}

TEST(WalkForward, GeneticOptimizerPerWindowIsDeterministic) {
  Series bars = generate_synthetic(6000, 27);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=2..20:2"),
                                parse_grid_spec("slow=25..85:10")};
  WfoConfig cfg = quiet_wfo(3, 0.75);
  cfg.optimizer = WfoOptimizer::Genetic;
  cfg.genetic.population = 12;
  cfg.genetic.maxGenerations = 5;
  cfg.genetic.elite = 2;
  cfg.genetic.seed = 8;
  cfg.genetic.quiet = true;

  WfoResult r1 = run_wfo_on(bars, cfg, axes);
  WfoResult r2 = run_wfo_on(bars, cfg, axes);

  ASSERT_EQ(r1.windows.size(), 3u);
  for (std::size_t w = 0; w < r1.windows.size(); ++w) {
    ASSERT_TRUE(r1.windows[w].optimized) << r1.windows[w].note;
    EXPECT_EQ(r1.windows[w].params.to_string(), r2.windows[w].params.to_string());
  }
  EXPECT_EQ(r1.isEvaluations, r2.isEvaluations);
  // The GA evaluates only a subset of the 10x7=70-variant space per window.
  EXPECT_LT(r1.isEvaluations, 3u * 70u);
}

TEST(WalkForward, WindowsWithoutValidVariantsContributeFlatEquity) {
  Series bars = generate_synthetic(3000, 33);
  // slow=5000 never fits into a 1500-bar IS segment -> every window skipped.
  std::vector<GridAxis> axes = {parse_grid_spec("fast=10"),
                                parse_grid_spec("slow=5000")};
  WfoConfig cfg = quiet_wfo(3, 0.75);
  WfoResult res = run_wfo_on(bars, cfg, axes);

  ASSERT_EQ(res.windows.size(), 3u);
  for (const auto& wr : res.windows) {
    EXPECT_FALSE(wr.optimized);
    EXPECT_FALSE(wr.note.empty());
  }
  for (double eq : res.stitchedEquity) {
    EXPECT_DOUBLE_EQ(eq, cfg.grid.exec.capital);
  }
  EXPECT_EQ(res.oosOverall.numTrades, 0);
  EXPECT_DOUBLE_EQ(res.oosOverall.netProfit, 0.0);
  EXPECT_TRUE(std::isnan(res.wfe));
}
