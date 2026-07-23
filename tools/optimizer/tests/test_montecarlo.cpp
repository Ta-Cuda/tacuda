/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>

#include "engine/MonteCarlo.h"

using namespace topt;

namespace {

std::vector<Trade> make_trades(std::initializer_list<double> pnls) {
  std::vector<Trade> trades;
  int bar = 0;
  for (double pnl : pnls) {
    Trade t;
    t.entryBar = bar;
    t.exitBar = bar + 1;
    t.dir = 1;
    t.entryPrice = 100.0;
    t.exitPrice = 100.0 + pnl;
    t.pnl = pnl;
    trades.push_back(t);
    bar += 2;
  }
  return trades;
}

McConfig make_cfg(int iterations, McMode mode, std::uint64_t seed = 42) {
  McConfig cfg;
  cfg.iterations = iterations;
  cfg.mode = mode;
  cfg.seed = seed;
  return cfg;
}

void expect_ordered(const McPercentiles& p) {
  EXPECT_LE(p.p5, p.p25);
  EXPECT_LE(p.p25, p.p50);
  EXPECT_LE(p.p50, p.p75);
  EXPECT_LE(p.p75, p.p95);
}

} // namespace

TEST(MonteCarlo, PermuteKeepsNetProfitConstant) {
  auto trades = make_trades({12.0, -7.5, 3.0, -1.0, 20.0, -14.0, 6.0});
  McResult res = run_monte_carlo(trades, 1000.0, make_cfg(500, McMode::Permute));

  const double total = 12.0 - 7.5 + 3.0 - 1.0 + 20.0 - 14.0 + 6.0;
  EXPECT_NEAR(res.netProfit.p5, total, 1e-9);
  EXPECT_NEAR(res.netProfit.p95, total, 1e-9);
  EXPECT_NEAR(res.netProfit.worst, total, 1e-9);
  EXPECT_NEAR(res.originalNetProfit, total, 1e-9);
  EXPECT_DOUBLE_EQ(res.probNetLoss, 0.0);
}

TEST(MonteCarlo, TwoTradePermutationHasKnownDrawdown) {
  // Orders (+10,-5) and (-5,+10) both have max drawdown 5.
  auto trades = make_trades({10.0, -5.0});
  McResult res = run_monte_carlo(trades, 100.0, make_cfg(64, McMode::Permute));

  EXPECT_NEAR(res.maxDrawdown.p5, 5.0, 1e-9);
  EXPECT_NEAR(res.maxDrawdown.p50, 5.0, 1e-9);
  EXPECT_NEAR(res.maxDrawdown.worst, 5.0, 1e-9);
  EXPECT_NEAR(res.netProfit.p50, 5.0, 1e-9);
}

TEST(MonteCarlo, OriginalStatsMatchChronologicalOrder) {
  // Path 100 -> 110 -> 90 -> 105: peak 110, deepest valley 90.
  auto trades = make_trades({10.0, -20.0, 15.0});
  McResult res = run_monte_carlo(trades, 100.0, make_cfg(10, McMode::Permute));

  EXPECT_NEAR(res.originalNetProfit, 5.0, 1e-9);
  EXPECT_NEAR(res.originalMaxDrawdown, 20.0, 1e-9);
}

TEST(MonteCarlo, PercentilesAreOrderedAndNonNegative) {
  auto trades = make_trades({5.0, -3.0, 8.0, -6.0, 2.0, -1.5, 9.0, -4.0, 1.0,
                             -2.0, 7.0, -5.0});
  for (McMode mode : {McMode::Permute, McMode::Resample}) {
    McResult res = run_monte_carlo(trades, 500.0, make_cfg(1000, mode));
    expect_ordered(res.maxDrawdown);
    expect_ordered(res.netProfit);
    EXPECT_GE(res.maxDrawdown.p5, 0.0);
    EXPECT_GE(res.maxDrawdown.worst, res.maxDrawdown.p95);
    EXPECT_LE(res.netProfit.worst, res.netProfit.p5);
    EXPECT_GE(res.probNetLoss, 0.0);
    EXPECT_LE(res.probNetLoss, 1.0);
    EXPECT_GE(res.probDdAboveOriginal, 0.0);
    EXPECT_LE(res.probDdAboveOriginal, 1.0);
  }
}

TEST(MonteCarlo, DeterministicForSameSeed) {
  auto trades = make_trades({4.0, -2.0, 6.0, -8.0, 3.0, 5.0, -1.0});
  McResult a = run_monte_carlo(trades, 200.0, make_cfg(300, McMode::Resample, 7));
  McResult b = run_monte_carlo(trades, 200.0, make_cfg(300, McMode::Resample, 7));

  EXPECT_DOUBLE_EQ(a.maxDrawdown.p50, b.maxDrawdown.p50);
  EXPECT_DOUBLE_EQ(a.maxDrawdown.worst, b.maxDrawdown.worst);
  EXPECT_DOUBLE_EQ(a.netProfit.p5, b.netProfit.p5);
  EXPECT_DOUBLE_EQ(a.probNetLoss, b.probNetLoss);
  EXPECT_DOUBLE_EQ(a.probDdAboveOriginal, b.probDdAboveOriginal);
}

TEST(MonteCarlo, ResampleVariesNetProfit) {
  auto trades = make_trades({15.0, -12.0, 9.0, -7.0, 11.0, -10.0, 4.0, -3.0});
  McResult res = run_monte_carlo(trades, 300.0, make_cfg(800, McMode::Resample));

  EXPECT_LT(res.netProfit.p5, res.netProfit.p95);
  EXPECT_GT(res.probNetLoss, 0.0); // bootstrap can draw the losing trades
}

TEST(MonteCarlo, EmptyTradesYieldZeroResult) {
  McResult res = run_monte_carlo({}, 1000.0, make_cfg(100, McMode::Permute));
  EXPECT_EQ(res.numTrades, 0);
  EXPECT_EQ(res.iterations, 100);
  EXPECT_DOUBLE_EQ(res.maxDrawdown.worst, 0.0);
  EXPECT_DOUBLE_EQ(res.originalNetProfit, 0.0);
}

TEST(MonteCarlo, RejectsBadIterations) {
  auto trades = make_trades({1.0});
  EXPECT_THROW(run_monte_carlo(trades, 100.0, make_cfg(0, McMode::Permute)),
               std::invalid_argument);
  EXPECT_THROW(run_monte_carlo(trades, 100.0, make_cfg(-5, McMode::Permute)),
               std::invalid_argument);
}

TEST(MonteCarlo, ModeParsing) {
  EXPECT_EQ(parse_mc_mode("permute"), McMode::Permute);
  EXPECT_EQ(parse_mc_mode("resample"), McMode::Resample);
  EXPECT_THROW(parse_mc_mode("shuffle"), std::invalid_argument);
  EXPECT_STREQ(to_string(McMode::Permute), "permute");
  EXPECT_STREQ(to_string(McMode::Resample), "resample");
}
