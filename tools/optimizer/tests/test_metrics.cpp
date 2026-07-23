/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <cmath>

#include "engine/Metrics.h"

using namespace topt;

namespace {

SimResult make_sim() {
  SimResult sim;
  sim.initialCapital = 1000.0;
  sim.equity = {1000.0, 1010.0, 990.0, 1020.0, 1005.0, 1030.0};
  sim.finalEquity = sim.equity.back();
  sim.barsInPosition = 4;
  Trade t1{0, 1, 1, 100.0, 110.0, 25.0};
  Trade t2{1, 2, -1, 110.0, 115.0, -10.0};
  Trade t3{3, 5, 1, 100.0, 115.0, 15.0};
  sim.trades = {t1, t2, t3};
  return sim;
}

} // namespace

TEST(Metrics, TradeAggregates) {
  MetricsResult m = compute_metrics(make_sim(), 252);
  EXPECT_DOUBLE_EQ(m.netProfit, 30.0);
  EXPECT_DOUBLE_EQ(m.grossProfit, 40.0);
  EXPECT_DOUBLE_EQ(m.grossLoss, 10.0);
  EXPECT_DOUBLE_EQ(m.profitFactor, 4.0);
  EXPECT_EQ(m.numTrades, 3);
  EXPECT_NEAR(m.winRate, 2.0 / 3.0, 1e-12);
  EXPECT_DOUBLE_EQ(m.avgTrade, 10.0);
  EXPECT_NEAR(m.exposure, 4.0 / 6.0, 1e-12);
}

TEST(Metrics, DrawdownFromEquityCurve) {
  MetricsResult m = compute_metrics(make_sim(), 252);
  // Peak 1010 -> trough 990 gives 20 absolute; 1020 -> 1005 gives 15.
  EXPECT_DOUBLE_EQ(m.maxDrawdown, 20.0);
  EXPECT_NEAR(m.maxDrawdownPct, 20.0 / 1010.0, 1e-12);
}

TEST(Metrics, SharpeMatchesHandComputation) {
  SimResult sim;
  sim.initialCapital = 0.0;
  sim.equity = {0.0, 1.0, 3.0, 2.0}; // diffs: 1, 2, -1
  sim.finalEquity = 2.0;
  MetricsResult m = compute_metrics(sim, 252);
  double mean = (1.0 + 2.0 - 1.0) / 3.0;
  double var = ((1 - mean) * (1 - mean) + (2 - mean) * (2 - mean) +
                (-1 - mean) * (-1 - mean)) / 3.0;
  EXPECT_NEAR(m.sharpe, mean / std::sqrt(var) * std::sqrt(252.0), 1e-12);
  // Sortino uses only the -1 diff: sqrt(mean(min(d,0)^2)) = sqrt(1/3).
  EXPECT_NEAR(m.sortino, mean / std::sqrt(1.0 / 3.0) * std::sqrt(252.0), 1e-12);
}

TEST(Metrics, NoTradesNoDivisionByZero) {
  SimResult sim;
  sim.initialCapital = 100.0;
  sim.equity = {100.0, 100.0, 100.0};
  sim.finalEquity = 100.0;
  MetricsResult m = compute_metrics(sim, 252);
  EXPECT_DOUBLE_EQ(m.netProfit, 0.0);
  EXPECT_DOUBLE_EQ(m.profitFactor, 0.0);
  EXPECT_DOUBLE_EQ(m.winRate, 0.0);
  EXPECT_DOUBLE_EQ(m.sharpe, 0.0);
  EXPECT_DOUBLE_EQ(m.maxDrawdown, 0.0);
}

TEST(Metrics, ProfitFactorCappedWithoutLosses) {
  SimResult sim;
  sim.initialCapital = 0.0;
  sim.equity = {0.0, 10.0};
  sim.finalEquity = 10.0;
  sim.trades = {Trade{0, 1, 1, 1.0, 2.0, 10.0}};
  MetricsResult m = compute_metrics(sim, 252);
  EXPECT_DOUBLE_EQ(m.profitFactor, 1e9);
}
