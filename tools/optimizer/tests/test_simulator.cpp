/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include "engine/Simulator.h"

using namespace topt;

namespace {

// open/close pairs; high/low derived, volume zero.
Series make_series(std::initializer_list<std::pair<double, double>> ocPairs) {
  Series s;
  for (auto [o, c] : ocPairs) {
    s.open.push_back(static_cast<float>(o));
    s.close.push_back(static_cast<float>(c));
    s.high.push_back(static_cast<float>(std::max(o, c) + 0.25));
    s.low.push_back(static_cast<float>(std::min(o, c) - 0.25));
    s.volume.push_back(0.0f);
    s.ts.push_back(0);
  }
  return s;
}

const Series kBars = make_series({
    {10.0, 10.5}, {11.0, 11.5}, {12.0, 12.5}, {13.0, 13.5}, {14.0, 12.0}});

} // namespace

TEST(Simulator, NextBarOpenExecutionWithReversalNoCosts) {
  // desired: flat, long, hold, reverse to short, (last bar decision ignored)
  std::vector<Signal> desired = {0, 1, 1, -1, 0};
  ExecConfig cfg;
  cfg.capital = 100.0;

  SimResult r = simulate(kBars, desired.data(), cfg);

  ASSERT_EQ(r.trades.size(), 2u);
  const Trade& t1 = r.trades[0];
  EXPECT_EQ(t1.dir, 1);
  EXPECT_EQ(t1.entryBar, 2); // decided at close of bar 1, filled at open[2]
  EXPECT_DOUBLE_EQ(t1.entryPrice, 12.0);
  EXPECT_EQ(t1.exitBar, 4);
  EXPECT_DOUBLE_EQ(t1.exitPrice, 14.0);
  EXPECT_DOUBLE_EQ(t1.pnl, 2.0);

  const Trade& t2 = r.trades[1]; // short from the reversal, force-closed at last close
  EXPECT_EQ(t2.dir, -1);
  EXPECT_DOUBLE_EQ(t2.entryPrice, 14.0);
  EXPECT_DOUBLE_EQ(t2.exitPrice, 12.0);
  EXPECT_DOUBLE_EQ(t2.pnl, 2.0);

  EXPECT_DOUBLE_EQ(r.finalEquity, 104.0);
  // Equity marked at each close: no position until open[2].
  EXPECT_DOUBLE_EQ(r.equity[0], 100.0);
  EXPECT_DOUBLE_EQ(r.equity[1], 100.0);
  EXPECT_DOUBLE_EQ(r.equity[2], 100.5);
  EXPECT_DOUBLE_EQ(r.equity[3], 101.5);
  EXPECT_DOUBLE_EQ(r.equity[4], 104.0);
}

TEST(Simulator, CommissionAndSlippageReduceProfit) {
  std::vector<Signal> desired = {0, 1, 1, -1, 0};
  ExecConfig cfg;
  cfg.capital = 100.0;
  cfg.commissionBps = 10.0;   // 0.1% per side
  cfg.commissionFixed = 0.5;  // per order
  cfg.slippageTicks = 1.0;
  cfg.tickSize = 0.01;

  SimResult r = simulate(kBars, desired.data(), cfg);
  ASSERT_EQ(r.trades.size(), 2u);

  // Long: buy at 12.01, reversal order sells 2 units at 13.99.
  // Reversal commission (27.98*0.001 + 0.5) splits per unit between trades.
  double buyCost = 12.01 * 0.001 + 0.5;
  double revPerUnit = (2 * 13.99 * 0.001 + 0.5) / 2.0;
  double expectedLongPnl = (13.99 - 12.01) - buyCost - revPerUnit;
  EXPECT_NEAR(r.trades[0].pnl, expectedLongPnl, 1e-12);

  // Short: entry 13.99, force-close buys at 12.0 + slip = 12.01.
  double closeCost = 12.01 * 0.001 + 0.5;
  double expectedShortPnl = (13.99 - 12.01) - revPerUnit - closeCost;
  EXPECT_NEAR(r.trades[1].pnl, expectedShortPnl, 1e-12);

  double expectedFinal = 100.0 + expectedLongPnl + expectedShortPnl;
  EXPECT_NEAR(r.finalEquity, expectedFinal, 1e-12);
}

TEST(Simulator, SignalOnLastBarIsNeverFilled) {
  std::vector<Signal> desired = {0, 0, 0, 0, 1};
  ExecConfig cfg;
  SimResult r = simulate(kBars, desired.data(), cfg);
  EXPECT_TRUE(r.trades.empty());
  EXPECT_DOUBLE_EQ(r.finalEquity, cfg.capital);
  EXPECT_EQ(r.barsInPosition, 0);
}

TEST(Simulator, LongFromFirstSignalFillsAtSecondOpen) {
  std::vector<Signal> desired = {1, 1, 1, 1, 1};
  ExecConfig cfg;
  cfg.capital = 100.0;
  SimResult r = simulate(kBars, desired.data(), cfg);
  // Decision at close of bar 0 -> fill at open[1]; equity[0] untouched.
  EXPECT_DOUBLE_EQ(r.equity[0], 100.0);
  ASSERT_EQ(r.trades.size(), 1u);
  EXPECT_EQ(r.trades[0].entryBar, 1);
  EXPECT_DOUBLE_EQ(r.trades[0].entryPrice, 11.0);
  // Force-closed at last close 12.0.
  EXPECT_DOUBLE_EQ(r.trades[0].exitPrice, 12.0);
  EXPECT_DOUBLE_EQ(r.finalEquity, 101.0);
  EXPECT_EQ(r.barsInPosition, 4);
}

TEST(Simulator, ShiftedSignalsChangeTheOutcome) {
  // Look-ahead guard: delaying every signal by one bar must change fills.
  std::vector<Signal> base = {0, 1, 1, -1, 0};
  std::vector<Signal> shifted = {0, 0, 1, 1, -1};
  ExecConfig cfg;
  cfg.capital = 100.0;
  SimResult a = simulate(kBars, base.data(), cfg);
  SimResult b = simulate(kBars, shifted.data(), cfg);
  EXPECT_NE(a.finalEquity, b.finalEquity);
}

TEST(Simulator, QtyScalesPnl) {
  std::vector<Signal> desired = {0, 1, 1, -1, 0};
  ExecConfig cfg;
  cfg.capital = 1000.0;
  cfg.qty = 5.0;
  SimResult r = simulate(kBars, desired.data(), cfg);
  EXPECT_DOUBLE_EQ(r.finalEquity, 1000.0 + 4.0 * 5.0);
}

TEST(Simulator, RejectsBadInputs) {
  std::vector<Signal> desired = {0, 0, 0, 0, 0};
  ExecConfig cfg;
  cfg.qty = 0.0;
  EXPECT_THROW(simulate(kBars, desired.data(), cfg), std::invalid_argument);
  ExecConfig ok;
  EXPECT_THROW(simulate(kBars, nullptr, ok), std::invalid_argument);
  Series empty;
  EXPECT_THROW(simulate(empty, desired.data(), ok), std::invalid_argument);
}

TEST(Simulator, DeterministicAcrossRuns) {
  std::vector<Signal> desired = {0, 1, -1, 1, 0};
  ExecConfig cfg;
  cfg.commissionBps = 2.5;
  cfg.slippageTicks = 2.0;
  SimResult a = simulate(kBars, desired.data(), cfg);
  SimResult b = simulate(kBars, desired.data(), cfg);
  ASSERT_EQ(a.trades.size(), b.trades.size());
  EXPECT_DOUBLE_EQ(a.finalEquity, b.finalEquity);
  for (std::size_t i = 0; i < a.trades.size(); ++i) {
    EXPECT_DOUBLE_EQ(a.trades[i].pnl, b.trades[i].pnl);
  }
}

TEST(Simulator, RangedRunTradesOnlyInsideRange) {
  std::vector<Signal> desired = {1, 1, 1, 1, 1};
  ExecConfig cfg;
  cfg.capital = 100.0;
  SimResult r = simulate(kBars, desired.data(), cfg, SimRange{1, 4});

  // Boundary decision desired[0] fills at open[1]=11; force close at close[3].
  ASSERT_EQ(r.trades.size(), 1u);
  EXPECT_EQ(r.trades[0].entryBar, 1); // absolute bar index
  EXPECT_DOUBLE_EQ(r.trades[0].entryPrice, 11.0);
  EXPECT_EQ(r.trades[0].exitBar, 3);
  EXPECT_DOUBLE_EQ(r.trades[0].exitPrice, 13.5);
  EXPECT_DOUBLE_EQ(r.trades[0].pnl, 2.5);

  // Equity is local to the range: bars 1, 2, 3.
  ASSERT_EQ(r.equity.size(), 3u);
  EXPECT_DOUBLE_EQ(r.equity[0], 100.5);
  EXPECT_DOUBLE_EQ(r.equity[1], 101.5);
  EXPECT_DOUBLE_EQ(r.equity[2], 102.5);
  EXPECT_DOUBLE_EQ(r.finalEquity, 102.5);
  EXPECT_EQ(r.barsInPosition, 3);
}

TEST(Simulator, RangedRunUsesBoundaryDecisionOnly) {
  // Only the pre-range decision is long; inside the range everything is flat,
  // so the position opened at the boundary closes at the next open.
  std::vector<Signal> desired = {1, 0, 0, 0, 0};
  ExecConfig cfg;
  cfg.capital = 100.0;
  SimResult r = simulate(kBars, desired.data(), cfg, SimRange{1, 5});

  ASSERT_EQ(r.trades.size(), 1u);
  EXPECT_EQ(r.trades[0].entryBar, 1);
  EXPECT_DOUBLE_EQ(r.trades[0].entryPrice, 11.0);
  EXPECT_EQ(r.trades[0].exitBar, 2);
  EXPECT_DOUBLE_EQ(r.trades[0].exitPrice, 12.0);
  EXPECT_DOUBLE_EQ(r.trades[0].pnl, 1.0);
  EXPECT_DOUBLE_EQ(r.finalEquity, 101.0);
}

TEST(Simulator, ExplicitFullRangeMatchesDefault) {
  std::vector<Signal> desired = {0, 1, 1, -1, 0};
  ExecConfig cfg;
  cfg.capital = 100.0;
  SimResult a = simulate(kBars, desired.data(), cfg);
  SimResult b = simulate(kBars, desired.data(), cfg, SimRange{0, 5});
  ASSERT_EQ(a.trades.size(), b.trades.size());
  ASSERT_EQ(a.equity.size(), b.equity.size());
  EXPECT_DOUBLE_EQ(a.finalEquity, b.finalEquity);
  for (std::size_t i = 0; i < a.equity.size(); ++i) {
    EXPECT_DOUBLE_EQ(a.equity[i], b.equity[i]);
  }
}

TEST(Simulator, RejectsBadRanges) {
  std::vector<Signal> desired = {0, 0, 0, 0, 0};
  ExecConfig cfg;
  EXPECT_THROW(simulate(kBars, desired.data(), cfg, SimRange{-1, 3}), std::invalid_argument);
  EXPECT_THROW(simulate(kBars, desired.data(), cfg, SimRange{3, 3}), std::invalid_argument);
  EXPECT_THROW(simulate(kBars, desired.data(), cfg, SimRange{2, 6}), std::invalid_argument);
  EXPECT_THROW(simulate(kBars, desired.data(), cfg, SimRange{5, 5}), std::invalid_argument);
}
