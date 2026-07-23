/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "TestBackend.h"
#include "backend/BatchBackend.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"
#include "engine/MonteCarlo.h"
#include "optimize/GridSearch.h"
#include "portfolio/Portfolio.h"
#include "strategy/IStrategy.h"

using namespace topt;
using topt::testing::TestBackend;

namespace {

std::vector<Series> make_series(int count, int bars, std::uint64_t seedBase = 1) {
  std::vector<Series> out;
  for (int i = 0; i < count; ++i) {
    out.push_back(generate_synthetic(bars, seedBase + static_cast<std::uint64_t>(i)));
  }
  return out;
}

std::vector<std::string> make_names(int count) {
  std::vector<std::string> names;
  for (int i = 0; i < count; ++i) names.push_back("s" + std::to_string(i + 1));
  return names;
}

std::unique_ptr<IBatchIndicatorBackend> make_test_batch(
    const std::vector<Series>& series) {
  std::vector<std::unique_ptr<IIndicatorBackend>> per;
  per.reserve(series.size());
  for (const Series& s : series) {
    per.push_back(std::make_unique<TestBackend>(s));
  }
  return std::make_unique<PerSymbolBatchBackend>(std::move(per),
                                                 "test-per-symbol");
}

ParamSet ma_params(int fast, int slow) {
  ParamSet ps;
  ps.values = {{"fast", static_cast<double>(fast)},
               {"slow", static_cast<double>(slow)}};
  return ps;
}

ExecConfig exec_with_capital(double capital) {
  ExecConfig exec;
  exec.capital = capital;
  exec.commissionBps = 1.0;
  return exec;
}

void drop_bars(Series& s, int from, int count) {
  s.ts.erase(s.ts.begin() + from, s.ts.begin() + from + count);
  s.open.erase(s.open.begin() + from, s.open.begin() + from + count);
  s.high.erase(s.high.begin() + from, s.high.begin() + from + count);
  s.low.erase(s.low.begin() + from, s.low.begin() + from + count);
  s.close.erase(s.close.begin() + from, s.close.begin() + from + count);
  s.volume.erase(s.volume.begin() + from, s.volume.begin() + from + count);
}

} // namespace

TEST(PortfolioAlign, EqualLengthWithoutTimestampsPasses) {
  std::vector<Series> series = make_series(2, 300);
  for (Series& s : series) std::fill(s.ts.begin(), s.ts.end(), 0);

  std::string log;
  align_portfolio(series, log);
  EXPECT_EQ(series[0].size(), 300);
  EXPECT_EQ(series[1].size(), 300);
  EXPECT_NE(log.find("bar index"), std::string::npos);
}

TEST(PortfolioAlign, MismatchedLengthWithoutTimestampsThrows) {
  std::vector<Series> series = make_series(2, 300);
  drop_bars(series[1], 100, 10);
  for (Series& s : series) std::fill(s.ts.begin(), s.ts.end(), 0);

  std::string log;
  EXPECT_THROW(align_portfolio(series, log), std::invalid_argument);
}

TEST(PortfolioAlign, IntersectsByTimestamps) {
  std::vector<Series> series = make_series(3, 200);
  // Punch different holes into two of the three series.
  drop_bars(series[1], 20, 5);
  drop_bars(series[2], 50, 7);

  const Series original = series[0];
  std::string log;
  align_portfolio(series, log);

  ASSERT_EQ(series[0].size(), 188); // 200 - 5 - 7, holes do not overlap
  ASSERT_EQ(series[1].size(), 188);
  ASSERT_EQ(series[2].size(), 188);
  for (int i = 0; i < series[0].size(); ++i) {
    EXPECT_EQ(series[0].ts[static_cast<std::size_t>(i)],
              series[1].ts[static_cast<std::size_t>(i)]);
    EXPECT_EQ(series[1].ts[static_cast<std::size_t>(i)],
              series[2].ts[static_cast<std::size_t>(i)]);
  }
  // Kept bars preserve their original OHLC values.
  std::size_t k = 0;
  for (std::size_t i = 0; i < original.ts.size() && k < series[0].ts.size(); ++i) {
    if (original.ts[i] != series[0].ts[k]) continue;
    EXPECT_FLOAT_EQ(original.close[i], series[0].close[k]);
    ++k;
  }
  EXPECT_EQ(k, series[0].ts.size());
}

TEST(PortfolioAlign, NoOverlapThrows) {
  std::vector<Series> series = make_series(2, 100);
  for (auto& t : series[1].ts) t += 1; // shift off the 300s grid
  std::string log;
  EXPECT_THROW(align_portfolio(series, log), std::invalid_argument);
}

TEST(Portfolio, AggregatesSymbolsExactly) {
  const int bars = 4000;
  const int numSymbols = 3;
  std::vector<Series> series = make_series(numSymbols, bars);
  auto names = make_names(numSymbols);
  auto strategy = make_strategy("ma_cross");
  ParamSet ps = ma_params(10, 30);

  PortfolioConfig cfg;
  cfg.exec = exec_with_capital(9000.0);
  auto backend = make_test_batch(series);
  PortfolioResult res =
      run_portfolio(series, names, *strategy, ps, *backend, cfg, nullptr);

  ASSERT_EQ(res.validSymbols, numSymbols);
  ASSERT_EQ(res.bars, bars);

  // Reference: each symbol simulated on its own with the capital share.
  ExecConfig symExec = cfg.exec;
  symExec.capital = 3000.0;
  double npSum = 0.0;
  int tradeSum = 0;
  std::vector<std::vector<double>> equities;
  for (int s = 0; s < numSymbols; ++s) {
    TestBackend one(series[static_cast<std::size_t>(s)]);
    IndicatorCache cache(series[static_cast<std::size_t>(s)], one);
    SimResult sim;
    VariantResult ref = run_variant(series[static_cast<std::size_t>(s)],
                                    *strategy, ps, cache, symExec, nullptr, &sim);
    ASSERT_TRUE(ref.valid);
    const SymbolResult& sr = res.symbols[static_cast<std::size_t>(s)];
    EXPECT_TRUE(sr.valid);
    EXPECT_NEAR(sr.metrics.netProfit, ref.metrics.netProfit, 1e-6);
    EXPECT_EQ(sr.metrics.numTrades, ref.metrics.numTrades);
    npSum += ref.metrics.netProfit;
    tradeSum += ref.metrics.numTrades;
    equities.push_back(sim.equity);
  }

  EXPECT_NEAR(res.metrics.netProfit, npSum, 1e-6);
  EXPECT_EQ(res.metrics.numTrades, tradeSum);
  EXPECT_EQ(static_cast<int>(res.trades.size()), tradeSum);
  ASSERT_EQ(static_cast<int>(res.equity.size()), bars);
  for (int t = 0; t < bars; t += 97) {
    double want = 0.0;
    for (const auto& eq : equities) want += eq[static_cast<std::size_t>(t)];
    EXPECT_NEAR(res.equity[static_cast<std::size_t>(t)], want, 1e-6);
  }
  EXPECT_NEAR(res.equity.front(), 9000.0, 1.0); // starts near total capital

  // Exposure is the mean of per-symbol exposures.
  double expMean = 0.0;
  for (const SymbolResult& sr : res.symbols) expMean += sr.metrics.exposure;
  expMean /= numSymbols;
  EXPECT_NEAR(res.metrics.exposure, expMean, 1e-12);
}

TEST(Portfolio, MergedTradesAreChronological) {
  std::vector<Series> series = make_series(4, 3000);
  auto strategy = make_strategy("ma_cross");
  PortfolioConfig cfg;
  cfg.exec = exec_with_capital(8000.0);
  auto backend = make_test_batch(series);
  PortfolioResult res = run_portfolio(series, make_names(4), *strategy,
                                      ma_params(8, 21), *backend, cfg, nullptr);

  ASSERT_GT(res.trades.size(), 0u);
  for (std::size_t i = 1; i < res.trades.size(); ++i) {
    EXPECT_LE(res.trades[i - 1].exitBar, res.trades[i].exitBar);
  }

  // Net profit equals the sum of merged trade PnLs (positions force-close).
  double pnlSum = 0.0;
  for (const Trade& t : res.trades) pnlSum += t.pnl;
  EXPECT_NEAR(res.metrics.netProfit, pnlSum, 1e-6);
}

TEST(Portfolio, InvalidParamsThrow) {
  std::vector<Series> series = make_series(2, 500);
  auto strategy = make_strategy("ma_cross");
  PortfolioConfig cfg;
  auto backend = make_test_batch(series);
  EXPECT_THROW(run_portfolio(series, make_names(2), *strategy, ma_params(50, 10),
                             *backend, cfg, nullptr),
               std::invalid_argument);
}

TEST(Portfolio, BackendSymbolCountMismatchThrows) {
  std::vector<Series> series = make_series(3, 500);
  std::vector<Series> two(series.begin(), series.begin() + 2);
  auto strategy = make_strategy("ma_cross");
  PortfolioConfig cfg;
  auto backend = make_test_batch(two);
  EXPECT_THROW(run_portfolio(series, make_names(3), *strategy, ma_params(5, 20),
                             *backend, cfg, nullptr),
               std::invalid_argument);
}

TEST(Portfolio, DeterministicAcrossRuns) {
  std::vector<Series> series = make_series(3, 2000);
  auto strategy = make_strategy("ma_cross");
  PortfolioConfig cfg;
  cfg.exec = exec_with_capital(6000.0);
  auto b1 = make_test_batch(series);
  auto b2 = make_test_batch(series);
  PortfolioResult a = run_portfolio(series, make_names(3), *strategy,
                                    ma_params(12, 48), *b1, cfg, nullptr);
  PortfolioResult b = run_portfolio(series, make_names(3), *strategy,
                                    ma_params(12, 48), *b2, cfg, nullptr);
  EXPECT_DOUBLE_EQ(a.metrics.netProfit, b.metrics.netProfit);
  EXPECT_EQ(a.trades.size(), b.trades.size());
  EXPECT_EQ(a.equity, b.equity);
}

TEST(Portfolio, MonteCarloOverPortfolioTrades) {
  std::vector<Series> series = make_series(3, 3000);
  auto strategy = make_strategy("ma_cross");
  PortfolioConfig cfg;
  cfg.exec = exec_with_capital(9000.0);
  auto backend = make_test_batch(series);
  PortfolioResult res = run_portfolio(series, make_names(3), *strategy,
                                      ma_params(10, 40), *backend, cfg, nullptr);
  ASSERT_GT(res.trades.size(), 1u);

  McConfig mcCfg;
  mcCfg.iterations = 200;
  mcCfg.mode = McMode::Permute;
  McResult mc = run_monte_carlo(res.trades, cfg.exec.capital, mcCfg);

  EXPECT_EQ(mc.numTrades, static_cast<int>(res.trades.size()));
  EXPECT_NEAR(mc.originalNetProfit, res.metrics.netProfit, 1e-6);
  EXPECT_LE(mc.maxDrawdown.p5, mc.maxDrawdown.p95);
}
