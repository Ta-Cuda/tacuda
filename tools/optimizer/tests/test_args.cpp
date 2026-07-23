/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include "cli/Args.h"

using namespace topt;

namespace {

CliOptions parse(std::initializer_list<const char*> args) {
  std::vector<const char*> argv(args);
  return parse_cli(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST(Args, ParsesGridCommand) {
  CliOptions opt = parse({"tacuda-opt", "grid", "--synthetic", "1000", "--strategy",
                          "ma_cross", "--grid", "fast=5..10:5", "--grid", "slow=20..40:20",
                          "--fitness", "sharpe", "--top", "5", "--backend", "talib",
                          "--commission-bps", "0.5", "--out-csv", "x.csv"});
  EXPECT_EQ(opt.command, "grid");
  EXPECT_EQ(opt.syntheticBars, 1000);
  EXPECT_EQ(opt.strategy, "ma_cross");
  ASSERT_EQ(opt.gridAxes.size(), 2u);
  EXPECT_EQ(opt.gridAxes[0].name, "fast");
  EXPECT_EQ(opt.fitness, "sharpe");
  EXPECT_EQ(opt.top, 5);
  EXPECT_EQ(opt.backend, BackendKind::Talib);
  EXPECT_DOUBLE_EQ(opt.exec.commissionBps, 0.5);
  EXPECT_EQ(opt.outCsv, "x.csv");
}

TEST(Args, ParsesInlineEqualsForm) {
  CliOptions opt = parse({"tacuda-opt", "backtest", "--synthetic=500",
                          "--strategy=rsi_reversion", "--set=period=7"});
  EXPECT_EQ(opt.syntheticBars, 500);
  EXPECT_EQ(opt.strategy, "rsi_reversion");
  ASSERT_EQ(opt.sets.size(), 1u);
  EXPECT_EQ(opt.sets[0].first, "period");
  EXPECT_DOUBLE_EQ(opt.sets[0].second, 7.0);
}

TEST(Args, RequiresExactlyOneDataSource) {
  EXPECT_THROW(parse({"tacuda-opt", "backtest", "--strategy", "ma_cross"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "backtest", "--strategy", "ma_cross", "--data",
                      "a.csv", "--synthetic", "100"}),
               std::invalid_argument);
}

TEST(Args, GridNeedsAxes) {
  EXPECT_THROW(parse({"tacuda-opt", "grid", "--synthetic", "100", "--strategy",
                      "ma_cross"}),
               std::invalid_argument);
}

TEST(Args, RejectsUnknownOptionAndCommand) {
  EXPECT_THROW(parse({"tacuda-opt", "fly"}), std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "grid", "--wat", "1"}), std::invalid_argument);
}

TEST(Args, HelpAndListNeedNoData) {
  EXPECT_EQ(parse({"tacuda-opt", "--help"}).command, "help");
  EXPECT_EQ(parse({"tacuda-opt", "list"}).command, "list");
}

TEST(Args, ParsesGeneticCommandWithGaFlags) {
  CliOptions opt = parse({"tacuda-opt", "genetic", "--synthetic", "1000", "--strategy",
                          "ma_cross", "--grid", "fast=2..10:2", "--population", "32",
                          "--generations", "15", "--elite", "3", "--tournament", "5",
                          "--crossover-prob", "0.8", "--mutation-prob", "0.2",
                          "--mutation-sigma", "1.5", "--stagnation", "6",
                          "--ga-seed", "99"});
  EXPECT_EQ(opt.command, "genetic");
  EXPECT_EQ(opt.population, 32);
  EXPECT_EQ(opt.generations, 15);
  EXPECT_EQ(opt.elite, 3);
  EXPECT_EQ(opt.tournament, 5);
  EXPECT_DOUBLE_EQ(opt.crossoverProb, 0.8);
  EXPECT_DOUBLE_EQ(opt.mutationProb, 0.2);
  EXPECT_DOUBLE_EQ(opt.mutationSigma, 1.5);
  EXPECT_EQ(opt.stagnation, 6);
  EXPECT_EQ(opt.gaSeed, 99u);
}

TEST(Args, GeneticNeedsAxes) {
  EXPECT_THROW(parse({"tacuda-opt", "genetic", "--synthetic", "100", "--strategy",
                      "ma_cross"}),
               std::invalid_argument);
}

TEST(Args, ParsesWfoCommand) {
  CliOptions opt = parse({"tacuda-opt", "wfo", "--synthetic", "10000", "--strategy",
                          "ma_cross", "--grid", "fast=5..20:5", "--windows", "8",
                          "--is", "0.8", "--optimizer", "genetic"});
  EXPECT_EQ(opt.command, "wfo");
  EXPECT_EQ(opt.windows, 8);
  EXPECT_DOUBLE_EQ(opt.isFraction, 0.8);
  EXPECT_EQ(opt.wfoOptimizer, "genetic");
}

TEST(Args, ParsesPortfolioCommand) {
  CliOptions opt = parse({"tacuda-opt", "portfolio", "--data", "a.csv", "--data",
                          "b.csv", "--data", "c.csv", "--strategy", "ma_cross",
                          "--set", "fast=10", "--mc", "500", "--mc-mode",
                          "resample", "--mc-seed", "7"});
  EXPECT_EQ(opt.command, "portfolio");
  ASSERT_EQ(opt.dataPaths.size(), 3u);
  EXPECT_EQ(opt.dataPaths[1], "b.csv");
  EXPECT_EQ(opt.mcIterations, 500);
  EXPECT_EQ(opt.mcMode, "resample");
  EXPECT_EQ(opt.mcSeed, 7u);
}

TEST(Args, ParsesPortfolioSyntheticSymbols) {
  CliOptions opt = parse({"tacuda-opt", "portfolio", "--synthetic", "1000",
                          "--symbols", "6", "--strategy", "ma_cross"});
  EXPECT_EQ(opt.syntheticBars, 1000);
  EXPECT_EQ(opt.symbols, 6);
}

TEST(Args, ParsesTacudaDeviceBackend) {
  CliOptions opt = parse({"tacuda-opt", "portfolio", "--synthetic", "1000",
                          "--strategy", "exp01_ma_crossover", "--backend",
                          "tacuda-device"});
  EXPECT_EQ(opt.backend, BackendKind::TacudaDevice);
}

TEST(Args, PortfolioAndMcValidation) {
  // Multiple --data files are portfolio-only.
  EXPECT_THROW(parse({"tacuda-opt", "grid", "--data", "a.csv", "--data", "b.csv",
                      "--strategy", "ma_cross", "--grid", "fast=5..10:5"}),
               std::invalid_argument);
  // --symbols needs portfolio + --synthetic.
  EXPECT_THROW(parse({"tacuda-opt", "backtest", "--synthetic", "100",
                      "--symbols", "4", "--strategy", "ma_cross"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "portfolio", "--data", "a.csv", "--symbols",
                      "4", "--strategy", "ma_cross"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "portfolio", "--synthetic", "100",
                      "--symbols", "0", "--strategy", "ma_cross"}),
               std::invalid_argument);
  // Portfolio takes fixed parameters, not grid axes.
  EXPECT_THROW(parse({"tacuda-opt", "portfolio", "--synthetic", "100",
                      "--strategy", "ma_cross", "--grid", "fast=5..10:5"}),
               std::invalid_argument);
  // Monte-Carlo flags are for backtest/portfolio and must be sane.
  EXPECT_THROW(parse({"tacuda-opt", "grid", "--synthetic", "100", "--strategy",
                      "ma_cross", "--grid", "fast=5..10:5", "--mc", "100"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "backtest", "--synthetic", "100",
                      "--strategy", "ma_cross", "--mc", "-1"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "backtest", "--synthetic", "100",
                      "--strategy", "ma_cross", "--mc", "10", "--mc-mode",
                      "shuffle"}),
               std::invalid_argument);
}

TEST(Args, BacktestAcceptsMc) {
  CliOptions opt = parse({"tacuda-opt", "backtest", "--synthetic", "1000",
                          "--strategy", "ma_cross", "--mc", "250"});
  EXPECT_EQ(opt.mcIterations, 250);
  EXPECT_EQ(opt.mcMode, "permute");
}

TEST(Args, WfoDefaultsAndValidation) {
  CliOptions opt = parse({"tacuda-opt", "wfo", "--synthetic", "10000", "--strategy",
                          "ma_cross", "--grid", "fast=5..20:5"});
  EXPECT_EQ(opt.windows, 6);
  EXPECT_DOUBLE_EQ(opt.isFraction, 0.75);
  EXPECT_EQ(opt.wfoOptimizer, "grid");

  EXPECT_THROW(parse({"tacuda-opt", "wfo", "--synthetic", "100", "--strategy",
                      "ma_cross", "--grid", "fast=5..20:5", "--windows", "0"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "wfo", "--synthetic", "100", "--strategy",
                      "ma_cross", "--grid", "fast=5..20:5", "--is", "1.5"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "wfo", "--synthetic", "100", "--strategy",
                      "ma_cross", "--grid", "fast=5..20:5", "--optimizer", "annealing"}),
               std::invalid_argument);
  EXPECT_THROW(parse({"tacuda-opt", "wfo", "--synthetic", "100", "--strategy",
                      "ma_cross"}),
               std::invalid_argument);
}
