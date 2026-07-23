/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include "core/Fitness.h"
#include "engine/Metrics.h"

using namespace topt;

namespace {

std::map<std::string, double> vars(std::initializer_list<std::pair<const std::string, double>> init) {
  return std::map<std::string, double>(init);
}

} // namespace

TEST(Fitness, SingleVariable) {
  auto e = FitnessExpr::parse("netprofit", fitness_variable_names());
  EXPECT_DOUBLE_EQ(e.eval(vars({{"netprofit", 123.5}})), 123.5);
}

TEST(Fitness, WeightedSumWithPrecedence) {
  auto e = FitnessExpr::parse("netprofit - 2*maxdd", fitness_variable_names());
  EXPECT_DOUBLE_EQ(e.eval(vars({{"netprofit", 100.0}, {"maxdd", 30.0}})), 40.0);
}

TEST(Fitness, ParenthesesAndUnaryMinus) {
  auto e = FitnessExpr::parse("-(sharpe + 1) * 2", fitness_variable_names());
  EXPECT_DOUBLE_EQ(e.eval(vars({{"sharpe", 0.5}})), -3.0);
}

TEST(Fitness, DivisionByZeroYieldsZero) {
  auto e = FitnessExpr::parse("netprofit / maxdd", fitness_variable_names());
  EXPECT_DOUBLE_EQ(e.eval(vars({{"netprofit", 10.0}, {"maxdd", 0.0}})), 0.0);
}

TEST(Fitness, CaseInsensitiveIdentifiers) {
  auto e = FitnessExpr::parse("NetProfit + SHARPE", fitness_variable_names());
  EXPECT_DOUBLE_EQ(e.eval(vars({{"netprofit", 1.0}, {"sharpe", 2.0}})), 3.0);
}

TEST(Fitness, RejectsUnknownVariable) {
  EXPECT_THROW(FitnessExpr::parse("bogus + 1", fitness_variable_names()),
               std::invalid_argument);
}

TEST(Fitness, RejectsSyntaxErrors) {
  EXPECT_THROW(FitnessExpr::parse("netprofit +", fitness_variable_names()),
               std::invalid_argument);
  EXPECT_THROW(FitnessExpr::parse("(netprofit", fitness_variable_names()),
               std::invalid_argument);
  EXPECT_THROW(FitnessExpr::parse("netprofit 5", fitness_variable_names()),
               std::invalid_argument);
}

TEST(Fitness, MetricsVariablesRoundTrip) {
  MetricsResult m;
  m.netProfit = 500.0;
  m.maxDrawdown = 120.0;
  m.sharpe = 1.4;
  auto v = fitness_variables(m);
  auto e = FitnessExpr::parse("netprofit - 2*maxdd + sharpe", fitness_variable_names());
  EXPECT_NEAR(e.eval(v), 500.0 - 240.0 + 1.4, 1e-12);
}
