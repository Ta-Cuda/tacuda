/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <cmath>

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

TEST(Fitness, FunctionsEvaluate) {
  const auto v = vars({{"netprofit", -40.0}, {"maxdd", 16.0}, {"trades", 9.0}});
  EXPECT_DOUBLE_EQ(FitnessExpr::parse("abs(netprofit)", fitness_variable_names()).eval(v),
                   40.0);
  EXPECT_DOUBLE_EQ(FitnessExpr::parse("sqrt(maxdd)", fitness_variable_names()).eval(v), 4.0);
  EXPECT_DOUBLE_EQ(FitnessExpr::parse("log(1)", fitness_variable_names()).eval(v), 0.0);
  EXPECT_DOUBLE_EQ(
      FitnessExpr::parse("MAX(netprofit, 0, -1)", fitness_variable_names()).eval(v), 0.0);
  EXPECT_DOUBLE_EQ(
      FitnessExpr::parse("min(trades, maxdd) * 2", fitness_variable_names()).eval(v), 18.0);
  // Functions nest and mix with the arithmetic grammar.
  EXPECT_DOUBLE_EQ(FitnessExpr::parse("netprofit / max(sqrt(maxdd), 1) + abs(-2)",
                                      fitness_variable_names())
                       .eval(v),
                   -8.0);
}

TEST(Fitness, FunctionsOutsideTheirDomainAreNotFinite) {
  const auto v = vars({{"netprofit", -4.0}});
  EXPECT_TRUE(std::isnan(
      FitnessExpr::parse("sqrt(netprofit)", fitness_variable_names()).eval(v)));
  EXPECT_FALSE(std::isfinite(
      FitnessExpr::parse("log(0)", fitness_variable_names()).eval(v)));
}

TEST(Fitness, FunctionArityAndSyntaxAreChecked) {
  EXPECT_THROW(FitnessExpr::parse("abs(1, 2)", fitness_variable_names()),
               std::invalid_argument);
  EXPECT_THROW(FitnessExpr::parse("max(1)", fitness_variable_names()),
               std::invalid_argument);
  EXPECT_THROW(FitnessExpr::parse("sqrt 4", fitness_variable_names()),
               std::invalid_argument);
  EXPECT_THROW(FitnessExpr::parse("min(1, 2", fitness_variable_names()),
               std::invalid_argument);
  EXPECT_THROW(FitnessExpr::parse("hypot(1, 2)", fitness_variable_names()),
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

TEST(Fitness, QualityMetricsAreVariables) {
  MetricsResult m;
  m.calmar = 2.5;
  m.sqn = 1.7;
  m.payoff = 1.2;
  m.maxConsecLosses = 4;
  m.maxDrawdownBars = 30;
  auto v = fitness_variables(m);
  auto e = FitnessExpr::parse("calmar + sqn + payoff - maxconseclosses - maxddbars",
                              fitness_variable_names());
  EXPECT_NEAR(e.eval(v), 2.5 + 1.7 + 1.2 - 4.0 - 30.0, 1e-12);
}
