/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>

#include "TestBackend.h"
#include "core/Fitness.h"
#include "core/ParamSpace.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"
#include "optimize/Genetic.h"
#include "optimize/GridSearch.h"
#include "strategy/IStrategy.h"

using namespace topt;
using topt::testing::TestBackend;

namespace {

GeneticConfig quiet_ga() {
  GeneticConfig cfg;
  cfg.eval.quiet = true;
  cfg.eval.threads = 4;
  cfg.quiet = true;
  return cfg;
}

GeneticResult run_ga(const Series& bars, const GeneticConfig& cfg,
                     const std::vector<GridAxis>& axes,
                     const std::string& fitnessExpr = "netprofit") {
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  auto fitness = FitnessExpr::parse(fitnessExpr, fitness_variable_names());
  return run_genetic(bars, *s, axes, {}, cache, cfg, fitness);
}

} // namespace

TEST(Genetic, DeterministicForFixedSeed) {
  Series bars = generate_synthetic(4000, 11);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=2..30:2"),
                                parse_grid_spec("slow=35..95:5")};
  GeneticConfig cfg = quiet_ga();
  cfg.population = 16;
  cfg.maxGenerations = 8;
  cfg.elite = 3;
  cfg.seed = 123;

  GeneticResult a = run_ga(bars, cfg, axes);
  GeneticResult b = run_ga(bars, cfg, axes);

  ASSERT_TRUE(a.has_best());
  ASSERT_TRUE(b.has_best());
  EXPECT_EQ(a.evaluations, b.evaluations);
  EXPECT_EQ(a.generations, b.generations);
  EXPECT_EQ(a.evaluated[a.bestIndex].params.to_string(),
            b.evaluated[b.bestIndex].params.to_string());
  EXPECT_DOUBLE_EQ(a.evaluated[a.bestIndex].fitness, b.evaluated[b.bestIndex].fitness);
  ASSERT_EQ(a.bestHistory.size(), b.bestHistory.size());
  for (std::size_t i = 0; i < a.bestHistory.size(); ++i) {
    EXPECT_DOUBLE_EQ(a.bestHistory[i], b.bestHistory[i]);
  }
}

TEST(Genetic, NeverEvaluatesTheSameGenomeTwice) {
  Series bars = generate_synthetic(3000, 21);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=2..20:2"),
                                parse_grid_spec("slow=30..60:5")};
  GeneticConfig cfg = quiet_ga();
  cfg.population = 20;
  cfg.maxGenerations = 10;
  cfg.elite = 4;
  cfg.seed = 7;

  GeneticResult res = run_ga(bars, cfg, axes);
  EXPECT_EQ(res.evaluations, res.evaluated.size());
  std::set<std::string> unique;
  for (const auto& v : res.evaluated) {
    unique.insert(v.params.to_string());
  }
  EXPECT_EQ(unique.size(), res.evaluated.size());
}

TEST(Genetic, BestFitnessIsMonotoneWithElitism) {
  Series bars = generate_synthetic(3000, 31);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=2..20:1"),
                                parse_grid_spec("slow=25..85:5")};
  GeneticConfig cfg = quiet_ga();
  cfg.population = 16;
  cfg.maxGenerations = 12;
  cfg.elite = 2;
  cfg.seed = 3;

  GeneticResult res = run_ga(bars, cfg, axes);
  ASSERT_FALSE(res.bestHistory.empty());
  for (std::size_t i = 1; i < res.bestHistory.size(); ++i) {
    EXPECT_GE(res.bestHistory[i], res.bestHistory[i - 1]) << "generation " << i;
  }
}

TEST(Genetic, MutationStaysInsideAxes) {
  Series bars = generate_synthetic(2000, 41);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=4..12:2"),
                                parse_grid_spec("slow=20..40:10")};
  GeneticConfig cfg = quiet_ga();
  cfg.population = 12;
  cfg.maxGenerations = 15;
  cfg.mutationProb = 0.9;      // mutate aggressively on purpose
  cfg.mutationSigmaSteps = 50; // way past the axis bounds
  cfg.elite = 1;
  cfg.seed = 17;

  GeneticResult res = run_ga(bars, cfg, axes);
  std::set<double> fastAllowed(axes[0].values.begin(), axes[0].values.end());
  std::set<double> slowAllowed(axes[1].values.begin(), axes[1].values.end());
  for (const auto& v : res.evaluated) {
    EXPECT_TRUE(fastAllowed.count(v.params.get("fast"))) << v.params.to_string();
    EXPECT_TRUE(slowAllowed.count(v.params.get("slow"))) << v.params.to_string();
  }
}

TEST(Genetic, InvalidCombinationsLoseButDoNotCrash) {
  Series bars = generate_synthetic(2000, 51);
  // Overlapping axes: many fast >= slow individuals are invalid.
  std::vector<GridAxis> axes = {parse_grid_spec("fast=10..60:10"),
                                parse_grid_spec("slow=20..40:10")};
  GeneticConfig cfg = quiet_ga();
  cfg.population = 12;
  cfg.maxGenerations = 8;
  cfg.elite = 2;
  cfg.seed = 29;

  GeneticResult res = run_ga(bars, cfg, axes);
  ASSERT_TRUE(res.has_best());
  const VariantResult& best = res.evaluated[res.bestIndex];
  EXPECT_TRUE(best.valid);
  EXPECT_LT(best.params.get("fast"), best.params.get("slow"));
  bool sawInvalid = false;
  for (const auto& v : res.evaluated) {
    if (!v.valid) sawInvalid = true;
  }
  EXPECT_TRUE(sawInvalid); // the overlapping space really was explored
}

TEST(Genetic, StopsEarlyOnStagnation) {
  Series bars = generate_synthetic(2000, 61);
  // Tiny space: 9 genomes; the GA exhausts it quickly and stalls.
  std::vector<GridAxis> axes = {parse_grid_spec("fast=2..4:1"),
                                parse_grid_spec("slow=10..12:1")};
  GeneticConfig cfg = quiet_ga();
  cfg.population = 6;
  cfg.maxGenerations = 50;
  cfg.elite = 1;
  cfg.stagnationLimit = 3;
  cfg.seed = 5;

  GeneticResult res = run_ga(bars, cfg, axes);
  EXPECT_LT(res.generations, 50);
  EXPECT_LE(res.evaluations, 9u);
}

TEST(Genetic, RejectsBadConfigAndDegenerateSpace) {
  Series bars = generate_synthetic(500, 71);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=2..10:2")};

  GeneticConfig bad = quiet_ga();
  bad.population = 1;
  EXPECT_THROW(run_ga(bars, bad, axes), std::invalid_argument);

  GeneticConfig badElite = quiet_ga();
  badElite.population = 8;
  badElite.elite = 8;
  EXPECT_THROW(run_ga(bars, badElite, axes), std::invalid_argument);

  // No axis with more than one value -> nothing to search.
  EXPECT_THROW(run_ga(bars, quiet_ga(), {parse_grid_spec("fast=5")}),
               std::invalid_argument);
}

// M2 acceptance: on the M1-style grid the GA must reach >= 95% of the
// exhaustive-grid best fitness while simulating <= 10% of the variants.
TEST(Genetic, ReachesGridQualityWithFractionOfEvaluations) {
  Series bars = generate_synthetic(20000, 42);
  std::vector<GridAxis> axes = {parse_grid_spec("fast=2..21:1"),
                                parse_grid_spec("slow=25..84:1")};
  auto s = make_strategy("ma_cross");
  auto fitness = FitnessExpr::parse("netprofit", fitness_variable_names());
  auto variants = enumerate_variants(s->params(), axes, {});
  ASSERT_EQ(variants.size(), 1200u);

  TestBackend gridBackend(bars);
  IndicatorCache gridCache(bars, gridBackend);
  GridConfig gcfg;
  gcfg.quiet = true;
  gcfg.threads = 4;
  auto gridResults = run_grid(bars, *s, variants, gridCache, gcfg, fitness);
  double gridBest = -1e300;
  for (const auto& v : gridResults) {
    if (v.valid) gridBest = std::max(gridBest, v.fitness);
  }
  ASSERT_GT(gridBest, 0.0) << "test landscape must have a positive best";

  GeneticConfig cfg = quiet_ga();
  cfg.population = 20;
  cfg.maxGenerations = 6;
  cfg.elite = 4;
  cfg.tournament = 4;
  cfg.stagnationLimit = 6;
  cfg.seed = 9;
  GeneticResult res = run_ga(bars, cfg, axes);

  ASSERT_TRUE(res.has_best());
  EXPECT_LE(res.evaluations, variants.size() / 10)
      << "GA budget must stay within 10% of the grid";
  EXPECT_GE(res.evaluated[res.bestIndex].fitness, 0.95 * gridBest)
      << "GA best " << res.evaluated[res.bestIndex].fitness << " vs grid best "
      << gridBest;
}
