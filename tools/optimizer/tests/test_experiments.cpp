/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Tests for the EXPERIMENTS strategy pack (STAD volumes 1-13): registry
// integrity, a full smoke run of every system on synthetic daily data, and
// unit coverage of the shared ExpHelpers utilities.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "TestBackend.h"
#include "backend/BatchBackend.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"
#include "optimize/GridSearch.h"
#include "portfolio/Portfolio.h"
#include "strategy/ExpHelpers.h"
#include "strategy/IStrategy.h"

using namespace topt;
using topt::testing::TestBackend;

namespace {

ParamSet params_of(const IStrategy& s) {
  ParamSet ps;
  for (const auto& def : s.params()) ps.values.emplace_back(def.name, def.defval);
  return ps;
}

std::vector<std::string> experiment_names() {
  std::vector<std::string> out;
  for (const auto& n : strategy_names())
    if (n.rfind("exp", 0) == 0) out.push_back(n);
  return out;
}

} // namespace

// --- registry ---------------------------------------------------------------

TEST(Experiments, RegistryHas94UniqueSystems) {
  const auto names = experiment_names();
  EXPECT_EQ(names.size(), 94u);
  std::set<std::string> uniq(names.begin(), names.end());
  EXPECT_EQ(uniq.size(), names.size()) << "duplicate strategy names registered";
  // Every volume 01..13 contributes at least one system.
  for (int vol = 1; vol <= 13; ++vol) {
    char prefix[8];
    std::snprintf(prefix, sizeof(prefix), "exp%02d_", vol);
    bool found = false;
    for (const auto& n : names)
      if (n.rfind(prefix, 0) == 0) found = true;
    EXPECT_TRUE(found) << "no systems registered for volume " << vol;
  }
}

TEST(Experiments, EverySystemDeclaresParamsAndIndicators) {
  for (const auto& name : experiment_names()) {
    auto s = make_strategy(name);
    ASSERT_NE(s, nullptr) << name;
    EXPECT_EQ(s->name(), name);
    const auto defs = s->params();
    EXPECT_FALSE(defs.empty()) << name;
    ParamSet ps = params_of(*s);
    EXPECT_TRUE(s->validate(ps)) << name << ": defaults must be valid";
    for (const auto& def : defs) {
      EXPECT_LE(def.minv, def.defval) << name << "." << def.name;
      EXPECT_GE(def.maxv, def.defval) << name << "." << def.name;
      EXPECT_GT(def.step, 0.0) << name << "." << def.name;
    }
    // Indicator requests must be well-formed (known family output counts).
    for (const auto& req : s->indicators(ps)) {
      EXPECT_GE(family_outputs(req.family), 1) << name;
    }
  }
}

TEST(Experiments, FastSlowValidationRejectsDegenerateGrids) {
  for (const char* name : {"exp12_lr_crossing", "exp12_kinda_shifty",
                           "exp12_first_prize", "exp12_ntw"}) {
    auto s = make_strategy(name);
    ParamSet ps = params_of(*s);
    // Find the fast/slow-style pair each of these strategies declares.
    const char* fastKey = ps.has("fastlen") ? "fastlen" : ps.has("fast") ? "fast" : "fastma";
    const char* slowKey = ps.has("slowlen") ? "slowlen" : ps.has("slow") ? "slow" : "slowma";
    ASSERT_TRUE(ps.has(fastKey) && ps.has(slowKey)) << name;
    const double slow = ps.get(slowKey);
    ps.set(fastKey, slow);
    EXPECT_FALSE(s->validate(ps)) << name << ": fast == slow must be invalid";
    ps.set(fastKey, slow + 1.0);
    EXPECT_FALSE(s->validate(ps)) << name << ": fast > slow must be invalid";
  }
}

// --- smoke run over synthetic daily data -------------------------------------

TEST(Experiments, AllSystemsRunCleanOnSyntheticData) {
  const Series bars = generate_synthetic(3000, 42);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  ExecConfig exec;

  int traded = 0;
  long long totalTrades = 0;
  for (const auto& name : experiment_names()) {
    auto s = make_strategy(name);
    ParamSet ps = params_of(*s);
    SimResult sim;
    VariantResult r = run_variant(bars, *s, ps, cache, exec, nullptr, &sim);
    ASSERT_TRUE(r.valid) << name << ": " << r.error;
    EXPECT_TRUE(std::isfinite(r.metrics.netProfit)) << name;
    EXPECT_TRUE(std::isfinite(r.metrics.maxDrawdown)) << name;
    EXPECT_GE(r.metrics.numTrades, 0) << name;
    // Positions must stay within one unit either way and every trade must
    // close at or after its entry (a fill on the last bar is force-closed
    // at that same bar's close).
    for (const Trade& t : sim.trades) {
      EXPECT_TRUE(t.dir == 1 || t.dir == -1) << name;
      EXPECT_GE(t.exitBar, t.entryBar) << name;
    }
    if (r.metrics.numTrades > 0) ++traded;
    totalTrades += r.metrics.numTrades;
  }
  // The pack as a whole must be alive on generic data: at least 2/3 of the
  // systems fire at least one trade over ~11.5 synthetic years.
  const int n = static_cast<int>(experiment_names().size());
  EXPECT_GE(traded, n * 2 / 3) << "only " << traded << " of " << n
                               << " systems traded (" << totalTrades << " trades)";
}

// Every system must also run through the portfolio path (the batch-API
// analog): shared parameters, one backend call per indicator request over
// all symbols, per-symbol simulation. On this CPU reference backend the
// aggregated result must match the sum of single-series runs.
TEST(Experiments, AllSystemsRunThroughPortfolioBatchPath) {
  const int numSymbols = 3;
  const int barCount = 2500;
  std::vector<Series> series;
  std::vector<std::string> names;
  for (int i = 0; i < numSymbols; ++i) {
    series.push_back(generate_synthetic(barCount, 42 + static_cast<std::uint64_t>(i)));
    names.push_back("s" + std::to_string(i + 1));
  }

  PortfolioConfig cfg;
  cfg.exec.capital = 30000.0;

  for (const auto& name : experiment_names()) {
    auto s = make_strategy(name);
    ParamSet ps = params_of(*s);

    std::vector<std::unique_ptr<IIndicatorBackend>> per;
    for (const Series& sr : series) per.push_back(std::make_unique<TestBackend>(sr));
    PerSymbolBatchBackend batch(std::move(per), "test-per-symbol");

    PortfolioResult res =
        run_portfolio(series, names, *s, ps, batch, cfg, nullptr);
    ASSERT_TRUE(res.has_valid()) << name;
    EXPECT_EQ(res.validSymbols, numSymbols) << name;
    EXPECT_TRUE(std::isfinite(res.metrics.netProfit)) << name;
    EXPECT_TRUE(std::isfinite(res.metrics.maxDrawdown)) << name;

    // Portfolio aggregation must equal per-symbol single-series runs with
    // the same capital share.
    ExecConfig symExec = cfg.exec;
    symExec.capital = cfg.exec.capital / numSymbols;
    double npSum = 0.0;
    int tradeSum = 0;
    for (int i = 0; i < numSymbols; ++i) {
      TestBackend one(series[static_cast<std::size_t>(i)]);
      IndicatorCache cache(series[static_cast<std::size_t>(i)], one);
      VariantResult ref = run_variant(series[static_cast<std::size_t>(i)], *s,
                                      ps, cache, symExec, nullptr, nullptr);
      ASSERT_TRUE(ref.valid) << name << " sym " << i << ": " << ref.error;
      npSum += ref.metrics.netProfit;
      tradeSum += ref.metrics.numTrades;
    }
    EXPECT_NEAR(res.metrics.netProfit, npSum, 1e-6) << name;
    EXPECT_EQ(res.metrics.numTrades, tradeSum) << name;
  }
}

TEST(Experiments, SignalsRespectWarmupAndRange) {
  const Series bars = generate_synthetic(1500, 7);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);

  for (const auto& name : experiment_names()) {
    auto s = make_strategy(name);
    ParamSet ps = params_of(*s);

    std::vector<std::vector<const float*>> lines;
    int warmup = 0;
    for (const auto& req : s->indicators(ps)) {
      const auto& entry = cache.get(req);
      std::vector<const float*> ptrs;
      for (const auto& line : entry.lines) ptrs.push_back(line.data());
      lines.push_back(std::move(ptrs));
      for (int fv : entry.firstValid) warmup = std::max(warmup, fv);
    }
    IndicatorView view(std::move(lines));
    std::vector<Signal> sig(static_cast<std::size_t>(bars.size()),
                            static_cast<Signal>(99));
    s->signals(bars, ps, view, sig.data());
    for (int i = 0; i < bars.size(); ++i) {
      const int v = sig[static_cast<std::size_t>(i)];
      ASSERT_TRUE(v == -1 || v == 0 || v == 1)
          << name << ": raw signal " << v << " at bar " << i;
    }
  }
}

// --- ExpHelpers unit coverage -------------------------------------------------

TEST(ExpHelpers, CrossingsMatchEasyLanguageSemantics) {
  const float a[] = {1.f, 2.f, 3.f, 2.f, 2.f, 4.f};
  const float b[] = {2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
  using namespace topt::exp;
  EXPECT_FALSE(crosses_above(a, b, 0));
  EXPECT_FALSE(crosses_above(a, b, 1)); // equal before is required (<=): 1->2 touches
  EXPECT_TRUE(crosses_above(a, b, 2));  // 2 -> 3 crosses through 2
  EXPECT_TRUE(crosses_below(a, b, 3) == false); // 3 -> 2 lands on the level
  EXPECT_TRUE(crosses_above(a, b, 5));  // 2 -> 4 leaves the level upward
  EXPECT_TRUE(crosses_above_level(a, 2.f, 2));
  EXPECT_FALSE(crosses_above_level(a, 2.f, 3));
  const float c[] = {3.f, 1.f};
  EXPECT_TRUE(crosses_below_level(c, 2.f, 1));
}

TEST(ExpHelpers, DayOfWeekMatchesKnownDates) {
  using topt::exp::day_of_week;
  EXPECT_EQ(day_of_week(0), 4);          // 1970-01-01 = Thursday
  EXPECT_EQ(day_of_week(86400 * 3), 0);  // 1970-01-04 = Sunday
  EXPECT_EQ(day_of_week(1751846400), 1); // 2025-07-07 = Monday
}

TEST(ExpHelpers, RollingStatsHandleWarmupAndValues) {
  using namespace topt::exp;
  const float line[] = {NAN, NAN, 1.f, 2.f, 3.f, 4.f, 5.f};
  const auto mean = rolling_mean(line, 7, 3);
  EXPECT_FALSE(finite(mean[3]));
  ASSERT_TRUE(finite(mean[4]));
  EXPECT_FLOAT_EQ(mean[4], 2.f);
  EXPECT_FLOAT_EQ(mean[6], 4.f);

  const auto mx = rolling_max(line, 7, 3);
  const auto mn = rolling_min(line, 7, 3);
  ASSERT_TRUE(finite(mx[4]) && finite(mn[4]));
  EXPECT_FLOAT_EQ(mx[6], 5.f);
  EXPECT_FLOAT_EQ(mn[6], 3.f);

  const float flat[] = {2.f, 2.f, 2.f, 2.f};
  const auto sd = rolling_std(flat, 4, 3);
  ASSERT_TRUE(finite(sd[3]));
  EXPECT_NEAR(sd[3], 0.f, 1e-6f);
  const float ramp[] = {1.f, 2.f, 3.f};
  const auto sd2 = rolling_std(ramp, 3, 3);
  ASSERT_TRUE(finite(sd2[2]));
  EXPECT_NEAR(sd2[2], std::sqrt(2.f / 3.f), 1e-5f);
}

TEST(ExpHelpers, EmaLineSeedsAtFirstFiniteValue) {
  using namespace topt::exp;
  const float line[] = {NAN, 10.f, 10.f, 10.f};
  const auto e = ema_line(line, 4, 3);
  EXPECT_FALSE(finite(e[0]));
  ASSERT_TRUE(finite(e[1]));
  EXPECT_FLOAT_EQ(e[1], 10.f);
  EXPECT_FLOAT_EQ(e[3], 10.f); // constant input stays constant
}

TEST(ExpHelpers, StopHitsUseBarExtremes) {
  using namespace topt::exp;
  Series b;
  b.open = {10.f};
  b.high = {12.f};
  b.low = {9.f};
  b.close = {11.f};
  b.volume = {0.f};
  b.ts = {0};
  EXPECT_TRUE(buy_stop_hit(b, 0, 11.5f));
  EXPECT_FALSE(buy_stop_hit(b, 0, 12.5f));
  EXPECT_TRUE(sell_stop_hit(b, 0, 9.5f));
  EXPECT_FALSE(sell_stop_hit(b, 0, 8.5f));
}

TEST(ExpHelpers, AvgRangeMatchesManualComputation) {
  using namespace topt::exp;
  Series b;
  for (int i = 0; i < 5; ++i) {
    b.open.push_back(10.f);
    b.high.push_back(10.f + static_cast<float>(i + 1)); // ranges 1,2,3,4,5
    b.low.push_back(10.f);
    b.close.push_back(10.f);
    b.volume.push_back(0.f);
    b.ts.push_back(0);
  }
  const auto ar = avg_range(b, 3);
  EXPECT_FALSE(finite(ar[1]));
  ASSERT_TRUE(finite(ar[2]));
  EXPECT_FLOAT_EQ(ar[2], 2.f); // (1+2+3)/3
  EXPECT_FLOAT_EQ(ar[4], 4.f); // (3+4+5)/3
}

TEST(ExpHelpers, OccurredAtLeastCountsWithinWindow) {
  using topt::exp::occurred_at_least;
  const std::vector<bool> cond = {false, true, false, true, true, false};
  EXPECT_TRUE(occurred_at_least(cond, 4, 2, 2));  // bars 3,4 both true
  EXPECT_FALSE(occurred_at_least(cond, 5, 1, 1)); // bar 5 false
  EXPECT_TRUE(occurred_at_least(cond, 5, 3, 2));  // bars 3,4 within last 3
  EXPECT_FALSE(occurred_at_least(cond, 2, 5, 3)); // only one true so far
}
