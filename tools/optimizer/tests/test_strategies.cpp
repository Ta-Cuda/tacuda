/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <cmath>

#include "TestBackend.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"
#include "optimize/GridSearch.h"
#include "strategy/IStrategy.h"

using namespace topt;
using topt::testing::TestBackend;

namespace {

ParamSet params_of(const IStrategy& s) {
  ParamSet ps;
  for (const auto& def : s.params()) {
    ps.values.emplace_back(def.name, def.defval);
  }
  return ps;
}

} // namespace

TEST(Strategies, RegistryKnowsCoreAndExperimentPacks) {
  auto names = strategy_names();
  // 3 core strategies + 94 EXPERIMENTS systems (STAD volumes 1-13).
  ASSERT_EQ(names.size(), 97u);
  for (const auto& n : names) {
    auto s = make_strategy(n);
    EXPECT_EQ(s->name(), n);
    EXPECT_FALSE(s->params().empty());
  }
  EXPECT_THROW(make_strategy("nope"), std::invalid_argument);
}

TEST(Strategies, MaCrossValidatesFastBelowSlow) {
  auto s = make_strategy("ma_cross");
  ParamSet ps = params_of(*s);
  ps.set("fast", 50);
  ps.set("slow", 20);
  EXPECT_FALSE(s->validate(ps));
  ps.set("slow", 50);
  EXPECT_FALSE(s->validate(ps)); // equal is degenerate too
  ps.set("slow", 51);
  EXPECT_TRUE(s->validate(ps));
}

TEST(Strategies, RsiReversionIsLongOnly) {
  Series bars = generate_synthetic(3000, 11);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("rsi_reversion");
  ExecConfig exec;

  SimResult sim;
  VariantResult r = run_variant(bars, *s, params_of(*s), cache, exec, nullptr, &sim);
  ASSERT_TRUE(r.valid) << r.error;
  for (const Trade& t : sim.trades) {
    EXPECT_EQ(t.dir, 1) << "rsi_reversion must never go short";
  }
}

TEST(Strategies, MaCrossTradesOnTrendReversal) {
  // 400 bars up, 400 bars down: the cross must flip the position sign.
  Series bars;
  for (int i = 0; i < 800; ++i) {
    double base = i < 400 ? 100.0 + i * 0.5 : 300.0 - (i - 400) * 0.5;
    bars.open.push_back(static_cast<float>(base));
    bars.close.push_back(static_cast<float>(base + 0.2));
    bars.high.push_back(static_cast<float>(base + 0.4));
    bars.low.push_back(static_cast<float>(base - 0.4));
    bars.volume.push_back(0.0f);
    bars.ts.push_back(0);
  }
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  ParamSet ps = params_of(*s);
  ps.set("fast", 10);
  ps.set("slow", 30);
  ExecConfig exec;

  SimResult sim;
  VariantResult r = run_variant(bars, *s, ps, cache, exec, nullptr, &sim);
  ASSERT_TRUE(r.valid) << r.error;
  ASSERT_GE(sim.trades.size(), 2u);
  EXPECT_EQ(sim.trades.front().dir, 1) << "uptrend should open a long first";
  bool hasShort = false;
  for (const Trade& t : sim.trades) hasShort |= t.dir == -1;
  EXPECT_TRUE(hasShort) << "downtrend leg should flip the position short";
}

TEST(Strategies, NoTradingDuringIndicatorWarmup) {
  Series bars = generate_synthetic(2000, 3);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  ParamSet ps = params_of(*s);
  ps.set("fast", 20);
  ps.set("slow", 100);
  ExecConfig exec;

  SimResult sim;
  VariantResult r = run_variant(bars, *s, ps, cache, exec, nullptr, &sim);
  ASSERT_TRUE(r.valid) << r.error;
  for (const Trade& t : sim.trades) {
    // Slow EMA(100) first valid index is 99; first possible fill is bar 100.
    EXPECT_GE(t.entryBar, 100);
  }
}

TEST(Strategies, InsufficientDataIsReportedNotCrashed) {
  Series bars = generate_synthetic(50, 5);
  TestBackend backend(bars);
  IndicatorCache cache(bars, backend);
  auto s = make_strategy("ma_cross");
  ParamSet ps = params_of(*s);
  ps.set("fast", 30);
  ps.set("slow", 200); // longer than the series
  ExecConfig exec;

  VariantResult r = run_variant(bars, *s, ps, cache, exec, nullptr);
  EXPECT_FALSE(r.valid);
  EXPECT_FALSE(r.error.empty());
}

TEST(Strategies, SignalsAreCausal) {
  // Two series identical for the first 300 bars, then diverging hard.
  Series a = generate_synthetic(600, 21);
  Series b = a;
  for (int i = 300; i < b.size(); ++i) {
    auto idx = static_cast<std::size_t>(i);
    b.close[idx] += 50.0f;
    b.open[idx] += 50.0f;
    b.high[idx] += 50.0f;
    b.low[idx] += 50.0f;
  }

  for (const auto& name : strategy_names()) {
    auto s = make_strategy(name);
    ParamSet ps = params_of(*s);

    TestBackend backendA(a), backendB(b);
    IndicatorCache cacheA(a, backendA), cacheB(b, backendB);

    auto compute_signals = [&](const Series& bars, IndicatorCache& cache) {
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
      std::vector<Signal> sig(static_cast<std::size_t>(bars.size()), 0);
      s->signals(bars, ps, view, sig.data());
      std::fill(sig.begin(), sig.begin() + warmup, static_cast<Signal>(0));
      return sig;
    };

    auto sigA = compute_signals(a, cacheA);
    auto sigB = compute_signals(b, cacheB);
    for (int i = 0; i < 300; ++i) {
      ASSERT_EQ(sigA[static_cast<std::size_t>(i)], sigB[static_cast<std::size_t>(i)])
          << name << ": signal at bar " << i << " depends on future data";
    }
  }
}
