/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "strategy/IStrategy.h"

#include <cmath>
#include <stdexcept>

#include "strategy/ExpRegistry.h"

namespace topt {

namespace {

bool finite(float v) { return std::isfinite(v); }

// --- ma_cross: EMA(fast) x EMA(slow), stop-and-reverse ---------------------
// Long while the fast EMA is above the slow one, short while below.
class MaCross final : public IStrategy {
public:
  std::string name() const override { return "ma_cross"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fast", 12, 2, 100, 1, true},
        {"slow", 48, 5, 250, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fast") < ps.get_int("slow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {static_cast<double>(ps.get_int("fast"))}},
        {Family::EMA, Column::Close, {static_cast<double>(ps.get_int("slow"))}},
    };
  }

  void signals(const Series& bars, const ParamSet&, const IndicatorView& ind,
               Signal* out) const override {
    const float* fast = ind.line(0);
    const float* slow = ind.line(1);
    const int n = bars.size();
    for (int i = 0; i < n; ++i) {
      if (!finite(fast[i]) || !finite(slow[i])) {
        out[i] = 0;
      } else if (fast[i] > slow[i]) {
        out[i] = 1;
      } else if (fast[i] < slow[i]) {
        out[i] = -1;
      } else {
        out[i] = i > 0 ? out[i - 1] : 0;
      }
    }
  }
};

// --- rsi_reversion: long-only mean reversion on RSI ------------------------
// Enter long when RSI drops below `low`, exit when it rises above `high`.
class RsiReversion final : public IStrategy {
public:
  std::string name() const override { return "rsi_reversion"; }

  std::vector<ParamDef> params() const override {
    return {
        {"period", 14, 2, 100, 1, true},
        {"low", 30, 5, 50, 1, true},
        {"high", 70, 50, 95, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("low") < ps.get("high");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::RSI, Column::Close, {static_cast<double>(ps.get_int("period"))}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* rsi = ind.line(0);
    const float low = static_cast<float>(ps.get("low"));
    const float high = static_cast<float>(ps.get("high"));
    const int n = bars.size();
    Signal pos = 0;
    for (int i = 0; i < n; ++i) {
      if (!finite(rsi[i])) {
        pos = 0;
      } else if (rsi[i] < low) {
        pos = 1;
      } else if (rsi[i] > high) {
        pos = 0;
      }
      out[i] = pos;
    }
  }
};

// --- bbands_breakout: Bollinger band breakout with middle-band exit --------
// Flat -> long on a close above the upper band, flat -> short on a close
// below the lower band; an open position exits when the close crosses back
// over the middle band.
class BbandsBreakout final : public IStrategy {
public:
  std::string name() const override { return "bbands_breakout"; }

  std::vector<ParamDef> params() const override {
    return {
        {"period", 20, 5, 100, 1, true},
        {"width", 2.0, 0.5, 4.0, 0.25, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("width") > 0.0;
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    double width = ps.get("width");
    return {
        {Family::BBANDS, Column::Close,
         {static_cast<double>(ps.get_int("period")), width, width}},
    };
  }

  void signals(const Series& bars, const ParamSet&, const IndicatorView& ind,
               Signal* out) const override {
    const float* upper = ind.line(0, 0);
    const float* middle = ind.line(0, 1);
    const float* lower = ind.line(0, 2);
    const int n = bars.size();
    Signal pos = 0;
    for (int i = 0; i < n; ++i) {
      float c = bars.close[static_cast<std::size_t>(i)];
      if (!finite(upper[i]) || !finite(middle[i]) || !finite(lower[i])) {
        pos = 0;
      } else if (pos == 0) {
        if (c > upper[i]) pos = 1;
        else if (c < lower[i]) pos = -1;
      } else if (pos == 1 && c < middle[i]) {
        pos = 0;
      } else if (pos == -1 && c > middle[i]) {
        pos = 0;
      }
      out[i] = pos;
    }
  }
};

} // namespace

namespace {

const std::vector<exp::Factory>& experiment_factories() {
  static const std::vector<exp::Factory> factories = [] {
    std::vector<exp::Factory> out;
    exp::register_vol01(out);
    exp::register_vol02(out);
    exp::register_vol03(out);
    exp::register_vol04(out);
    exp::register_vol05(out);
    exp::register_vol06(out);
    exp::register_vol07(out);
    exp::register_vol08(out);
    exp::register_vol09(out);
    exp::register_vol10(out);
    exp::register_vol11(out);
    exp::register_vol12(out);
    exp::register_vol13(out);
    return out;
  }();
  return factories;
}

} // namespace

std::vector<std::string> strategy_names() {
  std::vector<std::string> names = {"ma_cross", "rsi_reversion", "bbands_breakout"};
  for (const auto& f : experiment_factories()) {
    names.emplace_back(f.name);
  }
  return names;
}

std::unique_ptr<IStrategy> make_strategy(const std::string& name) {
  if (name == "ma_cross") return std::make_unique<MaCross>();
  if (name == "rsi_reversion") return std::make_unique<RsiReversion>();
  if (name == "bbands_breakout") return std::make_unique<BbandsBreakout>();
  for (const auto& f : experiment_factories()) {
    if (name == f.name) return f.make();
  }
  throw std::invalid_argument(
      "unknown strategy '" + name +
      "'; run `tacuda-opt list` for the built-in and EXPERIMENTS strategies");
}

} // namespace topt
