/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 8 (Omega Research STAD Club, Vol. 8).
// Systems: Escalator, Jack-in-the-Box, Ryan's Hope, Stampede.
// Skipped from this volume: ActivityBar Price Distribution (needs intrabar
// tick-distribution data); chapters on SystemBuilder, TradeStation-2000i
// forecasting and Fixed Ratio money management describe no OHLCV system.
// See docs/EXPERIMENTS.md for the adaptation notes.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// The four "Common Exits" from chapter 9, shared by Ryan's Hope and
// Stampede: protective ATR stop off the entry, trailing ATR stop off the
// position extreme, volatility ATR stop off the previous close, and the
// big-profit n-bar-extreme trail once open profit exceeds a threshold.
struct CommonExits {
  float entryPx = kNaN;
  float posHigh = kNaN;
  float posLow = kNaN;

  void on_entry(float fillPx, float hi, float lo) {
    entryPx = fillPx;
    posHigh = hi;
    posLow = lo;
  }

  // Returns true if any stop was hit at bar i (levels use atrPrev, the ATR
  // known at the previous close, as the EL orders were placed a bar ahead).
  bool step_long(const Series& bars, int i, float atrPrev, float prot,
                 float trail, float vol, float bigProfit, int exitBars) {
    posHigh = std::max(posHigh, H(bars, i));
    float stop = entryPx - prot * atrPrev;
    stop = std::max(stop, posHigh - trail * atrPrev);
    stop = std::max(stop, C(bars, i - 1) - vol * atrPrev);
    if (C(bars, i - 1) >= entryPx + bigProfit * atrPrev) {
      float lo = L(bars, i - 1);
      for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
        lo = std::min(lo, L(bars, i - 1 - k));
      stop = std::max(stop, lo);
    }
    return L(bars, i) <= stop;
  }

  bool step_short(const Series& bars, int i, float atrPrev, float prot,
                  float trail, float vol, float bigProfit, int exitBars) {
    posLow = std::min(posLow, L(bars, i));
    float stop = entryPx + prot * atrPrev;
    stop = std::min(stop, posLow + trail * atrPrev);
    stop = std::min(stop, C(bars, i - 1) + vol * atrPrev);
    if (C(bars, i - 1) <= entryPx - bigProfit * atrPrev) {
      float hi = H(bars, i - 1);
      for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
        hi = std::max(hi, H(bars, i - 1 - k));
      stop = std::min(stop, hi);
    }
    return H(bars, i) >= stop;
  }
};

// --- exp08_escalator ---------------------------------------------------------
// Weak-close/strong-close two-bar reversal in the direction of both moving
// averages; entry one tick beyond the two-bar extreme on the very next bar
// only. Bracket exit: stop at the pattern's other extreme, limit at twice
// the risk.
class EscalatorV8 final : public IStrategy {
public:
  std::string name() const override { return "exp08_escalator"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fastlen", 8, 3, 30, 1, true},
        {"slowlen", 40, 10, 100, 5, true},
        {"risklen", 2, 2, 6, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fastlen") < ps.get_int("slowlen");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("fastlen")}},
        {Family::SMA, Column::Close, {ps.get("slowlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* fast = ind.line(0);
    const float* slow = ind.line(1);
    const int n = bars.size();
    const int riskLen = ps.get_int("risklen");

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;   // valid one bar only
    float stopLvl = kNaN, target = kNaN;

    for (int i = 1; i < n; ++i) {
      // 1) Bracket exits for the open position.
      if (pos == 1) {
        if (L(bars, i) <= stopLvl) pos = 0;
        else if (finite(target) && H(bars, i) >= target) pos = 0;
      } else if (pos == -1) {
        if (H(bars, i) >= stopLvl) pos = 0;
        else if (finite(target) && L(bars, i) <= target) pos = 0;
      }
      // 2) One-bar entry stops from the previous close's setup.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        const float fill = std::max(O(bars, i), buyLvl);
        target = fill + 2.f * (fill - stopLvl);
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        const float fill = std::min(O(bars, i), sellLvl);
        target = fill - 2.f * (stopLvl - fill);
      }
      buyLvl = kNaN;
      sellLvl = kNaN;
      // 3) Escalator pattern at this close.
      if (!finite(fast[i]) || !finite(slow[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float rng = H(bars, i) - L(bars, i);
      const float rngPrev = H(bars, i - 1) - L(bars, i - 1);
      const bool weakPrev = C(bars, i - 1) <= L(bars, i - 1) + 0.25f * rngPrev;
      const bool strongNow = C(bars, i) >= H(bars, i) - 0.25f * rng;
      const bool strongPrev = C(bars, i - 1) >= H(bars, i - 1) - 0.25f * rngPrev;
      const bool weakNow = C(bars, i) <= L(bars, i) + 0.25f * rng;
      if (weakPrev && strongNow && C(bars, i) > fast[i] && C(bars, i) > slow[i] &&
          pos != 1) {
        buyLvl = std::max(H(bars, i), H(bars, i - 1));
        float lo = L(bars, i);
        for (int k = 1; k < riskLen && i - k >= 0; ++k) lo = std::min(lo, L(bars, i - k));
        stopLvl = lo;
      }
      if (strongPrev && weakNow && C(bars, i) < fast[i] && C(bars, i) < slow[i] &&
          pos != -1) {
        sellLvl = std::min(L(bars, i), L(bars, i - 1));
        float hi = H(bars, i);
        for (int k = 1; k < riskLen && i - k >= 0; ++k) hi = std::max(hi, H(bars, i - k));
        stopLvl = hi;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp08_jack_in_box -------------------------------------------------------
// A low popping above the high of the downtrend's lowest bar (or the mirror)
// arms an ATR-offset breakout entry for five bars. The exit is a modified
// parabolic seeded a multiple of ATR(3) beyond the entry bar.
class JackInBoxV8 final : public IStrategy {
public:
  std::string name() const override { return "exp08_jack_in_box"; }

  std::vector<ParamDef> params() const override {
    return {
        {"qbars", 5, 2, 12, 1, true},
        {"atrpcnt", 0.5, 0.1, 1.5, 0.1, false},
        {"accel", 0.02, 0.01, 0.1, 0.01, false},
        {"firstmult", 1.5, 0.5, 4.0, 0.25, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    (void)ps;
    return {
        {Family::ATR, Column::Close, {10.0}},
        {Family::ATR, Column::Close, {3.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr10 = ind.line(0);
    const float* atr3 = ind.line(1);
    const int n = bars.size();
    const int qBars = ps.get_int("qbars");
    const float pcnt = static_cast<float>(ps.get("atrpcnt"));
    const float accel = static_cast<float>(ps.get("accel"));
    const float firstMult = static_cast<float>(ps.get("firstmult"));

    int pos = 0;
    int trend = 0;
    float trendHigh = H(bars, 0), trendLow = L(bars, 0);
    int buyUntil = -1, sellUntil = -1;
    float sarStop = kNaN, af = 0.f, extreme = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(atr10[i]) || !finite(atr3[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Parabolic trailing exits from the previous bar's stop.
      if (pos == 1 && finite(sarStop)) {
        if (L(bars, i) <= sarStop) {
          pos = 0;
          sarStop = kNaN;
        } else {
          if (H(bars, i) > extreme) {
            extreme = H(bars, i);
            af = std::min(0.2f, af + accel);
          }
          sarStop = std::min(sarStop + af * (extreme - sarStop), L(bars, i));
        }
      } else if (pos == -1 && finite(sarStop)) {
        if (H(bars, i) >= sarStop) {
          pos = 0;
          sarStop = kNaN;
        } else {
          if (L(bars, i) < extreme) {
            extreme = L(bars, i);
            af = std::min(0.2f, af + accel);
          }
          sarStop = std::max(sarStop - af * (sarStop - extreme), H(bars, i));
        }
      }
      // 2) Armed ATR-offset entries (close proxies the unknown next open).
      if (pos != 1 && i <= buyUntil &&
          buy_stop_hit(bars, i, C(bars, i - 1) + pcnt * atr10[i - 1])) {
        pos = 1;
        sarStop = L(bars, i) - atr3[i] * firstMult;
        af = accel;
        extreme = H(bars, i);
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil &&
                 sell_stop_hit(bars, i, C(bars, i - 1) - pcnt * atr10[i - 1])) {
        pos = -1;
        sarStop = H(bars, i) + atr3[i] * firstMult;
        af = accel;
        extreme = L(bars, i);
        sellUntil = -1;
      }
      // 3) Trend flips and box pops at this close.
      if (L(bars, i) > trendHigh) {
        trend = 1;
        trendHigh = H(bars, i);
        trendLow = L(bars, i);
        if (pos != 1) buyUntil = i + qBars;
      } else if (H(bars, i) < trendLow) {
        trend = -1;
        trendHigh = H(bars, i);
        trendLow = L(bars, i);
        if (pos != -1) sellUntil = i + qBars;
      } else if (trend == 1 && H(bars, i) > trendHigh) {
        trendHigh = H(bars, i);
        trendLow = L(bars, i);
      } else if (trend == -1 && L(bars, i) < trendLow) {
        trendHigh = H(bars, i);
        trendLow = L(bars, i);
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp08_ryans_hope --------------------------------------------------------
// Rising SMA plus a pullback (close under the close of Length2 bars ago)
// with the longer trend intact enters at the next open; managed by the four
// common ATR exits.
class RyansHopeV8 final : public IStrategy {
public:
  std::string name() const override { return "exp08_ryans_hope"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length1", 20, 5, 60, 1, true},
        {"length2", 5, 2, 20, 1, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"volatrs", 2.0, 1.0, 6.0, 0.5, false},
        {"bigatrs", 7.0, 3.0, 12.0, 1.0, false},
        {"exitbars", 3, 2, 8, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("length1")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* avg = ind.line(0);
    const float* atr = ind.line(1);
    const int n = bars.size();
    const int len1 = ps.get_int("length1");
    const int len2 = ps.get_int("length2");
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));
    const float vol = static_cast<float>(ps.get("volatrs"));
    const float big = static_cast<float>(ps.get("bigatrs"));
    const int exitBars = ps.get_int("exitbars");

    int pos = 0;
    bool pendingLong = false, pendingShort = false;
    CommonExits ex;

    for (int i = 1; i < n; ++i) {
      // 1) Fill pending market entries at this open.
      if (pendingLong && pos != 1) {
        pos = 1;
        ex.on_entry(O(bars, i), H(bars, i), L(bars, i));
      } else if (pendingShort && pos != -1) {
        pos = -1;
        ex.on_entry(O(bars, i), H(bars, i), L(bars, i));
      } else if (pos != 0 && finite(atr[i - 1])) {
        // 2) Common ATR exits (skipped on the fill bar itself).
        if (pos == 1 && ex.step_long(bars, i, atr[i - 1], prot, trail, vol, big, exitBars))
          pos = 0;
        else if (pos == -1 &&
                 ex.step_short(bars, i, atr[i - 1], prot, trail, vol, big, exitBars))
          pos = 0;
      }
      pendingLong = pendingShort = false;
      // 3) Pullback setups at this close.
      const int back = len1 + len2;
      if (i >= back && finite(avg[i]) && finite(avg[i - len2])) {
        if (avg[i] > avg[i - len2] && C(bars, i) < C(bars, i - len2) &&
            C(bars, i) > C(bars, i - back) && pos != 1) {
          pendingLong = true;
        }
        if (avg[i] < avg[i - len2] && C(bars, i) > C(bars, i - len2) &&
            C(bars, i) < C(bars, i - back) && pos != -1) {
          pendingShort = true;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp08_stampede ----------------------------------------------------------
// Price-channel breakout (one tick beyond the n-bar extreme) managed by the
// four common ATR exits. The original pyramids up to nine entries; this
// adaptation trades a single unit per side.
class StampedeV8 final : public IStrategy {
public:
  std::string name() const override { return "exp08_stampede"; }

  std::vector<ParamDef> params() const override {
    return {
        {"chanlen", 30, 10, 60, 5, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"volatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"bigatrs", 8.0, 3.0, 14.0, 1.0, false},
        {"exitbars", 5, 2, 10, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("chanlen")}},
        {Family::MIN, Column::Low, {ps.get("chanlen")}},
        {Family::ATR, Column::Close, {ps.get("chanlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiCh = ind.line(0);
    const float* loCh = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));
    const float vol = static_cast<float>(ps.get("volatrs"));
    const float big = static_cast<float>(ps.get("bigatrs"));
    const int exitBars = ps.get_int("exitbars");

    int pos = 0;
    CommonExits ex;

    for (int i = 1; i < n; ++i) {
      if (!finite(hiCh[i - 1]) || !finite(loCh[i - 1]) || !finite(atr[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Common ATR exits.
      if (pos == 1 && ex.step_long(bars, i, atr[i - 1], prot, trail, vol, big, exitBars))
        pos = 0;
      else if (pos == -1 &&
               ex.step_short(bars, i, atr[i - 1], prot, trail, vol, big, exitBars))
        pos = 0;
      // 2) Channel breakout entries (stop-and-reverse capable).
      if (pos != 1 && buy_stop_hit(bars, i, hiCh[i - 1])) {
        pos = 1;
        ex.on_entry(std::max(O(bars, i), hiCh[i - 1]), H(bars, i), L(bars, i));
      } else if (pos != -1 && sell_stop_hit(bars, i, loCh[i - 1])) {
        pos = -1;
        ex.on_entry(std::min(O(bars, i), loCh[i - 1]), H(bars, i), L(bars, i));
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

} // namespace

void register_vol08(std::vector<Factory>& out) {
  out.push_back({"exp08_escalator",
                 [] { return std::unique_ptr<IStrategy>(new EscalatorV8); }});
  out.push_back({"exp08_jack_in_box",
                 [] { return std::unique_ptr<IStrategy>(new JackInBoxV8); }});
  out.push_back({"exp08_ryans_hope",
                 [] { return std::unique_ptr<IStrategy>(new RyansHopeV8); }});
  out.push_back({"exp08_stampede",
                 [] { return std::unique_ptr<IStrategy>(new StampedeV8); }});
}

} // namespace topt::exp
