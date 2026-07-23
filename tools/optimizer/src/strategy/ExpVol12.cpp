/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 12 (Omega Research STAD Club, Vol. 12).
// Systems: First Prize, In The Zone, Jailbreak, LR Crossing, Not That Way
// The Other Way, Seems Kinda Shifty, Springs & Upthrusts, Three Musketeers,
// Tigers in the Mist #2.
// Skipped from this volume: "Trendiness in the Futures Markets" (research
// note without trading rules). See docs/EXPERIMENTS.md for adaptations.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Per-trade modified parabolic stop (STAD "Parabolic ModStop"): seeded one
// ATR multiple past the entry bar's extreme, then accelerating toward the
// best excursion on every new extreme.
struct TradeParabolic {
  float stop = kNaN;
  float extreme = kNaN;
  float af = 0.f;
  float accel = 0.02f;

  void arm_long(float entryLow, float atrMult, float atr, float accelIn) {
    stop = entryLow - atrMult * atr;
    extreme = entryLow;
    af = accelIn;
    accel = accelIn;
  }
  void arm_short(float entryHigh, float atrMult, float atr, float accelIn) {
    stop = entryHigh + atrMult * atr;
    extreme = entryHigh;
    af = accelIn;
    accel = accelIn;
  }
  // Returns true when the stop is tagged intrabar.
  bool step_long(float high, float low) {
    if (low <= stop) return true;
    if (high > extreme) {
      extreme = high;
      af = std::min(af + accel, 0.4f);
    }
    stop += af * (extreme - stop);
    return false;
  }
  bool step_short(float high, float low) {
    if (high >= stop) return true;
    if (low < extreme) {
      extreme = low;
      af = std::min(af + accel, 0.4f);
    }
    stop -= af * (stop - extreme);
    return false;
  }
};

// --- exp12_first_prize -------------------------------------------------------
// Buys the first MACD pullback of a new up leg (Signal line above zero,
// MACD dipping under it once) above a rising EMA, entering on a return to
// the setup bar's high. Seven stops manage the trade: protective,
// breakeven, trailing, volatility, big-profit, countertrend and
// trend-reversal.
class FirstPrizeV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_first_prize"; }

  std::vector<ParamDef> params() const override {
    return {
        {"xavglen", 60, 20, 100, 5, true},
        {"fastma", 3, 2, 8, 1, true},
        {"slowma", 10, 6, 20, 1, true},
        {"signal", 16, 5, 30, 1, true},
        {"ips", 3.0, 1.0, 6.0, 0.5, false},
        {"bps", 9.0, 4.0, 14.0, 1.0, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("fastma") < ps.get("slowma");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MACD, Column::Close,
         {ps.get("fastma"), ps.get("slowma"), ps.get("signal")}},
        {Family::EMA, Column::Close, {ps.get("xavglen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* macd = ind.line(0, 0);
    const float* sig = ind.line(0, 1);
    const float* ema = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float ips = static_cast<float>(ps.get("ips"));
    const float bes = 3.f, ts = 6.f, vs = 3.f;
    const float bps = static_cast<float>(ps.get("bps"));
    const int nb = 5, barsIn = 2, barsReq = 2;
    const float atrPcnt = 0.6f;

    int pos = 0;
    bool buySetup = false, sellSetup = false;
    bool newBuySignal = false, newSellSignal = false;
    float cTrendLow = kNaN, cTrendHigh = kNaN;
    float buyLvl = kNaN, sellLvl = kNaN;
    float entryPx = kNaN, closeHigh = kNaN, closeLow = kNaN;
    int barsInPos = 0;

    for (int i = 1; i < n; ++i) {
      if (!finite(macd[i]) || !finite(sig[i]) || !finite(sig[i - 1]) || !finite(ema[i]) ||
          !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      const bool upTrendPrev = sig[i - 1] > 0.f;
      // 1) Long exit stack (levels computed on the previous bar).
      if (pos == 1) {
        ++barsInPos;
        float stop = entryPx - ips * atrPrev;
        if (closeHigh >= entryPx + bes * atrPrev) stop = std::max(stop, entryPx);
        stop = std::max(stop, closeHigh - ts * atrPrev);
        stop = std::max(stop, C(bars, i - 1) - vs * atrPrev);
        if (closeHigh >= entryPx + bps * atrPrev) {
          float lo = L(bars, i - 1);
          for (int k = 1; k < nb && i - 1 - k >= 0; ++k) lo = std::min(lo, L(bars, i - 1 - k));
          stop = std::max(stop, lo);
        }
        if (finite(cTrendLow)) stop = std::max(stop, cTrendLow - atrPcnt * atrPrev);
        if (!upTrendPrev && barsInPos >= barsIn + 1) stop = std::max(stop, L(bars, i - 1));
        if (L(bars, i) <= stop) pos = 0;
        closeHigh = std::max(closeHigh, C(bars, i));
      } else if (pos == -1) {
        ++barsInPos;
        float stop = entryPx + ips * atrPrev;
        if (closeLow <= entryPx - bes * atrPrev) stop = std::min(stop, entryPx);
        stop = std::min(stop, closeLow + ts * atrPrev);
        stop = std::min(stop, C(bars, i - 1) + vs * atrPrev);
        if (closeLow <= entryPx - bps * atrPrev) {
          float hi = H(bars, i - 1);
          for (int k = 1; k < nb && i - 1 - k >= 0; ++k) hi = std::max(hi, H(bars, i - 1 - k));
          stop = std::min(stop, hi);
        }
        if (finite(cTrendHigh)) stop = std::min(stop, cTrendHigh + atrPcnt * atrPrev);
        if (upTrendPrev && barsInPos >= barsIn + 1) stop = std::min(stop, H(bars, i - 1));
        if (H(bars, i) >= stop) pos = 0;
        closeLow = std::min(closeLow, C(bars, i));
      }
      // 2) Armed setup-bar-extreme entries.
      if (pos == 0 && buySetup && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        closeHigh = C(bars, i);
        barsInPos = 0;
        buySetup = false;
      } else if (pos == 0 && sellSetup && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        closeLow = C(bars, i);
        barsInPos = 0;
        sellSetup = false;
      }
      // 3) Trend / pullback bookkeeping at this close.
      const bool upTrend = sig[i] > 0.f;
      const bool downTrend = sig[i] < 0.f;
      if (upTrend) {
        if (!upTrendPrev) {
          newBuySignal = true;
          sellSetup = false;
          newSellSignal = false;
        }
        if (newBuySignal) {
          if (crosses_below(macd, sig, i)) {
            buySetup = true;
            cTrendLow = L(bars, i);
            newBuySignal = false;
          }
        } else if (buySetup && i >= barsReq) {
          bool allAbove = true;
          for (int k = 0; k < barsReq; ++k)
            if (!(macd[i - k] > sig[i - k])) allAbove = false;
          if (allAbove) buySetup = false;
        }
        if (macd[i] < sig[i] && finite(cTrendLow) && L(bars, i) < cTrendLow)
          cTrendLow = L(bars, i);
      }
      if (downTrend) {
        if (upTrendPrev || sig[i - 1] == 0.f) {
          newSellSignal = true;
          buySetup = false;
          newBuySignal = false;
        }
        if (newSellSignal) {
          if (crosses_above(macd, sig, i)) {
            sellSetup = true;
            cTrendHigh = H(bars, i);
            newSellSignal = false;
          }
        } else if (sellSetup && i >= barsReq) {
          bool allBelow = true;
          for (int k = 0; k < barsReq; ++k)
            if (!(macd[i - k] < sig[i - k])) allBelow = false;
          if (allBelow) sellSetup = false;
        }
        if (macd[i] > sig[i] && finite(cTrendHigh) && H(bars, i) > cTrendHigh)
          cTrendHigh = H(bars, i);
      }
      if (pos != 0) {
        buySetup = sellSetup = false;
      }
      buyLvl = (buySetup && H(bars, i) > ema[i]) ? H(bars, i) : kNaN;
      sellLvl = (sellSetup && L(bars, i) < ema[i]) ? L(bars, i) : kNaN;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp12_in_the_zone -------------------------------------------------------
// Stowell's zone trade: after a thrust close above the high of bar 3, a
// close back inside the support zone (below the previous close) arms a stop
// at the setup bar's high; protective stop below the setup bar, breakeven
// at a risk multiple and an ATR profit target above the thrust bar.
class InTheZoneV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_in_the_zone"; }

  std::vector<ParamDef> params() const override {
    return {
        {"ipsfactor", 3.0, 1.0, 6.0, 0.5, false},
        {"befactor", 3.0, 1.0, 6.0, 0.5, false},
        {"ptfactor", 5.0, 2.0, 10.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    (void)ps;
    return {{Family::ATR, Column::Close, {10.0}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const int n = bars.size();
    const float ips = static_cast<float>(ps.get("ipsfactor"));
    const float bef = static_cast<float>(ps.get("befactor"));
    const float ptf = static_cast<float>(ps.get("ptfactor"));

    int pos = 0;
    bool buySetup = false, sellSetup = false;
    float upperBuy = kNaN, lowerBuy = kNaN, upperSell = kNaN, lowerSell = kNaN;
    float buyLvl = kNaN, sellLvl = kNaN;
    float lTarget = kNaN, lStop = kNaN, sTarget = kNaN, sStop = kNaN;
    float entryPx = kNaN;
    bool lBreakeven = false, sBreakeven = false;

    for (int i = 4; i < n; ++i) {
      if (!finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits.
      if (pos == 1) {
        float stop = lBreakeven ? std::max(lStop, entryPx) : lStop;
        if (finite(stop) && L(bars, i) <= stop) pos = 0;
        else if (finite(lTarget) && H(bars, i) >= lTarget) pos = 0;
        else if (finite(lStop) && C(bars, i) >= entryPx + bef * (entryPx - lStop))
          lBreakeven = true;
      } else if (pos == -1) {
        float stop = sBreakeven ? std::min(sStop, entryPx) : sStop;
        if (finite(stop) && H(bars, i) >= stop) pos = 0;
        else if (finite(sTarget) && L(bars, i) <= sTarget) pos = 0;
        else if (finite(sStop) && C(bars, i) <= entryPx - bef * (sStop - entryPx))
          sBreakeven = true;
      }
      // 2) Armed entries.
      if (pos != 1 && buySetup && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        lBreakeven = false;
        buySetup = false;
      } else if (pos != -1 && sellSetup && finite(sellLvl) &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        sBreakeven = false;
        sellSetup = false;
      }
      // 3) Zone bookkeeping at this close.
      if (C(bars, i - 1) > H(bars, i - 3)) {
        upperBuy = H(bars, i - 3);
        lowerBuy = std::min(L(bars, i - 1), std::min(L(bars, i - 2), L(bars, i - 3)));
        if (C(bars, i) < upperBuy && C(bars, i) > lowerBuy && C(bars, i) < C(bars, i - 1)) {
          buySetup = true;
          buyLvl = H(bars, i);
          lTarget = H(bars, i - 1) + ptf * atr[i];
          lStop = L(bars, i) - ips * atr[i];
        }
      }
      if (finite(lowerBuy) && C(bars, i) < lowerBuy) buySetup = false;
      if (C(bars, i - 1) < L(bars, i - 3)) {
        lowerSell = L(bars, i - 3);
        upperSell = std::max(H(bars, i - 1), std::max(H(bars, i - 2), H(bars, i - 3)));
        if (C(bars, i) < upperSell && C(bars, i) > lowerSell && C(bars, i) > C(bars, i - 1)) {
          sellSetup = true;
          sellLvl = L(bars, i);
          sTarget = L(bars, i - 1) - ptf * atr[i];
          sStop = H(bars, i) + ips * atr[i];
        }
      }
      if (finite(upperSell) && C(bars, i) > upperSell) sellSetup = false;
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(4, n); ++i) out[i] = 0;
  }
};

// --- exp12_jailbreak ---------------------------------------------------------
// Channel breakout with decoupled lengths: enter on a Length1-bar extreme,
// exit on the shorter of the two channel lengths or an ATR protective stop.
class JailbreakV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_jailbreak"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length1", 50, 10, 100, 5, true},
        {"length2", 30, 5, 80, 5, true},
        {"ips", 4.0, 1.0, 8.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double l2 = std::min(ps.get("length1"), ps.get("length2"));
    return {
        {Family::MAX, Column::High, {ps.get("length1")}},
        {Family::MIN, Column::Low, {ps.get("length1")}},
        {Family::MAX, Column::High, {l2}},
        {Family::MIN, Column::Low, {l2}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hi1 = ind.line(0);
    const float* lo1 = ind.line(1);
    const float* hi2 = ind.line(2);
    const float* lo2 = ind.line(3);
    const float* atr = ind.line(4);
    const int n = bars.size();
    const float ips = static_cast<float>(ps.get("ips"));

    int pos = 0;
    float entryPx = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(hi1[i - 1]) || !finite(lo1[i - 1]) || !finite(atr[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits from the previous bar's channel levels.
      if (pos == 1) {
        const float stop = std::max(entryPx - ips * atr[i - 1], lo2[i - 1]);
        if (L(bars, i) <= stop) pos = 0;
      } else if (pos == -1) {
        const float stop = std::min(entryPx + ips * atr[i - 1], hi2[i - 1]);
        if (H(bars, i) >= stop) pos = 0;
      }
      // 2) Channel breakout entries (levels from the previous bar).
      if (pos != 1 && buy_stop_hit(bars, i, hi1[i - 1])) {
        pos = 1;
        entryPx = std::max(O(bars, i), hi1[i - 1]);
      } else if (pos != -1 && sell_stop_hit(bars, i, lo1[i - 1])) {
        pos = -1;
        entryPx = std::min(O(bars, i), lo1[i - 1]);
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp12_lr_crossing -------------------------------------------------------
// Zayas' linear-regression crossover: reverse when the fast LR endpoint
// crosses the slow one. The dollar-based stop/target of the original are
// exposed as optional ATR multiples (0 disables, matching the EL defaults).
class LrCrossingV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_lr_crossing"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fastlen", 9, 3, 30, 1, true},
        {"slowlen", 18, 6, 60, 1, true},
        {"stopatrs", 0.0, 0.0, 6.0, 0.5, false},
        {"targetatrs", 0.0, 0.0, 10.0, 0.5, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("fastlen") < ps.get("slowlen");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::LINEARREG, Column::Close, {ps.get("fastlen")}},
        {Family::LINEARREG, Column::Close, {ps.get("slowlen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* fast = ind.line(0);
    const float* slow = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float stopAtrs = static_cast<float>(ps.get("stopatrs"));
    const float targetAtrs = static_cast<float>(ps.get("targetatrs"));

    int pos = 0;
    int pending = 0;
    bool hasPending = false;
    float entryPx = kNaN;

    for (int i = 1; i < n; ++i) {
      // 1) Fill pending reversals at this open.
      if (hasPending) {
        pos = pending;
        entryPx = O(bars, i);
        hasPending = false;
      }
      // 2) Optional bracket exits.
      if (pos != 0 && finite(entryPx) && finite(atr[i - 1])) {
        if (pos == 1) {
          if (stopAtrs > 0.f && L(bars, i) <= entryPx - stopAtrs * atr[i - 1]) pos = 0;
          else if (targetAtrs > 0.f && H(bars, i) >= entryPx + targetAtrs * atr[i - 1]) pos = 0;
        } else {
          if (stopAtrs > 0.f && H(bars, i) >= entryPx + stopAtrs * atr[i - 1]) pos = 0;
          else if (targetAtrs > 0.f && L(bars, i) <= entryPx - targetAtrs * atr[i - 1]) pos = 0;
        }
      }
      // 3) Crossovers at this close queue a market order for the next open.
      if (finite(fast[i]) && finite(slow[i]) && finite(fast[i - 1]) && finite(slow[i - 1])) {
        if (crosses_above(fast, slow, i) && pos != 1) {
          pending = 1;
          hasPending = true;
        } else if (crosses_below(fast, slow, i) && pos != -1) {
          pending = -1;
          hasPending = true;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp12_ntw ---------------------------------------------------------------
// "Not That Way, The Other Way": reverse divergence. Short side: price puts
// in a lower swing high while the MACD makes a higher high; a MACD cross
// below its signal then arms a stop below the setup close scaled by the
// pivot spread. Long side is the mirror image. Includes the pivot-scaled
// protective stop, breakeven shift and measured profit target.
class NtwV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_ntw"; }

  std::vector<ParamDef> params() const override {
    return {
        {"strength", 2, 1, 4, 1, true},
        {"fast", 3, 2, 8, 1, true},
        {"slow", 10, 6, 20, 1, true},
        {"signal", 16, 5, 30, 1, true},
        {"length", 20, 10, 40, 2, true},
        {"pctenter", 50.0, 10.0, 100.0, 10.0, false},
        {"pctexit", 50.0, 10.0, 150.0, 10.0, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("fast") < ps.get("slow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::MACD, Column::Close,
             {ps.get("fast"), ps.get("slow"), ps.get("signal")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* macd = ind.line(0, 0);
    const float* sig = ind.line(0, 1);
    const int n = bars.size();
    const int s = ps.get_int("strength");
    const int maxSpan = ps.get_int("length");
    const float pctEnter = static_cast<float>(ps.get("pctenter")) / 100.f;
    const float pctExit = static_cast<float>(ps.get("pctexit")) / 100.f;

    struct Pivot {
      int bar = -1;
      float price = kNaN;
      float osc = kNaN;
    };
    // Latest two same-side pivots: [0] = older (x), [1] = newer (z).
    Pivot hi[2], lo[2];

    int pos = 0;
    bool checkShort = false, checkLong = false;
    float shortSpread = kNaN, longSpread = kNaN;
    int shortWindow = 0, longWindow = 0;
    float sellLvl = kNaN, buyLvl = kNaN;
    int sellUntil = -1, buyUntil = -1;
    float sExitSpread = kNaN, lExitSpread = kNaN;
    float sTarget = kNaN, lTarget = kNaN;
    float entryPx = kNaN;
    bool sBreakeven = false, lBreakeven = false;

    for (int i = 2 * s; i < n; ++i) {
      if (!finite(macd[i]) || !finite(sig[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits.
      if (pos == -1) {
        float stop = entryPx + sExitSpread;
        if (sBreakeven) stop = std::min(stop, entryPx);
        if (H(bars, i) >= stop) pos = 0;
        else if (finite(sTarget) && L(bars, i) <= sTarget) pos = 0;
        else if (C(bars, i) < entryPx - sExitSpread) sBreakeven = true;
      } else if (pos == 1) {
        float stop = entryPx - lExitSpread;
        if (lBreakeven) stop = std::max(stop, entryPx);
        if (L(bars, i) <= stop) pos = 0;
        else if (finite(lTarget) && H(bars, i) >= lTarget) pos = 0;
        else if (C(bars, i) > entryPx + lExitSpread) lBreakeven = true;
      }
      // 2) Armed entries.
      if (pos == 0 && i <= sellUntil && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        sBreakeven = false;
        sellUntil = -1;
      } else if (pos == 0 && i <= buyUntil && finite(buyLvl) &&
                 buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        lBreakeven = false;
        buyUntil = -1;
      }
      // 3) Pivot detection (confirmed s bars late).
      const int b = i - s;
      if (b >= s) {
        bool isHigh = true, isLow = true;
        for (int k = 1; k <= s; ++k) {
          if (!(H(bars, b) > H(bars, b - k) && H(bars, b) >= H(bars, b + k))) isHigh = false;
          if (!(L(bars, b) < L(bars, b - k) && L(bars, b) <= L(bars, b + k))) isLow = false;
        }
        if (isHigh) {
          hi[0] = hi[1];
          hi[1] = {b, H(bars, b), macd[b]};
          if (hi[0].bar >= 0 && b - hi[0].bar <= maxSpan) {
            // Bearish reverse divergence: lower price high, higher MACD high.
            if (hi[0].price > hi[1].price && hi[0].osc < hi[1].osc) {
              checkShort = true;
              shortSpread = hi[0].price - hi[1].price;
              shortWindow = std::max(1, (hi[1].bar - hi[0].bar) / 2);
            }
          }
        }
        if (isLow) {
          lo[0] = lo[1];
          lo[1] = {b, L(bars, b), macd[b]};
          if (lo[0].bar >= 0 && b - lo[0].bar <= maxSpan) {
            // Bullish reverse divergence: higher price low, lower MACD low.
            if (lo[0].price < lo[1].price && lo[0].osc > lo[1].osc) {
              checkLong = true;
              longSpread = lo[1].price - lo[0].price;
              longWindow = std::max(1, (lo[1].bar - lo[0].bar) / 2);
            }
          }
        }
      }
      // 4) MACD/Signal crossings convert a divergence into an armed setup.
      if (checkShort && crosses_below(macd, sig, i)) {
        checkShort = false;
        sellLvl = C(bars, i) - shortSpread * pctEnter;
        sExitSpread = shortSpread * pctExit;
        sTarget = C(bars, i) - 2.f * shortSpread;
        sellUntil = i + shortWindow;
      }
      if (checkLong && crosses_above(macd, sig, i)) {
        checkLong = false;
        buyLvl = C(bars, i) + longSpread * pctEnter;
        lExitSpread = longSpread * pctExit;
        lTarget = C(bars, i) + 2.f * longSpread;
        buyUntil = i + longWindow;
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2 * s, n); ++i) out[i] = 0;
  }
};

// --- exp12_kinda_shifty ------------------------------------------------------
// Dual-SMA system whose exit shifts between the averages: crossings under
// the fast SMA at successively lower levels exit at the fast average, while
// higher-level crossings push the exit back to a close through the slow
// average.
class KindaShiftyV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_kinda_shifty"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fast", 30, 5, 60, 5, true},
        {"slow", 60, 20, 120, 5, true},
        {"protatrs", 3.0, 1.0, 8.0, 0.5, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("fast") < ps.get("slow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("fast")}},
        {Family::SMA, Column::Close, {ps.get("slow")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* fastA = ind.line(0);
    const float* slowA = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float prot = static_cast<float>(ps.get("protatrs"));

    int pos = 0;
    float lastCrossUnder = kNaN, thisCrossUnder = kNaN;
    float lastCrossOver = kNaN, thisCrossOver = kNaN;
    float entryPx = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(fastA[i]) || !finite(slowA[i]) || !finite(fastA[i - 1]) ||
          !finite(slowA[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits armed on the previous close.
      if (pos == 1) {
        const bool fastExit = finite(thisCrossUnder) && finite(lastCrossUnder) &&
                              thisCrossUnder < lastCrossUnder;
        if (fastExit && sell_stop_hit(bars, i, fastA[i - 1])) pos = 0;
        else if (finite(atr[i - 1]) && finite(entryPx) &&
                 L(bars, i) <= entryPx - prot * atr[i - 1]) pos = 0;
      } else if (pos == -1) {
        const bool fastExit = finite(thisCrossOver) && finite(lastCrossOver) &&
                              thisCrossOver > lastCrossOver;
        if (fastExit && buy_stop_hit(bars, i, fastA[i - 1])) pos = 0;
        else if (finite(atr[i - 1]) && finite(entryPx) &&
                 H(bars, i) >= entryPx + prot * atr[i - 1]) pos = 0;
      }
      // 2) Entries armed on the previous close (price above/below slow SMA).
      if (pos != 1 && C(bars, i - 1) > slowA[i - 1] && buy_stop_hit(bars, i, fastA[i - 1])) {
        pos = 1;
        entryPx = std::max(O(bars, i), fastA[i - 1]);
      } else if (pos != -1 && C(bars, i - 1) < slowA[i - 1] &&
                 sell_stop_hit(bars, i, fastA[i - 1])) {
        pos = -1;
        entryPx = std::min(O(bars, i), fastA[i - 1]);
      }
      // 3) Slow-average closes exit at this close; crossing bookkeeping.
      if (pos == 1 && C(bars, i - 1) >= slowA[i - 1] && C(bars, i) < slowA[i]) pos = 0;
      if (pos == -1 && C(bars, i - 1) <= slowA[i - 1] && C(bars, i) > slowA[i]) pos = 0;
      if (C(bars, i - 1) >= fastA[i - 1] && C(bars, i) < fastA[i]) {
        lastCrossUnder = thisCrossUnder;
        thisCrossUnder = C(bars, i);
      }
      if (C(bars, i - 1) <= fastA[i - 1] && C(bars, i) > fastA[i]) {
        lastCrossOver = thisCrossOver;
        thisCrossOver = C(bars, i);
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp12_springs -----------------------------------------------------------
// Wyckoff springs and upthrusts: a wide-range bar violates a swing low but
// closes back above it near the top of its range — buy a push beyond a
// fraction of that bar's range; stop under the swing, modified parabolic
// trail. Upthrusts mirror the logic for shorts.
class SpringsV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_springs"; }

  std::vector<ParamDef> params() const override {
    return {
        {"strength", 5, 2, 10, 1, true},
        {"closepct", 25.0, 10.0, 50.0, 5.0, false},
        {"rangepct", 150.0, 100.0, 250.0, 25.0, false},
        {"entrypct", 50.0, 10.0, 100.0, 10.0, false},
        {"accel", 0.02, 0.01, 0.1, 0.01, false},
        {"firstmult", 1.5, 0.5, 3.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    (void)ps;
    return {{Family::ATR, Column::Close, {10.0}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const int n = bars.size();
    const int s = ps.get_int("strength");
    const float closePct = static_cast<float>(ps.get("closepct")) / 100.f;
    const float rangePct = static_cast<float>(ps.get("rangepct")) / 100.f;
    const float entryPct = static_cast<float>(ps.get("entrypct")) / 100.f;
    const float accel = static_cast<float>(ps.get("accel"));
    const float firstMult = static_cast<float>(ps.get("firstmult"));

    float pivotLow = kNaN, pivotHigh = kNaN;
    int pivotLowAge = 0, pivotHighAge = 0;
    constexpr int kMaxAge = 20;

    int pos = 0;
    float buyOffset = kNaN, sellOffset = kNaN;  // armed entry offsets
    float lStop = kNaN, sStop = kNaN;           // swing-based protective stops
    TradeParabolic par;

    for (int i = 2 * s; i < n; ++i) {
      if (!finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits: swing stop or modified parabolic.
      if (pos == 1) {
        if (par.step_long(H(bars, i), L(bars, i)) ||
            (finite(lStop) && L(bars, i) <= lStop))
          pos = 0;
      } else if (pos == -1) {
        if (par.step_short(H(bars, i), L(bars, i)) ||
            (finite(sStop) && H(bars, i) >= sStop))
          pos = 0;
      }
      // 2) Entries armed on the previous close: open-plus-offset stop.
      if (pos == 0 && finite(buyOffset)) {
        const float lvl = O(bars, i) + buyOffset;
        if (buy_stop_hit(bars, i, lvl)) {
          pos = 1;
          par.arm_long(L(bars, i), firstMult, atr[i], accel);
        }
        buyOffset = kNaN;
      } else if (pos == 0 && finite(sellOffset)) {
        const float lvl = O(bars, i) - sellOffset;
        if (sell_stop_hit(bars, i, lvl)) {
          pos = -1;
          par.arm_short(H(bars, i), firstMult, atr[i], accel);
        }
        sellOffset = kNaN;
      }
      // 3) Swing bookkeeping (confirmed s bars late) and setup detection.
      const int b = i - s;
      if (b >= s) {
        bool isHigh = true, isLow = true;
        for (int k = 1; k <= s; ++k) {
          if (!(H(bars, b) > H(bars, b - k) && H(bars, b) >= H(bars, b + k))) isHigh = false;
          if (!(L(bars, b) < L(bars, b - k) && L(bars, b) <= L(bars, b + k))) isLow = false;
        }
        if (isHigh) {
          pivotHigh = H(bars, b);
          pivotHighAge = s;
        }
        if (isLow) {
          pivotLow = L(bars, b);
          pivotLowAge = s;
        }
      }
      ++pivotLowAge;
      ++pivotHighAge;
      if (pivotLowAge > kMaxAge) pivotLow = kNaN;
      if (pivotHighAge > kMaxAge) pivotHigh = kNaN;
      const float rng = H(bars, i) - L(bars, i);
      // Spring: dip through the swing low, close back above it near the top
      // of a wide bar.
      if (pos == 0 && finite(pivotLow) && L(bars, i) < pivotLow && C(bars, i) > pivotLow &&
          C(bars, i) >= H(bars, i) - rng * closePct && rng >= atr[i] * rangePct) {
        buyOffset = rng * entryPct;
        lStop = pivotLow;
      }
      // Upthrust: pop through the swing high, close back below it near the
      // bottom of a wide bar.
      if (pos == 0 && finite(pivotHigh) && H(bars, i) > pivotHigh && C(bars, i) < pivotHigh &&
          C(bars, i) <= L(bars, i) + rng * closePct && rng >= atr[i] * rangePct) {
        sellOffset = rng * entryPct;
        sStop = pivotHigh;
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2 * s, n); ++i) out[i] = 0;
  }
};

// --- exp12_three_musketeers --------------------------------------------------
// Slow %D above 50 plus a dip under the low of bar 3 arms a stop at the
// setup bar's high; stop under the setup bar, breakeven after an ATR move
// and a profit target at a multiple of the initial risk.
class MusketeersV12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_three_musketeers"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 12, 6, 18, 2, true},
        {"target", 6.0, 2.0, 8.0, 1.0, false},
        {"breakeven", 3.0, 1.0, 6.0, 0.5, false},
        {"level", 50.0, 30.0, 70.0, 5.0, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // STOCH: kPeriod, dPeriod -> slowK, slowD.
    return {
        {Family::STOCH, Column::Close, {ps.get("length"), 3.0}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* dline = ind.line(0, 1);
    const float* atr = ind.line(1);
    const int n = bars.size();
    const float target = static_cast<float>(ps.get("target"));
    const float brk = static_cast<float>(ps.get("breakeven"));
    const float level = static_cast<float>(ps.get("level"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    float lRisk = kNaN, sRisk = kNaN;
    float entryPx = kNaN, riskLvl = kNaN;
    bool breakeven = false;

    for (int i = 2; i < n; ++i) {
      if (!finite(dline[i]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) Bracket exits.
      if (pos == 1 && finite(riskLvl)) {
        float stop = breakeven ? std::max(riskLvl, entryPx) : riskLvl;
        if (L(bars, i) <= stop) pos = 0;
        else if (H(bars, i) >= entryPx + target * (entryPx - riskLvl)) pos = 0;
        else if (C(bars, i) >= entryPx + brk * atrPrev) breakeven = true;
      } else if (pos == -1 && finite(riskLvl)) {
        float stop = breakeven ? std::min(riskLvl, entryPx) : riskLvl;
        if (H(bars, i) >= stop) pos = 0;
        else if (L(bars, i) <= entryPx - target * (riskLvl - entryPx)) pos = 0;
        else if (C(bars, i) <= entryPx - brk * atrPrev) breakeven = true;
      }
      // 2) Armed entries.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        riskLvl = lRisk;
        breakeven = false;
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        riskLvl = sRisk;
        breakeven = false;
      }
      // 3) Setups at this close.
      buyLvl = sellLvl = kNaN;
      if (pos != 1 && dline[i] > level && L(bars, i) < L(bars, i - 2)) {
        buyLvl = H(bars, i);
        lRisk = L(bars, i);
      }
      if (pos != -1 && dline[i] < level && H(bars, i) > H(bars, i - 2)) {
        sellLvl = L(bars, i);
        sRisk = H(bars, i);
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2, n); ++i) out[i] = 0;
  }
};

// --- exp12_mist2 -------------------------------------------------------------
// Walt Downs' Tigers in the Mist, pattern #2: after an n-bar breakout bar,
// the next bar gaps beyond that bar's extreme but closes back through it;
// enter on a return to the failed bar's extreme. Protective stop at the
// signal bar's opposite extreme plus a modified parabolic trail.
class Mist2V12 final : public IStrategy {
public:
  std::string name() const override { return "exp12_mist2"; }

  std::vector<ParamDef> params() const override {
    return {
        {"mistlen", 3, 1, 5, 1, true},
        {"accel", 0.03, 0.01, 0.05, 0.01, false},
        {"firstmult", 1.0, 0.5, 2.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("mistlen")}},
        {Family::MIN, Column::Low, {ps.get("mistlen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiC = ind.line(0);
    const float* loC = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float accel = static_cast<float>(ps.get("accel"));
    const float firstMult = static_cast<float>(ps.get("firstmult"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    float lStop = kNaN, sStop = kNaN;
    TradeParabolic par;

    for (int i = 2; i < n; ++i) {
      if (!finite(atr[i]) || !finite(hiC[i - 2]) || !finite(loC[i - 2])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits: signal-bar stop or modified parabolic.
      if (pos == 1) {
        if (par.step_long(H(bars, i), L(bars, i)) ||
            (finite(lStop) && L(bars, i) <= lStop))
          pos = 0;
      } else if (pos == -1) {
        if (par.step_short(H(bars, i), L(bars, i)) ||
            (finite(sStop) && H(bars, i) >= sStop))
          pos = 0;
      }
      // 2) Armed entries from the pattern completed on the previous bar.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        par.arm_long(L(bars, i), firstMult, atr[i], accel);
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        par.arm_short(H(bars, i), firstMult, atr[i], accel);
      }
      // 3) Pattern detection at this close. Bar i-1: n-bar breakout with a
      // confirming close; bar i: gap beyond bar i-1's extreme, close back
      // through it. (The EL's short setup compares Low > Lowest which we
      // read as the documented Low < Lowest breakout.)
      buyLvl = sellLvl = kNaN;
      const bool upBreak =
          H(bars, i - 1) > hiC[i - 2] && C(bars, i - 1) > C(bars, i - 2);
      const bool dnBreak =
          L(bars, i - 1) < loC[i - 2] && C(bars, i - 1) < C(bars, i - 2);
      if (pos != 1 && upBreak && O(bars, i) > H(bars, i - 1) && C(bars, i) < H(bars, i - 1)) {
        buyLvl = H(bars, i);
        lStop = L(bars, i);
      }
      if (pos != -1 && dnBreak && O(bars, i) < L(bars, i - 1) && C(bars, i) > L(bars, i - 1)) {
        sellLvl = L(bars, i);
        sStop = H(bars, i);
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2, n); ++i) out[i] = 0;
  }
};

} // namespace

void register_vol12(std::vector<Factory>& out) {
  out.push_back({"exp12_first_prize",
                 [] { return std::unique_ptr<IStrategy>(new FirstPrizeV12); }});
  out.push_back({"exp12_in_the_zone",
                 [] { return std::unique_ptr<IStrategy>(new InTheZoneV12); }});
  out.push_back({"exp12_jailbreak",
                 [] { return std::unique_ptr<IStrategy>(new JailbreakV12); }});
  out.push_back({"exp12_lr_crossing",
                 [] { return std::unique_ptr<IStrategy>(new LrCrossingV12); }});
  out.push_back({"exp12_ntw", [] { return std::unique_ptr<IStrategy>(new NtwV12); }});
  out.push_back({"exp12_kinda_shifty",
                 [] { return std::unique_ptr<IStrategy>(new KindaShiftyV12); }});
  out.push_back({"exp12_springs",
                 [] { return std::unique_ptr<IStrategy>(new SpringsV12); }});
  out.push_back({"exp12_three_musketeers",
                 [] { return std::unique_ptr<IStrategy>(new MusketeersV12); }});
  out.push_back({"exp12_mist2", [] { return std::unique_ptr<IStrategy>(new Mist2V12); }});
}

} // namespace topt::exp
