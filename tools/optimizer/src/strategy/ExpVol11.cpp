/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 11 (Omega Research STAD Club, Vol. 11).
// Systems: High Five, Hit the Bull's Eye, JK Night Train, Symmetrical
// Triangle, TrendScore, Triple Play, Win One for the Gapper.
// Skipped from this volume: MISER (needs a bonds Data2 series); the
// "Two Favorite Exit Strategies" and performance-report chapters describe
// no standalone system. See docs/EXPERIMENTS.md for the adaptation notes.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Buy-the-dip limit fill with close confirmation: the level was touched from
// above and the close settled back at or above it (mirror for sells).
inline bool buy_limit_hit(const Series& bars, int i, float level) {
  return L(bars, i) <= level && C(bars, i) >= level;
}
inline bool sell_limit_hit(const Series& bars, int i, float level) {
  return H(bars, i) >= level && C(bars, i) <= level;
}

// --- exp11_high_five ---------------------------------------------------------
// Five stacked SMAs (base length doubling each step) all rising arm a stop
// at the close plus a fraction of the bar's range; the position exits when
// three or more averages turn against it, plus protective/breakeven ATR
// stops.
class HighFiveV11 final : public IStrategy {
public:
  std::string name() const override { return "exp11_high_five"; }

  std::vector<ParamDef> params() const override {
    return {
        {"baselen", 5, 2, 12, 1, true},
        {"multiplier", 0.2, 0.1, 1.0, 0.1, false},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"breakatrs", 4.0, 1.0, 8.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double base = ps.get("baselen");
    return {
        {Family::SMA, Column::Close, {base}},
        {Family::SMA, Column::Close, {base * 2}},
        {Family::SMA, Column::Close, {base * 4}},
        {Family::SMA, Column::Close, {base * 8}},
        {Family::SMA, Column::Close, {base * 16}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const int n = bars.size();
    const float mult = static_cast<float>(ps.get("multiplier"));
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float* atr = ind.line(5);

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    float entryPx = kNaN, closeHigh = kNaN, closeLow = kNaN;

    for (int i = 1; i < n; ++i) {
      int up = 0, dn = 0;
      bool ready = finite(atr[i]);
      for (int a = 0; a < 5 && ready; ++a) {
        const float* line = ind.line(static_cast<std::size_t>(a));
        if (!finite(line[i]) || !finite(line[i - 1])) {
          ready = false;
          break;
        }
        if (line[i] > line[i - 1]) ++up;
        if (line[i] < line[i - 1]) ++dn;
      }
      if (!ready) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) ATR stops from the previous bar's levels.
      if (pos == 1 && finite(entryPx)) {
        float stop = entryPx - prot * atrPrev;
        if (closeHigh > entryPx + brk * atrPrev) stop = std::max(stop, entryPx);
        if (L(bars, i) <= stop) pos = 0;
        closeHigh = std::max(closeHigh, C(bars, i));
      } else if (pos == -1 && finite(entryPx)) {
        float stop = entryPx + prot * atrPrev;
        if (closeLow < entryPx - brk * atrPrev) stop = std::min(stop, entryPx);
        if (H(bars, i) >= stop) pos = 0;
        closeLow = std::min(closeLow, C(bars, i));
      }
      // 2) Armed range-fraction stop entries.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        closeHigh = C(bars, i);
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        closeLow = C(bars, i);
      }
      // 3) Majority-turn exits decided at this close.
      if (pos == 1 && dn >= 3) pos = 0;
      if (pos == -1 && up >= 3) pos = 0;
      // 4) All-five setups at this close.
      const float rng = H(bars, i) - L(bars, i);
      buyLvl = (up == 5 && pos != 1) ? C(bars, i) + rng * mult : kNaN;
      sellLvl = (dn == 5 && pos != -1) ? C(bars, i) - rng * mult : kNaN;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp11_bullseye ----------------------------------------------------------
// Buy the dip in an uptrend: a fresh n-bar high closing above the high two
// bars back with a rising EMA arms a limit buy at the midpoint of that
// earlier bar; profit target a multiple of the setup-bar excursion, plus a
// protective ATR stop.
class BullseyeV11 final : public IStrategy {
public:
  std::string name() const override { return "exp11_bullseye"; }

  std::vector<ParamDef> params() const override {
    return {
        {"rangelen", 5, 3, 20, 1, true},
        {"xavglen", 50, 20, 100, 5, true},
        {"barstoenter", 5, 2, 12, 1, true},
        {"factor", 3.0, 1.0, 6.0, 0.5, false},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("rangelen")}},
        {Family::MIN, Column::Low, {ps.get("rangelen")}},
        {Family::EMA, Column::Close, {ps.get("xavglen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiC = ind.line(0);
    const float* loC = ind.line(1);
    const float* ema = ind.line(2);
    const float* atr = ind.line(3);
    const int n = bars.size();
    const int barsToEnter = ps.get_int("barstoenter");
    const float factor = static_cast<float>(ps.get("factor"));
    const float prot = static_cast<float>(ps.get("protatrs"));

    int pos = 0;
    float buyEntry = kNaN, sellEntry = kNaN;
    float longTarget = kNaN, shortTarget = kNaN;
    int buyUntil = -1, sellUntil = -1;
    float entryPx = kNaN;

    for (int i = 2; i < n; ++i) {
      if (!finite(hiC[i]) || !finite(loC[i]) || !finite(ema[i]) || !finite(ema[i - 1]) ||
          !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) Exits: profit-target limit or protective ATR stop.
      if (pos == 1) {
        if (finite(longTarget) && H(bars, i) >= longTarget) pos = 0;
        else if (finite(entryPx) && L(bars, i) <= entryPx - prot * atrPrev) pos = 0;
      } else if (pos == -1) {
        if (finite(shortTarget) && L(bars, i) <= shortTarget) pos = 0;
        else if (finite(entryPx) && H(bars, i) >= entryPx + prot * atrPrev) pos = 0;
      }
      // 2) Armed midpoint limit entries.
      if (pos != 1 && i <= buyUntil && finite(buyEntry) &&
          buy_limit_hit(bars, i, buyEntry)) {
        pos = 1;
        entryPx = std::min(O(bars, i), buyEntry);
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && finite(sellEntry) &&
                 sell_limit_hit(bars, i, sellEntry)) {
        pos = -1;
        entryPx = std::max(O(bars, i), sellEntry);
        sellUntil = -1;
      }
      // 3) Fresh-extreme setups at this close.
      if (pos != 1 && H(bars, i) >= hiC[i] && C(bars, i) > H(bars, i - 2) &&
          ema[i] > ema[i - 1]) {
        buyEntry = (H(bars, i - 2) + L(bars, i - 2)) / 2.f;
        longTarget = H(bars, i) + factor * (H(bars, i) - buyEntry);
        buyUntil = i + barsToEnter;
      }
      if (pos != -1 && L(bars, i) <= loC[i] && C(bars, i) < L(bars, i - 2) &&
          ema[i] < ema[i - 1]) {
        sellEntry = (H(bars, i - 2) + L(bars, i - 2)) / 2.f;
        shortTarget = L(bars, i) - factor * (sellEntry - L(bars, i));
        sellUntil = i + barsToEnter;
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2, n); ++i) out[i] = 0;
  }
};

// --- exp11_night_train -------------------------------------------------------
// Krutsinger's Night Train: a close under the close two bars back arms a
// stop at the bar's high; exit at the first profitable open (Williams'
// Bailout), with a protective ATR stop.
class NightTrainV11 final : public IStrategy {
public:
  std::string name() const override { return "exp11_night_train"; }

  std::vector<ParamDef> params() const override {
    return {{"protatrs", 3.0, 1.0, 6.0, 0.5, false}};
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    (void)ps;
    return {{Family::ATR, Column::Close, {10.0}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const int n = bars.size();
    const float prot = static_cast<float>(ps.get("protatrs"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN, entryPx = kNaN;

    for (int i = 2; i < n; ++i) {
      // 1) Bailout: first profitable open exits at that open.
      if (pos == 1 && finite(entryPx) && O(bars, i) > entryPx) pos = 0;
      else if (pos == -1 && finite(entryPx) && O(bars, i) < entryPx) pos = 0;
      // 2) Protective ATR stop.
      if (pos == 1 && finite(atr[i - 1]) && L(bars, i) <= entryPx - prot * atr[i - 1])
        pos = 0;
      else if (pos == -1 && finite(atr[i - 1]) && H(bars, i) >= entryPx + prot * atr[i - 1])
        pos = 0;
      // 3) Armed entry stops from the previous close.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
      }
      // 4) Two-bar momentum setups at this close.
      buyLvl = (C(bars, i) < C(bars, i - 2) && pos != 1) ? H(bars, i) : kNaN;
      sellLvl = (C(bars, i) >= C(bars, i - 2) && pos != -1) ? L(bars, i) : kNaN;
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2, n); ++i) out[i] = 0;
  }
};

// --- exp11_sym_triangle ------------------------------------------------------
// Four alternating strength-1 swings squeezing into a symmetrical triangle
// during a DMI-confirmed trend; entry on the breakout through the resistance
// (support) trendline, stop past the last pivot, breakeven at a risk
// multiple and a measured-move profit target.
class SymTriangleV11 final : public IStrategy {
public:
  std::string name() const override { return "exp11_sym_triangle"; }

  std::vector<ParamDef> params() const override {
    return {
        {"evalrange", 40, 20, 80, 5, true},
        {"dmilen", 14, 5, 30, 1, true},
        {"strength", 1, 1, 3, 1, true},
        {"exitfactor", 0.5, 0.25, 2.0, 0.25, false},
        {"multiplier", 2.0, 1.0, 4.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::PLUS_DI, Column::Close, {ps.get("dmilen")}},
        {Family::MINUS_DI, Column::Close, {ps.get("dmilen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* diPlus = ind.line(0);
    const float* diMinus = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const int evalRange = ps.get_int("evalrange");
    const int s = ps.get_int("strength");
    const float exitFactor = static_cast<float>(ps.get("exitfactor"));
    const float mult = static_cast<float>(ps.get("multiplier"));

    // Swing pivots detected s bars late.
    struct Pivot {
      int bar = -1;
      float px = kNaN;
      float dmi = kNaN;
    };
    std::vector<Pivot> highs, lows;

    int pos = 0;
    float entryPx = kNaN, protStop = kNaN, target = kNaN;
    bool breakevenArmed = false;

    for (int i = 2 * s; i < n; ++i) {
      if (!finite(diPlus[i]) || !finite(diMinus[i]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits.
      if (pos == 1 && finite(protStop)) {
        float stop = protStop;
        if (breakevenArmed) stop = std::max(stop, entryPx);
        if (L(bars, i) <= stop) pos = 0;
        else if (finite(target) && H(bars, i) >= target) pos = 0;
        else if (C(bars, i) >= entryPx + mult * (entryPx - protStop))
          breakevenArmed = true;
      } else if (pos == -1 && finite(protStop)) {
        float stop = protStop;
        if (breakevenArmed) stop = std::min(stop, entryPx);
        if (H(bars, i) >= stop) pos = 0;
        else if (finite(target) && L(bars, i) <= target) pos = 0;
        else if (C(bars, i) <= entryPx - mult * (protStop - entryPx))
          breakevenArmed = true;
      }
      // 2) Register swings confirmed at this close.
      const int b = i - s;
      if (b >= s) {
        bool isHigh = true, isLow = true;
        for (int k = 1; k <= s; ++k) {
          if (!(H(bars, b) > H(bars, b - k) && H(bars, b) >= H(bars, b + k))) isHigh = false;
          if (!(L(bars, b) < L(bars, b - k) && L(bars, b) <= L(bars, b + k))) isLow = false;
        }
        if (isHigh)
          highs.push_back({b, H(bars, b), diPlus[b] - diMinus[b]});
        if (isLow)
          lows.push_back({b, L(bars, b), diPlus[b] - diMinus[b]});
        while (!highs.empty() && highs.front().bar < i - evalRange) highs.erase(highs.begin());
        while (!lows.empty() && lows.front().bar < i - evalRange) lows.erase(lows.begin());
      }
      // 3) Triangle detection and breakout at this close.
      if (pos == 0 && highs.size() >= 2 && lows.size() >= 2) {
        const Pivot& hA = highs[highs.size() - 2];  // older swing high (point A)
        const Pivot& hC = highs.back();             // newer swing high (point C)
        const Pivot& lB = lows[lows.size() - 2];    // older swing low (point B)
        const Pivot& lD = lows.back();              // newer swing low  (point D)
        // Bullish: descending highs, ascending lows, alternating A<B<C<D.
        if (hA.px > hC.px && lB.px < lD.px && hA.bar < lB.bar && lB.bar < hC.bar &&
            hC.bar < lD.bar && hA.dmi >= 15.f) {
          const float slopeR = (hC.px - hA.px) / static_cast<float>(hC.bar - hA.bar);
          const float resistance = hC.px + slopeR * static_cast<float>(i - hC.bar);
          const float slopeS = (lD.px - lB.px) / static_cast<float>(lD.bar - lB.bar);
          const float support = lB.px + slopeS * static_cast<float>(i - lB.bar);
          if (resistance > support && buy_stop_hit(bars, i, resistance)) {
            pos = 1;
            entryPx = std::max(O(bars, i), resistance);
            protStop = lD.px - atr[i] * exitFactor;
            target = entryPx + (hA.px - lB.px);
            breakevenArmed = false;
          }
        }
        // Bearish: ascending lows->highs mirrored, alternating A<B<C<D.
        const Pivot& lA = lows[lows.size() - 2];
        const Pivot& lC = lows.back();
        const Pivot& hB = highs[highs.size() - 2];
        const Pivot& hD = highs.back();
        if (pos == 0 && lA.px < lC.px && hB.px > hD.px && lA.bar < hB.bar &&
            hB.bar < lC.bar && lC.bar < hD.bar && lA.dmi <= -15.f) {
          const float slopeS = (lC.px - lA.px) / static_cast<float>(lC.bar - lA.bar);
          const float support = lC.px + slopeS * static_cast<float>(i - lC.bar);
          const float slopeR = (hD.px - hB.px) / static_cast<float>(hD.bar - hB.bar);
          const float resistance = hB.px + slopeR * static_cast<float>(i - hB.bar);
          if (resistance > support && sell_stop_hit(bars, i, support)) {
            pos = -1;
            entryPx = std::min(O(bars, i), support);
            protStop = hD.px + atr[i] * exitFactor;
            target = entryPx - (hB.px - lA.px);
            breakevenArmed = false;
          }
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2 * s, n); ++i) out[i] = 0;
  }
};

// --- exp11_trendscore --------------------------------------------------------
// Chande's TrendScore: +/-1 for each comparison of the close against closes
// lookback+1..lookback+10 bars back. Above its SMA with the close above its
// own SMA buys the next open; either falling back exits. Protective,
// breakeven and trailing ATR stops manage the risk.
class TrendScoreV11 final : public IStrategy {
public:
  std::string name() const override { return "exp11_trendscore"; }

  std::vector<ParamDef> params() const override {
    return {
        {"lookback", 10, 5, 40, 5, true},
        {"length", 18, 5, 50, 1, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"breakatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // TrendScore and its rolling mean chain as native (batchable) requests.
    IndicatorRequest scoreReq{Family::TRENDSCORE, Column::Close,
                              {ps.get("lookback")}};
    return {
        {Family::SMA, Column::Close, {ps.get("length")}},
        {Family::ATR, Column::Close, {10.0}},
        scoreReq,
        derived(Family::ROLLING_MEAN, {ps.get("length")}, scoreReq),
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* sma = ind.line(0);
    const float* atr = ind.line(1);
    const float* score = ind.line(2);
    const float* tsa = ind.line(3);
    const int n = bars.size();
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));

    int pos = 0;
    bool pendingLong = false, pendingShort = false, pendingExit = false;
    float entryPx = kNaN, posHigh = kNaN, posLow = kNaN;
    float closeHigh = kNaN, closeLow = kNaN;

    for (int i = 1; i < n; ++i) {
      // 1) Fill pending market orders at this open.
      if (pendingLong && pos != 1) {
        pos = 1;
        entryPx = O(bars, i);
        posHigh = H(bars, i);
        closeHigh = C(bars, i);
      } else if (pendingShort && pos != -1) {
        pos = -1;
        entryPx = O(bars, i);
        posLow = L(bars, i);
        closeLow = C(bars, i);
      } else if (pendingExit && pos != 0) {
        pos = 0;
      } else if (pos != 0 && finite(atr[i - 1])) {
        // 2) ATR stops from the previous bar's levels.
        if (pos == 1) {
          float stop = entryPx - prot * atr[i - 1];
          stop = std::max(stop, posHigh - trail * atr[i - 1]);
          if (closeHigh > entryPx + brk * atr[i - 1]) stop = std::max(stop, entryPx);
          if (L(bars, i) <= stop) pos = 0;
          posHigh = std::max(posHigh, H(bars, i));
          closeHigh = std::max(closeHigh, C(bars, i));
        } else {
          float stop = entryPx + prot * atr[i - 1];
          stop = std::min(stop, posLow + trail * atr[i - 1]);
          if (closeLow < entryPx - brk * atr[i - 1]) stop = std::min(stop, entryPx);
          if (H(bars, i) >= stop) pos = 0;
          posLow = std::min(posLow, L(bars, i));
          closeLow = std::min(closeLow, C(bars, i));
        }
      }
      pendingLong = pendingShort = pendingExit = false;
      // 3) TrendScore setups at this close.
      if (finite(score[i]) && finite(tsa[i]) && finite(sma[i])) {
        const bool up = score[i] > tsa[i] && C(bars, i) > sma[i];
        const bool dn = score[i] < tsa[i] && C(bars, i) < sma[i];
        if (up && pos != 1) pendingLong = true;
        else if (dn && pos != -1) pendingShort = true;
        else if (pos == 1 && !up) pendingExit = true;
        else if (pos == -1 && !dn) pendingExit = true;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp11_triple_play -------------------------------------------------------
// Lindsay's Trident with Bollinger confirmation: swing low (outside band),
// swing high (outside band), retracement swing low (inside bands, 38-62%)
// arm a stop entry a quarter-swing above point 3; stop past point 3,
// breakeven after target 1, trailing quarter-drop stop after price reaches
// point 2, and a measured target 2. Single unit (the original trades two).
class TriplePlayV11 final : public IStrategy {
public:
  std::string name() const override { return "exp11_triple_play"; }

  std::vector<ParamDef> params() const override {
    return {
        {"patternlen", 25, 10, 60, 5, true},
        {"strength", 7, 2, 12, 1, true},
        {"barstoenter", 10, 3, 20, 1, true},
        {"bblen", 11, 5, 30, 1, true},
        {"bbdev", 2.0, 1.0, 3.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // BBANDS params: period, devUp, devDn -> upper, middle, lower.
    return {{Family::BBANDS, Column::Close,
             {ps.get("bblen"), ps.get("bbdev"), ps.get("bbdev")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* upper = ind.line(0, 0);
    const float* lower = ind.line(0, 2);
    const int n = bars.size();
    const int patternLen = ps.get_int("patternlen");
    const int s = ps.get_int("strength");
    const int barsToEnter = ps.get_int("barstoenter");

    struct Swing {
      int bar = -1;
      float px = kNaN;
      bool outside = false;  // beyond the Bollinger band at its bar
    };
    Swing lastHigh, lastLow, prevLow, prevHigh;

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    float initStop = kNaN, target1 = kNaN, target2 = kNaN, trailTrig = kNaN, trailStep = kNaN;
    int buyUntil = -1, sellUntil = -1;
    float entryPx = kNaN, closeExt = kNaN;
    bool trailOn = false, protectAtEntry = false;

    for (int i = 2 * s; i < n; ++i) {
      // 1) Exits.
      if (pos == 1 && finite(initStop)) {
        float stop = protectAtEntry ? std::max(initStop, entryPx) : initStop;
        if (trailOn && finite(closeExt)) stop = std::max(stop, closeExt - trailStep);
        if (L(bars, i) <= stop) pos = 0;
        else if (finite(target2) && H(bars, i) >= target2) pos = 0;
        else {
          closeExt = finite(closeExt) ? std::max(closeExt, C(bars, i)) : C(bars, i);
          if (finite(trailTrig) && closeExt > trailTrig) trailOn = true;
          if (finite(target1) && C(bars, i) >= target1) protectAtEntry = true;
        }
      } else if (pos == -1 && finite(initStop)) {
        float stop = protectAtEntry ? std::min(initStop, entryPx) : initStop;
        if (trailOn && finite(closeExt)) stop = std::min(stop, closeExt + trailStep);
        if (H(bars, i) >= stop) pos = 0;
        else if (finite(target2) && L(bars, i) <= target2) pos = 0;
        else {
          closeExt = finite(closeExt) ? std::min(closeExt, C(bars, i)) : C(bars, i);
          if (finite(trailTrig) && closeExt < trailTrig) trailOn = true;
          if (finite(target1) && C(bars, i) <= target1) protectAtEntry = true;
        }
      }
      // 2) Armed stop entries.
      if (pos == 0 && i <= buyUntil && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        closeExt = C(bars, i);
        trailOn = false;
        protectAtEntry = false;
        buyUntil = -1;
      } else if (pos == 0 && i <= sellUntil && finite(sellLvl) &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        closeExt = C(bars, i);
        trailOn = false;
        protectAtEntry = false;
        sellUntil = -1;
      }
      // 3) Swing bookkeeping (confirmed s bars late).
      const int b = i - s;
      if (b >= s) {
        bool isHigh = true, isLow = true;
        for (int k = 1; k <= s; ++k) {
          if (!(H(bars, b) > H(bars, b - k) && H(bars, b) >= H(bars, b + k))) isHigh = false;
          if (!(L(bars, b) < L(bars, b - k) && L(bars, b) <= L(bars, b + k))) isLow = false;
        }
        if (isHigh) {
          prevHigh = lastHigh;
          lastHigh = {b, H(bars, b), finite(upper[b]) && H(bars, b) > upper[b]};
        }
        if (isLow) {
          prevLow = lastLow;
          lastLow = {b, L(bars, b), finite(lower[b]) && L(bars, b) < lower[b]};
        }
        // Bullish Trident: point1 = prevLow (outside), point2 = lastHigh
        // (outside, after point1), point3 = lastLow (inside, after point2).
        if (isLow && prevLow.bar >= 0 && lastHigh.bar > prevLow.bar &&
            lastLow.bar > lastHigh.bar && prevLow.outside && lastHigh.outside &&
            !lastLow.outside && lastHigh.bar - prevLow.bar <= patternLen &&
            lastLow.bar - prevLow.bar <= patternLen && pos == 0) {
          const float swing = lastHigh.px - prevLow.px;
          if (swing > 0.f) {
            const float retr = (lastHigh.px - lastLow.px) / swing;
            if (retr >= 0.382f && retr <= 0.618f) {
              buyLvl = lastLow.px + swing / 4.f;
              initStop = lastLow.px;
              target2 = lastLow.px + swing;
              target1 = (lastLow.px + target2) / 2.f;
              trailTrig = lastHigh.px;
              trailStep = (lastHigh.px - lastLow.px) / 4.f;
              buyUntil = i + barsToEnter;
            }
          }
        }
        // Bearish Trident (mirror).
        if (isHigh && prevHigh.bar >= 0 && lastLow.bar > prevHigh.bar &&
            lastHigh.bar > lastLow.bar && prevHigh.outside && lastLow.outside &&
            !lastHigh.outside && lastLow.bar - prevHigh.bar <= patternLen &&
            lastHigh.bar - prevHigh.bar <= patternLen && pos == 0) {
          const float swing = prevHigh.px - lastLow.px;
          if (swing > 0.f) {
            const float retr = (lastHigh.px - lastLow.px) / swing;
            if (retr >= 0.382f && retr <= 0.618f) {
              sellLvl = lastHigh.px - swing / 4.f;
              initStop = lastHigh.px;
              target2 = lastHigh.px - swing;
              target1 = (lastHigh.px + target2) / 2.f;
              trailTrig = lastLow.px;
              trailStep = (lastHigh.px - lastLow.px) / 4.f;
              sellUntil = i + barsToEnter;
            }
          }
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(2 * s, n); ++i) out[i] = 0;
  }
};

// --- exp11_win_gapper --------------------------------------------------------
// Williams' "Oops" gap fade in the trend direction: with ADX above 25 and
// rising and DMI+ over DMI-, a gap-down open recovering to the previous
// bar's low buys (mirror for shorts); the four common ATR stops manage the
// trade.
class WinGapperV11 final : public IStrategy {
public:
  std::string name() const override { return "exp11_win_gapper"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 14, 5, 30, 1, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"breakatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"bigatrs", 4.0, 2.0, 14.0, 1.0, false},
        {"exitbars", 3, 1, 8, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::ADX, Column::Close, {ps.get("length")}},
        {Family::PLUS_DI, Column::Close, {ps.get("length")}},
        {Family::MINUS_DI, Column::Close, {ps.get("length")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* adx = ind.line(0);
    const float* diP = ind.line(1);
    const float* diM = ind.line(2);
    const float* atr = ind.line(3);
    const int n = bars.size();
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));
    const float big = static_cast<float>(ps.get("bigatrs"));
    const int exitBars = ps.get_int("exitbars");

    int pos = 0;
    float entryPx = kNaN, posHigh = kNaN, posLow = kNaN, closeHigh = kNaN, closeLow = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(adx[i]) || !finite(adx[i - 1]) || !finite(diP[i]) || !finite(diM[i]) ||
          !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) Common ATR exits.
      if (pos == 1) {
        posHigh = std::max(posHigh, H(bars, i));
        float stop = entryPx - prot * atrPrev;
        stop = std::max(stop, posHigh - trail * atrPrev);
        if (closeHigh > entryPx + brk * atrPrev) stop = std::max(stop, entryPx);
        if (C(bars, i - 1) >= entryPx + big * atrPrev) {
          float lo = L(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            lo = std::min(lo, L(bars, i - 1 - k));
          stop = std::max(stop, lo);
        }
        if (L(bars, i) <= stop) pos = 0;
        closeHigh = std::max(closeHigh, C(bars, i));
      } else if (pos == -1) {
        posLow = std::min(posLow, L(bars, i));
        float stop = entryPx + prot * atrPrev;
        stop = std::min(stop, posLow + trail * atrPrev);
        if (closeLow < entryPx - brk * atrPrev) stop = std::min(stop, entryPx);
        if (C(bars, i - 1) <= entryPx - big * atrPrev) {
          float hi = H(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            hi = std::max(hi, H(bars, i - 1 - k));
          stop = std::min(stop, hi);
        }
        if (H(bars, i) >= stop) pos = 0;
        closeLow = std::min(closeLow, C(bars, i));
      }
      // 2) Oops entries: the gap and recovery are both visible at this
      // close; the fill happens at the next open per the engine.
      const bool trendFilter = adx[i - 1] > 25.f && adx[i - 1] > adx[i - 2 >= 0 ? i - 2 : 0];
      if (trendFilter && pos != 1 && diP[i - 1] > diM[i - 1] &&
          O(bars, i) < L(bars, i - 1) && buy_stop_hit(bars, i, L(bars, i - 1))) {
        pos = 1;
        entryPx = L(bars, i - 1);
        posHigh = H(bars, i);
        closeHigh = C(bars, i);
      } else if (trendFilter && pos != -1 && diP[i - 1] < diM[i - 1] &&
                 O(bars, i) > H(bars, i - 1) && sell_stop_hit(bars, i, H(bars, i - 1))) {
        pos = -1;
        entryPx = H(bars, i - 1);
        posLow = L(bars, i);
        closeLow = C(bars, i);
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

} // namespace

void register_vol11(std::vector<Factory>& out) {
  out.push_back({"exp11_high_five",
                 [] { return std::unique_ptr<IStrategy>(new HighFiveV11); }});
  out.push_back({"exp11_bullseye",
                 [] { return std::unique_ptr<IStrategy>(new BullseyeV11); }});
  out.push_back({"exp11_night_train",
                 [] { return std::unique_ptr<IStrategy>(new NightTrainV11); }});
  out.push_back({"exp11_sym_triangle",
                 [] { return std::unique_ptr<IStrategy>(new SymTriangleV11); }});
  out.push_back({"exp11_trendscore",
                 [] { return std::unique_ptr<IStrategy>(new TrendScoreV11); }});
  out.push_back({"exp11_triple_play",
                 [] { return std::unique_ptr<IStrategy>(new TriplePlayV11); }});
  out.push_back({"exp11_win_gapper",
                 [] { return std::unique_ptr<IStrategy>(new WinGapperV11); }});
}

} // namespace topt::exp
