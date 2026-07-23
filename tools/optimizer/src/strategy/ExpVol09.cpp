/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 9 (Omega Research STAD Club, Vol. 9).
// Systems: Double Your Fun, Luxor, No Hurry, OBV Revisited, Red Rover,
// Skinny Dipper, Swinger, Traffic Jam.
// Skipped from this volume: International Index Composite (needs seven
// world-index data series) and the RadarScreen indicator chapter (not a
// trading system). See docs/EXPERIMENTS.md for the adaptation notes.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Appendix-A common exits: protective stop off the entry, breakeven stop
// armed by the closing extreme, trailing stop off the position extreme and
// a profit-target limit frozen at entry. Any multiplier <= 0 disables the
// corresponding exit.
struct AtrExits {
  float entryPx = kNaN;
  float posHigh = kNaN, posLow = kNaN;      // intrabar extremes
  float closeHigh = kNaN, closeLow = kNaN;  // closing extremes (breakeven)
  float targetDist = kNaN;                  // frozen at entry

  void on_entry(float fillPx, const Series& bars, int i, float atrEntry,
                float targetAtrs) {
    entryPx = fillPx;
    posHigh = H(bars, i);
    posLow = L(bars, i);
    closeHigh = closeLow = C(bars, i);
    targetDist = targetAtrs > 0.f && finite(atrEntry) ? targetAtrs * atrEntry : kNaN;
  }

  // Levels use atrPrev (the ATR known when the orders were placed).
  bool step_long(const Series& bars, int i, float atrPrev, float prot,
                 float breakeven, float trail) {
    float stop = -std::numeric_limits<float>::infinity();
    if (prot > 0.f) stop = std::max(stop, entryPx - prot * atrPrev);
    if (trail > 0.f) stop = std::max(stop, posHigh - trail * atrPrev);
    if (breakeven > 0.f && closeHigh > entryPx + breakeven * atrPrev)
      stop = std::max(stop, entryPx);
    bool out = std::isfinite(stop) && L(bars, i) <= stop;
    if (!out && finite(targetDist) && H(bars, i) >= entryPx + targetDist) out = true;
    posHigh = std::max(posHigh, H(bars, i));
    closeHigh = std::max(closeHigh, C(bars, i));
    return out;
  }

  bool step_short(const Series& bars, int i, float atrPrev, float prot,
                  float breakeven, float trail) {
    float stop = std::numeric_limits<float>::infinity();
    if (prot > 0.f) stop = std::min(stop, entryPx + prot * atrPrev);
    if (trail > 0.f) stop = std::min(stop, posLow + trail * atrPrev);
    if (breakeven > 0.f && closeLow < entryPx - breakeven * atrPrev)
      stop = std::min(stop, entryPx);
    bool out = std::isfinite(stop) && H(bars, i) >= stop;
    if (!out && finite(targetDist) && L(bars, i) <= entryPx - targetDist) out = true;
    posLow = std::min(posLow, L(bars, i));
    closeLow = std::min(closeLow, C(bars, i));
    return out;
  }
};

// --- exp09_double_fun --------------------------------------------------------
// DiNapoli-style double penetration of a displaced moving average: cross
// above, quick cross back, second cross above completes the setup; entry one
// tick above the second-cross bar within four bars. Exit trails the n-bar
// extreme.
class DoubleFunV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_double_fun"; }

  std::vector<ParamDef> params() const override {
    return {
        {"avglen", 5, 3, 20, 1, true},
        {"disp", 5, 1, 15, 1, true},
        {"entrysetup", 4, 2, 10, 1, true},
        {"setupq1", 3, 1, 10, 1, true},
        {"setupq2", 5, 2, 12, 1, true},
        {"traillen", 5, 2, 12, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::SMA, Column::Close, {ps.get("avglen")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* sma = ind.line(0);
    const int n = bars.size();
    const int disp = ps.get_int("disp");
    const int entrySetup = ps.get_int("entrysetup");
    const int q1 = ps.get_int("setupq1");
    const int q2 = ps.get_int("setupq2");
    const int trailLen = ps.get_int("traillen");

    auto dma = [&](int i) { return i - disp >= 0 ? sma[i - disp] : kNaN; };

    int pos = 0;
    int lastUpBar = -1, lastDnBar = -1;
    float buyLvl = kNaN, sellLvl = kNaN;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 1; i < n; ++i) {
      const float d = dma(i), dPrev = dma(i - 1);
      if (!finite(d) || !finite(dPrev)) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) n-bar trailing exits from the previous bar's levels.
      if (pos == 1) {
        float lo = L(bars, i - 1);
        for (int k = 1; k < trailLen && i - 1 - k >= 0; ++k)
          lo = std::min(lo, L(bars, i - 1 - k));
        if (L(bars, i) <= lo) pos = 0;
      } else if (pos == -1) {
        float hi = H(bars, i - 1);
        for (int k = 1; k < trailLen && i - 1 - k >= 0; ++k)
          hi = std::max(hi, H(bars, i - 1 - k));
        if (H(bars, i) >= hi) pos = 0;
      }
      // 2) Armed double-cross entries.
      if (pos != 1 && i <= buyUntil && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && finite(sellLvl) &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sellUntil = -1;
      }
      // 3) Cross bookkeeping and setups at this close.
      const bool xUp = C(bars, i) > d && C(bars, i - 1) <= dPrev;
      const bool xDn = C(bars, i) < d && C(bars, i - 1) >= dPrev;
      if (xUp) {
        if (lastDnBar >= 0 && i - lastDnBar <= q2 && lastUpBar >= 0 &&
            lastDnBar - lastUpBar >= 0 && lastDnBar - lastUpBar <= q1 && pos != 1) {
          buyLvl = H(bars, i);
          buyUntil = i + entrySetup;
        }
        lastUpBar = i;
      }
      if (xDn) {
        if (lastUpBar >= 0 && i - lastUpBar <= q2 && lastDnBar >= 0 &&
            lastUpBar - lastDnBar >= 0 && lastUpBar - lastDnBar <= q1 && pos != -1) {
          sellLvl = L(bars, i);
          sellUntil = i + entrySetup;
        }
        lastDnBar = i;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp09_luxor -------------------------------------------------------------
// Triangular MA crossover: while the fast TMA is above the slow one, a stop
// order rides one tick above each bar's high; the slow TMA itself is the
// trailing exit. The original pyramids up to three units on ADX-confirmed
// pullbacks; this adaptation trades one unit.
class LuxorV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_luxor"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fastlen", 5, 2, 20, 1, true},
        {"slowlen", 20, 5, 60, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fastlen") < ps.get_int("slowlen");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // The original's ADX filter only gated the pyramiding add-ons, which the
    // single-unit adaptation drops - so no ADX request here.
    return {
        {Family::TRIMA, Column::Close, {ps.get("fastlen")}},
        {Family::TRIMA, Column::Close, {ps.get("slowlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* fast = ind.line(0);
    const float* slow = ind.line(1);
    const int n = bars.size();

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(fast[i]) || !finite(slow[i]) || !finite(slow[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Slow-TMA trailing exits (level from the previous bar).
      if (pos == 1 && L(bars, i) <= slow[i - 1]) pos = 0;
      else if (pos == -1 && H(bars, i) >= slow[i - 1]) pos = 0;
      // 2) Armed entry stops from the previous bar.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) pos = 1;
      else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) pos = -1;
      // 3) Re-arm while the crossover setup holds.
      buyLvl = (fast[i] > slow[i] && pos != 1) ? H(bars, i) : kNaN;
      sellLvl = (fast[i] < slow[i] && pos != -1) ? L(bars, i) : kNaN;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp09_no_hurry ----------------------------------------------------------
// Channel breakout delayed by n bars: entries alternate long/short (no
// re-entry on the same side), managed by protective, breakeven and trailing
// ATR stops.
class NoHurryV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_no_hurry"; }

  std::vector<ParamDef> params() const override {
    return {
        {"chanlen", 20, 5, 50, 5, true},
        {"chandelay", 15, 5, 30, 5, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"breakatrs", 2.0, 1.0, 5.0, 0.5, false},
        {"trailatrs", 5.0, 1.0, 8.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("chanlen")}},
        {Family::MIN, Column::Low, {ps.get("chanlen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiC = ind.line(0);
    const float* loC = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const int delay = ps.get_int("chandelay");
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));

    int pos = 0, lastDir = 0;
    AtrExits ex;

    for (int i = 1; i < n; ++i) {
      const int j = i - 1 - delay;
      if (j < 0 || !finite(hiC[j]) || !finite(loC[j]) || !finite(atr[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) ATR exits.
      if (pos == 1 && ex.step_long(bars, i, atr[i - 1], prot, brk, trail)) pos = 0;
      else if (pos == -1 && ex.step_short(bars, i, atr[i - 1], prot, brk, trail)) pos = 0;
      // 2) Alternating delayed-channel entries.
      if (pos != 1 && lastDir != 1 && buy_stop_hit(bars, i, hiC[j])) {
        pos = 1;
        lastDir = 1;
        ex.on_entry(std::max(O(bars, i), hiC[j]), bars, i, atr[i], 0.f);
      } else if (pos != -1 && lastDir != -1 && sell_stop_hit(bars, i, loC[j])) {
        pos = -1;
        lastDir = -1;
        ex.on_entry(std::min(O(bars, i), loC[j]), bars, i, atr[i], 0.f);
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp09_obv_revisited -----------------------------------------------------
// Weighted OBV (volume scaled by close-to-open position within the range)
// crossing its SMA arms a one-tick breakout of the crossover bar; the
// opposite cross exits at the close. Protective, breakeven and trailing ATR
// stops manage the trade.
class ObvRevisitedV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_obv_revisited"; }

  std::vector<ParamDef> params() const override {
    return {
        {"avglen", 25, 5, 60, 1, true},
        {"protatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"breakatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // Weighted OBV and its rolling mean as native (batchable) requests.
    IndicatorRequest wobvReq{Family::WOBV, Column::Close, {}};
    return {
        {Family::ATR, Column::Close, {20.0}},
        wobvReq,
        derived(Family::ROLLING_MEAN, {ps.get("avglen")}, wobvReq),
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const int n = bars.size();
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));

    const float* w = ind.line(1);
    const float* s = ind.line(2);

    int pos = 0;
    bool buySetup = false, sellSetup = false;
    float lePrice = kNaN, sePrice = kNaN;
    AtrExits ex;

    for (int i = 1; i < n; ++i) {
      if (!finite(s[i]) || !finite(s[i - 1]) || !finite(atr[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) ATR exits.
      if (pos == 1 && ex.step_long(bars, i, atr[i - 1], prot, brk, trail)) pos = 0;
      else if (pos == -1 && ex.step_short(bars, i, atr[i - 1], prot, brk, trail)) pos = 0;
      // 2) Armed breakout entries.
      if (pos != 1 && buySetup && finite(lePrice) && buy_stop_hit(bars, i, lePrice)) {
        pos = 1;
        buySetup = false;
        ex.on_entry(std::max(O(bars, i), lePrice), bars, i, atr[i], 0.f);
      } else if (pos != -1 && sellSetup && finite(sePrice) &&
                 sell_stop_hit(bars, i, sePrice)) {
        pos = -1;
        sellSetup = false;
        ex.on_entry(std::min(O(bars, i), sePrice), bars, i, atr[i], 0.f);
      }
      // 3) WOBV/SMA crosses at this close.
      const bool xUp = w[i] > s[i] && w[i - 1] <= s[i - 1];
      const bool xDn = w[i] < s[i] && w[i - 1] >= s[i - 1];
      if (xUp) {
        buySetup = true;
        lePrice = H(bars, i);
        sellSetup = false;
        if (pos == -1) pos = 0;  // exit short on this close
      }
      if (xDn) {
        sellSetup = true;
        sePrice = L(bars, i);
        buySetup = false;
        if (pos == 1) pos = 0;  // exit long on this close
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp09_red_rover ---------------------------------------------------------
// Pivot-style support/resistance from the weighted close (2*WC-Low resists,
// 2*WC-High supports): stop-and-reverse breakouts with an ATR profit target
// and the opposite line as the protective stop.
class RedRoverV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_red_rover"; }

  std::vector<ParamDef> params() const override {
    return {
        {"targetatrs", 2.0, 0.5, 5.0, 0.5, false},
        {"atrlen", 10, 3, 30, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::ATR, Column::Close, {ps.get("atrlen")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const int n = bars.size();
    const float targetAtrs = static_cast<float>(ps.get("targetatrs"));

    int pos = 0;
    float entryPx = kNaN, targetDist = kNaN;

    for (int i = 1; i < n; ++i) {
      // Levels computed from the previous bar.
      const float wc = (H(bars, i - 1) + L(bars, i - 1) + 2.f * C(bars, i - 1)) / 4.f;
      const float resistance = 2.f * wc - L(bars, i - 1);
      const float support = 2.f * wc - H(bars, i - 1);
      // 1) Profit target and protective line for the open position.
      if (pos == 1) {
        if (finite(targetDist) && H(bars, i) >= entryPx + targetDist) pos = 0;
        else if (L(bars, i) <= support) pos = 0;
      } else if (pos == -1) {
        if (finite(targetDist) && L(bars, i) <= entryPx - targetDist) pos = 0;
        else if (H(bars, i) >= resistance) pos = 0;
      }
      // 2) Stop-and-reverse breakout entries.
      if (pos != 1 && buy_stop_hit(bars, i, resistance)) {
        pos = 1;
        entryPx = std::max(O(bars, i), resistance);
        targetDist = finite(atr[i]) ? targetAtrs * atr[i] : kNaN;
      } else if (pos != -1 && sell_stop_hit(bars, i, support)) {
        pos = -1;
        entryPx = std::min(O(bars, i), support);
        targetDist = finite(atr[i]) ? targetAtrs * atr[i] : kNaN;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp09_skinny_dipper -----------------------------------------------------
// Naked close against a double-EMA trend arms a one-bar-only stop entry
// beyond the naked-close bar; protective and breakeven ATR stops plus an
// ATR profit target manage the trade.
class SkinnyDipperV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_skinny_dipper"; }

  std::vector<ParamDef> params() const override {
    return {
        {"longavg", 50, 20, 100, 5, true},
        {"shortavg", 15, 5, 40, 1, true},
        {"protatrs", 2.0, 1.0, 5.0, 0.5, false},
        {"breakatrs", 2.0, 1.0, 5.0, 0.5, false},
        {"targetatrs", 4.0, 1.0, 10.0, 0.5, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("shortavg") < ps.get_int("longavg");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("longavg")}},
        {Family::EMA, Column::Close, {ps.get("shortavg")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* emaL = ind.line(0);
    const float* emaS = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float target = static_cast<float>(ps.get("targetatrs"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;  // valid one bar only
    AtrExits ex;

    for (int i = 1; i < n; ++i) {
      if (!finite(emaL[i]) || !finite(emaS[i]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) ATR exits.
      if (pos == 1 && ex.step_long(bars, i, atr[i - 1], prot, brk, 0.f)) pos = 0;
      else if (pos == -1 && ex.step_short(bars, i, atr[i - 1], prot, brk, 0.f)) pos = 0;
      // 2) One-bar entry stops from the naked-close setup.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        ex.on_entry(std::max(O(bars, i), buyLvl), bars, i, atr[i], target);
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        ex.on_entry(std::min(O(bars, i), sellLvl), bars, i, atr[i], target);
      }
      buyLvl = kNaN;
      sellLvl = kNaN;
      // 3) Naked-close setups at this close.
      if (C(bars, i) > emaL[i] && C(bars, i) > emaS[i] && C(bars, i) < L(bars, i - 1) &&
          pos != 1) {
        buyLvl = H(bars, i);
      }
      if (C(bars, i) < emaL[i] && C(bars, i) < emaS[i] && C(bars, i) > H(bars, i - 1) &&
          pos != -1) {
        sellLvl = L(bars, i);
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp09_swinger -----------------------------------------------------------
// Price-oscillator momentum in the direction of an SMA trend filter: rising
// oscillator below zero buys the next open in an uptrend (mirror for
// shorts). When momentum fades, an n-bar extreme trailing stop activates; a
// protective ATR stop is always live.
class SwingerV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_swinger"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fastlen", 5, 2, 15, 1, true},
        {"slowlen", 20, 5, 50, 1, true},
        {"avglen", 50, 20, 100, 5, true},
        {"nbars", 3, 2, 8, 1, true},
        {"protatrs", 4.0, 1.0, 8.0, 0.5, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fastlen") < ps.get_int("slowlen");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("fastlen")}},
        {Family::SMA, Column::Close, {ps.get("slowlen")}},
        {Family::SMA, Column::Close, {ps.get("avglen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* fast = ind.line(0);
    const float* slow = ind.line(1);
    const float* filter = ind.line(2);
    const float* atr = ind.line(3);
    const int n = bars.size();
    const int nBars = ps.get_int("nbars");
    const float prot = static_cast<float>(ps.get("protatrs"));

    int pos = 0;
    bool pendingLong = false, pendingShort = false;
    float entryPx = kNaN;

    auto osc = [&](int i) {
      return finite(fast[i]) && finite(slow[i]) ? fast[i] - slow[i] : kNaN;
    };

    for (int i = 1; i < n; ++i) {
      const float o = osc(i), oPrev = osc(i - 1);
      // 1) Fill pending market entries at this open.
      if (pendingLong && pos != 1) {
        pos = 1;
        entryPx = O(bars, i);
      } else if (pendingShort && pos != -1) {
        pos = -1;
        entryPx = O(bars, i);
      } else if (pos != 0 && finite(oPrev) && i >= 2) {
        // 2) Exits from the previous close's decisions.
        const float oPrev2 = osc(i - 2);
        if (pos == 1) {
          bool exitHit = finite(atr[i - 1]) && finite(entryPx) &&
                         L(bars, i) <= entryPx - prot * atr[i - 1];
          if (!exitHit && finite(oPrev2) && oPrev < oPrev2) {
            float lo = L(bars, i - 1);
            for (int k = 1; k < nBars && i - 1 - k >= 0; ++k)
              lo = std::min(lo, L(bars, i - 1 - k));
            exitHit = L(bars, i) <= lo;
          }
          if (exitHit) pos = 0;
        } else if (pos == -1) {
          bool exitHit = finite(atr[i - 1]) && finite(entryPx) &&
                         H(bars, i) >= entryPx + prot * atr[i - 1];
          if (!exitHit && finite(oPrev2) && oPrev > oPrev2) {
            float hi = H(bars, i - 1);
            for (int k = 1; k < nBars && i - 1 - k >= 0; ++k)
              hi = std::max(hi, H(bars, i - 1 - k));
            exitHit = H(bars, i) >= hi;
          }
          if (exitHit) pos = 0;
        }
      }
      pendingLong = pendingShort = false;
      // 3) Momentum setups at this close.
      if (finite(o) && finite(oPrev) && finite(filter[i])) {
        if (o > oPrev && o < 0.f && C(bars, i) > filter[i] && pos != 1) pendingLong = true;
        if (o < oPrev && o > 0.f && C(bars, i) < filter[i] && pos != -1) pendingShort = true;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp09_traffic_jam -------------------------------------------------------
// Congestion fade: with ADX below its level and falling, three consecutive
// down closes buy this close (up closes sell short); exit on the close a
// fixed number of bars later or at a protective ATR stop.
class TrafficJamV9 final : public IStrategy {
public:
  std::string name() const override { return "exp09_traffic_jam"; }

  std::vector<ParamDef> params() const override {
    return {
        {"adxlen", 14, 5, 30, 1, true},
        {"adxlevel", 25, 10, 40, 5, true},
        {"consec", 3, 2, 6, 1, true},
        {"exitbars", 3, 1, 10, 1, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::ADX, Column::Close, {ps.get("adxlen")}},
        {Family::ATR, Column::Close, {5.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* adx = ind.line(0);
    const float* atr = ind.line(1);
    const int n = bars.size();
    const float level = static_cast<float>(ps.get_int("adxlevel"));
    const int consec = ps.get_int("consec");
    const int exitBars = ps.get_int("exitbars");
    const float prot = static_cast<float>(ps.get("protatrs"));

    int pos = 0, entryBar = -1;
    float entryPx = kNaN;

    for (int i = consec; i < n; ++i) {
      if (!finite(adx[i]) || i - consec < 0 || !finite(adx[i - consec]) ||
          !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits: protective ATR stop, then the fixed-bar close-out.
      if (pos != 0) {
        if (pos == 1 && finite(entryPx) && finite(atr[i - 1]) &&
            L(bars, i) <= entryPx - prot * atr[i - 1]) {
          pos = 0;
        } else if (pos == -1 && finite(entryPx) && finite(atr[i - 1]) &&
                   H(bars, i) >= entryPx + prot * atr[i - 1]) {
          pos = 0;
        } else if (i - entryBar >= exitBars) {
          pos = 0;
        }
      }
      // 2) Congestion entries decided at this close (filled next open).
      if (pos == 0) {
        const bool congestion = adx[i] < level && adx[i] < adx[i - consec];
        if (congestion) {
          bool allDown = true, allUp = true;
          for (int k = 0; k < consec; ++k) {
            if (!(C(bars, i - k) < C(bars, i - k - 1))) allDown = false;
            if (!(C(bars, i - k) > C(bars, i - k - 1))) allUp = false;
          }
          if (allDown) {
            pos = 1;
            entryBar = i;
            entryPx = C(bars, i);
          } else if (allUp) {
            pos = -1;
            entryBar = i;
            entryPx = C(bars, i);
          }
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(consec, n); ++i) out[i] = 0;
  }
};

} // namespace

void register_vol09(std::vector<Factory>& out) {
  out.push_back({"exp09_double_fun",
                 [] { return std::unique_ptr<IStrategy>(new DoubleFunV9); }});
  out.push_back({"exp09_luxor",
                 [] { return std::unique_ptr<IStrategy>(new LuxorV9); }});
  out.push_back({"exp09_no_hurry",
                 [] { return std::unique_ptr<IStrategy>(new NoHurryV9); }});
  out.push_back({"exp09_obv_revisited",
                 [] { return std::unique_ptr<IStrategy>(new ObvRevisitedV9); }});
  out.push_back({"exp09_red_rover",
                 [] { return std::unique_ptr<IStrategy>(new RedRoverV9); }});
  out.push_back({"exp09_skinny_dipper",
                 [] { return std::unique_ptr<IStrategy>(new SkinnyDipperV9); }});
  out.push_back({"exp09_swinger",
                 [] { return std::unique_ptr<IStrategy>(new SwingerV9); }});
  out.push_back({"exp09_traffic_jam",
                 [] { return std::unique_ptr<IStrategy>(new TrafficJamV9); }});
}

} // namespace topt::exp
