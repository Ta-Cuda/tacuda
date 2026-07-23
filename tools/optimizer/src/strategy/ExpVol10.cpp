/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 10 (Omega Research STAD Club, Vol. 10).
// Systems: JK Monday We Trade, Serendipity, Eight Ball, Going In Style,
// Who's Minding the Store, Reference Deviation, Superman.
// Skipped from this volume: Gemini (needs a weekly Data2 series) and the
// RadarScreen/Serendipity-trigger concept chapters (no standalone system).
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

// --- exp10_monday_trade ------------------------------------------------------
// Joe Krutsinger's Monday We Trade: on Friday's close, buy Monday's open if
// the close beats its 2-bar SMA, sell short otherwise; flat again a bar
// later. The dollar money-management stop and the exit-on-close collapse
// into a one-bar hold under the next-open execution model.
class MondayTradeV10 final : public IStrategy {
public:
  std::string name() const override { return "exp10_monday_trade"; }

  std::vector<ParamDef> params() const override {
    return {{"length", 2, 2, 10, 1, true}};
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::SMA, Column::Close, {ps.get("length")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* sma = ind.line(0);
    const int n = bars.size();
    for (int i = 0; i < n; ++i) {
      out[i] = 0;
      if (!finite(sma[i])) continue;
      if (day_of_week(bars.ts[static_cast<std::size_t>(i)]) == 5) {
        out[i] = C(bars, i) > sma[i] ? 1 : -1;  // held Monday open -> Tuesday open
      }
    }
  }
};

// --- exp10_serendipity -------------------------------------------------------
// LeBeau's Serendipity trigger: with RSI overbought, buy on a stop a
// fraction of ATR above the SETUP bar's open (not the entry bar's), so the
// setup day's own move already counts toward the trigger. Bracket exits:
// protective ATR stop, breakeven stop, ATR profit target.
class SerendipityV10 final : public IStrategy {
public:
  std::string name() const override { return "exp10_serendipity"; }

  std::vector<ParamDef> params() const override {
    return {
        {"rsilen", 9, 3, 25, 1, true},
        {"overbought", 70, 55, 90, 5, true},
        {"oversold", 30, 10, 45, 5, true},
        {"trigatr", 0.5, 0.1, 1.5, 0.1, false},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"breakatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"targetatrs", 4.0, 1.0, 8.0, 0.5, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("oversold") < ps.get_int("overbought");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::RSI, Column::Close, {ps.get("rsilen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* rsi = ind.line(0);
    const float* atr = ind.line(1);
    const int n = bars.size();
    const float overbought = static_cast<float>(ps.get_int("overbought"));
    const float oversold = static_cast<float>(ps.get_int("oversold"));
    const float trig = static_cast<float>(ps.get("trigatr"));
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float target = static_cast<float>(ps.get("targetatrs"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    float entryPx = kNaN, closeHigh = kNaN, closeLow = kNaN, atrEntry = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(rsi[i]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Bracket exits from levels known at the previous close.
      if (pos == 1 && finite(entryPx)) {
        float stop = entryPx - prot * atrEntry;
        if (closeHigh > entryPx + brk * atrEntry) stop = std::max(stop, entryPx);
        if (L(bars, i) <= stop) pos = 0;
        else if (H(bars, i) >= entryPx + target * atrEntry) pos = 0;
        closeHigh = std::max(closeHigh, C(bars, i));
      } else if (pos == -1 && finite(entryPx)) {
        float stop = entryPx + prot * atrEntry;
        if (closeLow < entryPx - brk * atrEntry) stop = std::min(stop, entryPx);
        if (H(bars, i) >= stop) pos = 0;
        else if (L(bars, i) <= entryPx - target * atrEntry) pos = 0;
        closeLow = std::min(closeLow, C(bars, i));
      }
      // 2) Serendipity stop entries armed at the previous close.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        closeHigh = C(bars, i);
        atrEntry = atr[i];
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        closeLow = C(bars, i);
        atrEntry = atr[i];
      }
      // 3) RSI setups: the trigger hangs off THIS bar's open (Serendipity).
      buyLvl = (rsi[i] > overbought && pos != 1) ? O(bars, i) + trig * atr[i] : kNaN;
      sellLvl = (rsi[i] < oversold && pos != -1) ? O(bars, i) - trig * atr[i] : kNaN;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp10_eight_ball --------------------------------------------------------
// Slow EMA rising while the fast EMA dips (a pullback within the trend)
// arms an ATR-trigger breakout entry; four ATR stops manage the trade
// (protective, trailing, volatility, big-profit n-bar trail).
class EightBallV10 final : public IStrategy {
public:
  std::string name() const override { return "exp10_eight_ball"; }

  std::vector<ParamDef> params() const override {
    return {
        {"slowlen", 40, 10, 100, 5, true},
        {"fastlen", 8, 3, 30, 1, true},
        {"trigger", 1.0, 0.25, 2.5, 0.25, false},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"volatrs", 2.0, 1.0, 6.0, 0.5, false},
        {"bigatrs", 7.0, 3.0, 12.0, 1.0, false},
        {"exitbars", 3, 2, 8, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fastlen") < ps.get_int("slowlen");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("slowlen")}},
        {Family::EMA, Column::Close, {ps.get("fastlen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* slow = ind.line(0);
    const float* fast = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float trig = static_cast<float>(ps.get("trigger"));
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));
    const float vol = static_cast<float>(ps.get("volatrs"));
    const float big = static_cast<float>(ps.get("bigatrs"));
    const int exitBars = ps.get_int("exitbars");

    int pos = 0;
    float entryPx = kNaN, posHigh = kNaN, posLow = kNaN;
    bool buyArmed = false, sellArmed = false;

    for (int i = 1; i < n; ++i) {
      if (!finite(slow[i]) || !finite(fast[i]) || !finite(slow[i - 1]) ||
          !finite(fast[i - 1]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) Four-stop exits.
      if (pos == 1) {
        posHigh = std::max(posHigh, H(bars, i));
        float stop = entryPx - prot * atrPrev;
        stop = std::max(stop, posHigh - trail * atrPrev);
        stop = std::max(stop, C(bars, i - 1) - vol * atrPrev);
        if (C(bars, i - 1) >= entryPx + big * atrPrev) {
          float lo = L(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            lo = std::min(lo, L(bars, i - 1 - k));
          stop = std::max(stop, lo);
        }
        if (L(bars, i) <= stop) pos = 0;
      } else if (pos == -1) {
        posLow = std::min(posLow, L(bars, i));
        float stop = entryPx + prot * atrPrev;
        stop = std::min(stop, posLow + trail * atrPrev);
        stop = std::min(stop, C(bars, i - 1) + vol * atrPrev);
        if (C(bars, i - 1) <= entryPx - big * atrPrev) {
          float hi = H(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            hi = std::max(hi, H(bars, i - 1 - k));
          stop = std::min(stop, hi);
        }
        if (H(bars, i) >= stop) pos = 0;
      }
      // 2) Armed trigger entries (this bar's open + trigger*ATR known a bar ago).
      if (pos != 1 && buyArmed) {
        const float lvl = O(bars, i) + trig * atrPrev;
        if (H(bars, i) >= lvl && C(bars, i) >= lvl) {
          pos = 1;
          entryPx = lvl;
          posHigh = H(bars, i);
        }
      } else if (pos != -1 && sellArmed) {
        const float lvl = O(bars, i) - trig * atrPrev;
        if (L(bars, i) <= lvl && C(bars, i) <= lvl) {
          pos = -1;
          entryPx = lvl;
          posLow = L(bars, i);
        }
      }
      // 3) Pullback-within-trend setups at this close (as coded in the EL).
      buyArmed = slow[i] > slow[i - 1] && fast[i] < fast[i - 1] && pos != 1;
      sellArmed = slow[i] < slow[i - 1] && fast[i] > fast[i - 1] && pos != -1;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp10_go_in_style -------------------------------------------------------
// Channel breakout with style: a fresh n-bar extreme arms a stop at the
// close +/- trigger*ATR for the next bar; a modified parabolic (seeded off
// the entry bar by an ATR(3) multiple) trails the trade.
class GoInStyleV10 final : public IStrategy {
public:
  std::string name() const override { return "exp10_go_in_style"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 20, 5, 50, 5, true},
        {"trigger", 0.66, 0.2, 1.5, 0.1, false},
        {"accel", 0.02, 0.01, 0.1, 0.01, false},
        {"firstmult", 1.5, 0.5, 4.0, 0.25, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::ATR, Column::Close, {ps.get("length")}},
        {Family::MAX, Column::High, {ps.get("length")}},
        {Family::MIN, Column::Low, {ps.get("length")}},
        {Family::ATR, Column::Close, {3.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const float* hiCh = ind.line(1);
    const float* loCh = ind.line(2);
    const float* atr3 = ind.line(3);
    const int n = bars.size();
    const float trig = static_cast<float>(ps.get("trigger"));
    const float accel = static_cast<float>(ps.get("accel"));
    const float firstMult = static_cast<float>(ps.get("firstmult"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;  // valid one bar
    float sarStop = kNaN, af = 0.f, extreme = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(atr[i]) || !finite(hiCh[i - 1]) || !finite(loCh[i - 1]) ||
          !finite(atr3[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Parabolic trailing exits.
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
      // 2) One-bar armed trigger entries.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        sarStop = L(bars, i) - atr3[i] * firstMult;
        af = accel;
        extreme = H(bars, i);
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sarStop = H(bars, i) + atr3[i] * firstMult;
        af = accel;
        extreme = L(bars, i);
      }
      buyLvl = kNaN;
      sellLvl = kNaN;
      // 3) Fresh-extreme setups at this close.
      if (H(bars, i) > hiCh[i - 1] && pos != 1) buyLvl = C(bars, i) + trig * atr[i];
      if (L(bars, i) < loCh[i - 1] && pos != -1) sellLvl = C(bars, i) - trig * atr[i];
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp10_store -------------------------------------------------------------
// Who's Minding the Store: a bullish Key Reversal (new n-bar low, up close)
// with slow stochastic oversold arms a 4-bar-high breakout for a few bars;
// initial stop at the n-bar extreme, then trailing and big-profit ATR stops.
class StoreV10 final : public IStrategy {
public:
  std::string name() const override { return "exp10_store"; }

  std::vector<ParamDef> params() const override {
    return {
        {"oversold", 40, 10, 45, 5, true},
        {"overbought", 60, 55, 90, 5, true},
        {"newlow", 8, 4, 20, 1, true},
        {"setuplen", 2, 1, 6, 1, true},
        {"entrybar_atrs", 1.0, 0.5, 3.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
        {"bigatrs", 7.0, 3.0, 12.0, 1.0, false},
        {"exitbars", 3, 2, 8, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("oversold") < ps.get_int("overbought");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::STOCH, Column::Close, {14.0, 3.0}},
        {Family::MAX, Column::High, {ps.get("newlow")}},
        {Family::MIN, Column::Low, {ps.get("newlow")}},
        {Family::ATR, Column::Close, {10.0}},
        {Family::MAX, Column::High, {4.0}},
        {Family::MIN, Column::Low, {4.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* stochK = ind.line(0, 0);
    const float* stochD = ind.line(0, 1);
    const float* hiN = ind.line(1);
    const float* loN = ind.line(2);
    const float* atr = ind.line(3);
    const float* hi4 = ind.line(4);
    const float* lo4 = ind.line(5);
    const int n = bars.size();
    const float oversold = static_cast<float>(ps.get_int("oversold"));
    const float overbought = static_cast<float>(ps.get_int("overbought"));
    const int setupLen = ps.get_int("setuplen");
    const float entryAtrs = static_cast<float>(ps.get("entrybar_atrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));
    const float big = static_cast<float>(ps.get("bigatrs"));
    const int exitBars = ps.get_int("exitbars");

    int pos = 0;
    int buyUntil = -1, sellUntil = -1;
    float entryPx = kNaN, posHigh = kNaN, posLow = kNaN, initStop = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(stochK[i]) || !finite(stochD[i]) || !finite(hiN[i - 1]) ||
          !finite(loN[i - 1]) || !finite(atr[i]) || !finite(hi4[i - 1]) ||
          !finite(lo4[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) Exits: initial n-bar extreme + entry-bar ATR floor, then trails.
      if (pos == 1) {
        posHigh = std::max(posHigh, H(bars, i));
        float stop = std::max(initStop, entryPx - entryAtrs * atrPrev);
        stop = std::max(stop, posHigh - trail * atrPrev);
        if (C(bars, i - 1) >= entryPx + big * atrPrev) {
          float lo = L(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            lo = std::min(lo, L(bars, i - 1 - k));
          stop = std::max(stop, lo);
        }
        if (L(bars, i) <= stop) pos = 0;
      } else if (pos == -1) {
        posLow = std::min(posLow, L(bars, i));
        float stop = std::min(initStop, entryPx + entryAtrs * atrPrev);
        stop = std::min(stop, posLow + trail * atrPrev);
        if (C(bars, i - 1) <= entryPx - big * atrPrev) {
          float hi = H(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            hi = std::max(hi, H(bars, i - 1 - k));
          stop = std::min(stop, hi);
        }
        if (H(bars, i) >= stop) pos = 0;
      }
      // 2) Armed 4-bar breakout entries.
      if (pos != 1 && i <= buyUntil && buy_stop_hit(bars, i, hi4[i - 1])) {
        pos = 1;
        entryPx = std::max(O(bars, i), hi4[i - 1]);
        posHigh = H(bars, i);
        initStop = loN[i - 1];
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && sell_stop_hit(bars, i, lo4[i - 1])) {
        pos = -1;
        entryPx = std::min(O(bars, i), lo4[i - 1]);
        posLow = L(bars, i);
        initStop = hiN[i - 1];
        sellUntil = -1;
      }
      // 3) Key-Reversal setups at this close.
      const bool keyRevUp = L(bars, i) < loN[i - 1] && C(bars, i) > C(bars, i - 1);
      const bool keyRevDn = H(bars, i) > hiN[i - 1] && C(bars, i) < C(bars, i - 1);
      if (keyRevUp && stochK[i] < stochD[i] && stochD[i] < oversold && pos != 1)
        buyUntil = i + setupLen;
      if (keyRevDn && stochK[i] > stochD[i] && stochD[i] > overbought && pos != -1)
        sellUntil = i + setupLen;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp10_ref_dev -----------------------------------------------------------
// Brower's Reference Deviation: RDV = 100 * sum(C-SMA) / sum(|C-SMA|) over
// the same window. Above +threshold buy, below -threshold sell short; the
// zero cross exits. Nearly always in the market.
class RefDevV10 final : public IStrategy {
public:
  std::string name() const override { return "exp10_ref_dev"; }

  std::vector<ParamDef> params() const override {
    return {
        {"rmalen", 15, 5, 55, 5, true},
        {"threshold", 5.0, 0.0, 40.0, 2.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // Reference Deviation as one native request: drd = close - SMA(rmalen),
    // RDV = 100 * sum(drd) / sum(|drd|) over the same window.
    return {{Family::REFDEV, Column::Close,
             {ps.get("rmalen"), ps.get("rmalen")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* rdvLine = ind.line(0);
    const int n = bars.size();
    const float et = static_cast<float>(ps.get("threshold"));
    (void)bars;

    int pos = 0;
    for (int i = 0; i < n; ++i) {
      if (!finite(rdvLine[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float rdv = rdvLine[i];
      if (pos == 1 && rdv < 0.f) pos = 0;
      else if (pos == -1 && rdv > 0.f) pos = 0;
      if (rdv > et) pos = 1;
      else if (rdv < -et) pos = -1;
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp10_superman ----------------------------------------------------------
// MarketStrength (net n-bar close change scaled by same-signed movement)
// pinned at +/-95 with decelerating older momentum arms an n-bar extreme
// breakout; trail at the m-bar extreme with a risk-multiple profit target.
class SupermanV10 final : public IStrategy {
public:
  std::string name() const override { return "exp10_superman"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 5, 3, 12, 1, true},
        {"stoplen", 6, 3, 15, 1, true},
        {"profitfactor", 1.0, 0.5, 4.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("length")}},
        {Family::MIN, Column::Low, {ps.get("length")}},
        {Family::MAX, Column::High, {ps.get("stoplen")}},
        {Family::MIN, Column::Low, {ps.get("stoplen")}},
        {Family::MARKET_STRENGTH, Column::Close, {ps.get("length")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiL = ind.line(0);
    const float* loL = ind.line(1);
    const float* hiS = ind.line(2);
    const float* loS = ind.line(3);
    const float* msLine = ind.line(4);
    const int n = bars.size();
    const float pf = static_cast<float>(ps.get("profitfactor"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    float stopLoss = kNaN, entryPx = kNaN;

    for (int i = 9; i < n; ++i) {
      if (!finite(hiL[i]) || !finite(loL[i]) || !finite(hiS[i]) || !finite(loS[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Stop-loss trail and profit objective from the previous bar.
      if (pos == 1 && finite(stopLoss)) {
        if (L(bars, i) <= stopLoss) pos = 0;
        else if (finite(entryPx) && H(bars, i) >= entryPx + (entryPx - stopLoss) * pf)
          pos = 0;
      } else if (pos == -1 && finite(stopLoss)) {
        if (H(bars, i) >= stopLoss) pos = 0;
        else if (finite(entryPx) && L(bars, i) <= entryPx - (stopLoss - entryPx) * pf)
          pos = 0;
      }
      // 2) Armed breakout entries.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        stopLoss = loS[i];
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        stopLoss = hiS[i];
      }
      buyLvl = kNaN;
      sellLvl = kNaN;
      // 3) Strength/speed setups at this close.
      const float ms = msLine[i];
      const float dpb1 = (C(bars, i) - C(bars, i - 4)) / 4.f;
      const float dpb2 = (C(bars, i - 4) - C(bars, i - 8)) / 5.f;
      if (finite(ms)) {
        if (ms >= 95.f && dpb1 >= 0.f && dpb2 < 0.f && H(bars, i) >= hiL[i] && pos != 1)
          buyLvl = hiL[i];
        if (ms <= -95.f && dpb1 < 0.f && dpb2 >= 0.f && L(bars, i) <= loL[i] && pos != -1)
          sellLvl = loL[i];
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(9, n); ++i) out[i] = 0;
  }
};

} // namespace

void register_vol10(std::vector<Factory>& out) {
  out.push_back({"exp10_monday_trade",
                 [] { return std::unique_ptr<IStrategy>(new MondayTradeV10); }});
  out.push_back({"exp10_serendipity",
                 [] { return std::unique_ptr<IStrategy>(new SerendipityV10); }});
  out.push_back({"exp10_eight_ball",
                 [] { return std::unique_ptr<IStrategy>(new EightBallV10); }});
  out.push_back({"exp10_go_in_style",
                 [] { return std::unique_ptr<IStrategy>(new GoInStyleV10); }});
  out.push_back({"exp10_store",
                 [] { return std::unique_ptr<IStrategy>(new StoreV10); }});
  out.push_back({"exp10_ref_dev",
                 [] { return std::unique_ptr<IStrategy>(new RefDevV10); }});
  out.push_back({"exp10_superman",
                 [] { return std::unique_ptr<IStrategy>(new SupermanV10); }});
}

} // namespace topt::exp
