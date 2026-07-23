/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 7 (Omega Research STAD Club, Vol. 7).
// Systems: Derivative Moving Average, Short-Term Support/Resistance,
// Moving Average Support/Resistance, Trading Range Breakout.
// Skipped from this volume: Dow Theory (needs Industrials/Transports/
// Utilities as Data2-4), Intraday Breakout and Fibonacci Retracement
// (time-of-day logic on intraday bars), Intermarket S&P/Bonds/Bills and
// S&P Implied Volatility (external Data2 series). See docs/EXPERIMENTS.md.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// --- exp07_deriv_ma ----------------------------------------------------------
// First/second discrete derivatives of an SMA: the second derivative's zero
// cross arms the setup, the first derivative's zero cross in the same
// direction fires a stop entry one tick beyond the close. Stop-and-reverse.
class DerivMaV7 final : public IStrategy {
public:
  std::string name() const override { return "exp07_deriv_ma"; }

  std::vector<ParamDef> params() const override {
    return {{"length", 9, 3, 40, 1, true}};
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::SMA, Column::Close, {ps.get("length")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* avg = ind.line(0);
    const int n = bars.size();
    const int len = ps.get_int("length");

    // DerivMA1[i] = (Avg[i] - Avg[i-len]) / len; DerivMA2 = its 1-bar change.
    std::vector<float> d1(static_cast<std::size_t>(n), kNaN);
    std::vector<float> d2(static_cast<std::size_t>(n), kNaN);
    for (int i = len; i < n; ++i) {
      if (finite(avg[i]) && finite(avg[i - len]))
        d1[static_cast<std::size_t>(i)] = (avg[i] - avg[i - len]) / static_cast<float>(len);
    }
    for (int i = 1; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (finite(d1[ui]) && finite(d1[ui - 1])) d2[ui] = d1[ui] - d1[ui - 1];
    }

    int pos = 0;
    bool buyFlag = false, sellFlag = false;
    float buyLvl = kNaN, sellLvl = kNaN;

    for (int i = 1; i < n; ++i) {
      // 1) Armed stop entries from the previous bar's signal.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        buyLvl = kNaN;
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sellLvl = kNaN;
      }
      // 2) Setup and trigger bookkeeping at this close.
      if (crosses_above_level(d2.data(), 0.f, i)) buyFlag = true;
      if (crosses_below_level(d2.data(), 0.f, i)) sellFlag = true;
      if (crosses_above_level(d1.data(), 0.f, i) && buyFlag) {
        buyLvl = C(bars, i);
        buyFlag = false;
      }
      if (crosses_below_level(d1.data(), 0.f, i) && sellFlag) {
        sellLvl = C(bars, i);
        sellFlag = false;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp07_sr_osc ------------------------------------------------------------
// Art Putt's intra-bar oscillator ((TrueHigh-Open)+(Close-TrueLow))/(2*TR):
// dipping under the buy level arms a long, re-crossing it enters at the next
// open; symmetric on the sell side. Exits when the oscillator breaks its
// pre-entry extreme or via a 2-bar extreme stop once it reaches the far band.
class SrOscV7 final : public IStrategy {
public:
  std::string name() const override { return "exp07_sr_osc"; }

  std::vector<ParamDef> params() const override {
    return {
        {"buylev", 0.30, 0.05, 0.45, 0.05, false},
        {"selllev", 0.70, 0.55, 0.95, 0.05, false},
        {"exitlen", 2, 1, 6, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("buylev") < ps.get("selllev");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    (void)ps;
    return {{Family::TRANGE, Column::Close, {}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* tr = ind.line(0);
    const int n = bars.size();
    const float buyLev = static_cast<float>(ps.get("buylev"));
    const float sellLev = static_cast<float>(ps.get("selllev"));
    const int exitLen = ps.get_int("exitlen");

    // SROsc, carried forward on zero-range bars as in the original.
    std::vector<float> osc(static_cast<std::size_t>(n), kNaN);
    float last = kNaN;
    for (int i = 0; i < n; ++i) {
      const float trueHigh = i > 0 ? std::max(H(bars, i), C(bars, i - 1)) : H(bars, i);
      const float trueLow = i > 0 ? std::min(L(bars, i), C(bars, i - 1)) : L(bars, i);
      if (finite(tr[i]) && tr[i] > 0.f) {
        last = ((trueHigh - O(bars, i)) + (C(bars, i) - trueLow)) / (2.f * tr[i]);
      }
      osc[static_cast<std::size_t>(i)] = last;
    }
    const float* o = osc.data();

    int pos = 0;
    float loOsc = kNaN, hiOsc = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(o[i]) || !finite(o[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Initial stop: oscillator breaks its pre-entry extreme (exit on close).
      if (pos == 1 && finite(loOsc) && o[i] < loOsc && o[i - 1] >= loOsc) pos = 0;
      if (pos == -1 && finite(hiOsc) && o[i] > hiOsc && o[i - 1] <= hiOsc) pos = 0;
      // 2) Trailing stop once the oscillator reaches the far band.
      if (pos == 1 && o[i - 1] > sellLev) {
        float lo = L(bars, i - 1);
        for (int k = 1; k < exitLen && i - 1 - k >= 0; ++k)
          lo = std::min(lo, L(bars, i - 1 - k));
        if (L(bars, i) <= lo) pos = 0;
      } else if (pos == -1 && o[i - 1] < buyLev) {
        float hi = H(bars, i - 1);
        for (int k = 1; k < exitLen && i - 1 - k >= 0; ++k)
          hi = std::max(hi, H(bars, i - 1 - k));
        if (H(bars, i) >= hi) pos = 0;
      }
      // 3) Pre-entry extreme bookkeeping.
      if (pos != 1) {
        if (o[i] < buyLev && o[i - 1] >= buyLev) loOsc = o[i];
        else if (o[i] < buyLev && finite(loOsc)) loOsc = std::min(loOsc, o[i]);
      }
      if (pos != -1) {
        if (o[i] > sellLev && o[i - 1] <= sellLev) hiOsc = o[i];
        else if (o[i] > sellLev && finite(hiOsc)) hiOsc = std::max(hiOsc, o[i]);
      }
      // 4) Market entries on re-crossing the trigger level.
      if (crosses_above_level(o, buyLev, i) && pos != 1) pos = 1;
      else if (crosses_below_level(o, sellLev, i) && pos != -1) pos = -1;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp07_ma_suppres --------------------------------------------------------
// Ratcheting support/resistance: each close-through-SMA flip freezes the last
// tracked extreme as the order level; a close through that level enters with
// a stop at the bar extreme. Exits combine an entry-anchored protective ATR
// stop and a position-extreme ATR trail.
class MaSuppResV7 final : public IStrategy {
public:
  std::string name() const override { return "exp07_ma_suppres"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 10, 4, 50, 1, true},
        {"p_atrs", 1.0, 0.5, 4.0, 0.25, false},
        {"t_atrs", 2.5, 0.5, 6.0, 0.25, false},
        {"atrlen", 10, 3, 30, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("length")}},
        {Family::ATR, Column::Close, {ps.get("atrlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* avg = ind.line(0);
    const float* atr = ind.line(1);
    const int n = bars.size();
    const float pAtrs = static_cast<float>(ps.get("p_atrs"));
    const float tAtrs = static_cast<float>(ps.get("t_atrs"));

    int pos = 0;
    float supp = kNaN, res = kNaN, lOrder = kNaN, sOrder = kNaN;
    bool suppFlag = false, resFlag = false;
    float entryPx = kNaN, posHigh = kNaN, posLow = kNaN;
    float buyLvl = kNaN, sellLvl = kNaN;

    for (int i = 1; i < n; ++i) {
      if (!finite(avg[i]) || !finite(avg[i - 1]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) ATR exits from the previous bar's levels.
      if (pos == 1 && finite(entryPx)) {
        const float prot = entryPx - pAtrs * atr[i];
        const float trail = posHigh - tAtrs * atr[i];
        if (L(bars, i) <= std::max(prot, trail)) pos = 0;
      } else if (pos == -1 && finite(entryPx)) {
        const float prot = entryPx + pAtrs * atr[i];
        const float trail = posLow + tAtrs * atr[i];
        if (H(bars, i) >= std::min(prot, trail)) pos = 0;
      }
      // 2) Armed breakout stops from the previous close's signal.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        posHigh = H(bars, i);
        buyLvl = kNaN;
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        posLow = L(bars, i);
        sellLvl = kNaN;
      }
      if (pos == 1) posHigh = std::max(posHigh, H(bars, i));
      if (pos == -1) posLow = std::min(posLow, L(bars, i));
      // 3) Support/resistance ratchet at this close.
      const bool xBelow = C(bars, i) < avg[i] && C(bars, i - 1) >= avg[i - 1];
      const bool xAbove = C(bars, i) > avg[i] && C(bars, i - 1) <= avg[i - 1];
      if (finite(res) && xBelow) {
        if (resFlag) {
          lOrder = res;
          resFlag = false;
        }
        suppFlag = true;
        supp = L(bars, i);
      }
      if (finite(supp) && xAbove) {
        if (suppFlag) {
          sOrder = supp;
          suppFlag = false;
        }
        resFlag = true;
        res = H(bars, i);
      }
      if (!finite(res) && xAbove) {
        resFlag = true;
        res = H(bars, i);
      }
      if (!finite(supp) && xBelow) {
        suppFlag = true;
        supp = L(bars, i);
      }
      if (C(bars, i) > avg[i] && finite(res) && H(bars, i) >= res) res = H(bars, i);
      if (C(bars, i) < avg[i] && finite(supp) && L(bars, i) <= supp) supp = L(bars, i);
      // 4) Close crossing the frozen order level arms the stop entry.
      if (finite(lOrder) && i >= 1 && C(bars, i) > lOrder && C(bars, i - 1) <= lOrder)
        buyLvl = H(bars, i);
      if (finite(sOrder) && i >= 1 && C(bars, i) < sOrder && C(bars, i - 1) >= sOrder)
        sellLvl = L(bars, i);
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp07_tr_breakout -------------------------------------------------------
// A dense trading range (large accumulated "no-trade" area) plus a true-range
// expansion bar closing beyond the range enters on that close. Protective
// stop at the opposite range edge; ATR trail off the position extreme.
class TrBreakoutV7 final : public IStrategy {
public:
  std::string name() const override { return "exp07_tr_breakout"; }

  std::vector<ParamDef> params() const override {
    return {
        {"rangelen", 7, 4, 20, 1, true},
        {"rngpcnt", 200, 50, 400, 25, true},
        {"atrs", 8, 1, 12, 1, true},
        {"atrlen", 2, 2, 10, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("rangelen")}},
        {Family::MIN, Column::Low, {ps.get("rangelen")}},
        {Family::ATR, Column::Close, {ps.get("atrlen")}},
        {Family::TRANGE, Column::Close, {}},
        {Family::ATR, Column::Close, {ps.get("rangelen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiC = ind.line(0);
    const float* loC = ind.line(1);
    const float* atr = ind.line(2);
    const float* tr = ind.line(3);
    const float* atrRange = ind.line(4);
    const int n = bars.size();
    const int rangeLen = ps.get_int("rangelen");
    const float mult = static_cast<float>(ps.get_int("rngpcnt")) / 100.f;
    const float atrs = static_cast<float>(ps.get_int("atrs"));

    int pos = 0;
    float longRisk = kNaN, longHigh = kNaN, shortRisk = kNaN, shortLow = kNaN;

    for (int i = 1; i < n; ++i) {
      const float rangeH = hiC[i - 1];
      const float rangeL = loC[i - 1];
      if (!finite(rangeH) || !finite(rangeL) || !finite(atr[i]) || !finite(tr[i]) ||
          i < 2 || !finite(atrRange[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits from the previous bar's levels.
      if (pos == 1) {
        longHigh = std::max(longHigh, H(bars, i));
        const float stop = std::max(longRisk, longHigh - atrs * atr[i]);
        if (L(bars, i) <= stop) pos = 0;
      } else if (pos == -1) {
        shortLow = std::min(shortLow, L(bars, i));
        const float stop = std::min(shortRisk, shortLow + atrs * atr[i]);
        if (H(bars, i) >= stop) pos = 0;
      }
      // 2) Breakout entries on this close.
      const float tRange = rangeH - rangeL;
      float noTrades = 0.f;
      for (int k = 1; k <= rangeLen && i - k >= 0; ++k) {
        if (H(bars, i - k) < rangeH) noTrades += rangeH - H(bars, i - k);
        if (L(bars, i - k) > rangeL) noTrades += L(bars, i - k) - rangeL;
      }
      const bool cond1 = noTrades >= tRange * mult;
      const bool cond2 = tr[i] > atrRange[i - 1];
      const float med = (H(bars, i) + L(bars, i)) / 2.f;
      if (cond1 && cond2) {
        if (C(bars, i) > rangeH && med > H(bars, i - 1) && pos != 1) {
          pos = 1;
          longRisk = rangeL;
          longHigh = H(bars, i);
        } else if (C(bars, i) < rangeL && med < L(bars, i - 1) && pos != -1) {
          pos = -1;
          shortRisk = rangeH;
          shortLow = L(bars, i);
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

} // namespace

void register_vol07(std::vector<Factory>& out) {
  out.push_back({"exp07_deriv_ma",
                 [] { return std::unique_ptr<IStrategy>(new DerivMaV7); }});
  out.push_back({"exp07_sr_osc",
                 [] { return std::unique_ptr<IStrategy>(new SrOscV7); }});
  out.push_back({"exp07_ma_suppres",
                 [] { return std::unique_ptr<IStrategy>(new MaSuppResV7); }});
  out.push_back({"exp07_tr_breakout",
                 [] { return std::unique_ptr<IStrategy>(new TrBreakoutV7); }});
}

} // namespace topt::exp
