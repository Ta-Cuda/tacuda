/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 5 (Omega Research STAD Club, Vol. 5).
// Systems: Fibonacci .382 Retracement, High/Low Oscillator, Stochastic
// Overbought/Oversold, Pivot Point Support & Resistance, Currency Four-Day,
// Channel Breakout with Pullback, Displaced Standard Deviation Bands.
// Skipped from this volume: Currency/Bond, Intermarket I, Intermarket II
// (all need a second symbol). See docs/EXPERIMENTS.md.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// --- exp05_fib_retrace -------------------------------------------------------
// A 38.2% retracement of the 50-bar range against the prior trend opens a
// 24-bar window; a hook (swing low) in the 15-bar EMA or a close back across
// it resumes the trend at the next open. Exits: percent trailing stop from
// the best close (the original used dollar-risk stops).
class FibRetraceV5 final : public IStrategy {
public:
  std::string name() const override { return "exp05_fib_retrace"; }

  std::vector<ParamDef> params() const override {
    return {
        {"xavglen", 15, 5, 40, 1, true},
        {"hilolen", 50, 20, 120, 5, true},
        {"retrace", 0.382, 0.2, 0.7, 0.02, false},
        {"setuplen", 24, 5, 60, 1, true},
        {"trailpct", 5.0, 1.0, 15.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("xavglen")}},
        {Family::MAX, Column::High, {ps.get("hilolen")}},
        {Family::MIN, Column::Low, {ps.get("hilolen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* xavg = ind.line(0);
    const float* hiHi = ind.line(1);
    const float* loLo = ind.line(2);
    const int n = bars.size();
    const int hiLoLen = ps.get_int("hilolen");
    const float retr = static_cast<float>(ps.get("retrace"));
    const int setupLen = ps.get_int("setuplen");
    const float trailPct = static_cast<float>(ps.get("trailpct")) / 100.f;

    int pos = 0;
    int buySetup = 1 << 20, sellSetup = 1 << 20;
    float bestClose = 0.f;

    for (int i = 0; i < n; ++i) {
      if (!finite(xavg[i]) || !finite(hiHi[i]) || !finite(loLo[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Percent trailing stop from the best close since entry.
      if (pos == 1) {
        bestClose = std::max(bestClose, C(bars, i));
        if (L(bars, i) <= bestClose * (1.f - trailPct)) pos = 0;
      } else if (pos == -1) {
        bestClose = std::min(bestClose, C(bars, i));
        if (H(bars, i) >= bestClose * (1.f + trailPct)) pos = 0;
      }
      // 2) Bars-since-extreme bookkeeping (most recent touch of the channel
      //    extremes inside the lookback window).
      int hiBar = -1, loBar = -1;
      for (int k = 0; k < hiLoLen && i - k >= 0; ++k) {
        if (hiBar < 0 && H(bars, i - k) >= hiHi[i]) hiBar = k;
        if (loBar < 0 && L(bars, i - k) <= loLo[i]) loBar = k;
        if (hiBar >= 0 && loBar >= 0) break;
      }
      if (hiBar < 0 || loBar < 0) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float retracement = (hiHi[i] - loLo[i]) * retr;
      if (C(bars, i) <= hiHi[i] - retracement && hiBar < loBar) buySetup = 0;
      if (C(bars, i) >= loLo[i] + retracement && hiBar > loBar) sellSetup = 0;
      ++buySetup;
      ++sellSetup;
      // 3) Trend-resumption triggers inside the setup window.
      const bool emaHook_up = i >= 2 && finite(xavg[i - 2]) &&
                              xavg[i - 1] < xavg[i - 2] && xavg[i - 1] <= xavg[i];
      const bool emaHook_dn = i >= 2 && finite(xavg[i - 2]) &&
                              xavg[i - 1] > xavg[i - 2] && xavg[i - 1] >= xavg[i];
      if (buySetup <= setupLen && (emaHook_up || C(bars, i) > xavg[i]) && pos != 1) {
        pos = 1;
        bestClose = C(bars, i);
        buySetup = setupLen + 1;
      } else if (sellSetup <= setupLen && (emaHook_dn || C(bars, i) < xavg[i]) &&
                 pos != -1) {
        pos = -1;
        bestClose = C(bars, i);
        sellSetup = setupLen + 1;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp05_hilo_osc ----------------------------------------------------------
// Trend filter from a slow close EMA; entries when the high/low oscillator
// (extreme minus its own EMA) curls back toward the trend. Exits: percent
// trailing stop off the close plus oscillator zero-cross timing exits.
class HiLoOscV5 final : public IStrategy {
public:
  std::string name() const override { return "exp05_hilo_osc"; }

  std::vector<ParamDef> params() const override {
    return {
        {"len1", 10, 3, 40, 1, true},
        {"len2", 60, 20, 120, 5, true},
        {"trailpct", 3.0, 0.5, 10.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::High, {ps.get("len1")}},
        {Family::EMA, Column::Low, {ps.get("len1")}},
        {Family::EMA, Column::Close, {ps.get("len2")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* emaH = ind.line(0);
    const float* emaL = ind.line(1);
    const float* emaC = ind.line(2);
    const int n = bars.size();
    const float trailPct = static_cast<float>(ps.get("trailpct")) / 100.f;

    auto hiOsc = [&](int i) {
      return finite(emaH[i]) ? H(bars, i) - emaH[i] : kNaN;
    };
    auto loOsc = [&](int i) {
      return finite(emaL[i]) ? L(bars, i) - emaL[i] : kNaN;
    };

    int pos = 0;
    for (int i = 0; i < n; ++i) {
      const bool ready = i >= 1 && finite(emaC[i]) && finite(emaC[i - 1]) &&
                         finite(hiOsc(i)) && finite(loOsc(i)) &&
                         finite(hiOsc(i - 1)) && finite(loOsc(i - 1));
      if (!ready) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits: close-anchored trailing stop plus oscillator timing.
      if (pos == 1) {
        const float trail = C(bars, i - 1) * (1.f - trailPct);
        if (L(bars, i) <= trail ||
            (hiOsc(i) < 0.f && hiOsc(i - 1) >= 0.f)) {
          pos = 0;
        }
      } else if (pos == -1) {
        const float trail = C(bars, i - 1) * (1.f + trailPct);
        if (H(bars, i) >= trail ||
            (loOsc(i) > 0.f && loOsc(i - 1) <= 0.f)) {
          pos = 0;
        }
      }
      // 2) Entries at this close.
      const bool buySetup = emaC[i] > emaC[i - 1] && C(bars, i) > emaC[i] &&
                            loOsc(i) < 0.f;
      const bool sellSetup = emaC[i] < emaC[i - 1] && C(bars, i) < emaC[i] &&
                             hiOsc(i) > 0.f;
      if (buySetup && loOsc(i) > loOsc(i - 1) && pos != 1) pos = 1;
      else if (sellSetup && hiOsc(i) < hiOsc(i - 1) && pos != -1) pos = -1;
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp05_stoch_obos --------------------------------------------------------
// SlowK hooking through SlowD inside the oversold/overbought band arms a
// two-bar-extreme entry stop for six bars, with the setup bar's far extreme
// as the protective stop. Opposite hooks outside the bands exit at the open.
class StochObosV5 final : public IStrategy {
public:
  std::string name() const override { return "exp05_stoch_obos"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 12, 5, 40, 1, true},
        {"obought", 70, 55, 90, 1, true},
        {"osold", 30, 10, 45, 1, true},
        {"setuplen", 6, 2, 15, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("osold") < ps.get("obought");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::STOCH, Column::Close, {ps.get("length"), 3}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* k = ind.line(0, 0);
    const float* d = ind.line(0, 1);
    const int n = bars.size();
    const float ob = static_cast<float>(ps.get("obought"));
    const float os = static_cast<float>(ps.get("osold"));
    const int setupLen = ps.get_int("setuplen");

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f, protStop = 0.f;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 0; i < n; ++i) {
      if (!finite(k[i]) || !finite(d[i]) || i < 1 || !finite(k[i - 1]) ||
          !finite(d[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const bool kXUp = k[i] > d[i] && k[i - 1] <= d[i - 1];
      const bool kXDn = k[i] < d[i] && k[i - 1] >= d[i - 1];
      // 1) Exits: protective stop or an opposite hook outside the band.
      if (pos == 1) {
        if (L(bars, i) <= protStop || (kXDn && k[i] > os)) pos = 0;
      } else if (pos == -1) {
        if (H(bars, i) >= protStop || (kXUp && k[i] < ob)) pos = 0;
      }
      // 2) Armed entry stops.
      if (pos != 1 && i <= buyUntil && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sellUntil = -1;
      }
      // 3) Fresh hooks inside the bands arm new stops.
      if (k[i] < os && kXUp) {
        buyLvl = std::max(H(bars, i), i >= 1 ? H(bars, i - 1) : H(bars, i));
        protStop = L(bars, i);
        buyUntil = i + setupLen;
      }
      if (k[i] > ob && kXDn) {
        sellLvl = std::min(L(bars, i), i >= 1 ? L(bars, i - 1) : L(bars, i));
        protStop = H(bars, i);
        sellUntil = i + setupLen;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp05_pivot_sr ----------------------------------------------------------
// A pivot low that is also the channel low, undercut and then reclaimed by a
// wide-range reversal bar, arms a stop above the reversal bar (EMA-filtered,
// 24-bar window). Initial stop under the reversal low, breakeven after twice
// the risk, profit-taking at the 44-bar extreme.
class PivotSrV5 final : public IStrategy {
public:
  std::string name() const override { return "exp05_pivot_sr"; }

  std::vector<ParamDef> params() const override {
    return {
        {"hilolen", 8, 4, 20, 1, true},
        {"atrlen", 10, 5, 30, 1, true},
        {"setupint", 4, 2, 10, 1, true},
        {"xavglen", 9, 3, 30, 1, true},
        {"setuplen", 24, 5, 60, 1, true},
        {"hilostop", 44, 10, 100, 2, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::ATR, Column::Close, {ps.get("atrlen")}},
        {Family::EMA, Column::Close, {ps.get("xavglen")}},
        {Family::TRANGE, Column::Close, {}},
        {Family::MAX, Column::High, {ps.get("hilostop")}},
        {Family::MIN, Column::Low, {ps.get("hilostop")}},
        {Family::MAX, Column::High, {ps.get("hilolen")}},
        {Family::MIN, Column::Low, {ps.get("hilolen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const float* ema = ind.line(1);
    const float* tr = ind.line(2);
    const float* hiStop = ind.line(3);
    const float* loStop = ind.line(4);
    const float* hiLen = ind.line(5);
    const float* loLen = ind.line(6);
    const int n = bars.size();
    const int setupInt = ps.get_int("setupint");
    const int setupLen = ps.get_int("setuplen");

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f;
    int buyUntil = -1, sellUntil = -1;
    float lExitLow = 0.f, sExitHigh = 0.f, entryRef = 0.f;
    bool breakevenL = false, breakevenS = false;

    // Rolling record of pivot-low / pivot-high events (bar index of event).
    std::vector<int> pivotLowAt, pivotHighAt;

    for (int i = 0; i < n; ++i) {
      const bool ready = finite(atr[i]) && finite(ema[i]) && finite(tr[i]) && i >= 2;
      // 1) Exits for the open position.
      if (pos == 1) {
        if (L(bars, i) <= lExitLow) {
          pos = 0;
        } else {
          if (!breakevenL && C(bars, i) > entryRef + 2.f * (entryRef - lExitLow))
            breakevenL = true;
          if (breakevenL && L(bars, i) <= entryRef) pos = 0;
          else if (i >= 1 && finite(hiStop[i - 1]) && H(bars, i) >= hiStop[i - 1])
            pos = 0; // profit target at the 44-bar high
        }
      } else if (pos == -1) {
        if (H(bars, i) >= sExitHigh) {
          pos = 0;
        } else {
          if (!breakevenS && C(bars, i) < entryRef - 2.f * (sExitHigh - entryRef))
            breakevenS = true;
          if (breakevenS && H(bars, i) >= entryRef) pos = 0;
          else if (i >= 1 && finite(loStop[i - 1]) && L(bars, i) <= loStop[i - 1])
            pos = 0;
        }
      }
      // 2) Armed entry stops (EMA-filtered).
      if (pos != 1 && i <= buyUntil && buyLvl > ema[i] &&
          buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryRef = buyLvl;
        breakevenL = false;
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && sellLvl < ema[i] &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryRef = sellLvl;
        breakevenS = false;
        sellUntil = -1;
      }
      if (!ready) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 3) Pivot bookkeeping at this close.
      const bool pivotLow = L(bars, i - 1) < L(bars, i - 2) &&
                            L(bars, i - 1) <= L(bars, i) &&
                            i >= 2 && finite(loLen[i - 1]) &&
                            L(bars, i - 1) <= loLen[i - 1];
      const bool pivotHigh = H(bars, i - 1) > H(bars, i - 2) &&
                             H(bars, i - 1) >= H(bars, i) &&
                             finite(hiLen[i - 1]) && H(bars, i - 1) >= hiLen[i - 1];
      if (pivotLow) pivotLowAt.push_back(i);
      if (pivotHigh) pivotHighAt.push_back(i);

      // 4) Reversal-bar setups within `setupint` bars of the pivot.
      auto recent = [&](const std::vector<int>& v) {
        return !v.empty() && i - v.back() >= 1 && i - v.back() < setupInt;
      };
      if (recent(pivotLowAt)) {
        const int pv = pivotLowAt.back();       // event bar; pivot low at pv-1
        const float pivLow = L(bars, pv - 1);
        float minSince = L(bars, i);
        for (int j = pv; j < i; ++j) minSince = std::min(minSince, L(bars, j));
        const bool undercut = minSince < pivLow;
        const bool strongBar = tr[i] > atr[i] && C(bars, i) > C(bars, i - 1) &&
                               C(bars, i) > O(bars, i) &&
                               C(bars, i) > L(bars, i) + 2.f * tr[i] / 3.f &&
                               C(bars, i) > pivLow;
        if (undercut && strongBar) {
          buyLvl = H(bars, i);
          lExitLow = L(bars, i);
          buyUntil = i + setupLen;
        }
      }
      if (recent(pivotHighAt)) {
        const int pv = pivotHighAt.back();
        const float pivHigh = H(bars, pv - 1);
        float maxSince = H(bars, i);
        for (int j = pv; j < i; ++j) maxSince = std::max(maxSince, H(bars, j));
        const bool overshoot = maxSince > pivHigh;
        const bool weakBar = tr[i] > atr[i] && C(bars, i) < C(bars, i - 1) &&
                             C(bars, i) < O(bars, i) &&
                             C(bars, i) < L(bars, i) + tr[i] / 3.f &&
                             C(bars, i) < pivHigh;
        if (overshoot && weakBar) {
          sellLvl = L(bars, i);
          sExitHigh = H(bars, i);
          sellUntil = i + setupLen;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp05_currency_4day -----------------------------------------------------
// Weekend momentum: on Friday's close, a 50-bar RSI above/below 50 buys/
// sells the Monday open; the trade is closed into Tuesday's open (Monday's
// touch of Friday's extreme takes profit at the same exit bar).
class Currency4DayV5 final : public IStrategy {
public:
  std::string name() const override { return "exp05_currency_4day"; }

  std::vector<ParamDef> params() const override {
    return {
        {"rsilen", 50, 10, 100, 2, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::RSI, Column::Close, {ps.get("rsilen")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* rsi = ind.line(0);
    const int n = bars.size();

    int pos = 0;
    int entryDecision = -1;
    for (int i = 0; i < n; ++i) {
      // Exit at the first close after the position bar (hold one session).
      if (pos != 0 && i > entryDecision + 1) pos = 0;
      if (pos == 0 && finite(rsi[i]) && !bars.ts.empty() &&
          day_of_week(bars.ts[static_cast<std::size_t>(i)]) == 5) {
        if (rsi[i] > 50.f) {
          pos = 1;
          entryDecision = i;
        } else if (rsi[i] < 50.f) {
          pos = -1;
          entryDecision = i;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp05_chan_pullback -----------------------------------------------------
// Close-channel breakout followed by a pullback to a fresh 4-bar extreme
// within ten bars enters with the trend at the next open. Exits on a snap to
// the opposite 8-bar extreme.
class ChanPullbackV5 final : public IStrategy {
public:
  std::string name() const override { return "exp05_chan_pullback"; }

  std::vector<ParamDef> params() const override {
    return {
        {"channel", 10, 5, 40, 1, true},
        {"retrbars", 10, 3, 30, 1, true},
        {"newlohi", 4, 2, 10, 1, true},
        {"trailstp", 8, 3, 20, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::Close, {ps.get("channel")}},
        {Family::MIN, Column::Close, {ps.get("channel")}},
        {Family::MAX, Column::High, {ps.get("newlohi")}},
        {Family::MIN, Column::Low, {ps.get("newlohi")}},
        {Family::MAX, Column::High, {ps.get("trailstp")}},
        {Family::MIN, Column::Low, {ps.get("trailstp")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiChan = ind.line(0);
    const float* loChan = ind.line(1);
    const float* hiNew = ind.line(2);
    const float* loNew = ind.line(3);
    const float* hiTrail = ind.line(4);
    const float* loTrail = ind.line(5);
    const int n = bars.size();
    const int retrBars = ps.get_int("retrbars");

    int lastBreakUp = -1000, lastBreakDn = -1000;
    int pos = 0;

    for (int i = 1; i < n; ++i) {
      // 1) Exits at this close.
      if (pos == 1 && finite(hiTrail[i - 1]) && H(bars, i) > hiTrail[i - 1]) pos = 0;
      else if (pos == -1 && finite(loTrail[i - 1]) && L(bars, i) < loTrail[i - 1]) pos = 0;
      // 2) Breakout bookkeeping.
      if (finite(hiChan[i - 1]) && C(bars, i) > hiChan[i - 1]) lastBreakUp = i;
      if (finite(loChan[i - 1]) && C(bars, i) < loChan[i - 1]) lastBreakDn = i;
      // 3) Pullback entries at this close.
      if (pos != 1 && i - lastBreakUp < retrBars && finite(loNew[i - 1]) &&
          L(bars, i) < loNew[i - 1]) {
        pos = 1;
      } else if (pos != -1 && i - lastBreakDn < retrBars && finite(hiNew[i - 1]) &&
                 H(bars, i) > hiNew[i - 1]) {
        pos = -1;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp05_disp_stddev -------------------------------------------------------
// Stop-and-reverse: both entry stops always live at a 3-bar moving average
// displaced 16 bars forward, +/- two current standard deviations.
class DispStdDevV5 final : public IStrategy {
public:
  std::string name() const override { return "exp05_disp_stddev"; }

  std::vector<ParamDef> params() const override {
    return {
        {"avglen", 3, 2, 15, 1, true},
        {"disp", 16, 4, 40, 1, true},
        {"sdlen", 12, 5, 40, 1, true},
        {"sdev", 2.0, 1.0, 3.0, 0.25, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("avglen")}},
        {Family::STDDEV, Column::Close, {ps.get("sdlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* sma = ind.line(0);
    const float* sd = ind.line(1);
    const int n = bars.size();
    const int disp = ps.get_int("disp");
    const float mult = static_cast<float>(ps.get("sdev"));

    int pos = 0;
    for (int i = 0; i < n; ++i) {
      // Levels armed at the previous close (displaced average, live bands).
      if (i >= 1 && i - 1 - disp >= 0 && finite(sma[i - 1 - disp]) &&
          finite(sd[i - 1])) {
        const float top = sma[i - 1 - disp] + mult * sd[i - 1];
        const float bot = sma[i - 1 - disp] - mult * sd[i - 1];
        const bool up = pos != 1 && buy_stop_hit(bars, i, top);
        const bool dn = pos != -1 && sell_stop_hit(bars, i, bot);
        if (up && dn) {
          pos = std::abs(C(bars, i) - top) <= std::abs(C(bars, i) - bot) ? 1 : -1;
        } else if (up) {
          pos = 1;
        } else if (dn) {
          pos = -1;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

} // namespace

void register_vol05(std::vector<Factory>& out) {
  out.push_back({"exp05_fib_retrace",
                 [] { return std::unique_ptr<IStrategy>(new FibRetraceV5); }});
  out.push_back({"exp05_hilo_osc",
                 [] { return std::unique_ptr<IStrategy>(new HiLoOscV5); }});
  out.push_back({"exp05_stoch_obos",
                 [] { return std::unique_ptr<IStrategy>(new StochObosV5); }});
  out.push_back({"exp05_pivot_sr",
                 [] { return std::unique_ptr<IStrategy>(new PivotSrV5); }});
  out.push_back({"exp05_currency_4day",
                 [] { return std::unique_ptr<IStrategy>(new Currency4DayV5); }});
  out.push_back({"exp05_chan_pullback",
                 [] { return std::unique_ptr<IStrategy>(new ChanPullbackV5); }});
  out.push_back({"exp05_disp_stddev",
                 [] { return std::unique_ptr<IStrategy>(new DispStdDevV5); }});
}

} // namespace topt::exp
