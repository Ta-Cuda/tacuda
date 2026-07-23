/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 4 (Omega Research STAD Club, Vol. 4).
// Systems: DMI-ADX, Keltner Channel, Linear Regression & Momentum, Herrick
// Payoff Index & Channel Breakout (open-interest-free HPI proxy),
// Consecutive Close, Momentum-Retracement, Three EMA Crossover, Displaced
// MA & Volume, Price Distribution.
// Skipped from this volume: Spread System Beta Up/Down (needs two symbols).
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

// --- exp04_dmi_adx -----------------------------------------------------------
// ADX crossing above the opposite DI line while DI+ leads DI- arms a stop at
// the setup bar's extreme (cancelled when ADX drops back). Exits: setup-bar
// protective stop, an ADX peak-and-turn close-out, and a DI-cross close-out.
class DmiAdxV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_dmi_adx"; }

  std::vector<ParamDef> params() const override {
    return {
        {"dmilen", 7, 3, 30, 1, true},
        {"adxlen", 7, 3, 30, 1, true},
        {"adxexit", 30, 15, 60, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::PLUS_DI, Column::Close, {ps.get("dmilen")}},
        {Family::MINUS_DI, Column::Close, {ps.get("dmilen")}},
        {Family::ADX, Column::Close, {ps.get("adxlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* dPlus = ind.line(0);
    const float* dMin = ind.line(1);
    const float* adx = ind.line(2);
    const int n = bars.size();
    const float adxExit = static_cast<float>(ps.get("adxexit"));

    int pos = 0, fillBar = -1;
    bool buySetup = false, sellSetup = false;
    float buyLvl = 0.f, sellLvl = 0.f, exitL = 0.f, exitS = 0.f;

    for (int i = 0; i < n; ++i) {
      if (!finite(dPlus[i]) || !finite(dMin[i]) || !finite(adx[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits for the open position.
      if (pos == 1) {
        if (L(bars, i) <= exitL) pos = 0;
        else if (i > fillBar + 1 &&
                 ((i >= 1 && finite(adx[i - 1]) && adx[i - 1] >= adxExit &&
                   adx[i] < adx[i - 1]) ||
                  dPlus[i] < dMin[i])) {
          pos = 0;
        }
      } else if (pos == -1) {
        if (H(bars, i) >= exitS) pos = 0;
        else if (i > fillBar + 1 &&
                 ((i >= 1 && finite(adx[i - 1]) && adx[i - 1] >= adxExit &&
                   adx[i] < adx[i - 1]) ||
                  dPlus[i] > dMin[i])) {
          pos = 0;
        }
      }
      // 2) Armed setup stops.
      if (buySetup && pos != 1 && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1; fillBar = i; buySetup = false;
      } else if (sellSetup && pos != -1 && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1; fillBar = i; sellSetup = false;
      }
      // 3) Setup detection and cancellation at this close.
      const bool adxXDMin = i >= 1 && finite(adx[i - 1]) && finite(dMin[i - 1]) &&
                            adx[i] > dMin[i] && adx[i - 1] <= dMin[i - 1];
      const bool adxXDPlus = i >= 1 && finite(adx[i - 1]) && finite(dPlus[i - 1]) &&
                             adx[i] > dPlus[i] && adx[i - 1] <= dPlus[i - 1];
      if (dPlus[i] > dMin[i] && adxXDMin && pos != 1) {
        buySetup = true;
        buyLvl = H(bars, i);
        exitL = L(bars, i);
      }
      if (adx[i] < dMin[i] || pos == 1) buySetup = false;
      if (dPlus[i] < dMin[i] && adxXDPlus && pos != -1) {
        sellSetup = true;
        sellLvl = L(bars, i);
        exitS = H(bars, i);
      }
      if (adx[i] < dPlus[i] || pos == -1) sellSetup = false;
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_keltner -----------------------------------------------------------
// A close outside the Keltner Channel arms a stop half a channel-width past
// the setup bar's extreme for five bars. Exits: a close back through the
// midline (at that close) or the 4-bar extreme trail.
class KeltnerV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_keltner"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 10, 5, 40, 1, true},
        {"const", 1.2, 0.6, 3.0, 0.1, false},
        {"chanpcnt", 0.5, 0.1, 1.0, 0.1, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("length")}},
        {Family::TRANGE, Column::Close, {}},
        {Family::MAX, Column::High, {4}},
        {Family::MIN, Column::Low, {4}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* mid = ind.line(0);
    const float* tr = ind.line(1);
    const float* hi4 = ind.line(2);
    const float* lo4 = ind.line(3);
    const int n = bars.size();
    const float cMult = static_cast<float>(ps.get("const"));
    const float pcnt = static_cast<float>(ps.get("chanpcnt"));
    const std::vector<float> atrV = rolling_mean(tr, n, ps.get_int("length"));
    const float* atr = atrV.data();

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 0; i < n; ++i) {
      if (!finite(mid[i]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float upper = mid[i] + atr[i] * cMult;
      const float lower = mid[i] - atr[i] * cMult;
      // 1) Exits.
      if (pos == 1 && i >= 1) {
        const bool crossMid = finite(mid[i - 1]) && C(bars, i) < mid[i] &&
                              C(bars, i - 1) >= mid[i - 1];
        if (crossMid || (finite(lo4[i - 1]) && L(bars, i) <= lo4[i - 1])) pos = 0;
      } else if (pos == -1 && i >= 1) {
        const bool crossMid = finite(mid[i - 1]) && C(bars, i) > mid[i] &&
                              C(bars, i - 1) <= mid[i - 1];
        if (crossMid || (finite(hi4[i - 1]) && H(bars, i) >= hi4[i - 1])) pos = 0;
      }
      // 2) Armed breakout stops (only while price stays outside the band).
      if (pos != 1 && i <= buyUntil && C(bars, i) > upper &&
          buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
      } else if (pos != -1 && i <= sellUntil && C(bars, i) < lower &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
      }
      // 3) Fresh band crossings at this close.
      if (i >= 1) {
        const float chanRng = atr[i] * cMult; // (upper-lower)/2
        const float prevUpper = finite(mid[i - 1]) && finite(atr[i - 1])
                                    ? mid[i - 1] + atr[i - 1] * cMult : kNaN;
        const float prevLower = finite(mid[i - 1]) && finite(atr[i - 1])
                                    ? mid[i - 1] - atr[i - 1] * cMult : kNaN;
        if (finite(prevUpper) && C(bars, i) > upper && C(bars, i - 1) <= prevUpper) {
          buyLvl = H(bars, i) + chanRng * pcnt;
          buyUntil = i + 5;
        }
        if (finite(prevLower) && C(bars, i) < lower && C(bars, i - 1) >= prevLower) {
          sellLvl = L(bars, i) - chanRng * pcnt;
          sellUntil = i + 5;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_linreg_mom --------------------------------------------------------
// A rising smoothed linear-regression line plus momentum curling up from
// below zero arms a stop halfway between the setup high and the line (four
// bars). Exits: initial ATR stop, then a market exit when the line turns and
// the 4-bar extreme breaks.
class LinRegMomV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_linreg_mom"; }

  std::vector<ParamDef> params() const override {
    return {
        {"lrlen", 20, 5, 60, 1, true},
        {"avglen", 15, 3, 40, 1, true},
        {"momlen", 10, 3, 40, 1, true},
        {"pcnt", 0.5, 0.1, 1.0, 0.1, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::LINEARREG, Column::Close, {ps.get("lrlen")}},
        {Family::MOM, Column::Close, {ps.get("momlen")}},
        {Family::ATR, Column::Close, {10}},
        {Family::MAX, Column::High, {4}},
        {Family::MIN, Column::Low, {4}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* linreg = ind.line(0);
    const float* mom = ind.line(1);
    const float* atr10 = ind.line(2);
    const float* hi4 = ind.line(3);
    const float* lo4 = ind.line(4);
    const int n = bars.size();
    const float pcnt = static_cast<float>(ps.get("pcnt"));
    const std::vector<float> xlr = ema_line(linreg, n, ps.get_int("avglen"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN, protStop = 0.f;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      const bool ready = finite(xlr[ui]) && i >= 1 && finite(xlr[ui - 1]) &&
                         finite(mom[i]) && i >= 1 && finite(mom[i - 1]);
      if (!ready) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const bool lineUp = xlr[ui] > xlr[ui - 1];
      const bool lineDn = xlr[ui] < xlr[ui - 1];
      // 1) Exits.
      if (pos == 1) {
        if (L(bars, i) <= protStop ||
            (lineDn && finite(lo4[i - 1]) && L(bars, i) < lo4[i - 1])) {
          pos = 0;
        }
      } else if (pos == -1) {
        if (H(bars, i) >= protStop ||
            (lineUp && finite(hi4[i - 1]) && H(bars, i) > hi4[i - 1])) {
          pos = 0;
        }
      }
      // 2) Armed entry stops.
      if (pos != 1 && i <= buyUntil && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        protStop = finite(atr10[i]) ? L(bars, i) - pcnt * atr10[i] : L(bars, i);
      } else if (pos != -1 && i <= sellUntil && finite(sellLvl) &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        protStop = finite(atr10[i]) ? H(bars, i) + pcnt * atr10[i] : H(bars, i);
      }
      // 3) Setups at this close.
      if (lineUp && mom[i] < 0.f && mom[i] > mom[i - 1]) {
        buyLvl = H(bars, i) + pcnt * (H(bars, i) - xlr[ui]);
        buyUntil = i + 4;
      }
      if (lineDn && mom[i] > 0.f && mom[i] < mom[i - 1]) {
        sellLvl = L(bars, i) - pcnt * (L(bars, i) - xlr[ui]);
        sellUntil = i + 4;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_hpi_breakout ------------------------------------------------------
// Channel breakout on close confirmed by a Herrick-Payoff-style money-flow
// line (mid-price change x volume, smoothed; open interest is unavailable on
// OHLCV data). Entry stops sit at the setup bar's extreme for five bars.
// Exits: 4-bar extreme trail plus a flat-out when the flow line flips sign.
class HpiBreakoutV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_hpi_breakout"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 10, 5, 40, 1, true},
        {"factor", 0.1, 0.02, 0.5, 0.02, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // The HPI flow line and its rolling extremes run as native requests;
    // the extremes chain on the HPI line inside the backend.
    IndicatorRequest hpiReq{Family::HPI, Column::Close, {ps.get("factor")}};
    return {
        {Family::MAX, Column::High, {ps.get("length")}},
        {Family::MIN, Column::Low, {ps.get("length")}},
        {Family::MAX, Column::High, {4}},
        {Family::MIN, Column::Low, {4}},
        hpiReq,
        derived(Family::ROLLING_MAX, {ps.get("length")}, hpiReq),
        derived(Family::ROLLING_MIN, {ps.get("length")}, hpiReq),
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* hiL = ind.line(0);
    const float* loL = ind.line(1);
    const float* hi4 = ind.line(2);
    const float* lo4 = ind.line(3);
    const float* hpi = ind.line(4);
    const float* hpiHi = ind.line(5);
    const float* hpiLo = ind.line(6);
    const int n = bars.size();

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 0; i < n; ++i) {
      const bool ready = finite(hpi[i]) && i >= 1;
      // 1) Exits.
      if (pos == 1 && ready &&
          ((finite(lo4[i - 1]) && L(bars, i) <= lo4[i - 1]) || hpi[i] < 0.f)) {
        pos = 0;
      } else if (pos == -1 && ready &&
                 ((finite(hi4[i - 1]) && H(bars, i) >= hi4[i - 1]) || hpi[i] > 0.f)) {
        pos = 0;
      }
      // 2) Armed breakout stops.
      if (pos != 1 && i <= buyUntil && buy_stop_hit(bars, i, buyLvl)) pos = 1;
      else if (pos != -1 && i <= sellUntil && sell_stop_hit(bars, i, sellLvl)) pos = -1;
      // 3) Setup detection at this close.
      if (ready && finite(hiL[i - 1]) && finite(loL[i - 1]) &&
          finite(hpiHi[i - 1]) && finite(hpiLo[i - 1])) {
        if (C(bars, i) > hiL[i - 1] && hpi[i] > hpiHi[i - 1] && hpi[i] > 0.f &&
            i > buyUntil) {
          buyLvl = H(bars, i);
          buyUntil = i + 5;
        }
        if (C(bars, i) < loL[i - 1] && hpi[i] < hpiLo[i - 1] && hpi[i] < 0.f &&
            i > sellUntil) {
          sellLvl = L(bars, i);
          sellUntil = i + 5;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_consec_close ------------------------------------------------------
// Three consecutive up closes followed by a down close arm a next-open
// breakout stop a quarter-ATR above the open. Exits: 10-bar extreme trail
// plus a two-strike scale-out once price clears the pattern extreme.
class ConsecCloseV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_consec_close"; }

  std::vector<ParamDef> params() const override {
    return {
        {"numbar", 3, 2, 6, 1, true},
        {"avglen", 8, 3, 30, 1, true},
        {"pcnt", 0.25, 0.05, 1.0, 0.05, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::ATR, Column::Close, {ps.get("avglen")}},
        {Family::MAX, Column::High, {10}},
        {Family::MIN, Column::Low, {10}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const float* hi10 = ind.line(1);
    const float* lo10 = ind.line(2);
    const int n = bars.size();
    const int numBar = ps.get_int("numbar");
    const float pcnt = static_cast<float>(ps.get("pcnt"));

    int pos = 0;
    bool buyArmed = false, sellArmed = false;
    float armOfs = 0.f, hh = 0.f, ll = 0.f;
    int lCount = 0, sCount = 0;

    auto upClose = [&](int j) { return j >= 1 && C(bars, j) > C(bars, j - 1); };

    for (int i = 0; i < n; ++i) {
      // 1) Exits.
      if (pos == 1) {
        if (i >= 1 && finite(lo10[i - 1]) && L(bars, i) <= lo10[i - 1]) {
          pos = 0; lCount = 0;
        } else {
          if (O(bars, i) > hh) ++lCount;
          if (lCount >= 2) { pos = 0; lCount = 0; }
        }
      } else if (pos == -1) {
        if (i >= 1 && finite(hi10[i - 1]) && H(bars, i) >= hi10[i - 1]) {
          pos = 0; sCount = 0;
        } else {
          if (C(bars, i) < ll) ++sCount;
          if (sCount >= 2) { pos = 0; sCount = 0; }
        }
      }
      // 2) Stops armed at the previous close, measured off this bar's open.
      if (buyArmed) {
        if (pos != 1 && H(bars, i) >= O(bars, i) + armOfs) { pos = 1; lCount = 0; }
        buyArmed = false;
      }
      if (sellArmed) {
        if (pos != -1 && L(bars, i) <= O(bars, i) - armOfs) { pos = -1; sCount = 0; }
        sellArmed = false;
      }
      // 3) Pattern detection at this close.
      if (i >= numBar + 1 && finite(atr[i])) {
        bool allUp = true, allDn = true;
        for (int k = 1; k <= numBar; ++k) {
          if (!upClose(i - k)) allUp = false;
          if (upClose(i - k)) allDn = false;
        }
        if (allUp && C(bars, i) < C(bars, i - 1)) {
          buyArmed = true;
          armOfs = atr[i] * pcnt;
          float m = H(bars, i);
          for (int k = 1; k <= numBar; ++k) m = std::max(m, H(bars, i - k));
          hh = m;
        }
        if (allDn && C(bars, i) > C(bars, i - 1)) {
          sellArmed = true;
          armOfs = atr[i] * pcnt;
          float m = L(bars, i);
          for (int k = 1; k <= numBar; ++k) m = std::min(m, L(bars, i - k));
          ll = m;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_mom_retrace -------------------------------------------------------
// The volume's flagship: a strong uptrend (EMA channel + MACD signal + RSI +
// DMI spread) followed by a three-way retracement arms a stop a third of the
// way back into the prior swing. Exits: ATR protective stop, swing/EMA
// trail, and a momentum-death close-out. Fully symmetric.
class MomRetraceV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_mom_retrace"; }

  std::vector<ParamDef> params() const override {
    return {
        {"xmalen", 5, 2, 15, 1, true},
        {"chanlen", 15, 5, 40, 1, true},
        {"rsith", 70, 55, 90, 1, true},
        {"dmith", 15, 5, 40, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("xmalen")}},
        {Family::EMA, Column::High, {ps.get("chanlen")}},
        {Family::EMA, Column::Low, {ps.get("chanlen")}},
        {Family::EMA, Column::Close, {3}},
        {Family::EMA, Column::Close, {10}},
        {Family::RSI, Column::Close, {5}},
        {Family::PLUS_DI, Column::Close, {15}},
        {Family::MINUS_DI, Column::Close, {15}},
        {Family::ATR, Column::Close, {10}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* priceXma = ind.line(0);
    const float* highXma = ind.line(1);
    const float* lowXma = ind.line(2);
    const float* ema3 = ind.line(3);
    const float* ema10 = ind.line(4);
    const float* rsi = ind.line(5);
    const float* diPlus = ind.line(6);
    const float* diMinus = ind.line(7);
    const float* atr10 = ind.line(8);
    const int n = bars.size();
    const float rsiTh = static_cast<float>(ps.get("rsith"));
    const float dmiTh = static_cast<float>(ps.get("dmith"));

    // MACD(3,10) with a 15-bar signal line, per the original.
    std::vector<float> macd(static_cast<std::size_t>(n), kNaN);
    for (int i = 0; i < n; ++i)
      if (finite(ema3[i]) && finite(ema10[i]))
        macd[static_cast<std::size_t>(i)] = ema3[i] - ema10[i];
    const std::vector<float> sig = ema_line(macd.data(), n, 15);

    // Long-side state.
    bool lFase1 = false, lFase2 = false, lSwitch = false;
    float highestHi = 0.f, highestRsi = 0.f, lowestLo = 0.f;
    // Short-side state.
    bool sFase1 = false, sFase2 = false, sSwitch = false;
    float lowestLo2 = 0.f, lowestRsi = 0.f, highestHi2 = 0.f;

    int pos = 0;
    float protStop = 0.f, swingStop = kNaN;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      const bool ready = finite(priceXma[i]) && finite(highXma[i]) &&
                         finite(lowXma[i]) && finite(sig[ui]) && finite(rsi[i]) &&
                         finite(diPlus[i]) && finite(diMinus[i]);
      if (!ready) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float dmiSpread = diPlus[i] - diMinus[i];

      // 1) Exits for the open position.
      if (pos == 1) {
        // Swing support (strength 2, confirmed 2 bars later) or the low EMA.
        if (i >= 4) {
          const int b = i - 2;
          bool isSwing = true;
          for (int k = 1; k <= 2; ++k)
            if (!(L(bars, b) < L(bars, b - k) && L(bars, b) <= L(bars, b + k)))
              isSwing = false;
          if (isSwing) swingStop = L(bars, b);
        }
        const float trail = finite(swingStop) ? std::max(swingStop, lowXma[i])
                                              : lowXma[i];
        if (L(bars, i) <= protStop || L(bars, i) <= trail ||
            (sig[ui] < 0.f && dmiSpread < 0.f)) {
          pos = 0;
        }
      } else if (pos == -1) {
        if (i >= 4) {
          const int b = i - 2;
          bool isSwing = true;
          for (int k = 1; k <= 2; ++k)
            if (!(H(bars, b) > H(bars, b - k) && H(bars, b) >= H(bars, b + k)))
              isSwing = false;
          if (isSwing) swingStop = H(bars, b);
        }
        const float trail = finite(swingStop) ? std::min(swingStop, highXma[i])
                                              : highXma[i];
        if (H(bars, i) >= protStop || H(bars, i) >= trail ||
            (sig[ui] > 0.f && dmiSpread > 0.f)) {
          pos = 0;
        }
      }

      // 2) Armed retracement stops.
      if (pos != 1 && lSwitch &&
          buy_stop_hit(bars, i, lowestLo + (highestHi - lowestLo) / 3.f)) {
        pos = 1;
        protStop = finite(atr10[i]) ? C(bars, i) - atr10[i] : L(bars, i);
        swingStop = kNaN;
        lSwitch = false; lFase1 = false; lFase2 = false;
      } else if (pos != -1 && sSwitch &&
                 sell_stop_hit(bars, i, highestHi2 - (highestHi2 - lowestLo2) / 3.f)) {
        pos = -1;
        protStop = finite(atr10[i]) ? C(bars, i) + atr10[i] : H(bars, i);
        swingStop = kNaN;
        sSwitch = false; sFase1 = false; sFase2 = false;
      }

      // 3) Long-side state machine.
      const bool trendUp = priceXma[i] > highXma[i] && sig[ui] > 0.f;
      const bool momUp = rsi[i] >= rsiTh && dmiSpread > dmiTh;
      if (trendUp && momUp) {
        if (!lFase1) { highestHi = 0.f; highestRsi = 0.f; }
        lFase1 = true;
        lFase2 = false;
        highestHi = std::max(highestHi, H(bars, i));
        highestRsi = std::max(highestRsi, rsi[i]);
      }
      const bool lRetr = lFase1 && priceXma[i] < highXma[i] &&
                         finite(macd[ui]) && macd[ui] - sig[ui] < 0.f &&
                         (rsi[i] <= 50.f || rsi[i] < highestRsi - 30.f);
      if (lRetr) {
        if (!lFase2) lowestLo = std::numeric_limits<float>::max();
        lFase2 = true;
        lFase1 = false;
      }
      if (lFase2) {
        lowestLo = std::min(lowestLo, L(bars, i));
        lSwitch = true;
      }
      if (priceXma[i] < lowXma[i] || sig[ui] < 0.f) {
        lFase1 = lFase2 = lSwitch = false;
      }

      // 4) Short-side state machine (mirror).
      const bool trendDn = priceXma[i] < lowXma[i] && sig[ui] < 0.f;
      const bool momDn = rsi[i] <= 100.f - rsiTh && dmiSpread < -dmiTh;
      if (trendDn && momDn) {
        if (!sFase1) { lowestLo2 = std::numeric_limits<float>::max(); lowestRsi = 100.f; }
        sFase1 = true;
        sFase2 = false;
        lowestLo2 = std::min(lowestLo2, L(bars, i));
        lowestRsi = std::min(lowestRsi, rsi[i]);
      }
      const bool sRetr = sFase1 && priceXma[i] > lowXma[i] &&
                         finite(macd[ui]) && macd[ui] - sig[ui] > 0.f &&
                         (rsi[i] >= 50.f || rsi[i] > lowestRsi + 30.f);
      if (sRetr) {
        if (!sFase2) highestHi2 = 0.f;
        sFase2 = true;
        sFase1 = false;
      }
      if (sFase2) {
        highestHi2 = std::max(highestHi2, H(bars, i));
        sSwitch = true;
      }
      if (priceXma[i] > highXma[i] || sig[ui] > 0.f) {
        sFase1 = sFase2 = sSwitch = false;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_three_ema ---------------------------------------------------------
// Fast EMA crossing the medium EMA with the slow EMA confirming enters at
// the next open; losing the fast/medium alignment exits, with a quarter-step
// converging trail as backstop.
class ThreeEmaV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_three_ema"; }

  std::vector<ParamDef> params() const override {
    return {
        {"len1", 6, 2, 20, 1, true},
        {"len2", 12, 4, 40, 1, true},
        {"len3", 28, 8, 80, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("len1") < ps.get_int("len2") &&
           ps.get_int("len2") < ps.get_int("len3");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("len1")}},
        {Family::EMA, Column::Close, {ps.get("len2")}},
        {Family::EMA, Column::Close, {ps.get("len3")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* a1 = ind.line(0);
    const float* a2 = ind.line(1);
    const float* a3 = ind.line(2);
    const int n = bars.size();
    const std::vector<float> ar4 = avg_range(bars, 4);

    int pos = 0;
    bool stopSet = false;
    float stop = 0.f;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (!finite(a1[i]) || !finite(a2[i]) || !finite(a3[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits: alignment loss or converging trail (quarter-step long,
      //    third-step short, as in the original listing).
      if (pos == 1) {
        if (a1[i] < a2[i]) {
          pos = 0; stopSet = false;
        } else if (!stopSet) {
          if (finite(ar4[ui])) { stop = L(bars, i) - ar4[ui]; stopSet = true; }
        } else if (L(bars, i) <= stop) {
          pos = 0; stopSet = false;
        } else {
          stop += (L(bars, i) - stop) / 4.f;
        }
      } else if (pos == -1) {
        if (a1[i] > a2[i]) {
          pos = 0; stopSet = false;
        } else if (!stopSet) {
          if (finite(ar4[ui])) { stop = H(bars, i) + ar4[ui]; stopSet = true; }
        } else if (H(bars, i) >= stop) {
          pos = 0; stopSet = false;
        } else {
          stop -= (stop - H(bars, i)) / 3.f;
        }
      }
      // 2) Entries at this close.
      if (crosses_above(a1, a2, i) && a2[i] > a3[i] && pos != 1) {
        pos = 1; stopSet = false;
      } else if (crosses_below(a1, a2, i) && a2[i] < a3[i] && pos != -1) {
        pos = -1; stopSet = false;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_dma_volume --------------------------------------------------------
// Close crossing a displaced moving average with volume above its own
// displaced average arms a next-open breakout stop 20% of ATR beyond the
// open. Exits: 5-bar extreme trail plus a market exit on the reverse cross.
class DmaVolumeV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_dma_volume"; }

  std::vector<ParamDef> params() const override {
    return {
        {"plen", 7, 3, 30, 1, true},
        {"pdisp", 5, 1, 15, 1, true},
        {"vlen", 7, 3, 30, 1, true},
        {"vdisp", 5, 1, 15, 1, true},
        {"atrpcnt", 0.2, 0.05, 1.0, 0.05, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("plen")}},
        {Family::SMA, Column::Volume, {ps.get("vlen")}},
        {Family::ATR, Column::Close, {10}},
        {Family::MAX, Column::High, {5}},
        {Family::MIN, Column::Low, {5}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* smaC = ind.line(0);
    const float* smaV = ind.line(1);
    const float* atr10 = ind.line(2);
    const float* hi5 = ind.line(3);
    const float* lo5 = ind.line(4);
    const int n = bars.size();
    const int pdisp = ps.get_int("pdisp");
    const int vdisp = ps.get_int("vdisp");
    const float pcnt = static_cast<float>(ps.get("atrpcnt"));

    auto dma = [&](int i) { return i - pdisp >= 0 ? smaC[i - pdisp] : kNaN; };
    auto dmav = [&](int i) { return i - vdisp >= 0 ? smaV[i - vdisp] : kNaN; };

    int pos = 0;
    bool buyArmed = false, sellArmed = false;
    float armOfs = 0.f;

    for (int i = 0; i < n; ++i) {
      const float d = dma(i), dv = dmav(i);
      const float dPrev = i >= 1 ? dma(i - 1) : kNaN;
      const bool ready = finite(d) && finite(dPrev) && finite(dv);
      const bool crossUp = ready && C(bars, i) > d && C(bars, i - 1) <= dPrev;
      const bool crossDn = ready && C(bars, i) < d && C(bars, i - 1) >= dPrev;

      // 1) Exits.
      if (pos == 1 && (crossDn || (i >= 1 && finite(lo5[i - 1]) &&
                                   L(bars, i) <= lo5[i - 1]))) {
        pos = 0;
      } else if (pos == -1 && (crossUp || (i >= 1 && finite(hi5[i - 1]) &&
                                           H(bars, i) >= hi5[i - 1]))) {
        pos = 0;
      }
      // 2) Stops armed at the previous close, measured off this bar's open.
      if (buyArmed) {
        if (pos != 1 && H(bars, i) >= O(bars, i) + armOfs) pos = 1;
        buyArmed = false;
      }
      if (sellArmed) {
        if (pos != -1 && L(bars, i) <= O(bars, i) - armOfs) pos = -1;
        sellArmed = false;
      }
      // 3) Setups at this close.
      if (ready && finite(atr10[i])) {
        if (crossUp && V(bars, i) > dv) {
          buyArmed = true;
          armOfs = pcnt * atr10[i];
        }
        if (crossDn && V(bars, i) < dv) {
          sellArmed = true;
          armOfs = pcnt * atr10[i];
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp04_price_dist --------------------------------------------------------
// Distribution shift: kurtosis of closes crossing under zero (choppy ->
// directional) enters in the direction of the skew. Exits: STAD converging
// trail at full range.
class PriceDistV4 final : public IStrategy {
public:
  std::string name() const override { return "exp04_price_dist"; }

  std::vector<ParamDef> params() const override {
    return {
        {"kurtlen", 40, 10, 100, 5, true},
        {"skewlen", 15, 5, 60, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // Rolling kurtosis/skewness of closes as native (batchable) requests.
    return {
        {Family::KURTOSIS, Column::Close, {ps.get("kurtlen")}},
        {Family::SKEW, Column::Close, {ps.get("skewlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* kurt = ind.line(0);
    const float* skew = ind.line(1);
    const int n = bars.size();
    const std::vector<float> ar4 = avg_range(bars, 4);

    int pos = 0;
    bool trailSet = false;
    float stop = 0.f;
    float prevKurt = kNaN;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) STAD trail.
      if (pos == 1) {
        if (!trailSet) {
          if (finite(ar4[ui])) { stop = L(bars, i) - ar4[ui]; trailSet = true; }
        } else if (L(bars, i) <= stop) {
          pos = 0; trailSet = false;
        } else {
          stop += (L(bars, i) - stop) / 3.f;
        }
      } else if (pos == -1) {
        if (!trailSet) {
          if (finite(ar4[ui])) { stop = H(bars, i) + ar4[ui]; trailSet = true; }
        } else if (H(bars, i) >= stop) {
          pos = 0; trailSet = false;
        } else {
          stop -= (stop - H(bars, i)) / 3.f;
        }
      }
      // 2) Distribution signals at this close.
      if (finite(kurt[i]) && finite(skew[i])) {
        if (finite(prevKurt) && kurt[i] < 0.f && prevKurt >= 0.f) {
          if (skew[i] > 0.f && pos != 1) {
            pos = 1; trailSet = false;
          } else if (skew[i] < 0.f && pos != -1) {
            pos = -1; trailSet = false;
          }
        }
        prevKurt = kurt[i];
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

} // namespace

void register_vol04(std::vector<Factory>& out) {
  out.push_back({"exp04_dmi_adx",
                 [] { return std::unique_ptr<IStrategy>(new DmiAdxV4); }});
  out.push_back({"exp04_keltner",
                 [] { return std::unique_ptr<IStrategy>(new KeltnerV4); }});
  out.push_back({"exp04_linreg_mom",
                 [] { return std::unique_ptr<IStrategy>(new LinRegMomV4); }});
  out.push_back({"exp04_hpi_breakout",
                 [] { return std::unique_ptr<IStrategy>(new HpiBreakoutV4); }});
  out.push_back({"exp04_consec_close",
                 [] { return std::unique_ptr<IStrategy>(new ConsecCloseV4); }});
  out.push_back({"exp04_mom_retrace",
                 [] { return std::unique_ptr<IStrategy>(new MomRetraceV4); }});
  out.push_back({"exp04_three_ema",
                 [] { return std::unique_ptr<IStrategy>(new ThreeEmaV4); }});
  out.push_back({"exp04_dma_volume",
                 [] { return std::unique_ptr<IStrategy>(new DmaVolumeV4); }});
  out.push_back({"exp04_price_dist",
                 [] { return std::unique_ptr<IStrategy>(new PriceDistV4); }});
}

} // namespace topt::exp
