/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 3 (Omega Research STAD Club, Vol. 3).
// Systems: Buy Dips/Sell Rallies, Bullish & Bearish Engulfing, Open-Close
// Histogram, Bollinger Bands, On Balance True Range, Micro-M & Micro-W,
// Variable Zones, Standard Deviation Volatility, Volatility Clusters.
// Skipped from this volume: Inside Day Breakout Failure (needs 30-minute
// bars plus a daily Data2 series). See docs/EXPERIMENTS.md.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// The converging STAD trail shared with volume 2 (duplicated locally to keep
// each volume's translation unit self-contained).
struct Trail {
  bool set = false;
  float stop = 0.f;
  void reset() { set = false; stop = 0.f; }
  bool step_long(const Series& b, int i, float ar, float mult, float conv) {
    if (!set) {
      if (!finite(ar)) return false;
      stop = L(b, i) - ar * mult;
      set = true;
      return false;
    }
    if (L(b, i) <= stop) return true;
    stop += conv * (L(b, i) - stop);
    return false;
  }
  bool step_short(const Series& b, int i, float ar, float mult, float conv) {
    if (!set) {
      if (!finite(ar)) return false;
      stop = H(b, i) + ar * mult;
      set = true;
      return false;
    }
    if (H(b, i) >= stop) return true;
    stop -= conv * (stop - H(b, i));
    return false;
  }
};

// --- exp03_buy_dips ----------------------------------------------------------
// RSI above/below 50 defines the trend; a DMI spread on the wrong side of
// zero flags the countertrend dip/rally and latches a stop at that bar's
// extreme until the opposite setup replaces it. Exits: STAD trail.
class BuyDipsV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_buy_dips"; }

  std::vector<ParamDef> params() const override {
    return {
        {"rsilen", 20, 5, 60, 1, true},
        {"rsilevel", 50, 30, 70, 1, true},
        {"dmilen", 5, 2, 30, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::RSI, Column::Close, {ps.get("rsilen")}},
        {Family::PLUS_DI, Column::Close, {ps.get("dmilen")}},
        {Family::MINUS_DI, Column::Close, {ps.get("dmilen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* rsi = ind.line(0);
    const float* diPlus = ind.line(1);
    const float* diMinus = ind.line(2);
    const int n = bars.size();
    const float level = static_cast<float>(ps.get("rsilevel"));
    const std::vector<float> ar4 = avg_range(bars, 4);

    int pos = 0;
    bool buySetup = false, sellSetup = false;
    float longEntry = 0.f, shortEntry = 0.f;
    Trail trail;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position.
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }
      // 2) Latched entry stops (they also reverse an opposite position).
      if (buySetup && pos != 1 && buy_stop_hit(bars, i, longEntry)) {
        pos = 1; trail.reset();
      } else if (sellSetup && pos != -1 && sell_stop_hit(bars, i, shortEntry)) {
        pos = -1; trail.reset();
      }
      // 3) Setup detection at this close.
      if (finite(rsi[i]) && finite(diPlus[i]) && finite(diMinus[i])) {
        const float spread = diPlus[i] - diMinus[i];
        if (rsi[i] >= level && spread < 0.f) {
          if (!buySetup) longEntry = H(bars, i);
          buySetup = true;
          sellSetup = false;
        } else if (rsi[i] < level && spread > 0.f) {
          if (!sellSetup) shortEntry = L(bars, i);
          sellSetup = true;
          buySetup = false;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_engulfing ---------------------------------------------------------
// Bullish/bearish engulfing candles against the EMA trend arm an entry stop
// at the pattern bar's extreme for three bars. Exits trail at the best close
// since entry -/+ half the 4-bar average range.
class EngulfingV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_engulfing"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 10, 4, 40, 1, true},
        {"bodylen", 10, 4, 40, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::EMA, Column::Close, {ps.get("length")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* ema = ind.line(0);
    const int n = bars.size();
    const int bodyLen = ps.get_int("bodylen");
    const std::vector<float> ar4 = avg_range(bars, 4);

    // Average absolute candle body.
    std::vector<float> avgBody(static_cast<std::size_t>(n), kNaN);
    {
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        sum += std::fabs(static_cast<double>(C(bars, i)) - O(bars, i));
        if (i >= bodyLen)
          sum -= std::fabs(static_cast<double>(C(bars, i - bodyLen)) - O(bars, i - bodyLen));
        if (i >= bodyLen - 1)
          avgBody[static_cast<std::size_t>(i)] = static_cast<float>(sum / bodyLen);
      }
    }

    int pos = 0;
    float longEntry = 0.f, shortEntry = 0.f;
    int longUntil = -1, shortUntil = -1;
    float bestClose = 0.f;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Trailing exit from the best close since entry.
      if (pos != 0 && i >= 1 && finite(ar4[ui - 1])) {
        if (pos == 1 && L(bars, i) <= bestClose - ar4[ui - 1] / 2.f) {
          pos = 0;
        } else if (pos == -1 && H(bars, i) >= bestClose + ar4[ui - 1] / 2.f) {
          pos = 0;
        }
      }
      if (pos == 1) bestClose = std::max(bestClose, C(bars, i));
      if (pos == -1) bestClose = std::min(bestClose, C(bars, i));
      // 2) Armed pattern stops.
      if (pos != 1 && i <= longUntil && buy_stop_hit(bars, i, longEntry)) {
        pos = 1;
        bestClose = C(bars, i);
      } else if (pos != -1 && i <= shortUntil && sell_stop_hit(bars, i, shortEntry)) {
        pos = -1;
        bestClose = C(bars, i);
      }
      // 3) Pattern detection at this close.
      if (i >= 1 && finite(ema[i]) && finite(ema[i - 1]) && finite(avgBody[ui])) {
        const bool upTrend = ema[i] > ema[i - 1];
        const bool downTrend = ema[i] < ema[i - 1];
        const float body = C(bars, i) - O(bars, i);
        const float prevBody = C(bars, i - 1) - O(bars, i - 1);
        const bool bull = body > 0.f && prevBody < 0.f && C(bars, i) > O(bars, i - 1) &&
                          O(bars, i) < C(bars, i - 1) && body > avgBody[ui] * 1.25f &&
                          -prevBody < avgBody[ui] * 0.75f;
        const bool bear = body < 0.f && prevBody > 0.f && C(bars, i) < O(bars, i - 1) &&
                          O(bars, i) > C(bars, i - 1) && -body > avgBody[ui] * 1.25f &&
                          prevBody < avgBody[ui] * 0.75f;
        if (downTrend && bull) {
          longEntry = H(bars, i);
          longUntil = i + 3;
        }
        if (upTrend && bear) {
          shortEntry = L(bars, i);
          shortUntil = i + 3;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_oc_histogram ------------------------------------------------------
// Histogram = EMA(close) - EMA(open). A zero-line cross arms a breakout stop
// half an average true range beyond the crossing bar for ten bars (while the
// histogram keeps its sign); the opposite cross exits at the close, with a
// protective stop from the pre-cross bar.
class OcHistogramV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_oc_histogram"; }

  std::vector<ParamDef> params() const override {
    return {
        {"openlen", 10, 4, 40, 1, true},
        {"closelen", 10, 4, 40, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("closelen")}},
        {Family::EMA, Column::Open, {ps.get("openlen")}},
        {Family::TRANGE, Column::Close, {}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* emaC = ind.line(0);
    const float* emaO = ind.line(1);
    const float* tr = ind.line(2);
    const int n = bars.size();
    const std::vector<float> atr10 = rolling_mean(tr, n, 10);

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f, longExit = 0.f, shortExit = 0.f;
    int buyUntil = -1, sellUntil = -1;

    auto hist = [&](int i) {
      return finite(emaC[i]) && finite(emaO[i]) ? emaC[i] - emaO[i] : kNaN;
    };

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      const float h = hist(i);
      const float hPrev = i >= 1 ? hist(i - 1) : kNaN;
      const bool crossUp = finite(h) && finite(hPrev) && h > 0.f && hPrev <= 0.f;
      const bool crossDn = finite(h) && finite(hPrev) && h < 0.f && hPrev >= 0.f;

      // 1) Exits: opposite zero-cross flattens; protective stop otherwise.
      if (pos == 1 && (crossDn || L(bars, i) <= longExit)) pos = 0;
      else if (pos == -1 && (crossUp || H(bars, i) >= shortExit)) pos = 0;

      // 2) Armed breakout stops (only while the histogram keeps its sign).
      if (pos != 1 && i <= buyUntil && finite(h) && h > 0.f &&
          buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
      } else if (pos != -1 && i <= sellUntil && finite(h) && h < 0.f &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
      }

      // 3) Fresh crossings arm new stops.
      if (crossUp && finite(atr10[ui])) {
        buyLvl = H(bars, i) + atr10[ui] * 0.5f;
        buyUntil = i + 10;
        if (i >= 1 && finite(atr10[ui - 1]))
          longExit = L(bars, i - 1) - atr10[ui - 1] / 2.f;
      }
      if (crossDn && finite(atr10[ui])) {
        sellLvl = L(bars, i) - atr10[ui] * 0.5f;
        sellUntil = i + 10;
        if (i >= 1 && finite(atr10[ui - 1]))
          shortExit = H(bars, i - 1) + atr10[ui - 1] / 2.f;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_bollinger ---------------------------------------------------------
// Mean reversion: two consecutive closes beyond a Bollinger band enter
// against the move at the next open (the original's scale-in limit orders are
// dropped in this single-position engine). Exits: STAD trail.
class BollingerV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_bollinger"; }

  std::vector<ParamDef> params() const override {
    return {
        {"bblen", 10, 5, 40, 1, true},
        {"bbdev", 2.0, 1.0, 3.0, 0.25, false},
        {"numbars", 2, 1, 5, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double dev = ps.get("bbdev");
    return {{Family::BBANDS, Column::Close, {ps.get("bblen"), dev, dev}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* upper = ind.line(0, 0);
    const float* lower = ind.line(0, 2);
    const int n = bars.size();
    const int numBars = ps.get_int("numbars");
    const std::vector<float> ar4 = avg_range(bars, 4);

    int pos = 0;
    Trail trail;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }

      bool below = true, above = true;
      for (int k = 0; k < numBars; ++k) {
        const int j = i - k;
        if (j < 0 || !finite(lower[j]) || !finite(upper[j])) {
          below = above = false;
          break;
        }
        if (!(C(bars, j) < lower[j])) below = false;
        if (!(C(bars, j) > upper[j])) above = false;
      }
      if (below && pos != 1) {
        pos = 1; trail.reset();
      } else if (above && pos != -1) {
        pos = -1; trail.reset();
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_obtr --------------------------------------------------------------
// The Bollinger dip-buy gated by quiet volatility: only when the on-balance
// true range sits under its own EMA. Long-only; exits with the STAD trail.
class ObtrV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_obtr"; }

  std::vector<ParamDef> params() const override {
    return {
        {"bblen", 10, 5, 40, 1, true},
        {"bbdev", 2.0, 1.0, 3.0, 0.25, false},
        {"obtrlen", 9, 3, 30, 1, true},
        {"numbars", 2, 1, 5, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double dev = ps.get("bbdev");
    // OBTR and its EL-average run as native (batchable) requests; the
    // XAverage chains on the OBTR line without leaving the backend.
    IndicatorRequest obtrReq{Family::OBTR, Column::Close, {}};
    return {
        {Family::BBANDS, Column::Close, {ps.get("bblen"), dev, dev}},
        obtrReq,
        derived(Family::XAVERAGE, {ps.get("obtrlen")}, obtrReq),
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* lower = ind.line(0, 2);
    const float* obtr = ind.line(1);
    const float* obtrEma = ind.line(2);
    const int n = bars.size();
    const int numBars = ps.get_int("numbars");
    const std::vector<float> ar4 = avg_range(bars, 4);

    int pos = 0;
    Trail trail;
    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }
      if (pos != 1 && finite(obtr[i]) && finite(obtrEma[i]) &&
          obtr[i] < obtrEma[i]) {
        bool below = true;
        for (int k = 0; k < numBars; ++k) {
          const int j = i - k;
          if (j < 0 || !finite(lower[j]) || !(C(bars, j) < lower[j])) {
            below = false;
            break;
          }
        }
        if (below) {
          pos = 1;
          trail.reset();
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_micro_mw ----------------------------------------------------------
// Micro-M tops / Micro-W bottoms: a price/oscillator swing divergence
// (RSI oscillator) followed within a few bars by the two-bar close hook arms
// a 4-bar-extreme entry stop. Exits: STAD trail.
class MicroMwV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_micro_mw"; }

  std::vector<ParamDef> params() const override {
    return {
        {"osclen", 10, 5, 30, 1, true},
        {"strength", 4, 2, 8, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::RSI, Column::Close, {ps.get("osclen")}},
        {Family::MAX, Column::High, {4}},
        {Family::MIN, Column::Low, {4}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* osc = ind.line(0);
    const float* hi4 = ind.line(1);
    const float* lo4 = ind.line(2);
    const int n = bars.size();
    const int s = ps.get_int("strength");
    const std::vector<float> ar4 = avg_range(bars, 4);

    struct Swing { int bar; float val; };
    std::deque<Swing> priceLows, oscLows, priceHighs, oscHighs;
    auto push = [](std::deque<Swing>& q, int bar, float v) {
      q.push_back({bar, v});
      if (q.size() > 11) q.pop_front();
    };

    int foundLowBar = -1000, foundHighBar = -1000;
    int pos = 0;
    Trail trail;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position.
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }

      // 2) Armed entry stops from the previous close.
      if (i >= 1) {
        const bool wPattern = i >= 3 &&
            ((C(bars, i - 2) > C(bars, i - 3) && C(bars, i - 1) < C(bars, i - 2)) ||
             (i >= 4 && C(bars, i - 3) > C(bars, i - 4) && C(bars, i - 2) < C(bars, i - 3)));
        const bool mPattern = i >= 3 &&
            ((C(bars, i - 2) < C(bars, i - 3) && C(bars, i - 1) > C(bars, i - 2)) ||
             (i >= 4 && C(bars, i - 3) < C(bars, i - 4) && C(bars, i - 2) > C(bars, i - 3)));
        if (pos != 1 && i - 1 - foundLowBar <= 5 && wPattern &&
            finite(hi4[i - 1]) && buy_stop_hit(bars, i, hi4[i - 1])) {
          pos = 1; trail.reset();
        } else if (pos != -1 && i - 1 - foundHighBar <= 5 && mPattern &&
                   finite(lo4[i - 1]) && sell_stop_hit(bars, i, lo4[i - 1])) {
          pos = -1; trail.reset();
        }
      }

      // 3) Swing bookkeeping (pivot confirmed `s` bars later).
      const int b = i - s;
      if (b >= s && finite(osc[i])) {
        bool priceLow = true, priceHigh = true, oscLow = true, oscHigh = true;
        for (int k = 1; k <= s; ++k) {
          if (!(L(bars, b) < L(bars, b - k) && L(bars, b) <= L(bars, b + k)))
            priceLow = false;
          if (!(H(bars, b) > H(bars, b - k) && H(bars, b) >= H(bars, b + k)))
            priceHigh = false;
          if (!finite(osc[b]) || !finite(osc[b - k]) || !finite(osc[b + k]) ||
              !(osc[b] < osc[b - k] && osc[b] <= osc[b + k]))
            oscLow = false;
          if (!finite(osc[b]) || !finite(osc[b - k]) || !finite(osc[b + k]) ||
              !(osc[b] > osc[b - k] && osc[b] >= osc[b + k]))
            oscHigh = false;
        }
        bool newSwing = false;
        if (priceLow) { push(priceLows, b, L(bars, b)); newSwing = true; }
        if (oscLow) { push(oscLows, b, osc[b]); newSwing = true; }
        if (priceHigh) { push(priceHighs, b, H(bars, b)); newSwing = true; }
        if (oscHigh) { push(oscHighs, b, osc[b]); newSwing = true; }

        // Divergence between the newest mated pair and any older mated pair.
        auto divergent = [](const std::deque<Swing>& p, const std::deque<Swing>& o) {
          if (p.empty() || o.empty()) return false;
          const Swing& p0 = p.back();
          const Swing& o0 = o.back();
          if (std::abs(p0.bar - o0.bar) > 5) return false;
          for (std::size_t a = 0; a + 1 < p.size(); ++a) {
            for (std::size_t c = 0; c + 1 < o.size(); ++c) {
              if (std::abs(p[a].bar - o[c].bar) > 5) continue;
              if ((p0.val > p[a].val && o0.val < o[c].val) ||
                  (p0.val < p[a].val && o0.val > o[c].val))
                return true;
            }
          }
          return false;
        };
        if (newSwing) {
          if ((priceLow || oscLow) && divergent(priceLows, oscLows))
            foundLowBar = i;
          if ((priceHigh || oscHigh) && divergent(priceHighs, oscHighs))
            foundHighBar = i;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_var_zones ---------------------------------------------------------
// Overbought/oversold zones rebuilt from the oscillator's own swing history:
// the RSI crossing up out of the adaptive bottom band buys, crossing down
// out of the top band sells. Exits: STAD trail.
class VarZonesV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_var_zones"; }

  std::vector<ParamDef> params() const override {
    return {
        {"osclen", 20, 5, 60, 1, true},
        {"swings", 4, 2, 8, 1, true},
        {"strength", 5, 2, 8, 1, true},
        {"factor", 0.75, 0.25, 1.5, 0.25, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {{Family::RSI, Column::Close, {ps.get("osclen")}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* osc = ind.line(0);
    const int n = bars.size();
    const int s = ps.get_int("strength");
    const int swings = ps.get_int("swings");
    const float factor = static_cast<float>(ps.get("factor"));
    const std::vector<float> ar4 = avg_range(bars, 4);

    std::deque<float> swingHighs, swingLows;
    float topBand = kNaN, botBand = kNaN;
    float prevOsc = kNaN, prevTop = kNaN, prevBot = kNaN;
    int pos = 0;
    Trail trail;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position.
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }

      // 2) Update swing history (pivot confirmed `s` bars later).
      const int b = i - s;
      if (b >= s && finite(osc[b])) {
        bool isHigh = true, isLow = true;
        for (int k = 1; k <= s; ++k) {
          if (!finite(osc[b - k]) || !finite(osc[b + k])) { isHigh = isLow = false; break; }
          if (!(osc[b] > osc[b - k] && osc[b] >= osc[b + k])) isHigh = false;
          if (!(osc[b] < osc[b - k] && osc[b] <= osc[b + k])) isLow = false;
        }
        if (isHigh) {
          swingHighs.push_back(osc[b]);
          if (static_cast<int>(swingHighs.size()) > swings) swingHighs.pop_front();
        }
        if (isLow) {
          swingLows.push_back(osc[b]);
          if (static_cast<int>(swingLows.size()) > swings) swingLows.pop_front();
        }
        if (static_cast<int>(swingHighs.size()) == swings) {
          float sum = 0.f, mx = swingHighs.front();
          for (float v : swingHighs) { sum += v; mx = std::max(mx, v); }
          const float avg = sum / swings;
          topBand = avg + factor * (mx - avg);
        }
        if (static_cast<int>(swingLows.size()) == swings) {
          float sum = 0.f, mn = swingLows.front();
          for (float v : swingLows) { sum += v; mn = std::min(mn, v); }
          const float avg = sum / swings;
          botBand = avg - factor * (avg - mn);
        }
      }

      // 3) Band crossings at this close.
      if (finite(osc[i]) && finite(prevOsc)) {
        if (finite(botBand) && finite(prevBot) && osc[i] > botBand &&
            prevOsc <= prevBot && pos != 1) {
          pos = 1; trail.reset();
        } else if (finite(topBand) && finite(prevTop) && osc[i] < topBand &&
                   prevOsc >= prevTop && pos != -1) {
          pos = -1; trail.reset();
        }
      }
      prevOsc = osc[i];
      prevTop = topBand;
      prevBot = botBand;
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_stddev_vol --------------------------------------------------------
// Always-armed volatility stops: entry levels sit a band-width fraction away
// from the close on both sides, every bar. Exits at the first profitable
// open (position flips are the only other way out).
class StdDevVolV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_stddev_vol"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 9, 4, 40, 1, true},
        {"sdev", 2.0, 1.0, 3.0, 0.25, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("length")}},
        {Family::STDDEV, Column::Close, {ps.get("length")}},
        {Family::SMA, Column::Close, {4}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* sma = ind.line(0);
    const float* sd = ind.line(1);
    const float* sma4 = ind.line(2);
    const int n = bars.size();
    const float mult = static_cast<float>(ps.get("sdev"));

    int pos = 0;
    float entryRef = 0.f;
    int fillBar = -1;
    float buyLvl = kNaN, sellLvl = kNaN;

    for (int i = 0; i < n; ++i) {
      // 1) First profitable open.
      if (pos == 1 && i > fillBar && O(bars, i) > entryRef) pos = 0;
      else if (pos == -1 && i > fillBar && O(bars, i) < entryRef) pos = 0;

      // 2) Stops armed at the previous close.
      if (finite(buyLvl) && finite(sellLvl)) {
        const bool up = pos != 1 && buy_stop_hit(bars, i, buyLvl);
        const bool dn = pos != -1 && sell_stop_hit(bars, i, sellLvl);
        if (up && dn) {
          pos = std::abs(C(bars, i) - buyLvl) <= std::abs(C(bars, i) - sellLvl) ? 1 : -1;
          entryRef = C(bars, i);
          fillBar = i + 1;
        } else if (up) {
          pos = 1; entryRef = C(bars, i); fillBar = i + 1;
        } else if (dn) {
          pos = -1; entryRef = C(bars, i); fillBar = i + 1;
        }
      }

      // 3) Re-arm both stops at this close.
      if (finite(sma[i]) && finite(sd[i]) && finite(sma4[i]) && sma4[i] > 0.f) {
        const float bandRange = 2.f * mult * sd[i];
        buyLvl = C(bars, i) * (1.f + bandRange / sma4[i]);
        sellLvl = C(bars, i) * (1.f - bandRange / sma4[i]);
      } else {
        buyLvl = sellLvl = kNaN;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp03_vol_clusters ------------------------------------------------------
// A close-to-close jump or bar range beyond mean + k standard deviations
// flags a volatility cluster; both breakout stops go an average-range away
// from that close for the next bar. Exits at the first profitable open.
class VolClustersV3 final : public IStrategy {
public:
  std::string name() const override { return "exp03_vol_clusters"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 18, 6, 60, 1, true},
        {"sdev", 2.0, 1.0, 3.0, 0.25, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // Close-to-close and range statistics are computed from raw bars.
    (void)ps;
    return {};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ind;
    const int n = bars.size();
    const int length = ps.get_int("length");
    const float mult = static_cast<float>(ps.get("sdev"));
    const std::vector<float> ar3 = avg_range(bars, 3);

    std::vector<float> ctc(static_cast<std::size_t>(n), kNaN);
    std::vector<float> rng(static_cast<std::size_t>(n), kNaN);
    for (int i = 0; i < n; ++i) {
      if (i >= 1) ctc[static_cast<std::size_t>(i)] = std::fabs(C(bars, i) - C(bars, i - 1));
      rng[static_cast<std::size_t>(i)] = H(bars, i) - L(bars, i);
    }
    const std::vector<float> ctcMean = rolling_mean(ctc.data(), n, length);
    const std::vector<float> ctcStd = rolling_std(ctc.data(), n, length);
    const std::vector<float> rngMean = rolling_mean(rng.data(), n, length);
    const std::vector<float> rngStd = rolling_std(rng.data(), n, length);

    int pos = 0;
    float entryRef = 0.f;
    int fillBar = -1;
    float buyLvl = kNaN, sellLvl = kNaN;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) First profitable open.
      if (pos == 1 && i > fillBar && O(bars, i) > entryRef) pos = 0;
      else if (pos == -1 && i > fillBar && O(bars, i) < entryRef) pos = 0;

      // 2) Stops armed at the previous close (single-bar validity).
      if (finite(buyLvl) && finite(sellLvl)) {
        const bool up = pos != 1 && buy_stop_hit(bars, i, buyLvl);
        const bool dn = pos != -1 && sell_stop_hit(bars, i, sellLvl);
        if (up && dn) {
          pos = std::abs(C(bars, i) - buyLvl) <= std::abs(C(bars, i) - sellLvl) ? 1 : -1;
          entryRef = C(bars, i);
          fillBar = i + 1;
        } else if (up) {
          pos = 1; entryRef = C(bars, i); fillBar = i + 1;
        } else if (dn) {
          pos = -1; entryRef = C(bars, i); fillBar = i + 1;
        }
      }
      buyLvl = sellLvl = kNaN;

      // 3) Cluster detection at this close arms next-bar stops.
      if (i >= 1 && finite(ctc[ui]) && finite(ctcMean[ui - 1]) &&
          finite(ctcStd[ui - 1]) && finite(rngMean[ui - 1]) && finite(rngStd[ui - 1]) &&
          finite(ar3[ui - 1])) {
        const bool big = ctc[ui] > ctcMean[ui - 1] + mult * ctcStd[ui - 1] ||
                         rng[ui] > rngMean[ui - 1] + mult * rngStd[ui - 1];
        if (big) {
          buyLvl = C(bars, i) + ar3[ui - 1];
          sellLvl = C(bars, i) - ar3[ui - 1];
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

} // namespace

void register_vol03(std::vector<Factory>& out) {
  out.push_back({"exp03_buy_dips",
                 [] { return std::unique_ptr<IStrategy>(new BuyDipsV3); }});
  out.push_back({"exp03_engulfing",
                 [] { return std::unique_ptr<IStrategy>(new EngulfingV3); }});
  out.push_back({"exp03_oc_histogram",
                 [] { return std::unique_ptr<IStrategy>(new OcHistogramV3); }});
  out.push_back({"exp03_bollinger",
                 [] { return std::unique_ptr<IStrategy>(new BollingerV3); }});
  out.push_back({"exp03_obtr",
                 [] { return std::unique_ptr<IStrategy>(new ObtrV3); }});
  out.push_back({"exp03_micro_mw",
                 [] { return std::unique_ptr<IStrategy>(new MicroMwV3); }});
  out.push_back({"exp03_var_zones",
                 [] { return std::unique_ptr<IStrategy>(new VarZonesV3); }});
  out.push_back({"exp03_stddev_vol",
                 [] { return std::unique_ptr<IStrategy>(new StdDevVolV3); }});
  out.push_back({"exp03_vol_clusters",
                 [] { return std::unique_ptr<IStrategy>(new VolClustersV3); }});
}

} // namespace topt::exp
