/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 13 (Omega Research STAD Club, Vol. 13).
// Systems: Open-Close Histogram, DMA & Range Leaders, Linear Regression &
// Momentum, Momentum Retracement, VolEx, Volatility Clusters.
// Skipped from this volume: Luxor (pyramiding variant of the Vol. 9 system
// already implemented as exp09_luxor), Currency/Bonds/Dollar Index and
// Advance-Decline Divergence (need secondary data series), Triple Play
// (already implemented from Vol. 11), chapters 1-3 (educational).

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// --- exp13_oc_histogram ------------------------------------------------------
// EMA(close) minus EMA(open) as a trend histogram. A zero-line upcross arms
// a stop at that bar's high plus a fraction of ATR for ten bars; the
// protective stop sits below the crossover bar, and an opposite zero-cross
// exits at that close.
class OcHistogramV13 final : public IStrategy {
public:
  std::string name() const override { return "exp13_oc_histogram"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 10, 5, 40, 1, true},
        {"entryfactor", 0.5, 0.25, 1.5, 0.25, false},
        {"exitfactor", 0.5, 0.25, 1.5, 0.25, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("length")}},
        {Family::EMA, Column::Open, {ps.get("length")}},
        {Family::ATR, Column::Close, {ps.get("length")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* emaC = ind.line(0);
    const float* emaO = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const float entryF = static_cast<float>(ps.get("entryfactor"));
    const float exitF = static_cast<float>(ps.get("exitfactor"));
    constexpr int kWindow = 10;

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN, lStop = kNaN, sStop = kNaN;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 1; i < n; ++i) {
      if (!finite(emaC[i]) || !finite(emaO[i]) || !finite(emaC[i - 1]) ||
          !finite(emaO[i - 1]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float hist = emaC[i] - emaO[i];
      const float histPrev = emaC[i - 1] - emaO[i - 1];
      // 1) Protective stops from the crossover bar.
      if (pos == 1 && finite(lStop) && L(bars, i) <= lStop) pos = 0;
      else if (pos == -1 && finite(sStop) && H(bars, i) >= sStop) pos = 0;
      // 2) Armed entries while the histogram stays on the right side.
      if (pos != 1 && i <= buyUntil && finite(buyLvl) && histPrev > 0.f &&
          buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && finite(sellLvl) && histPrev < 0.f &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sellUntil = -1;
      }
      // 3) Zero-line crossings: exit at this close and arm the reverse side.
      if (histPrev <= 0.f && hist > 0.f) {
        if (pos == -1) pos = 0;
        buyLvl = H(bars, i) + atr[i] * entryF;
        lStop = L(bars, i) - atr[i] * exitF;
        buyUntil = i + kWindow;
        sellUntil = -1;
      } else if (histPrev >= 0.f && hist < 0.f) {
        if (pos == 1) pos = 0;
        sellLvl = L(bars, i) - atr[i] * entryF;
        sStop = H(bars, i) + atr[i] * exitF;
        sellUntil = i + kWindow;
        buyUntil = -1;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp13_dma_range_leaders -------------------------------------------------
// Displaced moving-average channel plus the Range Leader bar: a wide-range
// bar whose midpoint clears the previous high closing above the displaced
// high average buys at that bar's high. Dollar stops of the original are
// replaced by ATR protective/breakeven/trailing stops.
class DmaRangeLeadersV13 final : public IStrategy {
public:
  std::string name() const override { return "exp13_dma_range_leaders"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 20, 10, 50, 2, true},
        {"displace", 5, 1, 12, 1, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"breakatrs", 3.0, 1.0, 8.0, 0.5, false},
        {"trailatrs", 4.0, 1.0, 8.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::High, {ps.get("length")}},
        {Family::SMA, Column::Low, {ps.get("length")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* avgH = ind.line(0);
    const float* avgL = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const int disp = ps.get_int("displace");
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = static_cast<float>(ps.get("breakatrs"));
    const float trail = static_cast<float>(ps.get("trailatrs"));

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    float entryPx = kNaN, closeHigh = kNaN, closeLow = kNaN;

    for (int i = 1; i < n; ++i) {
      const int d = i - disp;
      if (d < 0 || !finite(avgH[d]) || !finite(avgL[d]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) ATR exit stack.
      if (pos == 1) {
        float stop = entryPx - prot * atrPrev;
        stop = std::max(stop, closeHigh - trail * atrPrev);
        if (closeHigh >= entryPx + brk * atrPrev) stop = std::max(stop, entryPx);
        if (L(bars, i) <= stop) pos = 0;
        closeHigh = std::max(closeHigh, C(bars, i));
      } else if (pos == -1) {
        float stop = entryPx + prot * atrPrev;
        stop = std::min(stop, closeLow + trail * atrPrev);
        if (closeLow <= entryPx - brk * atrPrev) stop = std::min(stop, entryPx);
        if (H(bars, i) >= stop) pos = 0;
        closeLow = std::min(closeLow, C(bars, i));
      }
      // 2) Armed Range Leader entries.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        closeHigh = C(bars, i);
      } else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        closeLow = C(bars, i);
      }
      // 3) Range Leader detection at this close (displaced channel).
      const float mid = (H(bars, i) + L(bars, i)) / 2.f;
      const float rng = H(bars, i) - L(bars, i);
      const float rngPrev = H(bars, i - 1) - L(bars, i - 1);
      buyLvl = sellLvl = kNaN;
      if (mid > H(bars, i - 1) && rng > rngPrev && C(bars, i) > avgH[d])
        buyLvl = H(bars, i);
      if (mid < L(bars, i - 1) && rng > rngPrev && C(bars, i) < avgL[d])
        sellLvl = L(bars, i);
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp13_linreg_momentum ---------------------------------------------------
// Smoothed linear-regression trend plus a momentum pullback: EMA of the LR
// endpoint rising, close above it, momentum negative but turning up arms an
// ATR-offset stop above the setup bar for a few bars. The common five-stop
// ATR suite manages the position.
class LinRegMomV13 final : public IStrategy {
public:
  std::string name() const override { return "exp13_linreg_momentum"; }

  std::vector<ParamDef> params() const override {
    return {
        {"regression", 20, 10, 60, 5, true},
        {"xmalen", 15, 5, 40, 5, true},
        {"momlen", 10, 5, 30, 1, true},
        {"entrypct", 0.25, 0.1, 1.0, 0.05, false},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
        {"bigatrs", 8.0, 4.0, 12.0, 1.0, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::LINEARREG, Column::Close, {ps.get("regression")}},
        {Family::MOM, Column::Close, {ps.get("momlen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* lr = ind.line(0);
    const float* mom = ind.line(1);
    const float* atr = ind.line(2);
    const int n = bars.size();
    const int xmaLen = ps.get_int("xmalen");
    const float entryPct = static_cast<float>(ps.get("entrypct"));
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = 3.f, trail = 5.f, vol = 3.f;
    const float big = static_cast<float>(ps.get("bigatrs"));
    const int exitBars = 4, entryWindow = 4;

    const std::vector<float> xlr = ema_line(lr, n, xmaLen);

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;
    int buyUntil = -1, sellUntil = -1;
    float entryPx = kNaN, closeHigh = kNaN, closeLow = kNaN;

    for (int i = 1; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (!finite(xlr[ui]) || !finite(xlr[ui - 1]) || !finite(mom[i]) ||
          !finite(mom[i - 1]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) Five-stop ATR suite.
      if (pos == 1) {
        float stop = entryPx - prot * atrPrev;
        if (closeHigh >= entryPx + brk * atrPrev) stop = std::max(stop, entryPx);
        stop = std::max(stop, closeHigh - trail * atrPrev);
        stop = std::max(stop, C(bars, i - 1) - vol * atrPrev);
        if (closeHigh >= entryPx + big * atrPrev) {
          float lo = L(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            lo = std::min(lo, L(bars, i - 1 - k));
          stop = std::max(stop, lo);
        }
        if (L(bars, i) <= stop) pos = 0;
        closeHigh = std::max(closeHigh, C(bars, i));
      } else if (pos == -1) {
        float stop = entryPx + prot * atrPrev;
        if (closeLow <= entryPx - brk * atrPrev) stop = std::min(stop, entryPx);
        stop = std::min(stop, closeLow + trail * atrPrev);
        stop = std::min(stop, C(bars, i - 1) + vol * atrPrev);
        if (closeLow <= entryPx - big * atrPrev) {
          float hi = H(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            hi = std::max(hi, H(bars, i - 1 - k));
          stop = std::min(stop, hi);
        }
        if (H(bars, i) >= stop) pos = 0;
        closeLow = std::min(closeLow, C(bars, i));
      }
      // 2) Armed entries within the setup window.
      if (pos != 1 && i <= buyUntil && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        closeHigh = C(bars, i);
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && finite(sellLvl) &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        closeLow = C(bars, i);
        sellUntil = -1;
      }
      // 3) Setups at this close.
      if (xlr[ui] > xlr[ui - 1] && C(bars, i) > xlr[ui] && mom[i] < 0.f &&
          mom[i] > mom[i - 1]) {
        buyLvl = H(bars, i) + entryPct * atr[i];
        buyUntil = i + entryWindow;
      }
      if (xlr[ui] < xlr[ui - 1] && C(bars, i) < xlr[ui] && mom[i] > 0.f &&
          mom[i] < mom[i - 1]) {
        sellLvl = L(bars, i) - entryPct * atr[i];
        sellUntil = i + entryWindow;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp13_mom_retrace -------------------------------------------------------
// Momentum-Retracement (MRTS): EMA channel + MACD signal define the trend,
// RSI/DMI-spread momentum arms a retracement check; a decline into the
// channel, a MACD downcross and an RSI drop within the window of the trend
// extreme arm a two-bar-high stop entry. ATR stop suite plus a
// trend-failure market exit.
class MomRetraceV13 final : public IStrategy {
public:
  std::string name() const override { return "exp13_mom_retrace"; }

  std::vector<ParamDef> params() const override {
    return {
        {"channellen", 15, 5, 30, 1, true},
        {"closelen", 5, 3, 15, 1, true},
        {"rsilen", 5, 3, 14, 1, true},
        {"dmilen", 15, 5, 30, 1, true},
        {"retracebars", 10, 5, 20, 1, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::High, {ps.get("channellen")}},
        {Family::EMA, Column::Low, {ps.get("channellen")}},
        {Family::EMA, Column::Close, {ps.get("closelen")}},
        {Family::MACD, Column::Close, {3.0, 10.0, 15.0}},
        {Family::RSI, Column::Close, {ps.get("rsilen")}},
        {Family::PLUS_DI, Column::Close, {ps.get("dmilen")}},
        {Family::MINUS_DI, Column::Close, {ps.get("dmilen")}},
        {Family::ATR, Column::Close, {10.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiBand = ind.line(0);
    const float* loBand = ind.line(1);
    const float* closeLine = ind.line(2);
    const float* macd = ind.line(3, 0);
    const float* sig = ind.line(3, 1);
    const float* rsi = ind.line(4);
    const float* diP = ind.line(5);
    const float* diM = ind.line(6);
    const float* atr = ind.line(7);
    const int n = bars.size();
    const int retraceBars = ps.get_int("retracebars");
    const float prot = static_cast<float>(ps.get("protatrs"));
    const float brk = 4.f, trail = 6.f, vol = 3.f, big = 9.f;
    const int exitBars = 4, bestEntry = 2;
    const float obZone = 70.f, osZone = 30.f, rsiChange = 30.f;

    int pos = 0;
    bool chanUp = false, chanDn = false;
    bool upTrendPrev = false, dnTrendPrev = false;
    float upHigh = kNaN, dnLow = kNaN;
    int highCounter = 0, lowCounter = 0;
    float highRsi = kNaN, lowRsi = kNaN;
    bool checkRetrace = false, ctDecline = false, ctRally = false;
    int lastDipBar = -1, lastMacdDn = -1, lastRsiDn = -1;
    int lastPopBar = -1, lastMacdUp = -1, lastRsiUp = -1;
    float buyLvl = kNaN, sellLvl = kNaN;
    float entryPx = kNaN, closeHigh = kNaN, closeLow = kNaN;
    bool pendingExit = false;

    for (int i = 1; i < n; ++i) {
      if (!finite(hiBand[i]) || !finite(loBand[i]) || !finite(closeLine[i]) ||
          !finite(macd[i]) || !finite(sig[i]) || !finite(rsi[i]) || !finite(diP[i]) ||
          !finite(diM[i]) || !finite(atr[i]) || !finite(rsi[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const float atrPrev = finite(atr[i - 1]) ? atr[i - 1] : atr[i];
      // 1) Pending trend-failure market exit fills at this open.
      if (pendingExit && pos != 0) pos = 0;
      pendingExit = false;
      // 2) ATR stop suite.
      if (pos == 1) {
        float stop = entryPx - prot * atrPrev;
        if (closeHigh >= entryPx + brk * atrPrev) stop = std::max(stop, entryPx);
        stop = std::max(stop, closeHigh - trail * atrPrev);
        stop = std::max(stop, C(bars, i - 1) - vol * atrPrev);
        if (closeHigh >= entryPx + big * atrPrev) {
          float lo = L(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            lo = std::min(lo, L(bars, i - 1 - k));
          stop = std::max(stop, lo);
        }
        if (L(bars, i) <= stop) pos = 0;
        closeHigh = std::max(closeHigh, C(bars, i));
      } else if (pos == -1) {
        float stop = entryPx + prot * atrPrev;
        if (closeLow <= entryPx - brk * atrPrev) stop = std::min(stop, entryPx);
        stop = std::min(stop, closeLow + trail * atrPrev);
        stop = std::min(stop, C(bars, i - 1) + vol * atrPrev);
        if (closeLow <= entryPx - big * atrPrev) {
          float hi = H(bars, i - 1);
          for (int k = 1; k < exitBars && i - 1 - k >= 0; ++k)
            hi = std::max(hi, H(bars, i - 1 - k));
          stop = std::min(stop, hi);
        }
        if (H(bars, i) >= stop) pos = 0;
        closeLow = std::min(closeLow, C(bars, i));
      }
      // 3) Armed retracement entries.
      if (pos != 1 && ctDecline && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
        closeHigh = C(bars, i);
        ctDecline = false;
      } else if (pos != -1 && ctRally && finite(sellLvl) &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
        closeLow = C(bars, i);
        ctRally = false;
      }
      if (pos != 0) {
        checkRetrace = false;
        if (pos == 1) ctDecline = false;
        else ctRally = false;
      }
      // 4) Trend bookkeeping at this close.
      if (crosses_above(closeLine, hiBand, i)) {
        chanUp = true;
        chanDn = false;
      }
      if (crosses_below(closeLine, loBand, i)) {
        chanDn = true;
        chanUp = false;
      }
      const bool upTrend = chanUp && sig[i] > 0.f;
      const bool dnTrend = chanDn && sig[i] < 0.f;
      const float spread = diP[i] - diM[i];
      // Condition trackers for the retracement window.
      if (L(bars, i) < hiBand[i]) lastDipBar = i;
      if (crosses_below(macd, sig, i)) lastMacdDn = i;
      if ((rsi[i - 1] >= 50.f && rsi[i] < 50.f) ||
          (finite(highRsi) && rsi[i - 1] >= highRsi - rsiChange && rsi[i] < highRsi - rsiChange))
        lastRsiDn = i;
      if (H(bars, i) > loBand[i]) lastPopBar = i;
      if (crosses_above(macd, sig, i)) lastMacdUp = i;
      if ((rsi[i - 1] <= 50.f && rsi[i] > 50.f) ||
          (finite(lowRsi) && rsi[i - 1] <= lowRsi + rsiChange && rsi[i] > lowRsi + rsiChange))
        lastRsiUp = i;
      if (upTrend) {
        if (!upTrendPrev) {
          upHigh = H(bars, i);
          highCounter = 0;
          highRsi = rsi[i];
          checkRetrace = false;
          ctDecline = false;
        } else {
          if (H(bars, i) > upHigh) {
            upHigh = H(bars, i);
            highCounter = 0;
          } else if (highCounter < retraceBars) {
            ++highCounter;
          }
          if (rsi[i] > highRsi) highRsi = rsi[i];
        }
        if (rsi[i] > obZone && spread > 15.f) checkRetrace = true;
        if (checkRetrace && highCounter != 0 && highCounter < retraceBars) {
          const int from = i - highCounter;
          if (lastDipBar >= from && lastMacdDn >= from && lastRsiDn >= from) {
            ctDecline = true;
            float hh = H(bars, i);
            for (int k = 1; k < bestEntry && i - k >= 0; ++k) hh = std::max(hh, H(bars, i - k));
            buyLvl = hh;
            checkRetrace = false;
          }
        }
      }
      if (dnTrend) {
        if (!dnTrendPrev) {
          dnLow = L(bars, i);
          lowCounter = 0;
          lowRsi = rsi[i];
          checkRetrace = false;
          ctRally = false;
        } else {
          if (L(bars, i) < dnLow) {
            dnLow = L(bars, i);
            lowCounter = 0;
          } else if (lowCounter < retraceBars) {
            ++lowCounter;
          }
          if (rsi[i] < lowRsi) lowRsi = rsi[i];
        }
        if (rsi[i] < osZone && spread < -15.f) checkRetrace = true;
        if (checkRetrace && lowCounter != 0 && lowCounter < retraceBars) {
          const int from = i - lowCounter;
          if (lastPopBar >= from && lastMacdUp >= from && lastRsiUp >= from) {
            ctRally = true;
            float ll = L(bars, i);
            for (int k = 1; k < bestEntry && i - k >= 0; ++k) ll = std::min(ll, L(bars, i - k));
            sellLvl = ll;
            checkRetrace = false;
          }
        }
      }
      upTrendPrev = upTrend;
      dnTrendPrev = dnTrend;
      // 5) Trend-failure exit queues a market order for the next open.
      if (pos == 1 && sig[i] < 0.f && spread < 0.f) pendingExit = true;
      if (pos == -1 && sig[i] > 0.f && spread > 0.f) pendingExit = true;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp13_volex -------------------------------------------------------------
// Cruz/Williams/Wright volatility expansion: every bar arms stops at the
// open plus/minus a factor of the four-bar average range (short factor
// wider). Exits: first profitable open, four-bar extreme trail, stale-trade
// market exit and a protective ATR stop.
class VolExV13 final : public IStrategy {
public:
  std::string name() const override { return "exp13_volex"; }

  std::vector<ParamDef> params() const override {
    return {
        {"longfactor", 1.2, 0.6, 2.4, 0.2, false},
        {"shortfactor", 1.8, 0.8, 3.0, 0.2, false},
        {"stalebars", 4, 2, 10, 1, true},
        {"protatrs", 3.0, 1.0, 6.0, 0.5, false},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    (void)ps;
    return {
        {Family::ATR, Column::Close, {10.0}},
        {Family::MIN, Column::Low, {4.0}},
        {Family::MAX, Column::High, {4.0}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* atr = ind.line(0);
    const float* lo4 = ind.line(1);
    const float* hi4 = ind.line(2);
    const int n = bars.size();
    const float lf = static_cast<float>(ps.get("longfactor"));
    const float sf = static_cast<float>(ps.get("shortfactor"));
    const int staleBars = ps.get_int("stalebars");
    const float prot = static_cast<float>(ps.get("protatrs"));

    int pos = 0;
    float entryPx = kNaN;
    int barsInPos = 0;

    for (int i = 4; i < n; ++i) {
      // 1) Exits.
      if (pos != 0) {
        ++barsInPos;
        bool exitNow = false;
        // First profitable open.
        if (pos == 1 && O(bars, i) > entryPx) exitNow = true;
        if (pos == -1 && O(bars, i) < entryPx) exitNow = true;
        // Four-bar extreme trail and protective ATR stop.
        if (!exitNow && pos == 1 &&
            (L(bars, i) <= lo4[i - 1] ||
             (finite(atr[i - 1]) && L(bars, i) <= entryPx - prot * atr[i - 1])))
          exitNow = true;
        if (!exitNow && pos == -1 &&
            (H(bars, i) >= hi4[i - 1] ||
             (finite(atr[i - 1]) && H(bars, i) >= entryPx + prot * atr[i - 1])))
          exitNow = true;
        // Stale position: too long without profit, exit at this open.
        if (!exitNow && barsInPos > staleBars) {
          if (pos == 1 && C(bars, i - 1) <= entryPx) exitNow = true;
          if (pos == -1 && C(bars, i - 1) >= entryPx) exitNow = true;
        }
        if (exitNow) pos = 0;
      }
      // 2) Entries: expansion from this bar's open, confirmed at the close.
      if (pos == 0) {
        float avgRng = 0.f;
        for (int k = 1; k <= 4; ++k) avgRng += H(bars, i - k) - L(bars, i - k);
        avgRng /= 4.f;
        const float buyLvl = O(bars, i) + avgRng * lf;
        const float sellLvl = O(bars, i) - avgRng * sf;
        if (buy_stop_hit(bars, i, buyLvl)) {
          pos = 1;
          entryPx = buyLvl;
          barsInPos = 0;
        } else if (sell_stop_hit(bars, i, sellLvl)) {
          pos = -1;
          entryPx = sellLvl;
          barsInPos = 0;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(4, n); ++i) out[i] = 0;
  }
};

// --- exp13_vol_clusters ------------------------------------------------------
// Volatility clusters: a close-to-close move or range beyond mean plus
// N standard deviations arms breakout stops at the close plus/minus the
// prior three-bar average range. Exit at the first profitable open or a
// protective ATR stop; entries can reverse the position.
class VolClustersV13 final : public IStrategy {
public:
  std::string name() const override { return "exp13_vol_clusters"; }

  std::vector<ParamDef> params() const override {
    return {
        {"deviations", 2.0, 1.0, 3.0, 0.5, false},
        {"length", 15, 5, 25, 5, true},
        {"protatrs", 2.0, 1.0, 6.0, 0.5, false},
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
    const float dev = static_cast<float>(ps.get("deviations"));
    const int len = ps.get_int("length");
    const float prot = static_cast<float>(ps.get("protatrs"));

    // Close-to-close absolute move and bar range series.
    std::vector<float> ctc(static_cast<std::size_t>(n), kNaN);
    std::vector<float> rng(static_cast<std::size_t>(n), kNaN);
    for (int i = 0; i < n; ++i) {
      rng[static_cast<std::size_t>(i)] = H(bars, i) - L(bars, i);
      if (i > 0) ctc[static_cast<std::size_t>(i)] = std::fabs(C(bars, i) - C(bars, i - 1));
    }
    const std::vector<float> ctcMean = rolling_mean(ctc.data(), n, len);
    const std::vector<float> ctcStd = rolling_std(ctc.data(), n, len);
    const std::vector<float> rngMean = rolling_mean(rng.data(), n, len);
    const std::vector<float> rngStd = rolling_std(rng.data(), n, len);

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN, entryPx = kNaN;

    for (int i = 4; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Exits: first profitable open or protective ATR stop.
      if (pos == 1 && finite(entryPx)) {
        if (O(bars, i) > entryPx) pos = 0;
        else if (finite(atr[i - 1]) && L(bars, i) <= entryPx - prot * atr[i - 1]) pos = 0;
      } else if (pos == -1 && finite(entryPx)) {
        if (O(bars, i) < entryPx) pos = 0;
        else if (finite(atr[i - 1]) && H(bars, i) >= entryPx + prot * atr[i - 1]) pos = 0;
      }
      // 2) Armed breakout entries (either side; may reverse).
      if (finite(buyLvl) && pos != 1 && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        entryPx = std::max(O(bars, i), buyLvl);
      } else if (finite(sellLvl) && pos != -1 && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        entryPx = std::min(O(bars, i), sellLvl);
      }
      // 3) Volatility spike detection at this close.
      buyLvl = sellLvl = kNaN;
      if (finite(ctcMean[ui - 1]) && finite(ctcStd[ui - 1]) && finite(rngMean[ui - 1]) &&
          finite(rngStd[ui - 1])) {
        const bool bigVol = ctc[ui] > ctcMean[ui - 1] + dev * ctcStd[ui - 1] ||
                            rng[ui] > rngMean[ui - 1] + dev * rngStd[ui - 1];
        if (bigVol) {
          float avg3 = 0.f;
          for (int k = 1; k <= 3; ++k) avg3 += rng[static_cast<std::size_t>(i - k)];
          avg3 /= 3.f;
          buyLvl = C(bars, i) + avg3;
          sellLvl = C(bars, i) - avg3;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    for (int i = 0; i < std::min(4, n); ++i) out[i] = 0;
  }
};

} // namespace

void register_vol13(std::vector<Factory>& out) {
  out.push_back({"exp13_oc_histogram",
                 [] { return std::unique_ptr<IStrategy>(new OcHistogramV13); }});
  out.push_back({"exp13_dma_range_leaders",
                 [] { return std::unique_ptr<IStrategy>(new DmaRangeLeadersV13); }});
  out.push_back({"exp13_linreg_momentum",
                 [] { return std::unique_ptr<IStrategy>(new LinRegMomV13); }});
  out.push_back({"exp13_mom_retrace",
                 [] { return std::unique_ptr<IStrategy>(new MomRetraceV13); }});
  out.push_back({"exp13_volex",
                 [] { return std::unique_ptr<IStrategy>(new VolExV13); }});
  out.push_back({"exp13_vol_clusters",
                 [] { return std::unique_ptr<IStrategy>(new VolClustersV13); }});
}

} // namespace topt::exp
