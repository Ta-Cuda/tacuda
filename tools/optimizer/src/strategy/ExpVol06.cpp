/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 6 (Omega Research STAD Club, Vol. 6).
// Systems: Accumulation Swing Index, ADX & Moving-Average Channel,
// Displaced MA Channel & Range Leaders, Four Sets of MA Crossover,
// CCI Average, Minor Reaction/Trend Resumption, First Pullback,
// Parabolic & Volume, Volume-Weighted Momentum. Nothing skipped.
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

// The Accumulation Swing Index and the EL-style parabolic used by this
// volume are native TaCuda families since 0.4.0 (Family::ASI /
// Family::SAR_EL) and are requested through the backend.

// --- exp06_asi ---------------------------------------------------------------
// A fresh price swing high whose Accumulation Swing Index beats the previous
// swing's reading arms a two-bar-high breakout stop for five bars (mirrored
// for swing lows). Initial stop at the two-bar extreme; trail at the most
// recent strength-1 swing.
class AsiV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_asi"; }

  std::vector<ParamDef> params() const override {
    return {
        {"strength", 2, 1, 5, 1, true},
        {"setuplen", 5, 2, 15, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // ASI and the confirmed swing pivots run as native (batchable)
    // requests; a pivot line is non-NaN exactly at its confirmation bar.
    return {
        {Family::ASI, Column::Close, {}},
        {Family::PIVOT_HIGH, Column::High, {ps.get("strength")}},
        {Family::PIVOT_LOW, Column::Low, {ps.get("strength")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* asi = ind.line(0);
    const float* ph = ind.line(1);
    const float* pl = ind.line(2);
    const int n = bars.size();
    const int s = ps.get_int("strength");
    const int setupLen = ps.get_int("setuplen");

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f, initL = 0.f, initS = 0.f;
    int buyUntil = -1, sellUntil = -1;
    float prevSwingHighAsi = kNaN, prevSwingLowAsi = kNaN;
    float trailLow = kNaN, trailHigh = kNaN;

    for (int i = 0; i < n; ++i) {
      // 1) Exits.
      if (pos == 1) {
        if (L(bars, i) <= initL) pos = 0;
        else if (finite(trailLow) && L(bars, i) <= trailLow) pos = 0;
      } else if (pos == -1) {
        if (H(bars, i) >= initS) pos = 0;
        else if (finite(trailHigh) && H(bars, i) >= trailHigh) pos = 0;
      }
      // 2) Armed breakout stops.
      if (pos != 1 && i <= buyUntil && buy_stop_hit(bars, i, buyLvl)) pos = 1;
      else if (pos != -1 && i <= sellUntil && sell_stop_hit(bars, i, sellLvl)) pos = -1;

      // 3) Swing bookkeeping at this close: a finite pivot line value means
      // the swing at bar i-s was confirmed on this bar.
      if (i >= s) {
        const int b = i - s;
        if (finite(ph[i])) {
          if (finite(prevSwingHighAsi) && finite(asi[b]) &&
              asi[b] > prevSwingHighAsi && pos != 1) {
            buyLvl = std::max(H(bars, i), H(bars, i - 1));
            initL = std::min(L(bars, i), L(bars, i - 1));
            buyUntil = i + setupLen;
          }
          prevSwingHighAsi = asi[b];
        }
        if (finite(pl[i])) {
          if (finite(prevSwingLowAsi) && finite(asi[b]) &&
              asi[b] < prevSwingLowAsi && pos != -1) {
            sellLvl = std::min(L(bars, i), L(bars, i - 1));
            initS = std::max(H(bars, i), H(bars, i - 1));
            sellUntil = i + setupLen;
          }
          prevSwingLowAsi = asi[b];
        }
      }
      // 4) Strength-1 swing trail levels.
      if (i >= 2) {
        if (L(bars, i - 1) < L(bars, i - 2) && L(bars, i - 1) <= L(bars, i))
          trailLow = L(bars, i - 1);
        if (H(bars, i - 1) > H(bars, i - 2) && H(bars, i - 1) >= H(bars, i))
          trailHigh = H(bars, i - 1);
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp06_mac_adx -----------------------------------------------------------
// Close beyond the EMA high/low channel with a rising ADX arms a stop half a
// channel-height beyond that close for two bars. Exits when the close falls
// back through the opposite channel line.
class MacAdxV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_mac_adx"; }

  std::vector<ParamDef> params() const override {
    return {
        {"avglen", 30, 10, 80, 2, true},
        {"adxlen", 12, 5, 40, 1, true},
        {"entrybar", 2, 1, 6, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::High, {ps.get("avglen")}},
        {Family::EMA, Column::Low, {ps.get("avglen")}},
        {Family::ADX, Column::Close, {ps.get("adxlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* emaH = ind.line(0);
    const float* emaL = ind.line(1);
    const float* adx = ind.line(2);
    const int n = bars.size();
    const int entryBar = ps.get_int("entrybar");

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 1; i < n; ++i) {
      // Channel values are the previous bar's EMAs, as in the original.
      const float upper = emaH[i - 1];
      const float lower = emaL[i - 1];
      if (!finite(upper) || !finite(lower) || !finite(adx[i]) || !finite(adx[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits on a close through the opposite channel line.
      if (pos == 1 && C(bars, i) < lower) pos = 0;
      else if (pos == -1 && C(bars, i) > upper) pos = 0;
      // 2) Armed entry stops.
      if (pos != 1 && i <= buyUntil && buy_stop_hit(bars, i, buyLvl)) pos = 1;
      else if (pos != -1 && i <= sellUntil && sell_stop_hit(bars, i, sellLvl)) pos = -1;
      // 3) Setups at this close.
      const float spread = (upper - lower) / 2.f;
      if (C(bars, i) > upper && adx[i] > adx[i - 1]) {
        buyLvl = C(bars, i) + spread;
        buyUntil = i + entryBar;
      }
      if (C(bars, i) < lower && adx[i] > adx[i - 1]) {
        sellLvl = C(bars, i) - spread;
        sellUntil = i + entryBar;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp06_dma_chan ----------------------------------------------------------
// A Range Leader bar (midpoint beyond the prior bar's extreme on expanding
// range) closing outside the displaced MA channel enters at that close. The
// exit trails the displaced median-price average early, then hugs the
// channel's own side.
class DmaChanV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_dma_chan"; }

  std::vector<ParamDef> params() const override {
    return {
        {"avglen", 20, 5, 60, 1, true},
        {"disp", 5, 1, 15, 1, true},
        {"exitbar", 5, 2, 15, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::High, {ps.get("avglen")}},
        {Family::SMA, Column::Low, {ps.get("avglen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* smaH = ind.line(0);
    const float* smaL = ind.line(1);
    const int n = bars.size();
    const int disp = ps.get_int("disp");
    const int avgLen = ps.get_int("avglen");
    const int exitBar = ps.get_int("exitbar");

    // Displaced moving average of the median price, computed locally.
    std::vector<float> medAvg(static_cast<std::size_t>(n), kNaN);
    {
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        const double med = (static_cast<double>(H(bars, i)) + L(bars, i)) / 2.0;
        sum += med;
        if (i >= avgLen)
          sum -= (static_cast<double>(H(bars, i - avgLen)) + L(bars, i - avgLen)) / 2.0;
        if (i >= avgLen - 1)
          medAvg[static_cast<std::size_t>(i)] = static_cast<float>(sum / avgLen);
      }
    }
    auto upper = [&](int i) { return i - disp >= 0 ? smaH[i - disp] : kNaN; };
    auto lower = [&](int i) { return i - disp >= 0 ? smaL[i - disp] : kNaN; };
    auto exitAvg = [&](int i) {
      return i - disp >= 0 ? medAvg[static_cast<std::size_t>(i - disp)] : kNaN;
    };

    int pos = 0, fillBar = -1;
    for (int i = 1; i < n; ++i) {
      const float up = upper(i), lo = lower(i), ex = exitAvg(i);
      if (!finite(up) || !finite(lo) || !finite(ex)) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Exits: displaced median average early, channel side afterwards.
      if (pos != 0 && i > fillBar) {
        const bool early = i - fillBar < exitBar;
        if (pos == 1 && (early ? L(bars, i) <= ex : L(bars, i) <= up)) pos = 0;
        else if (pos == -1 && (early ? H(bars, i) >= ex : H(bars, i) >= lo)) pos = 0;
      }
      // 2) Range Leader entries at this close.
      const float med = (H(bars, i) + L(bars, i)) / 2.f;
      const float rng = H(bars, i) - L(bars, i);
      const float rngPrev = H(bars, i - 1) - L(bars, i - 1);
      const bool leadB = med > H(bars, i - 1) && rng > rngPrev;
      const bool leadS = med < L(bars, i - 1) && rng > rngPrev;
      if (leadB && C(bars, i) > up && pos != 1) {
        pos = 1;
        fillBar = i + 1;
      } else if (leadS && C(bars, i) < lo && pos != -1) {
        pos = -1;
        fillBar = i + 1;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp06_ma_pairs ----------------------------------------------------------
// Four independently tuned SMA pairs: one pair arms the long entry stop at
// the bar high, another exits the long on its own crossover; two more do the
// same for shorts.
class MaPairsV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_ma_pairs"; }

  std::vector<ParamDef> params() const override {
    return {
        {"lefast", 7, 2, 30, 1, true},  {"leslow", 23, 5, 80, 1, true},
        {"lxfast", 4, 2, 30, 1, true},  {"lxslow", 8, 5, 80, 1, true},
        {"sefast", 6, 2, 30, 1, true},  {"seslow", 18, 5, 80, 1, true},
        {"sxfast", 2, 2, 30, 1, true},  {"sxslow", 9, 5, 80, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("lefast") < ps.get_int("leslow") &&
           ps.get_int("lxfast") < ps.get_int("lxslow") &&
           ps.get_int("sefast") < ps.get_int("seslow") &&
           ps.get_int("sxfast") < ps.get_int("sxslow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("lefast")}},
        {Family::SMA, Column::Close, {ps.get("leslow")}},
        {Family::SMA, Column::Close, {ps.get("lxfast")}},
        {Family::SMA, Column::Close, {ps.get("lxslow")}},
        {Family::SMA, Column::Close, {ps.get("sefast")}},
        {Family::SMA, Column::Close, {ps.get("seslow")}},
        {Family::SMA, Column::Close, {ps.get("sxfast")}},
        {Family::SMA, Column::Close, {ps.get("sxslow")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ps;
    const float* leF = ind.line(0);
    const float* leS = ind.line(1);
    const float* lxF = ind.line(2);
    const float* lxS = ind.line(3);
    const float* seF = ind.line(4);
    const float* seS = ind.line(5);
    const float* sxF = ind.line(6);
    const float* sxS = ind.line(7);
    const int n = bars.size();

    int pos = 0;
    float buyLvl = kNaN, sellLvl = kNaN;

    for (int i = 0; i < n; ++i) {
      // 1) Crossover exits at this close.
      if (pos == 1 && crosses_below(lxF, lxS, i)) pos = 0;
      if (pos == -1 && crosses_above(sxF, sxS, i)) pos = 0;
      // 2) Stops armed at the previous close.
      if (pos != 1 && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) pos = 1;
      else if (pos != -1 && finite(sellLvl) && sell_stop_hit(bars, i, sellLvl)) pos = -1;
      // 3) Re-arm while the entry pair stays aligned.
      buyLvl = (finite(leF[i]) && finite(leS[i]) && leF[i] > leS[i]) ? H(bars, i) : kNaN;
      sellLvl = (finite(seF[i]) && finite(seS[i]) && seF[i] < seS[i]) ? L(bars, i) : kNaN;
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp06_cci_avg -----------------------------------------------------------
// Smoothed CCI escaping the +/-100 band arms a 9-bar channel breakout stop
// for four bars. The trail is a channel stop whose lookback shrinks by one
// every time the trade makes a new extreme.
class CciAvgV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_cci_avg"; }

  std::vector<ParamDef> params() const override {
    return {
        {"ccilen", 30, 8, 60, 1, true},
        {"avglen", 5, 2, 15, 1, true},
        {"hilolen", 9, 4, 20, 1, true},
        {"entrybar", 4, 2, 10, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::CCI, Column::Close, {ps.get("ccilen")}},
        {Family::MAX, Column::High, {ps.get("hilolen")}},
        {Family::MIN, Column::Low, {ps.get("hilolen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* cci = ind.line(0);
    const float* hiCh = ind.line(1);
    const float* loCh = ind.line(2);
    const int n = bars.size();
    const int hiLoLen = ps.get_int("hilolen");
    const int entryBar = ps.get_int("entrybar");
    const std::vector<float> avgCci =
        rolling_mean(cci, n, ps.get_int("avglen"));

    int pos = 0;
    int buyUntil = -1, sellUntil = -1;
    int trailLen = hiLoLen;

    auto rollLow = [&](int i, int len) {
      float m = L(bars, i);
      for (int k = 1; k < len && i - k >= 0; ++k) m = std::min(m, L(bars, i - k));
      return m;
    };
    auto rollHigh = [&](int i, int len) {
      float m = H(bars, i);
      for (int k = 1; k < len && i - k >= 0; ++k) m = std::max(m, H(bars, i - k));
      return m;
    };

    for (int i = 1; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Shrinking-channel trail.
      if (pos == 1) {
        if (L(bars, i) <= rollLow(i - 1, trailLen)) {
          pos = 0;
        } else if (finite(hiCh[i - 1]) && H(bars, i) > hiCh[i - 1] && trailLen > 1) {
          --trailLen;
        }
      } else if (pos == -1) {
        if (H(bars, i) >= rollHigh(i - 1, trailLen)) {
          pos = 0;
        } else if (finite(loCh[i - 1]) && L(bars, i) < loCh[i - 1] && trailLen > 1) {
          --trailLen;
        }
      }
      // 2) Armed channel-breakout entries.
      if (finite(avgCci[ui])) {
        if (pos != 1 && i <= buyUntil && avgCci[ui] > -100.f &&
            finite(hiCh[i - 1]) && buy_stop_hit(bars, i, hiCh[i - 1])) {
          pos = 1;
          trailLen = hiLoLen;
        } else if (pos != -1 && i <= sellUntil && avgCci[ui] < 100.f &&
                   finite(loCh[i - 1]) && sell_stop_hit(bars, i, loCh[i - 1])) {
          pos = -1;
          trailLen = hiLoLen;
        }
        // 3) Band-escape setups at this close.
        if (finite(avgCci[ui - 1])) {
          if (avgCci[ui] > -100.f && avgCci[ui - 1] <= -100.f) buyUntil = i + entryBar;
          if (avgCci[ui] < 100.f && avgCci[ui - 1] >= 100.f) sellUntil = i + entryBar;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp06_mrtr --------------------------------------------------------------
// Schwager's Minor Reaction/Trend Resumption: a channel breakout defines the
// trend, four nested reaction bars define the pullback, two thrust bars
// re-arm with the trend; the reaction extreme is the protective stop.
class MrtrV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_mrtr"; }

  std::vector<ParamDef> params() const override {
    return {
        {"chanlen", 40, 10, 100, 5, true},
        {"reaction", 4, 2, 8, 1, true},
        {"tcount", 2, 1, 5, 1, true},
        {"setup", 4, 2, 10, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("chanlen")}},
        {Family::MIN, Column::Low, {ps.get("chanlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiChan = ind.line(0);
    const float* loChan = ind.line(1);
    const int n = bars.size();
    const int reactionReq = ps.get_int("reaction");
    const int tCount = ps.get_int("tcount");
    const int setupLen = ps.get_int("setup");

    int pos = 0;
    int lReaction = 0, sReaction = 0, lThrust = 0, sThrust = 0;
    float lReacHigh = 0.f, lReacLow = 0.f, sReacHigh = 0.f, sReacLow = 0.f;
    float lExit = 0.f, sExit = 0.f, lEntry = 0.f, sEntry = 0.f;
    int lArmedUntil = -1, sArmedUntil = -1;

    for (int i = 1; i < n; ++i) {
      if (!finite(hiChan[i - 1]) || !finite(loChan[i - 1])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) Protective stops at the reaction extreme.
      if (pos == 1 && L(bars, i) <= lExit) pos = 0;
      else if (pos == -1 && H(bars, i) >= sExit) pos = 0;
      // 2) Armed thrust entries.
      if (pos != 1 && i <= lArmedUntil && lReaction >= reactionReq &&
          buy_stop_hit(bars, i, lEntry)) {
        pos = 1;
        lArmedUntil = -1;
      } else if (pos != -1 && i <= sArmedUntil && sReaction >= reactionReq &&
                 sell_stop_hit(bars, i, sEntry)) {
        pos = -1;
        sArmedUntil = -1;
      }
      // 3) Channel breakouts (re)start the reaction counts.
      if (pos != 1 && H(bars, i) > hiChan[i - 1]) {
        lReaction = 1;
        lReacHigh = H(bars, i);
        lReacLow = L(bars, i);
        lExit = lReacLow;
        lThrust = 0;
        sReaction = 0;
      } else if (pos == 1) {
        lReaction = 0;
        lThrust = 0;
      }
      if (pos != -1 && L(bars, i) < loChan[i - 1]) {
        sReaction = 1;
        sReacHigh = H(bars, i);
        sReacLow = L(bars, i);
        sExit = sReacHigh;
        sThrust = 0;
        lReaction = 0;
      } else if (pos == -1) {
        sReaction = 0;
        sThrust = 0;
      }
      // 4) Reaction bookkeeping.
      if (lReaction >= 1) {
        if (L(bars, i) < lExit) {
          lExit = L(bars, i);
          lThrust = 0;
        }
        if (H(bars, i) < lReacHigh && L(bars, i) < lReacLow) {
          ++lReaction;
          lReacHigh = H(bars, i);
          lReacLow = L(bars, i);
        }
      }
      if (sReaction >= 1) {
        if (H(bars, i) > sExit) {
          sExit = H(bars, i);
          sThrust = 0;
        }
        if (L(bars, i) > sReacLow && H(bars, i) > sReacHigh) {
          ++sReaction;
          sReacLow = L(bars, i);
          sReacHigh = H(bars, i);
        }
      }
      // 5) Thrust bars complete the setup.
      if (lReaction >= reactionReq && C(bars, i) > H(bars, i - 1)) {
        ++lThrust;
        if (lThrust == tCount) {
          lEntry = H(bars, i);
          lArmedUntil = i + setupLen;
        }
      }
      if (sReaction >= reactionReq && C(bars, i) < L(bars, i - 1)) {
        ++sThrust;
        if (sThrust == tCount) {
          sEntry = L(bars, i);
          sArmedUntil = i + setupLen;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp06_first_pullback ----------------------------------------------------
// Raschke-style first pullback: the MACD signal line's zero cross sets the
// trend; the first MACD/signal cross against it arms a volatility stop entry
// above the close. Exits at the pullback extreme, a volatility stop, or the
// trend flipping back.
class FirstPullbackV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_first_pullback"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fast", 4, 2, 12, 1, true},
        {"slow", 10, 5, 30, 1, true},
        {"avg", 16, 5, 40, 1, true},
        {"atrlen", 10, 5, 30, 1, true},
        {"eatr", 1.25, 0.25, 3.0, 0.25, false},
        {"xatr", 2.0, 0.5, 4.0, 0.25, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fast") < ps.get_int("slow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MACD, Column::Close, {ps.get("fast"), ps.get("slow"), ps.get("avg")}},
        {Family::ATR, Column::Close, {ps.get("atrlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* macd = ind.line(0, 0);
    const float* sig = ind.line(0, 1);
    const float* atr = ind.line(1);
    const int n = bars.size();
    const float eAtr = static_cast<float>(ps.get("eatr"));
    const float xAtr = static_cast<float>(ps.get("xatr"));

    bool upTrend = false, dnTrend = false, buySetup = false, sellSetup = false;
    bool signalFlag = false;
    float cTrendLow = kNaN, cTrendHigh = kNaN;
    float buyLvl = kNaN, sellLvl = kNaN;
    int pos = 0;

    for (int i = 1; i < n; ++i) {
      if (!finite(macd[i]) || !finite(sig[i]) || !finite(macd[i - 1]) ||
          !finite(sig[i - 1]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const bool sigXUp = sig[i] > 0.f && sig[i - 1] <= 0.f;
      const bool sigXDn = sig[i] < 0.f && sig[i - 1] >= 0.f;
      // 1) Exits.
      if (pos == 1) {
        const bool volStop = i >= 1 && finite(atr[i - 1]) &&
                             L(bars, i) <= C(bars, i - 1) - xAtr * atr[i - 1];
        if ((finite(cTrendLow) && L(bars, i) <= cTrendLow) || volStop || sigXDn) {
          pos = 0;
        }
      } else if (pos == -1) {
        const bool volStop = i >= 1 && finite(atr[i - 1]) &&
                             H(bars, i) >= C(bars, i - 1) + xAtr * atr[i - 1];
        if ((finite(cTrendHigh) && H(bars, i) >= cTrendHigh) || volStop || sigXUp) {
          pos = 0;
        }
      }
      // 2) Armed pullback entries (stop level from the previous close).
      if (pos != 1 && buySetup && finite(buyLvl) && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        buySetup = false;
        signalFlag = true;
      } else if (pos != -1 && sellSetup && finite(sellLvl) &&
                 sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sellSetup = false;
        signalFlag = true;
      }
      // 3) Trend and pullback bookkeeping at this close.
      if (sigXUp) {
        upTrend = true;
        dnTrend = false;
        sellSetup = false;
        signalFlag = false;
      }
      if (sigXDn) {
        dnTrend = true;
        upTrend = false;
        buySetup = false;
        signalFlag = false;
      }
      if (upTrend) {
        const bool macdXDn = macd[i] < sig[i] && macd[i - 1] >= sig[i - 1];
        if (macdXDn && !signalFlag && !buySetup) {
          buySetup = true;
          cTrendLow = L(bars, i);
        }
        if (macd[i] < sig[i] && finite(cTrendLow)) cTrendLow = std::min(cTrendLow, L(bars, i));
      }
      if (dnTrend) {
        const bool macdXUp = macd[i] > sig[i] && macd[i - 1] <= sig[i - 1];
        if (macdXUp && !signalFlag && !sellSetup) {
          sellSetup = true;
          cTrendHigh = H(bars, i);
        }
        if (macd[i] > sig[i] && finite(cTrendHigh)) cTrendHigh = std::max(cTrendHigh, H(bars, i));
      }
      buyLvl = buySetup ? C(bars, i) + eAtr * atr[i] : kNaN;
      sellLvl = sellSetup ? C(bars, i) - eAtr * atr[i] : kNaN;
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp06_parabolic_vol -----------------------------------------------------
// Parabolic SAR flips confirmed by above-average volume arm an entry stop a
// fraction of the old-to-new SAR distance beyond the flip close (four bars).
// The SAR itself is the trailing stop.
class ParabolicVolV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_parabolic_vol"; }

  std::vector<ParamDef> params() const override {
    return {
        {"accfact", 0.03, 0.01, 0.1, 0.01, false},
        {"volavg", 5, 2, 20, 1, true},
        {"pcntdiff", 0.3, 0.1, 1.0, 0.1, false},
        {"setuplen", 4, 2, 10, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // EL-style parabolic (close-seeded direction, 0.2 cap) as a native
    // (batchable) request.
    return {
        {Family::SMA, Column::Volume, {ps.get("volavg")}},
        {Family::SAR_EL, Column::Close, {ps.get("accfact"), 0.2}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* vAvg = ind.line(0);
    const float* sar = ind.line(1);
    const int n = bars.size();
    const float pcnt = static_cast<float>(ps.get("pcntdiff"));
    const int setupLen = ps.get_int("setuplen");

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 2; i < n; ++i) {
      const float p = sar[i - 1];      // Parabolic offset by one bar
      const float pPrev = sar[i - 2];
      if (!finite(p) || !finite(pPrev) || !finite(vAvg[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      // 1) SAR trailing exits.
      if (pos == 1 && L(bars, i) <= p) pos = 0;
      else if (pos == -1 && H(bars, i) >= p) pos = 0;
      // 2) Armed flip entries.
      if (pos != 1 && i <= buyUntil && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sellUntil = -1;
      }
      // 3) Volume-confirmed SAR touches at this bar.
      const bool volOk = V(bars, i) > vAvg[i];
      const bool hiCross = H(bars, i) > p && H(bars, i - 1) <= pPrev;
      const bool loCross = L(bars, i) < p && L(bars, i - 1) >= pPrev;
      if (hiCross && volOk) {
        buyLvl = C(bars, i) + pcnt * std::fabs(pPrev - p);
        buyUntil = i + setupLen;
      }
      if (loCross && volOk) {
        sellLvl = C(bars, i) - pcnt * std::fabs(pPrev - p);
        sellUntil = i + setupLen;
      }
      out[i] = static_cast<Signal>(pos);
    }
    out[0] = 0;
    if (n > 1) out[1] = 0;
  }
};

// --- exp06_vwm ---------------------------------------------------------------
// Volume-weighted momentum (EMA of volume x momentum) crossing zero arms a
// quarter-ATR breakout above the crossing close for five bars, with an ATR
// stop under the setup bar; the opposite cross exits at the next open.
class VwmV6 final : public IStrategy {
public:
  std::string name() const override { return "exp06_vwm"; }

  std::vector<ParamDef> params() const override {
    return {
        {"momlen", 5, 2, 20, 1, true},
        {"avglen", 20, 5, 60, 1, true},
        {"atrlen", 5, 2, 20, 1, true},
        {"atrpcnt", 0.25, 0.05, 1.0, 0.05, false},
        {"setuplen", 5, 2, 12, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MOM, Column::Close, {ps.get("momlen")}},
        {Family::ATR, Column::Close, {ps.get("atrlen")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* mom = ind.line(0);
    const float* atr = ind.line(1);
    const int n = bars.size();
    const float pcnt = static_cast<float>(ps.get("atrpcnt"));
    const int setupLen = ps.get_int("setuplen");

    std::vector<float> vm(static_cast<std::size_t>(n), kNaN);
    for (int i = 0; i < n; ++i)
      if (finite(mom[i])) vm[static_cast<std::size_t>(i)] = V(bars, i) * mom[i];
    const std::vector<float> vwm = ema_line(vm.data(), n, ps.get_int("avglen"));

    int pos = 0;
    float buyLvl = 0.f, sellLvl = 0.f, initStop = 0.f;
    int buyUntil = -1, sellUntil = -1;

    for (int i = 1; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (!finite(vwm[ui]) || !finite(vwm[ui - 1]) || !finite(atr[i])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const bool crossUp = vwm[ui] > 0.f && vwm[ui - 1] <= 0.f;
      const bool crossDn = vwm[ui] < 0.f && vwm[ui - 1] >= 0.f;
      // 1) Exits: initial stop or the opposite zero cross.
      if (pos == 1 && (L(bars, i) <= initStop || crossDn)) pos = 0;
      else if (pos == -1 && (H(bars, i) >= initStop || crossUp)) pos = 0;
      // 2) Armed entries.
      if (pos != 1 && i <= buyUntil && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        buyUntil = -1;
      } else if (pos != -1 && i <= sellUntil && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        sellUntil = -1;
      }
      // 3) Zero-cross setups at this close.
      if (crossUp) {
        buyLvl = C(bars, i) + pcnt * atr[i];
        initStop = L(bars, i) - pcnt * atr[i];
        buyUntil = i + setupLen;
      }
      if (crossDn) {
        sellLvl = C(bars, i) - pcnt * atr[i];
        initStop = H(bars, i) + pcnt * atr[i];
        sellUntil = i + setupLen;
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

} // namespace

void register_vol06(std::vector<Factory>& out) {
  out.push_back({"exp06_asi",
                 [] { return std::unique_ptr<IStrategy>(new AsiV6); }});
  out.push_back({"exp06_mac_adx",
                 [] { return std::unique_ptr<IStrategy>(new MacAdxV6); }});
  out.push_back({"exp06_dma_chan",
                 [] { return std::unique_ptr<IStrategy>(new DmaChanV6); }});
  out.push_back({"exp06_ma_pairs",
                 [] { return std::unique_ptr<IStrategy>(new MaPairsV6); }});
  out.push_back({"exp06_cci_avg",
                 [] { return std::unique_ptr<IStrategy>(new CciAvgV6); }});
  out.push_back({"exp06_mrtr",
                 [] { return std::unique_ptr<IStrategy>(new MrtrV6); }});
  out.push_back({"exp06_first_pullback",
                 [] { return std::unique_ptr<IStrategy>(new FirstPullbackV6); }});
  out.push_back({"exp06_parabolic_vol",
                 [] { return std::unique_ptr<IStrategy>(new ParabolicVolV6); }});
  out.push_back({"exp06_vwm",
                 [] { return std::unique_ptr<IStrategy>(new VwmV6); }});
}

} // namespace topt::exp
