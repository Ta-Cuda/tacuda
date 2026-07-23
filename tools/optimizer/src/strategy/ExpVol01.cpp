/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 1 (Omega Research STAD Club, Vol. 1).
// Systems: Moving Average Crossover, MACD Breakout, Trend Follower,
// Support & Resistance Levels, Pivot Points, Channel Breakout, VolEx.
// Skipped from this volume: Trend Line Automatic (drawing-object trendlines),
// Spread (needs a second data series), Pivot Points Intra-Day (intraday
// sessions). See docs/EXPERIMENTS.md for the adaptation notes.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

// --- exp01_ma_crossover ----------------------------------------------------
// SMA(fast) x SMA(slow) crossover arms a breakout stop 3% above the recent
// high (long) / 3% below the recent low (short) for `chlen` bars; trailing
// exits at the `trailbar` extreme; after a stop-out the trade re-arms at the
// 10-bar extreme for `rebars` bars.
class MaCrossoverV1 final : public IStrategy {
public:
  std::string name() const override { return "exp01_ma_crossover"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fast", 9, 3, 50, 1, true},
        {"slow", 18, 6, 120, 1, true},
        {"chlen", 12, 4, 40, 1, true},
        {"trailbar", 8, 2, 30, 1, true},
        {"rebars", 15, 0, 40, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fast") < ps.get_int("slow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double fast = ps.get("fast"), slow = ps.get("slow");
    const double chlen = ps.get("chlen"), trail = ps.get("trailbar");
    return {
        {Family::SMA, Column::Close, {fast}},
        {Family::SMA, Column::Close, {slow}},
        {Family::MAX, Column::High, {chlen}},
        {Family::MIN, Column::Low, {chlen}},
        {Family::MAX, Column::High, {10}},
        {Family::MIN, Column::Low, {10}},
        {Family::MAX, Column::High, {trail}},
        {Family::MIN, Column::Low, {trail}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* fast = ind.line(0);
    const float* slow = ind.line(1);
    const float* hiCh = ind.line(2);
    const float* loCh = ind.line(3);
    const float* hi10 = ind.line(4);
    const float* lo10 = ind.line(5);
    const float* hiTrail = ind.line(6);
    const float* loTrail = ind.line(7);
    const int n = bars.size();
    const int chlen = ps.get_int("chlen");
    const int rebars = ps.get_int("rebars");

    int pos = 0;
    float longLvl = 0.f, shortLvl = 0.f;
    int longUntil = -1, shortUntil = -1; // last bar the armed stop stays active
    int reDir = 0, reLeft = 0;

    for (int i = 0; i < n; ++i) {
      if (!finite(fast[i]) || !finite(slow[i])) {
        out[i] = 0;
        continue;
      }
      // 1) Trailing stops armed at the previous close.
      if (pos != 0 && i > 0 && finite(loTrail[i - 1]) && finite(hiTrail[i - 1])) {
        if (pos == 1 && sell_stop_hit(bars, i, loTrail[i - 1])) {
          pos = 0; reDir = 1; reLeft = rebars;
        } else if (pos == -1 && buy_stop_hit(bars, i, hiTrail[i - 1])) {
          pos = 0; reDir = -1; reLeft = rebars;
        }
      }
      // 2) Crossover breakout stops armed at earlier closes.
      if (pos != 1 && i <= longUntil && buy_stop_hit(bars, i, longLvl)) {
        pos = 1; reDir = 0;
      } else if (pos != -1 && i <= shortUntil && sell_stop_hit(bars, i, shortLvl)) {
        pos = -1; reDir = 0;
      } else if (pos == 0 && reDir != 0 && reLeft > 0 && i > 0) {
        // 3) Re-entry stop at the 10-bar extreme after a stop-out.
        --reLeft;
        if (reDir == 1 && finite(hi10[i - 1]) && buy_stop_hit(bars, i, hi10[i - 1])) {
          pos = 1; reDir = 0;
        } else if (reDir == -1 && finite(lo10[i - 1]) &&
                   sell_stop_hit(bars, i, lo10[i - 1])) {
          pos = -1; reDir = 0;
        }
      }
      // 4) New crossovers observed at this close re-arm the entries.
      if (crosses_above(fast, slow, i) && finite(hiCh[i])) {
        longLvl = hiCh[i] * 1.03f;
        longUntil = i + chlen;
      }
      if (crosses_below(fast, slow, i) && finite(loCh[i])) {
        shortLvl = loCh[i] * 0.97f;
        shortUntil = i + chlen;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp01_macd_breakout ---------------------------------------------------
// Channel built from the highs/lows of the last `ncos` zero-crossings of the
// MACD histogram; when all of them happened within `nbars` bars, breakout
// stops go half an average true range beyond the channel. Trailing exits at
// the `trailbar` extreme.
class MacdBreakoutV1 final : public IStrategy {
public:
  std::string name() const override { return "exp01_macd_breakout"; }

  std::vector<ParamDef> params() const override {
    return {
        {"fast", 12, 4, 40, 1, true},
        {"slow", 26, 10, 80, 1, true},
        {"sig", 9, 3, 30, 1, true},
        {"ncos", 4, 2, 8, 1, true},
        {"nbars", 50, 10, 150, 5, true},
        {"trailbar", 6, 2, 20, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fast") < ps.get_int("slow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MACD, Column::Close,
         {ps.get("fast"), ps.get("slow"), ps.get("sig")}},
        {Family::MAX, Column::High, {ps.get("trailbar")}},
        {Family::MIN, Column::Low, {ps.get("trailbar")}},
        {Family::TRANGE, Column::Close, {}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hist = ind.line(0, 2);
    const float* hiTrail = ind.line(1);
    const float* loTrail = ind.line(2);
    const int n = bars.size();
    const int ncos = ps.get_int("ncos");
    const int nbars = ps.get_int("nbars");
    std::vector<float> atr4 = rolling_mean(ind.line(3), n, 4);

    struct Cross { int bar; float high, low; };
    std::deque<Cross> crossings;
    int pos = 0;

    for (int i = 0; i < n; ++i) {
      if (!finite(hist[i])) {
        out[i] = 0;
        continue;
      }
      // Trailing stops armed at the previous close (always active).
      if (pos != 0 && i > 0 && finite(loTrail[i - 1]) && finite(hiTrail[i - 1])) {
        if (pos == 1 && sell_stop_hit(bars, i, loTrail[i - 1])) pos = 0;
        else if (pos == -1 && buy_stop_hit(bars, i, hiTrail[i - 1])) pos = 0;
      }
      // Breakout stops armed at the previous close when the channel was live.
      if (static_cast<int>(crossings.size()) == ncos && i > 0 &&
          finite(atr4[static_cast<std::size_t>(i - 1)])) {
        const int span = (i - 1) - crossings.front().bar;
        if (span < nbars) {
          float hh = crossings[0].high, ll = crossings[0].low;
          for (const Cross& c : crossings) {
            hh = std::max(hh, c.high);
            ll = std::min(ll, c.low);
          }
          const float pad = 0.5f * atr4[static_cast<std::size_t>(i - 1)];
          if (pos != 1 && buy_stop_hit(bars, i, hh + pad)) pos = 1;
          else if (pos != -1 && sell_stop_hit(bars, i, ll - pad)) pos = -1;
        }
      }
      // Record histogram zero crossings observed at this close.
      if (i > 0 && finite(hist[i - 1]) &&
          ((hist[i] > 0.f && hist[i - 1] <= 0.f) ||
           (hist[i] < 0.f && hist[i - 1] >= 0.f))) {
        crossings.push_back({i, bars.high[static_cast<std::size_t>(i)],
                             bars.low[static_cast<std::size_t>(i)]});
        while (static_cast<int>(crossings.size()) > ncos) crossings.pop_front();
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp01_trend_follower --------------------------------------------------
// Ratcheting volatility bands: EMA(xmalen) closing beyond the active band in
// the direction of the 4/9/18 SMA stack enters and shifts the bands by
// mult * Average(TrueRange, vollen). Protective stop at the opposite band.
class TrendFollowerV1 final : public IStrategy {
public:
  std::string name() const override { return "exp01_trend_follower"; }

  std::vector<ParamDef> params() const override {
    return {
        {"xmalen", 4, 2, 20, 1, true},
        {"vollen", 8, 3, 30, 1, true},
        {"mult", 2.0, 0.5, 5.0, 0.25, false},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("mult") > 0.0;
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::Close, {ps.get("xmalen")}},
        {Family::SMA, Column::Close, {4}},
        {Family::SMA, Column::Close, {9}},
        {Family::SMA, Column::Close, {18}},
        {Family::TRANGE, Column::Close, {}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* xavg = ind.line(0);
    const float* fast = ind.line(1);
    const float* med = ind.line(2);
    const float* slow = ind.line(3);
    const int n = bars.size();
    const float mult = static_cast<float>(ps.get("mult"));
    std::vector<float> vol = rolling_mean(ind.line(4), n, ps.get_int("vollen"));

    int pos = 0;
    bool init = false;
    float top = 0.f, bot = 0.f;

    for (int i = 0; i < n; ++i) {
      const bool ready = finite(xavg[i]) && finite(slow[i]) &&
                         finite(vol[static_cast<std::size_t>(i)]);
      if (!ready) {
        out[i] = 0;
        continue;
      }
      if (!init) {
        top = bars.open[static_cast<std::size_t>(i)] +
              vol[static_cast<std::size_t>(i)] * mult;
        bot = bars.open[static_cast<std::size_t>(i)] -
              vol[static_cast<std::size_t>(i)] * mult;
        init = true;
      }
      // Protective stops at the opposite band, armed at the previous close.
      if (pos == 1 && sell_stop_hit(bars, i, bot)) pos = 0;
      else if (pos == -1 && buy_stop_hit(bars, i, top)) pos = 0;

      const bool bull = fast[i] > med[i] && med[i] > slow[i];
      const bool bear = fast[i] < med[i] && med[i] < slow[i];
      if (xavg[i] > top && bull) {
        pos = 1;
        bot = top;
        top = top + vol[static_cast<std::size_t>(i)] * mult;
      } else if (xavg[i] < bot && bear) {
        pos = -1;
        top = bot;
        bot = bot - vol[static_cast<std::size_t>(i)] * mult;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp01_sr_levels ---------------------------------------------------------
// Support/resistance from the average of the last `numswing` swing highs and
// lows (strength-bar pivots). Inside a wide-enough range, EMA(Low,4) crossing
// back above support buys, EMA(High,4) crossing under resistance sells short.
// Trailing exits only once the position shows a profit.
class SupportResistanceV1 final : public IStrategy {
public:
  std::string name() const override { return "exp01_sr_levels"; }

  std::vector<ParamDef> params() const override {
    return {
        {"strength", 5, 2, 10, 1, true},
        {"numswing", 3, 2, 6, 1, true},
        {"trailbar", 5, 2, 20, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::EMA, Column::High, {4}},
        {Family::EMA, Column::Low, {4}},
        {Family::MAX, Column::High, {ps.get("trailbar")}},
        {Family::MIN, Column::Low, {ps.get("trailbar")}},
        {Family::TRANGE, Column::Close, {}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* emaHigh = ind.line(0);
    const float* emaLow = ind.line(1);
    const float* hiTrail = ind.line(2);
    const float* loTrail = ind.line(3);
    const int n = bars.size();
    const int strength = ps.get_int("strength");
    const int numswing = ps.get_int("numswing");
    std::vector<float> atr10 = rolling_mean(ind.line(4), n, 10);

    std::deque<float> swingHighs, swingLows;
    int pos = 0;
    float entryRef = 0.f;
    float prevR = std::numeric_limits<float>::quiet_NaN();
    float prevS = std::numeric_limits<float>::quiet_NaN();
    float prevEmaH = std::numeric_limits<float>::quiet_NaN();
    float prevEmaL = std::numeric_limits<float>::quiet_NaN();

    auto is_swing_high = [&](int b) {
      for (int k = 1; k <= strength; ++k) {
        if (bars.high[static_cast<std::size_t>(b)] <
            bars.high[static_cast<std::size_t>(b - k)]) return false;
        if (bars.high[static_cast<std::size_t>(b)] <=
            bars.high[static_cast<std::size_t>(b + k)]) return false;
      }
      return true;
    };
    auto is_swing_low = [&](int b) {
      for (int k = 1; k <= strength; ++k) {
        if (bars.low[static_cast<std::size_t>(b)] >
            bars.low[static_cast<std::size_t>(b - k)]) return false;
        if (bars.low[static_cast<std::size_t>(b)] >=
            bars.low[static_cast<std::size_t>(b + k)]) return false;
      }
      return true;
    };

    for (int i = 0; i < n; ++i) {
      // Confirm swings `strength` bars back, using only data up to bar i.
      const int b = i - strength;
      if (b - strength >= 0) {
        if (is_swing_high(b)) {
          swingHighs.push_front(bars.high[static_cast<std::size_t>(b)]);
          while (static_cast<int>(swingHighs.size()) > numswing) swingHighs.pop_back();
        }
        if (is_swing_low(b)) {
          swingLows.push_front(bars.low[static_cast<std::size_t>(b)]);
          while (static_cast<int>(swingLows.size()) > numswing) swingLows.pop_back();
        }
      }

      // Profit-gated trailing stops armed at the previous close.
      if (pos != 0 && i > 0 && finite(hiTrail[i - 1]) && finite(loTrail[i - 1])) {
        const float pc = bars.close[static_cast<std::size_t>(i - 1)];
        const bool profitable = pos == 1 ? pc > entryRef : pc < entryRef;
        if (profitable) {
          if (pos == 1 && sell_stop_hit(bars, i, loTrail[i - 1])) pos = 0;
          else if (pos == -1 && buy_stop_hit(bars, i, hiTrail[i - 1])) pos = 0;
        }
      }

      if (static_cast<int>(swingHighs.size()) == numswing &&
          static_cast<int>(swingLows.size()) == numswing &&
          finite(atr10[static_cast<std::size_t>(i)]) && finite(emaHigh[i]) &&
          finite(emaLow[i])) {
        float rLevel = 0.f, sLevel = 0.f, hh = swingHighs[0], ll = swingLows[0];
        for (float v : swingHighs) { rLevel += v; hh = std::max(hh, v); }
        for (float v : swingLows) { sLevel += v; ll = std::min(ll, v); }
        rLevel /= numswing;
        sLevel /= numswing;

        const float c = bars.close[static_cast<std::size_t>(i)];
        const float span = hh - ll;
        const bool cond1 =
            rLevel - sLevel > 2.f * atr10[static_cast<std::size_t>(i)];
        const bool cond2 = span > 0.f && (rLevel - ll) / span > 0.66f &&
                           (sLevel - ll) / span < 0.33f;
        const bool cond3 = c < rLevel && c > sLevel;
        if (cond1 && cond2 && cond3 && finite(prevEmaH) && finite(prevR)) {
          if (emaHigh[i] < rLevel && prevEmaH >= prevR) {
            pos = -1;
            entryRef = c;
          }
          if (emaLow[i] > sLevel && prevEmaL <= prevS) {
            pos = 1;
            entryRef = c;
          }
        }
        prevR = rLevel;
        prevS = sLevel;
      }
      prevEmaH = emaHigh[i];
      prevEmaL = emaLow[i];
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp01_pivot_points ------------------------------------------------------
// Classic floor-trader pivots from the previous bar; in a 4/9/18 SMA bull
// stack a breakout through R2 buys, in a bear stack a break through S2 sells.
// Exits: first profitable open, else the `trailbar` extreme.
class PivotPointsV1 final : public IStrategy {
public:
  std::string name() const override { return "exp01_pivot_points"; }

  std::vector<ParamDef> params() const override {
    return {
        {"trailbar", 4, 2, 20, 1, true},
        {"fast", 4, 2, 20, 1, true},
        {"med", 9, 3, 60, 1, true},
        {"slow", 18, 5, 120, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("fast") < ps.get_int("med") &&
           ps.get_int("med") < ps.get_int("slow");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::SMA, Column::Close, {ps.get("fast")}},
        {Family::SMA, Column::Close, {ps.get("med")}},
        {Family::SMA, Column::Close, {ps.get("slow")}},
        {Family::MAX, Column::High, {ps.get("trailbar")}},
        {Family::MIN, Column::Low, {ps.get("trailbar")}},
    };
  }

  void signals(const Series& bars, const ParamSet&, const IndicatorView& ind,
               Signal* out) const override {
    const float* fast = ind.line(0);
    const float* med = ind.line(1);
    const float* slow = ind.line(2);
    const float* hiTrail = ind.line(3);
    const float* loTrail = ind.line(4);
    const int n = bars.size();

    int pos = 0;
    float entryRef = 0.f;

    for (int i = 0; i < n; ++i) {
      if (i == 0 || !finite(slow[i])) {
        out[i] = 0;
        continue;
      }
      // Exits armed at the previous close.
      if (pos != 0) {
        const float o = bars.open[static_cast<std::size_t>(i)];
        const bool profitOpen = pos == 1 ? o > entryRef : o < entryRef;
        if (profitOpen) {
          pos = 0;
        } else if (finite(loTrail[i - 1]) && finite(hiTrail[i - 1])) {
          if (pos == 1 && sell_stop_hit(bars, i, loTrail[i - 1])) pos = 0;
          else if (pos == -1 && buy_stop_hit(bars, i, hiTrail[i - 1])) pos = 0;
        }
      }
      // Entry stops armed at the previous close from that bar's pivots.
      if (finite(slow[i - 1])) {
        const auto prev = static_cast<std::size_t>(i - 1);
        const float pivot =
            (bars.low[prev] + bars.high[prev] + bars.close[prev]) / 3.f;
        const float range = bars.high[prev] - bars.low[prev];
        const float r2 = pivot + range;
        const float s2 = pivot - range;
        const bool bull = fast[i - 1] > med[i - 1] && med[i - 1] > slow[i - 1];
        const bool bear = fast[i - 1] < med[i - 1] && med[i - 1] < slow[i - 1];
        if (pos != 1 && bull && buy_stop_hit(bars, i, r2)) {
          pos = 1;
          entryRef = bars.close[static_cast<std::size_t>(i)];
        } else if (pos != -1 && bear && sell_stop_hit(bars, i, s2)) {
          pos = -1;
          entryRef = bars.close[static_cast<std::size_t>(i)];
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp01_channel_breakout --------------------------------------------------
// Donchian-style channel; the `brkouts`-th same-side breakout (with no
// opposite breakout in between) enters at the close. Trailing exits at the
// `trailstp` extreme.
class ChannelBreakoutV1 final : public IStrategy {
public:
  std::string name() const override { return "exp01_channel_breakout"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 10, 4, 60, 1, true},
        {"brkouts", 2, 1, 5, 1, true},
        {"trailstp", 6, 2, 30, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("length")}},
        {Family::MIN, Column::Low, {ps.get("length")}},
        {Family::MAX, Column::High, {ps.get("trailstp")}},
        {Family::MIN, Column::Low, {ps.get("trailstp")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiCh = ind.line(0);
    const float* loCh = ind.line(1);
    const float* hiTrail = ind.line(2);
    const float* loTrail = ind.line(3);
    const int n = bars.size();
    const int need = ps.get_int("brkouts");

    int pos = 0;
    int upCount = 0, downCount = 0;

    for (int i = 0; i < n; ++i) {
      if (i == 0 || !finite(hiCh[i - 1]) || !finite(loCh[i - 1])) {
        out[i] = 0;
        continue;
      }
      // Unconditional trailing stops armed at the previous close.
      if (pos != 0 && finite(loTrail[i - 1]) && finite(hiTrail[i - 1])) {
        if (pos == 1 && sell_stop_hit(bars, i, loTrail[i - 1])) pos = 0;
        else if (pos == -1 && buy_stop_hit(bars, i, hiTrail[i - 1])) pos = 0;
      }
      if (bars.high[static_cast<std::size_t>(i)] > hiCh[i - 1]) {
        ++upCount;
        downCount = 0;
      }
      if (bars.low[static_cast<std::size_t>(i)] < loCh[i - 1]) {
        ++downCount;
        upCount = 0;
      }
      if (upCount >= need) {
        pos = 1;
        upCount = 1;
      }
      if (downCount >= need) {
        pos = -1;
        downCount = 1;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp01_volex --------------------------------------------------------------
// Volatility expansion: breakout stops at today's open +/- mult * 4-bar
// average range. Exits: first profitable open, `trailbar` extreme, or a
// stale-position exit after 4 bars.
class VolExV1 final : public IStrategy {
public:
  std::string name() const override { return "exp01_volex"; }

  std::vector<ParamDef> params() const override {
    return {
        {"lmult", 1.2, 0.4, 4.0, 0.1, false},
        {"smult", 1.8, 0.4, 4.0, 0.1, false},
        {"trailbar", 4, 2, 20, 1, true},
        {"maxbars", 4, 1, 20, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::MAX, Column::High, {ps.get("trailbar")}},
        {Family::MIN, Column::Low, {ps.get("trailbar")}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiTrail = ind.line(0);
    const float* loTrail = ind.line(1);
    const int n = bars.size();
    const float lmult = static_cast<float>(ps.get("lmult"));
    const float smult = static_cast<float>(ps.get("smult"));
    const int maxBars = ps.get_int("maxbars");
    std::vector<float> rng4 = avg_range(bars, 4);

    int pos = 0, barsIn = 0;
    float entryRef = 0.f;

    for (int i = 0; i < n; ++i) {
      if (i == 0 || !finite(rng4[static_cast<std::size_t>(i - 1)])) {
        out[i] = 0;
        continue;
      }
      if (pos != 0) {
        ++barsIn;
        const float o = bars.open[static_cast<std::size_t>(i)];
        const bool profitOpen = pos == 1 ? o > entryRef : o < entryRef;
        bool exited = false;
        if (profitOpen) {
          pos = 0;
          exited = true;
        } else if (finite(loTrail[i - 1]) && finite(hiTrail[i - 1])) {
          if (pos == 1 && sell_stop_hit(bars, i, loTrail[i - 1])) { pos = 0; exited = true; }
          else if (pos == -1 && buy_stop_hit(bars, i, hiTrail[i - 1])) { pos = 0; exited = true; }
        }
        if (!exited && barsIn > maxBars) pos = 0;
      }
      // Breakout stops off today's open; the level uses yesterday's average
      // range, exactly as the original next-bar order does.
      const float o = bars.open[static_cast<std::size_t>(i)];
      const float buyLvl = o + rng4[static_cast<std::size_t>(i - 1)] * lmult;
      const float sellLvl = o - rng4[static_cast<std::size_t>(i - 1)] * smult;
      if (pos != 1 && buy_stop_hit(bars, i, buyLvl)) {
        pos = 1;
        barsIn = 0;
        entryRef = bars.close[static_cast<std::size_t>(i)];
      } else if (pos != -1 && sell_stop_hit(bars, i, sellLvl)) {
        pos = -1;
        barsIn = 0;
        entryRef = bars.close[static_cast<std::size_t>(i)];
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

} // namespace

void register_vol01(std::vector<Factory>& out) {
  out.push_back({"exp01_ma_crossover",
                 [] { return std::unique_ptr<IStrategy>(new MaCrossoverV1); }});
  out.push_back({"exp01_macd_breakout",
                 [] { return std::unique_ptr<IStrategy>(new MacdBreakoutV1); }});
  out.push_back({"exp01_trend_follower",
                 [] { return std::unique_ptr<IStrategy>(new TrendFollowerV1); }});
  out.push_back({"exp01_sr_levels",
                 [] { return std::unique_ptr<IStrategy>(new SupportResistanceV1); }});
  out.push_back({"exp01_pivot_points",
                 [] { return std::unique_ptr<IStrategy>(new PivotPointsV1); }});
  out.push_back({"exp01_channel_breakout",
                 [] { return std::unique_ptr<IStrategy>(new ChannelBreakoutV1); }});
  out.push_back({"exp01_volex",
                 [] { return std::unique_ptr<IStrategy>(new VolExV1); }});
}

} // namespace topt::exp
