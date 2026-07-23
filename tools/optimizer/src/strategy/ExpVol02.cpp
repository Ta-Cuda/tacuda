/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// EXPERIMENTS pack, volume 2 (Omega Research STAD Club, Vol. 2).
// Systems: ADX Differential, Head & Shoulders, OBV New Highs/Lows,
// Advance/Decline Divergence, Stochastics, CCI Bollinger Bands,
// Percent Change Channel, Bull Beater.
// Skipped from this volume: Opening Gap (needs intraday bars plus a daily
// Data2 series). See docs/EXPERIMENTS.md for the adaptation notes.

#include "strategy/ExpRegistry.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <limits>
#include <vector>

#include "strategy/ExpHelpers.h"

namespace topt::exp {

namespace {

// The classic STAD exit used across this volume: at the first close in a
// position the stop starts `mult` average-ranges beyond the extreme, then
// converges toward price by `conv` of the remaining distance every bar.
struct StadTrail {
  bool set = false;
  float stop = 0.f;

  void reset() { set = false; stop = 0.f; }

  // Returns true when the protective stop was touched during bar i.
  bool step_long(const Series& bars, int i, float ar4, float mult, float conv) {
    if (!set) {
      if (!finite(ar4)) return false;
      stop = L(bars, i) - ar4 * mult;
      set = true;
      return false;
    }
    if (L(bars, i) <= stop) return true;
    stop += conv * (L(bars, i) - stop);
    return false;
  }
  bool step_short(const Series& bars, int i, float ar4, float mult, float conv) {
    if (!set) {
      if (!finite(ar4)) return false;
      stop = H(bars, i) + ar4 * mult;
      set = true;
      return false;
    }
    if (H(bars, i) >= stop) return true;
    stop -= conv * (stop - H(bars, i));
    return false;
  }
};

// When buy and sell stops are both touched inside one bar, pick the side the
// close settled nearer to (the intrabar path is unknown on OHLC data).
inline int pick_side(const Series& bars, int i, float buyLvl, float sellLvl) {
  return std::abs(C(bars, i) - buyLvl) <= std::abs(C(bars, i) - sellLvl) ? 1 : -1;
}

// --- exp02_adx_diff ----------------------------------------------------------
// A weighted average of the 1-bar ADX differential below zero (or crossing
// back above it) marks a dying trend; breakout stops go half an average
// range above/below that close for `ch` bars. Exits: 6-bar channel on the
// entry bar, then the STAD converging trail.
class AdxDiffV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_adx_diff"; }

  std::vector<ParamDef> params() const override {
    return {
        {"adxlen", 14, 5, 40, 1, true},
        {"avglen", 4, 2, 12, 1, true},
        {"ch", 8, 2, 20, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::ADX, Column::Close, {ps.get("adxlen")}},
        {Family::MAX, Column::High, {6}},
        {Family::MIN, Column::Low, {6}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* adx = ind.line(0);
    const float* hi6 = ind.line(1);
    const float* lo6 = ind.line(2);
    const int n = bars.size();
    const int avglen = ps.get_int("avglen");
    const int ch = ps.get_int("ch");
    const std::vector<float> ar4 = avg_range(bars, 4);

    // WAverage of the ADX 1-bar differential (weights avglen..1, newest first).
    std::vector<float> deltaAvg(static_cast<std::size_t>(n),
                                std::numeric_limits<float>::quiet_NaN());
    for (int i = 1; i < n; ++i) {
      double num = 0.0, den = 0.0;
      bool ok = true;
      for (int k = 0; k < avglen; ++k) {
        const int j = i - k;
        if (j < 1 || !finite(adx[j]) || !finite(adx[j - 1])) { ok = false; break; }
        const double w = avglen - k;
        num += w * (static_cast<double>(adx[j]) - adx[j - 1]);
        den += w;
      }
      if (ok) deltaAvg[static_cast<std::size_t>(i)] = static_cast<float>(num / den);
    }

    int pos = 0;
    float upCh = 0.f, dnCh = 0.f;
    int armBar = -1;             // bar whose close armed the entry stops
    int fillBar = -1;            // decision bar of the current entry
    StadTrail trail;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position.
      if (pos != 0) {
        if (!trail.set && i == fillBar + 1) {
          // Entry-bar channel exit armed with the pre-entry 6-bar extreme.
          if (pos == 1 && i >= 1 && finite(lo6[i - 1]) && L(bars, i) <= lo6[i - 1]) {
            pos = 0; trail.reset();
          } else if (pos == -1 && i >= 1 && finite(hi6[i - 1]) &&
                     H(bars, i) >= hi6[i - 1]) {
            pos = 0; trail.reset();
          }
        }
        if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
          pos = 0; trail.reset();
        } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
          pos = 0; trail.reset();
        }
      }
      // 2) Pending breakout stops (active for `ch` bars after arming).
      if (pos == 0 && armBar >= 0 && i > armBar && i <= armBar + ch) {
        const bool up = buy_stop_hit(bars, i, upCh);
        const bool dn = sell_stop_hit(bars, i, dnCh);
        if (up || dn) {
          pos = (up && dn) ? pick_side(bars, i, upCh, dnCh) : (up ? 1 : -1);
          fillBar = i;
          armBar = -1;
          trail.reset();
        }
      }
      // 3) Re-arm on trend-death signals (flat only).
      if (pos == 0 && finite(deltaAvg[ui]) && finite(ar4[ui])) {
        const bool crossUp = i >= 1 && finite(deltaAvg[ui - 1]) &&
                             deltaAvg[ui] > 0.f && deltaAvg[ui - 1] <= 0.f;
        if (deltaAvg[ui] < 0.f || crossUp) {
          upCh = C(bars, i) + ar4[ui] / 2.f;
          dnCh = C(bars, i) - ar4[ui] / 2.f;
          armBar = i;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp02_head_shoulders ----------------------------------------------------
// Swing-based Head & Shoulders detection: three swing highs where the middle
// one towers over both shoulders arm a short at the neckline drawn through
// the two swing lows between them (mirrored for bottoms). Exits use the
// STAD converging trail.
class HeadShouldersV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_head_shoulders"; }

  std::vector<ParamDef> params() const override {
    return {
        {"stren", 5, 2, 8, 1, true},
        {"maxage", 40, 10, 100, 5, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    // Swing detection and the average range run on raw bars; no precomputed
    // indicator lines are needed.
    (void)ps;
    return {};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    (void)ind;
    const int n = bars.size();
    const int s = ps.get_int("stren");
    const int maxAge = ps.get_int("maxage");
    const std::vector<float> ar4 = avg_range(bars, 4);

    struct Swing { int bar; float price; };
    std::deque<Swing> highs, lows;

    // Armed necklines: y(i) = p1 + slope * (i - b1), valid until crossed,
    // replaced, or `maxage` bars old.
    bool topArmed = false, botArmed = false;
    int topB1 = 0, botB1 = 0, topBirth = 0, botBirth = 0;
    float topP1 = 0.f, topSlope = 0.f, botP1 = 0.f, botSlope = 0.f;

    int pos = 0;
    StadTrail trail;

    auto neck = [](float p1, float slope, int b1, int i) {
      return p1 + slope * static_cast<float>(i - b1);
    };

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position.
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }

      // 2) Neckline crossings observed at this close (entries reverse).
      if (topArmed && i - topBirth > maxAge) topArmed = false;
      if (botArmed && i - botBirth > maxAge) botArmed = false;
      if (topArmed && i >= 1) {
        const float y = neck(topP1, topSlope, topB1, i);
        const float yPrev = neck(topP1, topSlope, topB1, i - 1);
        if (C(bars, i) < y && C(bars, i - 1) >= yPrev && pos != -1) {
          pos = -1; trail.reset(); topArmed = false;
        }
      }
      if (botArmed && i >= 1) {
        const float y = neck(botP1, botSlope, botB1, i);
        const float yPrev = neck(botP1, botSlope, botB1, i - 1);
        if (C(bars, i) > y && C(bars, i - 1) <= yPrev && pos != 1) {
          pos = 1; trail.reset(); botArmed = false;
        }
      }

      // 3) Swing detection confirmed at bar i (pivot sits at bar i - s).
      const int b = i - s;
      if (b >= s) {
        bool isHigh = true, isLow = true;
        for (int k = 1; k <= s && (isHigh || isLow); ++k) {
          if (!(H(bars, b) > H(bars, b - k) && H(bars, b) >= H(bars, b + k)))
            isHigh = false;
          if (!(L(bars, b) < L(bars, b - k) && L(bars, b) <= L(bars, b + k)))
            isLow = false;
        }
        if (isHigh) {
          highs.push_back({b, H(bars, b)});
          if (highs.size() > 3) highs.pop_front();
          // Head & Shoulders top: middle swing high above both neighbours.
          if (highs.size() == 3 && highs[1].price > highs[0].price &&
              highs[1].price > highs[2].price) {
            // Neckline through the two most recent swing lows.
            if (lows.size() >= 2) {
              const Swing& a = lows[lows.size() - 2];
              const Swing& c = lows[lows.size() - 1];
              if (c.bar > a.bar) {
                topP1 = a.price;
                topSlope = (c.price - a.price) / static_cast<float>(c.bar - a.bar);
                topB1 = a.bar;
                topBirth = i;
                topArmed = true;
              }
            }
          }
        }
        if (isLow) {
          lows.push_back({b, L(bars, b)});
          if (lows.size() > 3) lows.pop_front();
          // Head & Shoulders bottom: middle swing low below both neighbours.
          if (lows.size() == 3 && lows[1].price < lows[0].price &&
              lows[1].price < lows[2].price) {
            if (highs.size() >= 2) {
              const Swing& a = highs[highs.size() - 2];
              const Swing& c = highs[highs.size() - 1];
              if (c.bar > a.bar) {
                botP1 = a.price;
                botSlope = (c.price - a.price) / static_cast<float>(c.bar - a.bar);
                botB1 = a.bar;
                botBirth = i;
                botArmed = true;
              }
            }
          }
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp02_obv_extremes ------------------------------------------------------
// Long-only: buy when On Balance Volume printed `newhighs` window-highs (or
// `newlows` window-lows, the contrarian double-size entry of the original)
// within the last `barsback` bars. Exits use the STAD converging trail.
class ObvExtremesV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_obv_extremes"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 15, 5, 50, 1, true},
        {"barsback", 6, 2, 12, 1, true},
        {"newhighs", 5, 1, 6, 1, true},
        {"newlows", 3, 1, 6, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get_int("newhighs") <= ps.get_int("barsback") &&
           ps.get_int("newlows") <= ps.get_int("barsback");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    (void)ps;
    return {{Family::OBV, Column::Close, {}}};
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* obv = ind.line(0);
    const int n = bars.size();
    const int length = ps.get_int("length");
    const int barsback = ps.get_int("barsback");
    const int newhighs = ps.get_int("newhighs");
    const int newlows = ps.get_int("newlows");
    const std::vector<float> ar4 = avg_range(bars, 4);
    const std::vector<float> hi = rolling_max(obv, n, length);
    const std::vector<float> lo = rolling_min(obv, n, length);

    std::vector<bool> isNewHigh(static_cast<std::size_t>(n), false);
    std::vector<bool> isNewLow(static_cast<std::size_t>(n), false);
    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (finite(obv[i]) && finite(hi[ui])) isNewHigh[ui] = obv[i] >= hi[ui];
      if (finite(obv[i]) && finite(lo[ui])) isNewLow[ui] = obv[i] <= lo[ui];
    }

    int pos = 0;
    StadTrail trail;
    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0;
        trail.reset();
      }
      if (pos == 0 && finite(hi[ui]) && finite(lo[ui])) {
        if (occurred_at_least(isNewHigh, i, barsback, newhighs) ||
            occurred_at_least(isNewLow, i, barsback, newlows)) {
          pos = 1;
          trail.reset();
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp02_ad_divergence -----------------------------------------------------
// Volume-flow divergence: a new `lookback`-bar price low taken without (or
// against) a matching low in the smoothed volume-move line buys; the mirror
// image sells. Exits use the STAD converging trail.
class AdDivergenceV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_ad_divergence"; }

  std::vector<ParamDef> params() const override {
    return {
        {"len", 9, 3, 30, 1, true},
        {"lookback", 30, 10, 60, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double lb = ps.get("lookback");
    return {
        {Family::MAX, Column::High, {lb}},
        {Family::MIN, Column::Low, {lb}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiLb = ind.line(0);
    const float* loLb = ind.line(1);
    const int n = bars.size();
    const int len = ps.get_int("len");
    const int lookback = ps.get_int("lookback");
    const std::vector<float> ar4 = avg_range(bars, 4);

    // VolumeMove = XAverage((C-O)/(H-L) * Volume, len).
    std::vector<float> raw(static_cast<std::size_t>(n), 0.f);
    for (int i = 0; i < n; ++i) {
      const float range = H(bars, i) - L(bars, i);
      raw[static_cast<std::size_t>(i)] =
          range > 0.f ? (C(bars, i) - O(bars, i)) / range * V(bars, i) : 0.f;
    }
    const std::vector<float> vm = ema_line(raw.data(), n, len);
    const std::vector<float> vmHi = rolling_max(vm.data(), n, lookback);
    const std::vector<float> vmLo = rolling_min(vm.data(), n, lookback);

    std::vector<bool> vmNewHigh(static_cast<std::size_t>(n), false);
    std::vector<bool> vmNewLow(static_cast<std::size_t>(n), false);
    for (int i = 1; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (finite(vm[ui]) && finite(vmHi[ui - 1])) vmNewHigh[ui] = vm[ui] > vmHi[ui - 1];
      if (finite(vm[ui]) && finite(vmLo[ui - 1])) vmNewLow[ui] = vm[ui] < vmLo[ui - 1];
    }

    int pos = 0;
    StadTrail trail;
    for (int i = 1; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }

      if (!finite(hiLb[i - 1]) || !finite(loLb[i - 1]) || !finite(vm[ui])) {
        out[i] = static_cast<Signal>(pos);
        continue;
      }
      const bool newPriceLow = L(bars, i) < loLb[i - 1];
      const bool newPriceHigh = H(bars, i) > hiLb[i - 1];
      int dir = 0;
      if (newPriceLow &&
          (!occurred_at_least(vmNewLow, i, 5, 1) || vmNewHigh[ui])) {
        dir = 1;
      } else if (newPriceHigh &&
                 (!occurred_at_least(vmNewHigh, i, 5, 1) || vmNewLow[ui])) {
        dir = -1;
      }
      if (dir != 0 && dir != pos) {
        pos = dir;
        trail.reset();
      }
      out[i] = static_cast<Signal>(pos);
    }
    if (n > 0) out[0] = 0;
  }
};

// --- exp02_stochastics -------------------------------------------------------
// SlowK pinned beyond the overbought/oversold band for 3 straight bars puts
// on a with-trend position; the converging reversal stop flips it to a
// double-size fade, and the fade leg exits on a 5-bar channel break.
class StochasticsV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_stochastics"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 14, 5, 40, 1, true},
        {"ob", 75, 55, 95, 1, true},
        {"os", 25, 5, 45, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("os") < ps.get("ob");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    return {
        {Family::STOCH, Column::Close, {ps.get("length"), 3}},
        {Family::MAX, Column::High, {5}},
        {Family::MIN, Column::Low, {5}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* slowK = ind.line(0, 0);
    const float* hi5 = ind.line(1);
    const float* lo5 = ind.line(2);
    const int n = bars.size();
    const float ob = static_cast<float>(ps.get("ob"));
    const float os = static_cast<float>(ps.get("os"));
    const std::vector<float> ar4 = avg_range(bars, 4);

    enum class St { Flat, LongSig, ShortSig, ShortRev, LongRev };
    St st = St::Flat;
    int pos = 0;
    int fillBar = -1;
    bool rpSet = false;
    float rp = 0.f;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position.
      switch (st) {
      case St::LongSig:
        if (i > fillBar) {
          if (!rpSet) {
            if (finite(ar4[ui])) { rp = L(bars, i) - ar4[ui]; rpSet = true; }
          } else if (L(bars, i) <= rp) {
            st = St::ShortRev; pos = -1; rpSet = false; // reversal fill next open
          } else {
            rp += (L(bars, i) - rp) / 3.f;
          }
        }
        break;
      case St::ShortSig:
        if (i > fillBar) {
          if (!rpSet) {
            if (finite(ar4[ui])) { rp = H(bars, i) + ar4[ui]; rpSet = true; }
          } else if (H(bars, i) >= rp) {
            st = St::LongRev; pos = 1; rpSet = false;
          } else {
            rp -= (rp - H(bars, i)) / 3.f;
          }
        }
        break;
      case St::ShortRev:
        if (i >= 1 && finite(hi5[i - 1]) && H(bars, i) >= hi5[i - 1]) {
          st = St::Flat; pos = 0;
        }
        break;
      case St::LongRev:
        if (i >= 1 && finite(lo5[i - 1]) && L(bars, i) <= lo5[i - 1]) {
          st = St::Flat; pos = 0;
        }
        break;
      case St::Flat:
        break;
      }
      // 2) Fresh band signals from a flat book.
      if (st == St::Flat && i >= 2 && finite(slowK[i]) && finite(slowK[i - 1]) &&
          finite(slowK[i - 2])) {
        const bool pinnedHigh =
            slowK[i] >= ob && slowK[i - 1] >= ob && slowK[i - 2] >= ob;
        const bool pinnedLow =
            slowK[i] <= os && slowK[i - 1] <= os && slowK[i - 2] <= os;
        if (pinnedHigh) {
          st = St::LongSig; pos = 1; fillBar = i + 1; rpSet = false;
        } else if (pinnedLow) {
          st = St::ShortSig; pos = -1; fillBar = i + 1; rpSet = false;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp02_cci_bollinger -----------------------------------------------------
// Prices poking outside the Bollinger Bands set the hunting mode; a smoothed
// CCI hooking back from beyond +/-100 arms a 3-bar-extreme entry stop that
// stays live for five bars. Exits use a tightened converging trail.
class CciBollingerV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_cci_bollinger"; }

  std::vector<ParamDef> params() const override {
    return {
        {"bblen", 9, 5, 30, 1, true},
        {"bbdev", 2.0, 1.0, 3.0, 0.25, false},
        {"ccilen", 14, 5, 40, 1, true},
        {"cciavg", 2, 1, 8, 1, true},
    };
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double dev = ps.get("bbdev");
    return {
        {Family::BBANDS, Column::Close, {ps.get("bblen"), dev, dev}},
        {Family::CCI, Column::Close, {ps.get("ccilen")}},
        {Family::MAX, Column::High, {3}},
        {Family::MIN, Column::Low, {3}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* upper = ind.line(0, 0);
    const float* lower = ind.line(0, 2);
    const float* cci = ind.line(1);
    const float* hi3 = ind.line(2);
    const float* lo3 = ind.line(3);
    const int n = bars.size();
    const std::vector<float> ar4 = avg_range(bars, 4);
    const std::vector<float> cciAvg = ema_line(cci, n, ps.get_int("cciavg"));

    int bbStatus = 0;
    float prevPosLine = std::numeric_limits<float>::quiet_NaN();
    int pos = 0;
    int longUntil = -1, shortUntil = -1;
    float lep = 0.f, sep = 0.f;
    StadTrail trail;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position (tightened STAD trail: 2/3 range, 0.3).
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 2.f / 3.f, 0.3f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 2.f / 3.f, 0.3f)) {
        pos = 0; trail.reset();
      }
      // 2) Armed entry stops, live for 5 bars after the signal.
      if (pos == 0) {
        const bool canLong = i <= longUntil && buy_stop_hit(bars, i, lep);
        const bool canShort = i <= shortUntil && sell_stop_hit(bars, i, sep);
        if (canLong || canShort) {
          pos = (canLong && canShort) ? pick_side(bars, i, lep, sep)
                                      : (canLong ? 1 : -1);
          trail.reset();
          longUntil = shortUntil = -1;
        }
      }
      // 3) Band status bookkeeping (same precedence as the original).
      if (finite(upper[i]) && finite(lower[i]) && upper[i] != lower[i]) {
        const float posLine = (C(bars, i) - lower[i]) / (upper[i] - lower[i]);
        if (i >= 1 && finite(prevPosLine)) {
          if ((posLine > 0.5f && prevPosLine <= 0.5f) ||
              (posLine < 0.5f && prevPosLine >= 0.5f)) {
            bbStatus = 0;
          }
          if (i >= 1 && finite(lower[i - 1]) && L(bars, i) < lower[i] &&
              L(bars, i - 1) >= lower[i - 1]) {
            bbStatus = -1;
          }
          if (i >= 1 && finite(upper[i - 1]) && H(bars, i) > upper[i] &&
              H(bars, i - 1) <= upper[i - 1]) {
            bbStatus = 1;
          }
        }
        prevPosLine = posLine;
      }
      // 4) CCI hooks arm new entry stops (only meaningful when flat).
      const auto avgAt = [&](int j) {
        return j >= 0 ? cciAvg[static_cast<std::size_t>(j)]
                      : std::numeric_limits<float>::quiet_NaN();
      };
      if (pos == 0 && i >= 4) {
        const bool turnsUp = finite(avgAt(i)) && finite(avgAt(i - 1)) &&
                             finite(avgAt(i - 2)) && avgAt(i - 1) < avgAt(i - 2) &&
                             avgAt(i - 1) <= avgAt(i) && avgAt(i - 1) < -100.f;
        const bool turnsDown = finite(avgAt(i)) && finite(avgAt(i - 4)) &&
                               avgAt(i - 2) > avgAt(i - 3) &&
                               avgAt(i - 2) > avgAt(i - 4) &&
                               avgAt(i - 2) >= avgAt(i - 1) &&
                               avgAt(i - 2) >= avgAt(i) && avgAt(i - 2) > 100.f;
        if (bbStatus == -1 && turnsUp && finite(hi3[i])) {
          lep = hi3[i];
          longUntil = i + 5;
        }
        if (bbStatus == 1 && turnsDown && finite(lo3[i])) {
          sep = lo3[i];
          shortUntil = i + 5;
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp02_pct_change_channel ------------------------------------------------
// A move of more than ~2% over `pchng` bars arms a fade-the-extreme stop at
// the recent 3-bar extreme for three bars. Exits: first profitable open or
// a 5-bar channel break.
class PctChangeChannelV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_pct_change_channel"; }

  std::vector<ParamDef> params() const override {
    return {
        {"pchng", 5, 2, 20, 1, true},
        {"obought", 102.0, 100.5, 110.0, 0.5, false},
        {"osold", 98.0, 90.0, 99.5, 0.5, false},
        {"bands", 3, 2, 10, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("osold") < 100.0 && ps.get("obought") > 100.0;
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double bands = ps.get("bands");
    return {
        {Family::MAX, Column::High, {bands}},
        {Family::MIN, Column::Low, {bands}},
        {Family::MAX, Column::High, {5}},
        {Family::MIN, Column::Low, {5}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* hiB = ind.line(0);
    const float* loB = ind.line(1);
    const float* hi5 = ind.line(2);
    const float* lo5 = ind.line(3);
    const int n = bars.size();
    const int pchng = ps.get_int("pchng");
    const float ob = static_cast<float>(ps.get("obought"));
    const float os = static_cast<float>(ps.get("osold"));

    int pos = 0;
    float entryRef = 0.f;
    int fillBar = -1;
    float highBand = 0.f, lowBand = 0.f;
    int osUntil = -1, obUntil = -1;

    for (int i = 0; i < n; ++i) {
      // 1) Exits: first profitable open, then the 5-bar channel.
      if (pos != 0 && i > fillBar) {
        if (pos == 1 && O(bars, i) > entryRef) {
          pos = 0;
        } else if (pos == -1 && O(bars, i) < entryRef) {
          pos = 0;
        } else if (pos == 1 && i >= 1 && finite(lo5[i - 1]) &&
                   L(bars, i) <= lo5[i - 1]) {
          pos = 0;
        } else if (pos == -1 && i >= 1 && finite(hi5[i - 1]) &&
                   H(bars, i) >= hi5[i - 1]) {
          pos = 0;
        }
      }
      // 2) Armed fade stops (live for ~3 bars, as the MRO reset works out).
      if (pos == 0) {
        const bool canLong = i <= osUntil && buy_stop_hit(bars, i, highBand);
        const bool canShort = i <= obUntil && sell_stop_hit(bars, i, lowBand);
        if (canLong || canShort) {
          pos = (canLong && canShort) ? pick_side(bars, i, highBand, lowBand)
                                      : (canLong ? 1 : -1);
          entryRef = C(bars, i);
          fillBar = i + 1;
          osUntil = obUntil = -1;
        }
      }
      // 3) Percent-change setup at this close.
      if (i >= pchng) {
        const float prev = C(bars, i - pchng);
        if (prev > 0.f) {
          const float pcnt = C(bars, i) / prev * 100.f;
          if (pcnt < os && finite(hiB[i])) {
            highBand = hiB[i];
            osUntil = i + 3;
          }
          if (pcnt > ob && finite(loB[i])) {
            lowBand = loB[i];
            obUntil = i + 3;
          }
        }
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

// --- exp02_bull_beater -------------------------------------------------------
// Bollinger squeeze (band width under 1.5 average ranges) plus a fast RSI
// tilt arms 5-bar-extreme breakout stops for five bars. Exits use the STAD
// converging trail. The intraday session filter is dropped on daily bars.
class BullBeaterV2 final : public IStrategy {
public:
  std::string name() const override { return "exp02_bull_beater"; }

  std::vector<ParamDef> params() const override {
    return {
        {"length", 9, 5, 30, 1, true},
        {"sdev", 2.0, 1.0, 3.0, 0.25, false},
        {"rsihi", 55, 45, 80, 1, true},
        {"rsilo", 45, 20, 55, 1, true},
    };
  }

  bool validate(const ParamSet& ps) const override {
    return ps.get("rsilo") <= ps.get("rsihi");
  }

  std::vector<IndicatorRequest> indicators(const ParamSet& ps) const override {
    const double dev = ps.get("sdev");
    return {
        {Family::BBANDS, Column::Close, {ps.get("length"), dev, dev}},
        {Family::RSI, Column::Close, {4}},
        {Family::MAX, Column::High, {5}},
        {Family::MIN, Column::Low, {5}},
    };
  }

  void signals(const Series& bars, const ParamSet& ps, const IndicatorView& ind,
               Signal* out) const override {
    const float* upper = ind.line(0, 0);
    const float* lower = ind.line(0, 2);
    const float* rsi = ind.line(1);
    const float* hi5 = ind.line(2);
    const float* lo5 = ind.line(3);
    const int n = bars.size();
    const float rsiHi = static_cast<float>(ps.get("rsihi"));
    const float rsiLo = static_cast<float>(ps.get("rsilo"));
    const std::vector<float> ar4 = avg_range(bars, 4);

    int pos = 0;
    int squeezeBar = -1;
    StadTrail trail;

    for (int i = 0; i < n; ++i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      // 1) Manage the open position.
      if (pos == 1 && trail.step_long(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      } else if (pos == -1 && trail.step_short(bars, i, ar4[ui], 1.f, 1.f / 3.f)) {
        pos = 0; trail.reset();
      }
      // 2) Breakout stops armed at the previous close, live 5 bars after
      //    the squeeze.
      if (pos == 0 && squeezeBar >= 0 && i > squeezeBar && i <= squeezeBar + 5 &&
          i >= 1 && finite(rsi[i - 1]) && finite(hi5[i - 1]) && finite(lo5[i - 1])) {
        const bool canLong = rsi[i - 1] < rsiLo && buy_stop_hit(bars, i, hi5[i - 1]);
        const bool canShort = rsi[i - 1] > rsiHi && sell_stop_hit(bars, i, lo5[i - 1]);
        if (canLong || canShort) {
          pos = (canLong && canShort)
                    ? pick_side(bars, i, hi5[i - 1], lo5[i - 1])
                    : (canLong ? 1 : -1);
          trail.reset();
        }
      }
      // 3) Squeeze detection at this close.
      if (finite(upper[i]) && finite(lower[i]) && finite(ar4[ui]) &&
          (upper[i] - lower[i]) < 1.5f * ar4[ui]) {
        squeezeBar = i;
      }
      out[i] = static_cast<Signal>(pos);
    }
  }
};

} // namespace

void register_vol02(std::vector<Factory>& out) {
  out.push_back({"exp02_adx_diff",
                 [] { return std::unique_ptr<IStrategy>(new AdxDiffV2); }});
  out.push_back({"exp02_head_shoulders",
                 [] { return std::unique_ptr<IStrategy>(new HeadShouldersV2); }});
  out.push_back({"exp02_obv_extremes",
                 [] { return std::unique_ptr<IStrategy>(new ObvExtremesV2); }});
  out.push_back({"exp02_ad_divergence",
                 [] { return std::unique_ptr<IStrategy>(new AdDivergenceV2); }});
  out.push_back({"exp02_stochastics",
                 [] { return std::unique_ptr<IStrategy>(new StochasticsV2); }});
  out.push_back({"exp02_cci_bollinger",
                 [] { return std::unique_ptr<IStrategy>(new CciBollingerV2); }});
  out.push_back({"exp02_pct_change_channel",
                 [] { return std::unique_ptr<IStrategy>(new PctChangeChannelV2); }});
  out.push_back({"exp02_bull_beater",
                 [] { return std::unique_ptr<IStrategy>(new BullBeaterV2); }});
}

} // namespace topt::exp
