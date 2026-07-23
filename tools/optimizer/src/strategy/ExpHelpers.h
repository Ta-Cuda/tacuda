/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_STRATEGY_EXPHELPERS_H
#define TACUDA_OPT_STRATEGY_EXPHELPERS_H

// Shared helpers for the EXPERIMENTS strategy pack (STAD volumes 1-13
// adapted to tacuda-opt's next-bar-open engine).
//
// Execution-model adaptation, used consistently by every strategy here:
// the original systems place intrabar stop/limit orders ("buy next bar at
// X stop"). This engine decides a target position at each bar's close and
// fills it at the next bar's open, so a pending stop level L armed at bar
// i-1 is considered touched when bar i trades through it (high >= L for
// buy stops, low <= L for sell stops); the position change then executes
// at the open of bar i+1. Entries therefore lag the original fills by
// less than one bar and use open prices instead of exact stop prices.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/Types.h"

namespace topt::exp {

inline bool finite(float v) { return std::isfinite(v); }

// Shorthand accessors so strategy loops stay readable.
inline float O(const Series& b, int i) { return b.open[static_cast<std::size_t>(i)]; }
inline float H(const Series& b, int i) { return b.high[static_cast<std::size_t>(i)]; }
inline float L(const Series& b, int i) { return b.low[static_cast<std::size_t>(i)]; }
inline float C(const Series& b, int i) { return b.close[static_cast<std::size_t>(i)]; }
inline float V(const Series& b, int i) { return b.volume[static_cast<std::size_t>(i)]; }

// EasyLanguage-style crossings evaluated at bar i (requires i >= 1 and
// finite values at i-1; callers guard warm-up).
inline bool crosses_above(const float* a, const float* b, int i) {
  return i >= 1 && a[i] > b[i] && a[i - 1] <= b[i - 1];
}
inline bool crosses_below(const float* a, const float* b, int i) {
  return i >= 1 && a[i] < b[i] && a[i - 1] >= b[i - 1];
}
inline bool crosses_above_level(const float* a, float level, int i) {
  return i >= 1 && a[i] > level && a[i - 1] <= level;
}
inline bool crosses_below_level(const float* a, float level, int i) {
  return i >= 1 && a[i] < level && a[i - 1] >= level;
}

// EasyLanguage DayOfWeek convention: 0=Sunday .. 6=Saturday.
inline int day_of_week(std::int64_t unixTs) {
  return static_cast<int>(((unixTs / 86400) % 7 + 7 + 4) % 7);
}

// EasyLanguage Average(Range, n): rolling simple mean of high-low, NaN during
// warm-up. Computed CPU-side because it is a derived series the indicator
// backends do not expose.
inline std::vector<float> avg_range(const Series& bars, int n) {
  const int size = bars.size();
  std::vector<float> out(static_cast<std::size_t>(size),
                         std::numeric_limits<float>::quiet_NaN());
  double sum = 0.0;
  for (int i = 0; i < size; ++i) {
    sum += static_cast<double>(bars.high[static_cast<std::size_t>(i)]) -
           bars.low[static_cast<std::size_t>(i)];
    if (i >= n) {
      sum -= static_cast<double>(bars.high[static_cast<std::size_t>(i - n)]) -
             bars.low[static_cast<std::size_t>(i - n)];
    }
    if (i >= n - 1) out[static_cast<std::size_t>(i)] = static_cast<float>(sum / n);
  }
  return out;
}

// Rolling simple mean over an indicator line; NaN until the line and the
// window are both ready. Assumes NaNs only during the leading warm-up.
inline std::vector<float> rolling_mean(const float* line, int size, int n) {
  std::vector<float> out(static_cast<std::size_t>(size),
                         std::numeric_limits<float>::quiet_NaN());
  double sum = 0.0;
  int have = 0;
  for (int i = 0; i < size; ++i) {
    if (!finite(line[i])) continue;
    sum += line[i];
    ++have;
    if (have > n) sum -= line[i - n];
    if (have >= n) out[static_cast<std::size_t>(i)] = static_cast<float>(sum / n);
  }
  return out;
}

// Rolling highest/lowest of a derived line over the trailing n values ending
// at bar i (EasyLanguage Highest/Lowest applied to a function series).
inline std::vector<float> rolling_max(const float* line, int size, int n) {
  std::vector<float> out(static_cast<std::size_t>(size),
                         std::numeric_limits<float>::quiet_NaN());
  for (int i = 0; i < size; ++i) {
    if (!finite(line[i])) continue;
    float m = line[i];
    bool ok = true;
    for (int k = 1; k < n; ++k) {
      const int j = i - k;
      if (j < 0 || !finite(line[j])) { ok = false; break; }
      m = std::max(m, line[j]);
    }
    if (ok) out[static_cast<std::size_t>(i)] = m;
  }
  return out;
}
inline std::vector<float> rolling_min(const float* line, int size, int n) {
  std::vector<float> out(static_cast<std::size_t>(size),
                         std::numeric_limits<float>::quiet_NaN());
  for (int i = 0; i < size; ++i) {
    if (!finite(line[i])) continue;
    float m = line[i];
    bool ok = true;
    for (int k = 1; k < n; ++k) {
      const int j = i - k;
      if (j < 0 || !finite(line[j])) { ok = false; break; }
      m = std::min(m, line[j]);
    }
    if (ok) out[static_cast<std::size_t>(i)] = m;
  }
  return out;
}

// Rolling population standard deviation of a derived line (EasyLanguage
// StdDev over a function series).
inline std::vector<float> rolling_std(const float* line, int size, int n) {
  std::vector<float> out(static_cast<std::size_t>(size),
                         std::numeric_limits<float>::quiet_NaN());
  for (int i = n - 1; i < size; ++i) {
    double sum = 0.0, sq = 0.0;
    bool ok = true;
    for (int k = 0; k < n; ++k) {
      const float v = line[i - k];
      if (!finite(v)) { ok = false; break; }
      sum += v;
      sq += static_cast<double>(v) * v;
    }
    if (!ok) continue;
    const double mean = sum / n;
    const double var = std::max(0.0, sq / n - mean * mean);
    out[static_cast<std::size_t>(i)] = static_cast<float>(std::sqrt(var));
  }
  return out;
}

// EasyLanguage XAverage over a derived line (seeded with the first finite
// value, alpha = 2/(n+1)).
inline std::vector<float> ema_line(const float* line, int size, int n) {
  std::vector<float> out(static_cast<std::size_t>(size),
                         std::numeric_limits<float>::quiet_NaN());
  const double alpha = 2.0 / (static_cast<double>(n) + 1.0);
  double ema = 0.0;
  bool seeded = false;
  for (int i = 0; i < size; ++i) {
    if (!finite(line[i])) continue;
    if (!seeded) {
      ema = line[i];
      seeded = true;
    } else {
      ema = alpha * line[i] + (1.0 - alpha) * ema;
    }
    out[static_cast<std::size_t>(i)] = static_cast<float>(ema);
  }
  return out;
}

// EasyLanguage MRO(cond, len, instances) > -1: cond was true at least
// `instances` times within the last `len` bars ending at bar i.
inline bool occurred_at_least(const std::vector<bool>& cond, int i, int len,
                              int instances) {
  int cnt = 0;
  for (int k = 0; k < len; ++k) {
    const int j = i - k;
    if (j < 0) break;
    if (cond[static_cast<std::size_t>(j)] && ++cnt >= instances) return true;
  }
  return false;
}

// Pending stop-entry order armed at a level; helper mirrors the
// close-confirmation adaptation described above.
inline bool buy_stop_hit(const Series& bars, int i, float level) {
  return bars.high[static_cast<std::size_t>(i)] >= level;
}
inline bool sell_stop_hit(const Series& bars, int i, float level) {
  return bars.low[static_cast<std::size_t>(i)] <= level;
}

// Position-state tracker shared by strategies that manage protective
// stops/targets expressed in price levels. All checks happen on bar
// close data, keeping signals causal.
struct PosTracker {
  int pos = 0;            // -1, 0, +1
  int barsInPos = 0;      // bars since entry decision
  float entryPrice = 0.f; // reference price at entry decision (close)
  float stop = 0.f;       // protective stop level (0 = unset)
  float target = 0.f;     // profit target level (0 = unset)

  void flat() { pos = 0; barsInPos = 0; entryPrice = 0.f; stop = 0.f; target = 0.f; }

  void enter(int dir, float refPrice) {
    pos = dir;
    barsInPos = 0;
    entryPrice = refPrice;
    stop = 0.f;
    target = 0.f;
  }

  // Returns true when a protective level was touched during bar i.
  bool stop_hit(const Series& bars, int i) const {
    if (pos > 0 && stop > 0.f) return bars.low[static_cast<std::size_t>(i)] <= stop;
    if (pos < 0 && stop > 0.f) return bars.high[static_cast<std::size_t>(i)] >= stop;
    return false;
  }
  bool target_hit(const Series& bars, int i) const {
    if (pos > 0 && target > 0.f) return bars.high[static_cast<std::size_t>(i)] >= target;
    if (pos < 0 && target > 0.f) return bars.low[static_cast<std::size_t>(i)] <= target;
    return false;
  }
};

} // namespace topt::exp

#endif // TACUDA_OPT_STRATEGY_EXPHELPERS_H
