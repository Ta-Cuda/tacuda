/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_BACKEND_CLUBCPU_H
#define TACUDA_OPT_BACKEND_CLUBCPU_H

// Plain-C++ implementations of the STAD club-pack families (Family::ASI ..
// Family::ROLLING_STD) shared by the CPU backends (TalibBackend fallback,
// TestBackend reference). Semantics mirror TaCuda's ClubPack.cu kernels,
// which in turn port the original EXPERIMENTS pack code: double
// accumulation, identical NaN warm-up conventions, so CPU and GPU paths
// stay numerically comparable.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/Types.h"

namespace topt::club {

inline constexpr float kClubNaN = std::numeric_limits<float>::quiet_NaN();

inline void asi(const Series& b, float* out) {
  const int n = b.size();
  std::fill(out, out + n, kClubNaN);
  double acc = 0.0;
  for (int i = 1; i < n; ++i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    const double h = b.high[ui], l = b.low[ui], c = b.close[ui], o = b.open[ui];
    const double cy = b.close[ui - 1], oy = b.open[ui - 1];
    const double hc = std::fabs(h - cy), lc = std::fabs(l - cy);
    const double hl = h - l, coy = std::fabs(cy - oy);
    double r;
    if (hc >= lc && hc >= hl) r = hc - 0.5 * lc + 0.25 * coy;
    else if (lc >= hc && lc >= hl) r = lc - 0.5 * hc + 0.25 * coy;
    else r = hl + 0.25 * coy;
    if (r > 1e-12) {
      const double k = std::max(hc, lc);
      acc += 50.0 * ((c - cy) + 0.5 * (c - o) + 0.25 * (cy - oy)) / r *
             (k / std::max(1e-12, hl > 0.0 ? hl : k));
    }
    out[i] = static_cast<float>(acc);
  }
}

inline void sar_el(const Series& b, float accFact, float maxAf, float* out) {
  const int n = b.size();
  std::fill(out, out + n, kClubNaN);
  if (n < 2) return;
  bool isLong = b.close[1] >= b.close[0];
  float af = accFact;
  float ep = isLong ? b.high[1] : b.low[1];
  float sar = isLong ? b.low[0] : b.high[0];
  out[1] = sar;
  for (int i = 2; i < n; ++i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    sar = sar + af * (ep - sar);
    if (isLong) {
      sar = std::min(sar, std::min(b.low[ui - 1], b.low[ui - 2]));
      if (b.low[ui] < sar) {
        isLong = false;
        sar = ep;
        ep = b.low[ui];
        af = accFact;
      } else if (b.high[ui] > ep) {
        ep = b.high[ui];
        af = std::min(maxAf, af + accFact);
      }
    } else {
      sar = std::max(sar, std::max(b.high[ui - 1], b.high[ui - 2]));
      if (b.high[ui] > sar) {
        isLong = true;
        sar = ep;
        ep = b.high[ui];
        af = accFact;
      } else if (b.low[ui] < ep) {
        ep = b.low[ui];
        af = std::min(maxAf, af + accFact);
      }
    }
    out[i] = sar;
  }
}

inline void obtr(const Series& b, float* out) {
  const int n = b.size();
  std::fill(out, out + n, kClubNaN);
  double acc = 0.0;
  for (int i = 1; i < n; ++i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    const double cy = b.close[ui - 1];
    const double tr = std::max(static_cast<double>(b.high[ui]) - b.low[ui],
                               std::max(std::fabs(b.high[ui] - cy),
                                        std::fabs(b.low[ui] - cy)));
    if (b.close[ui] > b.close[ui - 1]) acc += tr;
    else if (b.close[ui] < b.close[ui - 1]) acc -= tr;
    out[i] = static_cast<float>(acc);
  }
}

inline void wobv(const Series& b, float* out) {
  const int n = b.size();
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    const float rng = b.high[ui] - b.low[ui];
    if (rng > 0.f) {
      acc += (static_cast<double>(b.close[ui]) - b.open[ui]) / rng * b.volume[ui];
    }
    out[i] = static_cast<float>(acc);
  }
}

inline void hpi(const Series& b, float factor, float* out) {
  const int n = b.size();
  std::fill(out, out + n, kClubNaN);
  double acc = 0.0;
  for (int i = 1; i < n; ++i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    const double mid = (static_cast<double>(b.high[ui]) + b.low[ui]) / 2.0;
    const double midPrev =
        (static_cast<double>(b.high[ui - 1]) + b.low[ui - 1]) / 2.0;
    acc += static_cast<double>(factor) * ((mid - midPrev) * b.volume[ui] - acc);
    out[i] = static_cast<float>(acc);
  }
}

inline void market_strength(const float* in, int n, int len, float* out) {
  std::fill(out, out + n, kClubNaN);
  if (len <= 0 || len >= n) return;
  for (int i = len; i < n; ++i) {
    double total = 0.0, ups = 0.0, dns = 0.0;
    for (int k = 0; k < len; ++k) {
      const double d = static_cast<double>(in[i - k]) - in[i - k - 1];
      total += d;
      if (d > 0.0) ups += d;
      else dns += -d;
    }
    if (total >= 0.0)
      out[i] = ups > 0.0 ? static_cast<float>(100.0 * total / ups) : 0.f;
    else
      out[i] = dns > 0.0 ? static_cast<float>(100.0 * total / dns) : 0.f;
  }
}

inline void trendscore(const float* in, int n, int lookback, float* out) {
  std::fill(out, out + n, kClubNaN);
  for (int i = lookback + 10; i < n; ++i) {
    int t = 0;
    for (int k = 1; k <= 10; ++k) t += in[i] >= in[i - lookback - k] ? 1 : -1;
    out[i] = static_cast<float>(t);
  }
}

inline void refdev(const float* in, int n, int smaLen, int sumLen, float* out) {
  std::fill(out, out + n, kClubNaN);
  if (smaLen <= 0 || sumLen <= 0) return;
  for (int i = smaLen - 1 + sumLen - 1; i < n; ++i) {
    double ndv = 0.0, tdv = 0.0;
    for (int k = 0; k < sumLen; ++k) {
      const int j = i - k;
      double sma = 0.0;
      for (int m = 0; m < smaLen; ++m) sma += in[j - m];
      const double drd = in[j] - sma / smaLen;
      ndv += drd;
      tdv += std::fabs(drd);
    }
    if (tdv > 0.0) out[i] = static_cast<float>(100.0 * ndv / tdv);
  }
}

inline void moments(const float* in, int n, int len, bool wantKurtosis,
                    float* out) {
  std::fill(out, out + n, kClubNaN);
  if (len <= 0 || len > n) return;
  for (int i = len - 1; i < n; ++i) {
    double mean = 0.0;
    for (int k = 0; k < len; ++k) mean += in[i - k];
    mean /= len;
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int k = 0; k < len; ++k) {
      const double d = static_cast<double>(in[i - k]) - mean;
      const double d2 = d * d;
      m2 += d2;
      m3 += d2 * d;
      m4 += d2 * d2;
    }
    m2 /= len;
    m3 /= len;
    m4 /= len;
    if (m2 <= 0.0) out[i] = 0.f;
    else if (wantKurtosis) out[i] = static_cast<float>(m4 / (m2 * m2) - 3.0);
    else out[i] = static_cast<float>(m3 / std::pow(m2, 1.5));
  }
}

inline void pivot(const float* in, int n, int strength, bool wantHigh,
                  float* out) {
  std::fill(out, out + n, kClubNaN);
  if (strength <= 0) return;
  for (int i = 2 * strength; i < n; ++i) {
    const int b = i - strength;
    bool ok = true;
    for (int k = 1; k <= strength && ok; ++k) {
      if (wantHigh) ok = in[b] > in[b - k] && in[b] >= in[b + k];
      else ok = in[b] < in[b - k] && in[b] <= in[b + k];
    }
    if (ok) out[i] = in[b];
  }
}

inline void xaverage(const float* in, int n, int period, float* out) {
  std::fill(out, out + n, kClubNaN);
  const double alpha = 2.0 / (static_cast<double>(period) + 1.0);
  double ema = 0.0;
  bool seeded = false;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(in[i])) continue;
    if (!seeded) {
      ema = in[i];
      seeded = true;
    } else {
      ema = alpha * in[i] + (1.0 - alpha) * ema;
    }
    out[i] = static_cast<float>(ema);
  }
}

// mode: 0=mean, 1=max, 2=min, 3=population stddev; NaN-strict windows.
inline void rolling_strict(const float* in, int n, int len, int mode,
                           float* out) {
  std::fill(out, out + n, kClubNaN);
  if (len <= 0 || len > n) return;
  for (int i = len - 1; i < n; ++i) {
    bool ok = true;
    double sum = 0.0, sq = 0.0;
    float mx = in[i], mn = in[i];
    for (int k = 0; k < len; ++k) {
      const float v = in[i - k];
      if (!std::isfinite(v)) {
        ok = false;
        break;
      }
      sum += v;
      sq += static_cast<double>(v) * v;
      mx = std::max(mx, v);
      mn = std::min(mn, v);
    }
    if (!ok) continue;
    if (mode == 0) out[i] = static_cast<float>(sum / len);
    else if (mode == 1) out[i] = mx;
    else if (mode == 2) out[i] = mn;
    else {
      const double mean = sum / len;
      out[i] = static_cast<float>(std::sqrt(std::max(0.0, sq / len - mean * mean)));
    }
  }
}

// Dispatches a club family over `in` (single-input families) or `bars`
// (fixed-column families) into outs[0]. Returns false when `f` is not a
// club family.
inline bool compute_family(const Series& bars, const IndicatorRequest& req,
                           const float* in, float* out) {
  const int n = bars.size();
  const int p0 = req.params.empty() ? 0 : static_cast<int>(req.params.at(0));
  switch (req.family) {
  case Family::ASI: asi(bars, out); return true;
  case Family::SAR_EL:
    sar_el(bars, static_cast<float>(req.params.at(0)),
           static_cast<float>(req.params.at(1)), out);
    return true;
  case Family::OBTR: obtr(bars, out); return true;
  case Family::WOBV: wobv(bars, out); return true;
  case Family::HPI:
    hpi(bars, static_cast<float>(req.params.at(0)), out);
    return true;
  case Family::MARKET_STRENGTH: market_strength(in, n, p0, out); return true;
  case Family::TRENDSCORE: trendscore(in, n, p0, out); return true;
  case Family::REFDEV:
    refdev(in, n, p0, static_cast<int>(req.params.at(1)), out);
    return true;
  case Family::SKEW: moments(in, n, p0, false, out); return true;
  case Family::KURTOSIS: moments(in, n, p0, true, out); return true;
  case Family::PIVOT_HIGH: pivot(in, n, p0, true, out); return true;
  case Family::PIVOT_LOW: pivot(in, n, p0, false, out); return true;
  case Family::XAVERAGE: xaverage(in, n, p0, out); return true;
  case Family::ROLLING_MEAN: rolling_strict(in, n, p0, 0, out); return true;
  case Family::ROLLING_MAX: rolling_strict(in, n, p0, 1, out); return true;
  case Family::ROLLING_MIN: rolling_strict(in, n, p0, 2, out); return true;
  case Family::ROLLING_STD: rolling_strict(in, n, p0, 3, out); return true;
  default:
    return false;
  }
}

// Resolves the effective input line of a (possibly chained) single-input
// request on a CPU backend: the OHLCV column, or the upstream line computed
// recursively through `self`.
template <typename BackendT>
inline std::vector<float> resolve_input(const Series& bars,
                                        const IndicatorRequest& req,
                                        BackendT& self) {
  validate_chain(req);
  if (!req.chain) {
    const std::vector<float>& col = bars.column(req.input);
    return {col.begin(), col.end()};
  }
  std::vector<float> line(static_cast<std::size_t>(bars.size()));
  std::vector<float*> outs{line.data()};
  self.compute(*req.chain, outs);
  return line;
}

} // namespace topt::club

#endif // TACUDA_OPT_BACKEND_CLUBCPU_H
