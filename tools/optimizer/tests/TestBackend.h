/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_TESTS_TESTBACKEND_H
#define TACUDA_OPT_TESTS_TESTBACKEND_H

// Plain-C++ reference backend for optimizer unit tests: TA-Lib-compatible
// warm-up conventions, double accumulation. Lets engine/strategy/grid tests
// run on machines without a GPU or a TA-Lib install, and doubles as the
// numeric reference for the guarded TacudaBackend comparison test.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "backend/Backend.h"
#include "backend/ClubCpu.h"

namespace topt::testing {

inline constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

inline void ref_sma(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += in[static_cast<std::size_t>(i)];
    if (i >= p) sum -= in[static_cast<std::size_t>(i - p)];
    if (i >= p - 1) out[i] = static_cast<float>(sum / p);
  }
}

inline void ref_ema(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  double seed = 0.0;
  for (int i = 0; i < p; ++i) seed += in[static_cast<std::size_t>(i)];
  double ema = seed / p;
  out[p - 1] = static_cast<float>(ema);
  const double alpha = 2.0 / (p + 1.0);
  for (int i = p; i < n; ++i) {
    ema = alpha * in[static_cast<std::size_t>(i)] + (1.0 - alpha) * ema;
    out[i] = static_cast<float>(ema);
  }
}

inline void ref_rsi(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p >= n) return;
  double avgGain = 0.0, avgLoss = 0.0;
  for (int i = 1; i <= p; ++i) {
    double d = static_cast<double>(in[static_cast<std::size_t>(i)]) -
               in[static_cast<std::size_t>(i - 1)];
    if (d > 0) avgGain += d; else avgLoss -= d;
  }
  avgGain /= p;
  avgLoss /= p;
  auto rsi_of = [](double g, double l) {
    double denom = g + l;
    return denom == 0.0 ? 0.0 : 100.0 * g / denom;
  };
  out[p] = static_cast<float>(rsi_of(avgGain, avgLoss));
  for (int i = p + 1; i < n; ++i) {
    double d = static_cast<double>(in[static_cast<std::size_t>(i)]) -
               in[static_cast<std::size_t>(i - 1)];
    double gain = d > 0 ? d : 0.0;
    double loss = d < 0 ? -d : 0.0;
    avgGain = (avgGain * (p - 1) + gain) / p;
    avgLoss = (avgLoss * (p - 1) + loss) / p;
    out[i] = static_cast<float>(rsi_of(avgGain, avgLoss));
  }
}

inline void ref_bbands(const std::vector<float>& in, int p, double up, double dn,
                       float* upper, float* middle, float* lower) {
  const int n = static_cast<int>(in.size());
  std::fill(upper, upper + n, kNaN);
  std::fill(middle, middle + n, kNaN);
  std::fill(lower, lower + n, kNaN);
  if (p <= 0 || p > n) return;
  for (int i = p - 1; i < n; ++i) {
    double mean = 0.0;
    for (int j = i - p + 1; j <= i; ++j) mean += in[static_cast<std::size_t>(j)];
    mean /= p;
    double var = 0.0;
    for (int j = i - p + 1; j <= i; ++j) {
      double d = in[static_cast<std::size_t>(j)] - mean;
      var += d * d;
    }
    double sd = std::sqrt(var / p); // population stddev, TA-Lib convention
    middle[i] = static_cast<float>(mean);
    upper[i] = static_cast<float>(mean + up * sd);
    lower[i] = static_cast<float>(mean - dn * sd);
  }
}

inline void ref_atr(const std::vector<float>& high, const std::vector<float>& low,
                    const std::vector<float>& close, int p, float* out) {
  const int n = static_cast<int>(high.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p >= n) return;
  std::vector<double> tr(static_cast<std::size_t>(n));
  tr[0] = static_cast<double>(high[0]) - low[0];
  for (int i = 1; i < n; ++i) {
    double hl = static_cast<double>(high[static_cast<std::size_t>(i)]) -
                low[static_cast<std::size_t>(i)];
    double hc = std::fabs(static_cast<double>(high[static_cast<std::size_t>(i)]) -
                          close[static_cast<std::size_t>(i - 1)]);
    double lc = std::fabs(static_cast<double>(low[static_cast<std::size_t>(i)]) -
                          close[static_cast<std::size_t>(i - 1)]);
    tr[static_cast<std::size_t>(i)] = std::max(hl, std::max(hc, lc));
  }
  double atr = 0.0;
  for (int i = 1; i <= p; ++i) atr += tr[static_cast<std::size_t>(i)];
  atr /= p;
  out[p] = static_cast<float>(atr);
  for (int i = p + 1; i < n; ++i) {
    atr = (atr * (p - 1) + tr[static_cast<std::size_t>(i)]) / p;
    out[i] = static_cast<float>(atr);
  }
}

inline void ref_wma(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  const double denom = static_cast<double>(p) * (p + 1) / 2.0;
  for (int i = p - 1; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < p; ++j) {
      sum += static_cast<double>(in[static_cast<std::size_t>(i - j)]) * (p - j);
    }
    out[i] = static_cast<float>(sum / denom);
  }
}

// TA-Lib TRIMA: triangular weights 1..m..1 (odd p, m=(p+1)/2, sum m^2) or
// 1..m,m..1 (even p, m=p/2, sum m*(m+1)).
inline void ref_trima(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  std::vector<double> w(static_cast<std::size_t>(p));
  double denom = 0.0;
  const int m = (p % 2 == 1) ? (p + 1) / 2 : p / 2;
  for (int j = 0; j < p; ++j) {
    // weight of the j-th oldest value in the window
    int wj = (p % 2 == 1) ? std::min(j + 1, p - j) : std::min(std::min(j + 1, p - j), m);
    w[static_cast<std::size_t>(j)] = wj;
    denom += wj;
  }
  for (int i = p - 1; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < p; ++j) {
      sum += static_cast<double>(in[static_cast<std::size_t>(i - p + 1 + j)]) *
             w[static_cast<std::size_t>(j)];
    }
    out[i] = static_cast<float>(sum / denom);
  }
}

inline void ref_mom(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p >= n) return;
  for (int i = p; i < n; ++i) {
    out[i] = static_cast<float>(static_cast<double>(in[static_cast<std::size_t>(i)]) -
                                in[static_cast<std::size_t>(i - p)]);
  }
}

inline void ref_roc(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p >= n) return;
  for (int i = p; i < n; ++i) {
    double prev = in[static_cast<std::size_t>(i - p)];
    out[i] = prev == 0.0
                 ? 0.0f
                 : static_cast<float>(
                       (static_cast<double>(in[static_cast<std::size_t>(i)]) / prev -
                        1.0) * 100.0);
  }
}

inline void ref_stddev(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  for (int i = p - 1; i < n; ++i) {
    double mean = 0.0;
    for (int j = i - p + 1; j <= i; ++j) mean += in[static_cast<std::size_t>(j)];
    mean /= p;
    double var = 0.0;
    for (int j = i - p + 1; j <= i; ++j) {
      double d = in[static_cast<std::size_t>(j)] - mean;
      var += d * d;
    }
    out[i] = static_cast<float>(std::sqrt(var / p));
  }
}

// TA-Lib LINEARREG family: x = bars back (weight k on the value k bars
// back) with the negated divider, which yields the standard ascending-time
// least-squares slope (positive in uptrends) and the regression value
// evaluated at the current bar.
inline void ref_linearreg(const std::vector<float>& in, int p, bool slopeOnly,
                          float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  const double sumX = static_cast<double>(p) * (p - 1) / 2.0;
  const double sumXSqr = static_cast<double>(p) * (p - 1) * (2 * p - 1) / 6.0;
  const double divisor = sumX * sumX - p * sumXSqr;
  for (int i = p - 1; i < n; ++i) {
    double sumY = 0.0, sumXY = 0.0;
    for (int k = 0; k < p; ++k) {
      double y = in[static_cast<std::size_t>(i - k)];
      sumY += y;
      sumXY += k * y;
    }
    double m = (p * sumXY - sumX * sumY) / divisor;
    if (slopeOnly) {
      out[i] = static_cast<float>(m);
    } else {
      double b = (sumY - m * sumX) / p;
      out[i] = static_cast<float>(b + m * (p - 1));
    }
  }
}

inline void ref_max(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  for (int i = p - 1; i < n; ++i) {
    float best = in[static_cast<std::size_t>(i)];
    for (int j = i - p + 1; j < i; ++j) {
      best = std::max(best, in[static_cast<std::size_t>(j)]);
    }
    out[i] = best;
  }
}

inline void ref_min(const std::vector<float>& in, int p, float* out) {
  const int n = static_cast<int>(in.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  for (int i = p - 1; i < n; ++i) {
    float best = in[static_cast<std::size_t>(i)];
    for (int j = i - p + 1; j < i; ++j) {
      best = std::min(best, in[static_cast<std::size_t>(j)]);
    }
    out[i] = best;
  }
}

inline void ref_trange(const std::vector<float>& high, const std::vector<float>& low,
                       const std::vector<float>& close, float* out) {
  const int n = static_cast<int>(high.size());
  std::fill(out, out + n, kNaN);
  for (int i = 1; i < n; ++i) {
    double hl = static_cast<double>(high[static_cast<std::size_t>(i)]) -
                low[static_cast<std::size_t>(i)];
    double hc = std::fabs(static_cast<double>(high[static_cast<std::size_t>(i)]) -
                          close[static_cast<std::size_t>(i - 1)]);
    double lc = std::fabs(static_cast<double>(low[static_cast<std::size_t>(i)]) -
                          close[static_cast<std::size_t>(i - 1)]);
    out[i] = static_cast<float>(std::max(hl, std::max(hc, lc)));
  }
}

inline void ref_obv(const std::vector<float>& close, const std::vector<float>& volume,
                    float* out) {
  const int n = static_cast<int>(close.size());
  if (n <= 0) return;
  double obv = volume[0];
  out[0] = static_cast<float>(obv);
  for (int i = 1; i < n; ++i) {
    float c = close[static_cast<std::size_t>(i)];
    float pc = close[static_cast<std::size_t>(i - 1)];
    if (c > pc) obv += volume[static_cast<std::size_t>(i)];
    else if (c < pc) obv -= volume[static_cast<std::size_t>(i)];
    out[i] = static_cast<float>(obv);
  }
}

// TA-Lib MACD parity: both EMAs are seeded with an SMA ending at slow-1
// (the fast EMA does NOT run from its own fast-1 lookback), then recurse
// together from index slow.
inline void ref_macd(const std::vector<float>& in, int fast, int slow, int sig,
                     float* macd, float* signal, float* hist) {
  const int n = static_cast<int>(in.size());
  std::fill(macd, macd + n, kNaN);
  std::fill(signal, signal + n, kNaN);
  std::fill(hist, hist + n, kNaN);
  if (fast > slow) std::swap(fast, slow);
  const int firstMacd = slow - 1;
  const int firstOut = slow + sig - 2;
  if (fast <= 0 || sig <= 0 || firstOut >= n) return;
  double fastEma = 0.0, slowEma = 0.0;
  for (int i = slow - fast; i < slow; ++i) fastEma += in[static_cast<std::size_t>(i)];
  fastEma /= fast;
  for (int i = 0; i < slow; ++i) slowEma += in[static_cast<std::size_t>(i)];
  slowEma /= slow;
  const double aF = 2.0 / (fast + 1.0), aS = 2.0 / (slow + 1.0);
  std::vector<double> line(static_cast<std::size_t>(n),
                           std::numeric_limits<double>::quiet_NaN());
  line[static_cast<std::size_t>(firstMacd)] = fastEma - slowEma;
  for (int i = slow; i < n; ++i) {
    fastEma = aF * in[static_cast<std::size_t>(i)] + (1.0 - aF) * fastEma;
    slowEma = aS * in[static_cast<std::size_t>(i)] + (1.0 - aS) * slowEma;
    line[static_cast<std::size_t>(i)] = fastEma - slowEma;
  }
  // Signal: EMA over the MACD line, SMA-seeded at its first `sig` values.
  double seed = 0.0;
  for (int i = firstMacd; i < firstMacd + sig; ++i) {
    seed += line[static_cast<std::size_t>(i)];
  }
  double sigEma = seed / sig;
  const double alpha = 2.0 / (sig + 1.0);
  signal[firstOut] = static_cast<float>(sigEma);
  macd[firstOut] = static_cast<float>(line[static_cast<std::size_t>(firstOut)]);
  hist[firstOut] = static_cast<float>(line[static_cast<std::size_t>(firstOut)] - sigEma);
  for (int i = firstOut + 1; i < n; ++i) {
    sigEma = alpha * line[static_cast<std::size_t>(i)] + (1.0 - alpha) * sigEma;
    macd[i] = static_cast<float>(line[static_cast<std::size_t>(i)]);
    signal[i] = static_cast<float>(sigEma);
    hist[i] = static_cast<float>(line[static_cast<std::size_t>(i)] - sigEma);
  }
}

inline void ref_stoch(const std::vector<float>& high, const std::vector<float>& low,
                      const std::vector<float>& close, int k, int d,
                      float* slowK, float* slowD) {
  const int n = static_cast<int>(high.size());
  std::fill(slowK, slowK + n, kNaN);
  std::fill(slowD, slowD + n, kNaN);
  const int firstFastK = k - 1;
  const int firstSlowK = firstFastK + d - 1;
  const int firstSlowD = firstSlowK + d - 1;
  if (k <= 0 || d <= 0 || firstSlowD >= n) return;
  std::vector<double> fastK(static_cast<std::size_t>(n), 0.0);
  for (int i = firstFastK; i < n; ++i) {
    double hh = high[static_cast<std::size_t>(i)];
    double ll = low[static_cast<std::size_t>(i)];
    for (int j = 1; j < k; ++j) {
      hh = std::max(hh, static_cast<double>(high[static_cast<std::size_t>(i - j)]));
      ll = std::min(ll, static_cast<double>(low[static_cast<std::size_t>(i - j)]));
    }
    double diff = hh - ll;
    fastK[static_cast<std::size_t>(i)] =
        diff == 0.0 ? 0.0
                    : 100.0 * (close[static_cast<std::size_t>(i)] - ll) / diff;
  }
  std::vector<double> sk(static_cast<std::size_t>(n), 0.0);
  for (int i = firstSlowK; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) s += fastK[static_cast<std::size_t>(i - j)];
    sk[static_cast<std::size_t>(i)] = s / d;
  }
  for (int i = firstSlowD; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) s += sk[static_cast<std::size_t>(i - j)];
    // TA-Lib reports both lines from the same index.
    slowK[i] = static_cast<float>(sk[static_cast<std::size_t>(i)]);
    slowD[i] = static_cast<float>(s / d);
  }
}

inline void ref_cci(const std::vector<float>& high, const std::vector<float>& low,
                    const std::vector<float>& close, int p, float* out) {
  const int n = static_cast<int>(high.size());
  std::fill(out, out + n, kNaN);
  if (p <= 0 || p > n) return;
  std::vector<double> tp(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    tp[static_cast<std::size_t>(i)] =
        (static_cast<double>(high[static_cast<std::size_t>(i)]) +
         low[static_cast<std::size_t>(i)] + close[static_cast<std::size_t>(i)]) /
        3.0;
  }
  for (int i = p - 1; i < n; ++i) {
    double mean = 0.0;
    for (int j = i - p + 1; j <= i; ++j) mean += tp[static_cast<std::size_t>(j)];
    mean /= p;
    double mad = 0.0;
    for (int j = i - p + 1; j <= i; ++j) {
      mad += std::fabs(tp[static_cast<std::size_t>(j)] - mean);
    }
    mad /= p;
    out[i] = mad == 0.0 ? 0.0f
                        : static_cast<float>((tp[static_cast<std::size_t>(i)] - mean) /
                                             (0.015 * mad));
  }
}

namespace detail {

struct DmBar {
  double plusDM = 0.0, minusDM = 0.0, tr = 0.0;
};

inline DmBar dm_bar(const std::vector<float>& high, const std::vector<float>& low,
                    const std::vector<float>& close, int i) {
  DmBar b;
  double diffP = static_cast<double>(high[static_cast<std::size_t>(i)]) -
                 high[static_cast<std::size_t>(i - 1)];
  double diffM = static_cast<double>(low[static_cast<std::size_t>(i - 1)]) -
                 low[static_cast<std::size_t>(i)];
  if (diffP > 0 && diffP > diffM) b.plusDM = diffP;
  if (diffM > 0 && diffM > diffP) b.minusDM = diffM;
  double hl = static_cast<double>(high[static_cast<std::size_t>(i)]) -
              low[static_cast<std::size_t>(i)];
  double hc = std::fabs(static_cast<double>(high[static_cast<std::size_t>(i)]) -
                        close[static_cast<std::size_t>(i - 1)]);
  double lc = std::fabs(static_cast<double>(low[static_cast<std::size_t>(i)]) -
                        close[static_cast<std::size_t>(i - 1)]);
  b.tr = std::max(hl, std::max(hc, lc));
  return b;
}

} // namespace detail

// TA-Lib PLUS_DI / MINUS_DI (period > 1): the seed sums the first p-1
// DM/TR diffs, and every reported bar (from index p on) has been through
// the Wilder subtract-then-add update — including the first one.
inline void ref_di(const std::vector<float>& high, const std::vector<float>& low,
                   const std::vector<float>& close, int p, bool plus, float* out) {
  const int n = static_cast<int>(high.size());
  std::fill(out, out + n, kNaN);
  if (p <= 1 || p >= n) return;
  double dmSum = 0.0, trSum = 0.0;
  for (int i = 1; i <= p - 1; ++i) {
    auto b = detail::dm_bar(high, low, close, i);
    dmSum += plus ? b.plusDM : b.minusDM;
    trSum += b.tr;
  }
  for (int i = p; i < n; ++i) {
    auto b = detail::dm_bar(high, low, close, i);
    dmSum = dmSum - dmSum / p + (plus ? b.plusDM : b.minusDM);
    trSum = trSum - trSum / p + b.tr;
    out[i] = trSum == 0.0 ? 0.0f : static_cast<float>(100.0 * dmSum / trSum);
  }
}

// TA-Lib ADX: p-1 raw accumulation diffs, p smoothed DX values averaged into
// the seed at index 2p-1, then Wilder recursion.
inline void ref_adx(const std::vector<float>& high, const std::vector<float>& low,
                    const std::vector<float>& close, int p, float* out) {
  const int n = static_cast<int>(high.size());
  std::fill(out, out + n, kNaN);
  if (p <= 1 || 2 * p - 1 >= n) return;
  double plusDM = 0.0, minusDM = 0.0, tr = 0.0;
  int today = 0;
  for (int i = 0; i < p - 1; ++i) {
    ++today;
    auto b = detail::dm_bar(high, low, close, today);
    plusDM += b.plusDM;
    minusDM += b.minusDM;
    tr += b.tr;
  }
  auto dx_of = [](double pDM, double mDM, double trv) {
    if (trv == 0.0) return 0.0;
    double pDI = 100.0 * pDM / trv;
    double mDI = 100.0 * mDM / trv;
    double sum = pDI + mDI;
    return sum == 0.0 ? 0.0 : 100.0 * std::fabs(pDI - mDI) / sum;
  };
  double sumDX = 0.0;
  for (int i = 0; i < p; ++i) {
    ++today;
    auto b = detail::dm_bar(high, low, close, today);
    plusDM = plusDM - plusDM / p + b.plusDM;
    minusDM = minusDM - minusDM / p + b.minusDM;
    tr = tr - tr / p + b.tr;
    sumDX += dx_of(plusDM, minusDM, tr);
  }
  double adx = sumDX / p;
  out[today] = static_cast<float>(adx); // today == 2p-1
  while (today + 1 < n) {
    ++today;
    auto b = detail::dm_bar(high, low, close, today);
    plusDM = plusDM - plusDM / p + b.plusDM;
    minusDM = minusDM - minusDM / p + b.minusDM;
    tr = tr - tr / p + b.tr;
    adx = (adx * (p - 1) + dx_of(plusDM, minusDM, tr)) / p;
    out[today] = static_cast<float>(adx);
  }
}

// Reference backend usable anywhere IIndicatorBackend is expected.
class TestBackend final : public IIndicatorBackend {
public:
  explicit TestBackend(const Series& bars) : bars_(bars) {}

  const char* name() const override { return "test-cpu"; }

  void compute(const IndicatorRequest& req, const std::vector<float*>& outs) override {
    ++computeCalls;
    validate_chain(req);
    // Chained (derived-series) input or OHLCV column.
    std::vector<float> chained;
    const std::vector<float>* inPtr = nullptr;
    if (family_single_input(req.family)) {
      if (req.chain) {
        chained = club::resolve_input(bars_, req, *this);
        inPtr = &chained;
      } else {
        inPtr = &bars_.column(req.input);
      }
    }
    // Club-pack families share the CPU implementation with TalibBackend.
    if (club::compute_family(bars_, req, inPtr ? inPtr->data() : nullptr,
                             outs.at(0))) {
      return;
    }
    const std::vector<float>& in = inPtr ? *inPtr : bars_.column(req.input);
    const int p0 = req.params.empty() ? 0 : static_cast<int>(req.params.at(0));
    switch (req.family) {
    case Family::SMA:
      ref_sma(in, p0, outs.at(0));
      break;
    case Family::EMA:
      ref_ema(in, p0, outs.at(0));
      break;
    case Family::RSI:
      ref_rsi(in, p0, outs.at(0));
      break;
    case Family::BBANDS:
      ref_bbands(in, p0, req.params.at(1), req.params.at(2), outs.at(0),
                 outs.at(1), outs.at(2));
      break;
    case Family::ATR:
      ref_atr(bars_.high, bars_.low, bars_.close, p0, outs.at(0));
      break;
    case Family::WMA:
      ref_wma(in, p0, outs.at(0));
      break;
    case Family::TRIMA:
      ref_trima(in, p0, outs.at(0));
      break;
    case Family::MACD:
      ref_macd(in, p0, static_cast<int>(req.params.at(1)),
               static_cast<int>(req.params.at(2)), outs.at(0), outs.at(1),
               outs.at(2));
      break;
    case Family::STOCH:
      ref_stoch(bars_.high, bars_.low, bars_.close, p0,
                static_cast<int>(req.params.at(1)), outs.at(0), outs.at(1));
      break;
    case Family::CCI:
      ref_cci(bars_.high, bars_.low, bars_.close, p0, outs.at(0));
      break;
    case Family::ADX:
      ref_adx(bars_.high, bars_.low, bars_.close, p0, outs.at(0));
      break;
    case Family::PLUS_DI:
      ref_di(bars_.high, bars_.low, bars_.close, p0, true, outs.at(0));
      break;
    case Family::MINUS_DI:
      ref_di(bars_.high, bars_.low, bars_.close, p0, false, outs.at(0));
      break;
    case Family::OBV:
      ref_obv(bars_.close, bars_.volume, outs.at(0));
      break;
    case Family::MOM:
      ref_mom(in, p0, outs.at(0));
      break;
    case Family::ROC:
      ref_roc(in, p0, outs.at(0));
      break;
    case Family::STDDEV:
      ref_stddev(in, p0, outs.at(0));
      break;
    case Family::LINEARREG:
      ref_linearreg(in, p0, false, outs.at(0));
      break;
    case Family::LINEARREG_SLOPE:
      ref_linearreg(in, p0, true, outs.at(0));
      break;
    case Family::MAX:
      ref_max(in, p0, outs.at(0));
      break;
    case Family::MIN:
      ref_min(in, p0, outs.at(0));
      break;
    case Family::TRANGE:
      ref_trange(bars_.high, bars_.low, bars_.close, outs.at(0));
      break;
    default:
      // Club families returned above; anything else is a wiring bug.
      throw std::invalid_argument("test backend: unhandled family");
    }
  }

  int computeCalls = 0;

private:
  const Series& bars_;
};

} // namespace topt::testing

#endif // TACUDA_OPT_TESTS_TESTBACKEND_H
