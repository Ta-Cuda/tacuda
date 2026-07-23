/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Club-pack indicators (ClubPack.cu) vs plain-C++ references ported from
// the tacuda-opt EXPERIMENTS strategy pack (the semantics source of truth).

#include <tacuda.h>
#include "test_utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

struct Ohlcv {
  std::vector<float> open, high, low, close, volume;
  int size() const { return static_cast<int>(close.size()); }
};

Ohlcv make_bars(int n, int seed = 7) {
  Ohlcv b;
  b.open.resize(n);
  b.high.resize(n);
  b.low.resize(n);
  b.close.resize(n);
  b.volume.resize(n);
  float prev = 100.0f;
  for (int i = 0; i < n; ++i) {
    const float drift = 0.4f * std::sin(0.021f * i + seed) +
                        0.25f * std::cos(0.13f * i * (seed + 1));
    const float close = prev + drift;
    const float open = prev;
    const float range = 0.3f + 0.2f * std::fabs(std::sin(0.31f * i + seed));
    b.open[i] = open;
    b.close[i] = close;
    b.high[i] = std::max(open, close) + range;
    b.low[i] = std::min(open, close) - range;
    b.volume[i] = 1000.0f + 40.0f * (i % 53) + 15.0f * ((i * seed) % 17);
    prev = close;
  }
  return b;
}

void expect_close(const std::vector<float>& got, const std::vector<float>& ref,
                  float absTol = 1e-3f, float relTol = 1e-4f) {
  ASSERT_EQ(got.size(), ref.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const bool gNan = std::isnan(got[i]), rNan = std::isnan(ref[i]);
    EXPECT_EQ(gNan, rNan) << "NaN mismatch at " << i << " got=" << got[i]
                          << " ref=" << ref[i];
    if (gNan || rNan) continue;
    const float tol = absTol + relTol * std::fabs(ref[i]);
    EXPECT_NEAR(got[i], ref[i], tol) << "at index " << i;
  }
}

// --- references (ports of tools/optimizer/src/strategy implementations) ---

std::vector<float> ref_asi(const Ohlcv& b) {
  const int n = b.size();
  std::vector<float> out(n, kNaN);
  double acc = 0.0;
  for (int i = 1; i < n; ++i) {
    const double h = b.high[i], l = b.low[i], c = b.close[i], o = b.open[i];
    const double cy = b.close[i - 1], oy = b.open[i - 1];
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
  return out;
}

std::vector<float> ref_sar_el(const Ohlcv& b, float accFact, float maxAf) {
  const int n = b.size();
  std::vector<float> out(n, kNaN);
  if (n < 2) return out;
  bool isLong = b.close[1] >= b.close[0];
  float af = accFact;
  float ep = isLong ? b.high[1] : b.low[1];
  float sar = isLong ? b.low[0] : b.high[0];
  out[1] = sar;
  for (int i = 2; i < n; ++i) {
    sar = sar + af * (ep - sar);
    if (isLong) {
      sar = std::min(sar, std::min(b.low[i - 1], b.low[i - 2]));
      if (b.low[i] < sar) {
        isLong = false;
        sar = ep;
        ep = b.low[i];
        af = accFact;
      } else if (b.high[i] > ep) {
        ep = b.high[i];
        af = std::min(maxAf, af + accFact);
      }
    } else {
      sar = std::max(sar, std::max(b.high[i - 1], b.high[i - 2]));
      if (b.high[i] > sar) {
        isLong = true;
        sar = ep;
        ep = b.high[i];
        af = accFact;
      } else if (b.low[i] < ep) {
        ep = b.low[i];
        af = std::min(maxAf, af + accFact);
      }
    }
    out[i] = sar;
  }
  return out;
}

std::vector<float> ref_obtr(const Ohlcv& b) {
  const int n = b.size();
  std::vector<float> out(n, kNaN);
  double acc = 0.0;
  for (int i = 1; i < n; ++i) {
    const double cy = b.close[i - 1];
    const double tr =
        std::max({static_cast<double>(b.high[i]) - b.low[i],
                  std::fabs(b.high[i] - cy), std::fabs(b.low[i] - cy)});
    if (b.close[i] > b.close[i - 1]) acc += tr;
    else if (b.close[i] < b.close[i - 1]) acc -= tr;
    out[i] = static_cast<float>(acc);
  }
  return out;
}

std::vector<float> ref_wobv(const Ohlcv& b) {
  const int n = b.size();
  std::vector<float> out(n, 0.f);
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    const float rng = b.high[i] - b.low[i];
    if (rng > 0.f) {
      acc += (static_cast<double>(b.close[i]) - b.open[i]) / rng * b.volume[i];
    }
    out[i] = static_cast<float>(acc);
  }
  return out;
}

std::vector<float> ref_hpi(const Ohlcv& b, float factor) {
  const int n = b.size();
  std::vector<float> out(n, kNaN);
  double acc = 0.0;
  for (int i = 1; i < n; ++i) {
    const double mid = (static_cast<double>(b.high[i]) + b.low[i]) / 2.0;
    const double midPrev =
        (static_cast<double>(b.high[i - 1]) + b.low[i - 1]) / 2.0;
    acc += factor * ((mid - midPrev) * b.volume[i] - acc);
    out[i] = static_cast<float>(acc);
  }
  return out;
}

std::vector<float> ref_market_strength(const std::vector<float>& c, int len) {
  const int n = static_cast<int>(c.size());
  std::vector<float> out(n, kNaN);
  for (int i = len; i < n; ++i) {
    double total = 0.0, ups = 0.0, dns = 0.0;
    for (int k = 0; k < len; ++k) {
      const double d = static_cast<double>(c[i - k]) - c[i - k - 1];
      total += d;
      if (d > 0) ups += d;
      else dns += -d;
    }
    if (total >= 0.0)
      out[i] = ups > 0.0 ? static_cast<float>(100.0 * total / ups) : 0.f;
    else
      out[i] = dns > 0.0 ? static_cast<float>(100.0 * total / dns) : 0.f;
  }
  return out;
}

std::vector<float> ref_trendscore(const std::vector<float>& c, int lookback) {
  const int n = static_cast<int>(c.size());
  std::vector<float> out(n, kNaN);
  for (int i = lookback + 10; i < n; ++i) {
    int t = 0;
    for (int k = 1; k <= 10; ++k) t += c[i] >= c[i - lookback - k] ? 1 : -1;
    out[i] = static_cast<float>(t);
  }
  return out;
}

std::vector<float> ref_refdev(const std::vector<float>& c, int smaLen,
                              int sumLen) {
  const int n = static_cast<int>(c.size());
  std::vector<float> out(n, kNaN);
  for (int i = smaLen - 1 + sumLen - 1; i < n; ++i) {
    double ndv = 0.0, tdv = 0.0;
    for (int k = 0; k < sumLen; ++k) {
      const int j = i - k;
      double sma = 0.0;
      for (int m = 0; m < smaLen; ++m) sma += c[j - m];
      const double drd = c[j] - sma / smaLen;
      ndv += drd;
      tdv += std::fabs(drd);
    }
    if (tdv > 0.0) out[i] = static_cast<float>(100.0 * ndv / tdv);
  }
  return out;
}

void ref_moments(const std::vector<float>& c, int len,
                 std::vector<float>& skew, std::vector<float>& kurt) {
  const int n = static_cast<int>(c.size());
  skew.assign(n, kNaN);
  kurt.assign(n, kNaN);
  for (int i = len - 1; i < n; ++i) {
    double mean = 0.0;
    for (int k = 0; k < len; ++k) mean += c[i - k];
    mean /= len;
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int k = 0; k < len; ++k) {
      const double d = static_cast<double>(c[i - k]) - mean;
      m2 += d * d;
      m3 += d * d * d;
      m4 += d * d * d * d;
    }
    m2 /= len;
    m3 /= len;
    m4 /= len;
    if (m2 <= 0.0) {
      skew[i] = 0.f;
      kurt[i] = 0.f;
    } else {
      skew[i] = static_cast<float>(m3 / std::pow(m2, 1.5));
      kurt[i] = static_cast<float>(m4 / (m2 * m2) - 3.0);
    }
  }
}

std::vector<float> ref_pivot(const std::vector<float>& v, int s, bool high) {
  const int n = static_cast<int>(v.size());
  std::vector<float> out(n, kNaN);
  for (int i = 2 * s; i < n; ++i) {
    const int b = i - s;
    bool ok = true;
    for (int k = 1; k <= s && ok; ++k) {
      if (high) ok = v[b] > v[b - k] && v[b] >= v[b + k];
      else ok = v[b] < v[b - k] && v[b] <= v[b + k];
    }
    if (ok) out[i] = v[b];
  }
  return out;
}

std::vector<float> ref_xaverage(const std::vector<float>& line, int n_) {
  const int n = static_cast<int>(line.size());
  std::vector<float> out(n, kNaN);
  const double alpha = 2.0 / (static_cast<double>(n_) + 1.0);
  double ema = 0.0;
  bool seeded = false;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(line[i])) continue;
    if (!seeded) {
      ema = line[i];
      seeded = true;
    } else {
      ema = alpha * line[i] + (1.0 - alpha) * ema;
    }
    out[i] = static_cast<float>(ema);
  }
  return out;
}

} // namespace

TEST(ClubPack, AsiMatchesReference) {
  const auto b = make_bars(600);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_asi(b.open.data(), b.high.data(), b.low.data(), b.close.data(),
                   out.data(), b.size(), 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_asi(b), 5e-3f, 1e-4f);
}

TEST(ClubPack, SarElMatchesReference) {
  const auto b = make_bars(600, 11);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_sar_el(b.high.data(), b.low.data(), b.close.data(), out.data(),
                      b.size(), 0.02f, 0.2f, 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_sar_el(b, 0.02f, 0.2f), 1e-3f, 1e-5f);
}

TEST(ClubPack, ObtrMatchesReference) {
  const auto b = make_bars(600, 3);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_obtr(b.high.data(), b.low.data(), b.close.data(), out.data(),
                    b.size(), 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_obtr(b), 5e-3f, 1e-4f);
}

TEST(ClubPack, WobvMatchesReference) {
  const auto b = make_bars(600, 5);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_wobv(b.open.data(), b.high.data(), b.low.data(), b.close.data(),
                    b.volume.data(), out.data(), b.size(), 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_wobv(b), 0.5f, 1e-4f);
}

TEST(ClubPack, HpiMatchesReference) {
  const auto b = make_bars(600, 9);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_hpi(b.high.data(), b.low.data(), b.volume.data(), out.data(),
                   b.size(), 0.1f, 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_hpi(b, 0.1f), 0.05f, 1e-4f);
}

TEST(ClubPack, MarketStrengthMatchesReference) {
  const auto b = make_bars(600, 13);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_market_strength(b.close.data(), out.data(), b.size(), 5, 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_market_strength(b.close, 5), 5e-3f, 1e-4f);
}

TEST(ClubPack, TrendScoreMatchesReference) {
  const auto b = make_bars(600, 17);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_trendscore(b.close.data(), out.data(), b.size(), 10, 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_trendscore(b.close, 10), 0.0f, 0.0f);
}

TEST(ClubPack, RefDevMatchesReference) {
  const auto b = make_bars(600, 19);
  std::vector<float> out(600, 0.f);
  ASSERT_EQ(ct_refdev(b.close.data(), out.data(), b.size(), 15, 15, 0),
            CT_STATUS_SUCCESS);
  expect_close(out, ref_refdev(b.close, 15, 15), 5e-2f, 1e-3f);
}

TEST(ClubPack, SkewAndKurtosisMatchReference) {
  const auto b = make_bars(600, 23);
  std::vector<float> skewRef, kurtRef;
  ref_moments(b.close, 40, skewRef, kurtRef);

  std::vector<float> skew(600, 0.f), kurt(600, 0.f);
  ASSERT_EQ(ct_skew(b.close.data(), skew.data(), b.size(), 40, 0),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_kurtosis(b.close.data(), kurt.data(), b.size(), 40, 0),
            CT_STATUS_SUCCESS);
  expect_close(skew, skewRef, 2e-3f, 1e-3f);
  expect_close(kurt, kurtRef, 5e-3f, 1e-3f);
}

TEST(ClubPack, PivotsMatchReference) {
  const auto b = make_bars(600, 29);
  std::vector<float> hi(600, 0.f), lo(600, 0.f);
  ASSERT_EQ(ct_pivot_high(b.high.data(), hi.data(), b.size(), 2, 0),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_pivot_low(b.low.data(), lo.data(), b.size(), 2, 0),
            CT_STATUS_SUCCESS);
  expect_close(hi, ref_pivot(b.high, 2, true), 0.0f, 0.0f);
  expect_close(lo, ref_pivot(b.low, 2, false), 0.0f, 0.0f);
}

TEST(ClubPack, XAverageHandlesNanPrefix) {
  const auto b = make_bars(600, 31);
  // OBTR has a NaN at bar 0; XAverage must seed at the first finite value —
  // exactly the pack's ema_line(obtr) chain.
  std::vector<float> obtr(600, 0.f), ema(600, 0.f);
  ASSERT_EQ(ct_obtr(b.high.data(), b.low.data(), b.close.data(), obtr.data(),
                    b.size(), 0),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_xaverage(obtr.data(), ema.data(), b.size(), 9, 0),
            CT_STATUS_SUCCESS);
  expect_close(ema, ref_xaverage(ref_obtr(b), 9), 5e-3f, 1e-3f);
}

TEST(ClubPack, RollingStatsAreNanStrict) {
  const auto b = make_bars(400, 37);
  // Build a NaN-prefixed line (TrendScore warm-up) and roll stats over it.
  std::vector<float> score(400, 0.f);
  ASSERT_EQ(ct_trendscore(b.close.data(), score.data(), b.size(), 10, 0),
            CT_STATUS_SUCCESS);

  std::vector<float> mean(400, 0.f), mx(400, 0.f), mn(400, 0.f), sd(400, 0.f);
  ASSERT_EQ(ct_rolling_mean(score.data(), mean.data(), 400, 18, 0),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_rolling_max(score.data(), mx.data(), 400, 18, 0),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_rolling_min(score.data(), mn.data(), 400, 18, 0),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_rolling_std(score.data(), sd.data(), 400, 18, 0),
            CT_STATUS_SUCCESS);

  // Reference: strict window over the score line.
  const int len = 18;
  for (int i = 0; i < 400; ++i) {
    bool ok = i >= len - 1;
    double sum = 0.0, sq = 0.0;
    float refMax = -1e30f, refMin = 1e30f;
    for (int k = 0; ok && k < len; ++k) {
      const float v = score[static_cast<size_t>(i - k)];
      if (!std::isfinite(v)) ok = false;
      else {
        sum += v;
        sq += static_cast<double>(v) * v;
        refMax = std::max(refMax, v);
        refMin = std::min(refMin, v);
      }
    }
    if (!ok) {
      EXPECT_TRUE(std::isnan(mean[static_cast<size_t>(i)])) << i;
      EXPECT_TRUE(std::isnan(mx[static_cast<size_t>(i)])) << i;
      EXPECT_TRUE(std::isnan(mn[static_cast<size_t>(i)])) << i;
      EXPECT_TRUE(std::isnan(sd[static_cast<size_t>(i)])) << i;
    } else {
      const double m = sum / len;
      const double var = std::max(0.0, sq / len - m * m);
      EXPECT_NEAR(mean[static_cast<size_t>(i)], m, 1e-3) << i;
      EXPECT_EQ(mx[static_cast<size_t>(i)], refMax) << i;
      EXPECT_EQ(mn[static_cast<size_t>(i)], refMin) << i;
      EXPECT_NEAR(sd[static_cast<size_t>(i)], std::sqrt(var), 1e-3) << i;
    }
  }
}

TEST(ClubPack, BatchMatchesSingleSeries) {
  const int n = 500, numSeries = 3;
  std::vector<Ohlcv> bars;
  for (int s = 0; s < numSeries; ++s) bars.push_back(make_bars(n, 41 + s));

  auto matrix = [&](const std::vector<float> Ohlcv::* col) {
    std::vector<float> m(static_cast<size_t>(n) * numSeries);
    for (int s = 0; s < numSeries; ++s) {
      const auto& src = bars[static_cast<size_t>(s)].*col;
      std::copy(src.begin(), src.end(), m.begin() + static_cast<size_t>(s) * n);
    }
    return m;
  };
  const auto mo = matrix(&Ohlcv::open), mh = matrix(&Ohlcv::high),
             ml = matrix(&Ohlcv::low), mc = matrix(&Ohlcv::close),
             mv = matrix(&Ohlcv::volume);

  std::vector<float> outB(static_cast<size_t>(n) * numSeries, 0.f);
  ASSERT_EQ(ct_wobv_batch(mo.data(), mh.data(), ml.data(), mc.data(),
                          mv.data(), outB.data(), n, numSeries, 0, 0),
            CT_STATUS_SUCCESS);
  for (int s = 0; s < numSeries; ++s) {
    std::vector<float> one(n, 0.f);
    const auto& b = bars[static_cast<size_t>(s)];
    ASSERT_EQ(ct_wobv(b.open.data(), b.high.data(), b.low.data(),
                      b.close.data(), b.volume.data(), one.data(), n, 0),
              CT_STATUS_SUCCESS);
    std::vector<float> rowB(outB.begin() + static_cast<size_t>(s) * n,
                            outB.begin() + static_cast<size_t>(s + 1) * n);
    expect_close(rowB, one, 1e-4f, 1e-6f);
  }

  std::vector<float> asiB(static_cast<size_t>(n) * numSeries, 0.f);
  ASSERT_EQ(ct_asi_batch(mo.data(), mh.data(), ml.data(), mc.data(),
                         asiB.data(), n, numSeries, 0, 0),
            CT_STATUS_SUCCESS);
  for (int s = 0; s < numSeries; ++s) {
    std::vector<float> rowB(asiB.begin() + static_cast<size_t>(s) * n,
                            asiB.begin() + static_cast<size_t>(s + 1) * n);
    expect_close(rowB, ref_asi(bars[static_cast<size_t>(s)]), 5e-3f, 1e-4f);
  }
}
