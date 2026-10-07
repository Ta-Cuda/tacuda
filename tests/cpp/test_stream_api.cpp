/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Streaming API parity and lifetime tests.
//
// The contract under test: appending bars through a stream reproduces the
// tail of a full-series recompute. The NaN warm-up mask must match exactly;
// values are compared with the corpus oracle's mixed absolute/relative
// tolerance, because the stream state is double while several full-series
// kernels accumulate their recursion in float (documented in
// docs/DESIGN_STREAMING_API.md).

#include <tacuda.h>
#include <tacuda_stream.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

constexpr int kBars = 4096;

struct Series {
  std::vector<float> high, low, close, volume;
};

// Deterministic random walk with a flat stretch (exercises the zero-range
// and flat-window branches) and a hard drop.
Series make_series(int n) {
  Series s;
  s.high.reserve(n);
  s.low.reserve(n);
  s.close.reserve(n);
  s.volume.reserve(n);

  std::mt19937 rng(20260802u);
  std::normal_distribution<float> step(0.0f, 0.75f);
  std::uniform_real_distribution<float> vol(1000.0f, 5000.0f);

  float price = 100.0f;
  for (int i = 0; i < n; ++i) {
    if (i >= 500 && i < 540) {
      // flat stretch: identical bars
    } else if (i == 1200) {
      price *= 0.85f;
    } else {
      price += step(rng);
      if (price < 1.0f) price = 1.0f;
    }
    const float spread = 0.4f + std::fabs(step(rng)) * 0.2f;
    s.close.push_back(price);
    s.high.push_back(price + spread);
    s.low.push_back(price - spread);
    s.volume.push_back(vol(rng));
  }
  return s;
}

const Series &series() {
  static const Series s = make_series(kBars);
  return s;
}

// Corpus oracle: identical NaN masks, then |a-b| / max(1,|a|,|b|) <= tol.
void expect_stream_parity(const std::string &what, const std::vector<float> &full,
                          const std::vector<float> &streamed, int lookback,
                          double tol) {
  ASSERT_EQ(full.size(), streamed.size()) << what << ": size mismatch";

  double worst = 0.0;
  int worstIdx = -1;
  for (std::size_t i = 0; i < full.size(); ++i) {
    const bool nanFull = std::isnan(full[i]);
    const bool nanStream = std::isnan(streamed[i]);
    ASSERT_EQ(nanFull, nanStream)
        << what << ": NaN mask differs at " << i << " (full=" << full[i]
        << ", stream=" << streamed[i] << ")";
    if (nanFull) continue;

    const double a = static_cast<double>(full[i]);
    const double b = static_cast<double>(streamed[i]);
    const double denom = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    const double rel = std::fabs(a - b) / denom;
    if (rel > worst) {
      worst = rel;
      worstIdx = static_cast<int>(i);
    }
  }

  // Everything before the lookback must be warm-up NaN on both sides.
  for (int i = 0; i < lookback && i < static_cast<int>(full.size()); ++i) {
    EXPECT_TRUE(std::isnan(streamed[i]))
        << what << ": expected warm-up NaN at " << i;
  }

  EXPECT_LE(worst, tol) << what << ": worst deviation " << worst << " at index "
                        << worstIdx;
  std::cout << "[parity] " << what << ": worst=" << worst << " (idx " << worstIdx
            << ", tol " << tol << ")\n";
}

// RAII wrapper so a failing assertion cannot leak the handle.
class StreamHandle {
public:
  StreamHandle() = default;
  ~StreamHandle() { ct_stream_destroy(h_); }
  StreamHandle(const StreamHandle &) = delete;
  StreamHandle &operator=(const StreamHandle &) = delete;

  ctStream_t **out() { return &h_; }
  ctStream_t *get() const { return h_; }
  void release() {
    ct_stream_destroy(h_);
    h_ = nullptr;
  }

private:
  ctStream_t *h_ = nullptr;
};

// Feeds the whole series through `append` in fixed-size chunks.
template <typename AppendFn>
void feed(ctStream_t *s, int total, int chunk, AppendFn &&append) {
  for (int off = 0; off < total; off += chunk) {
    const int n = std::min(chunk, total - off);
    ASSERT_EQ(append(off, n), CT_STATUS_SUCCESS)
        << "append failed at offset " << off << " (" << ct_get_last_error() << ")";
  }
}

const int kChunks[] = {1, 3, 16, kBars};

}  // namespace

// ---------------------------------------------------------------------------
// Parity: single-output, close-only families
// ---------------------------------------------------------------------------

TEST(TacudaStream, SmaMatchesFullSeries) {
  const auto &s = series();
  const int period = 30;
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_sma(s.close.data(), full.data(), kBars, period), CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_sma_stream_create(period, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_sma_stream_append(h.get(), s.close.data() + off, n, got.data() + off);
    });
    expect_stream_parity("SMA chunk=" + std::to_string(chunk), full, got,
                         period - 1, 1e-6);
  }
}

TEST(TacudaStream, EmaMatchesFullSeries) {
  const auto &s = series();
  const int period = 21;
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_ema(s.close.data(), full.data(), kBars, period), CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_ema_stream_create(period, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_ema_stream_append(h.get(), s.close.data() + off, n, got.data() + off);
    });
    expect_stream_parity("EMA chunk=" + std::to_string(chunk), full, got,
                         period - 1, 1e-5);
  }
}

TEST(TacudaStream, WmaMatchesFullSeries) {
  const auto &s = series();
  const int period = 18;
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_wma(s.close.data(), full.data(), kBars, period), CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_wma_stream_create(period, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_wma_stream_append(h.get(), s.close.data() + off, n, got.data() + off);
    });
    expect_stream_parity("WMA chunk=" + std::to_string(chunk), full, got,
                         period - 1, 1e-6);
  }
}

TEST(TacudaStream, RsiMatchesFullSeries) {
  const auto &s = series();
  const int period = 14;
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_rsi(s.close.data(), full.data(), kBars, period), CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_rsi_stream_create(period, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_rsi_stream_append(h.get(), s.close.data() + off, n, got.data() + off);
    });
    expect_stream_parity("RSI chunk=" + std::to_string(chunk), full, got, period,
                         1e-4);
  }
}

// ---------------------------------------------------------------------------
// Parity: HLC / volume families
// ---------------------------------------------------------------------------

TEST(TacudaStream, AtrMatchesFullSeries) {
  const auto &s = series();
  const int period = 14;
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_atr(s.high.data(), s.low.data(), s.close.data(), full.data(), kBars,
                   period, 0.0f),
            CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_atr_stream_create(period, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_atr_stream_append(h.get(), s.high.data() + off, s.low.data() + off,
                                  s.close.data() + off, n, got.data() + off);
    });
    expect_stream_parity("ATR chunk=" + std::to_string(chunk), full, got, period,
                         1e-4);
  }
}

TEST(TacudaStream, NatrMatchesFullSeries) {
  const auto &s = series();
  const int period = 14;
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_natr(s.high.data(), s.low.data(), s.close.data(), full.data(), kBars,
                    period),
            CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_natr_stream_create(period, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_natr_stream_append(h.get(), s.high.data() + off, s.low.data() + off,
                                   s.close.data() + off, n, got.data() + off);
    });
    expect_stream_parity("NATR chunk=" + std::to_string(chunk), full, got, period,
                         1e-4);
  }
}

TEST(TacudaStream, ObvMatchesFullSeries) {
  const auto &s = series();
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_obv(s.close.data(), s.volume.data(), full.data(), kBars),
            CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_obv_stream_create(h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_obv_stream_append(h.get(), s.close.data() + off,
                                  s.volume.data() + off, n, got.data() + off);
    });
    expect_stream_parity("OBV chunk=" + std::to_string(chunk), full, got, 0, 1e-4);
  }
}

TEST(TacudaStream, AdMatchesFullSeries) {
  const auto &s = series();
  std::vector<float> full(kBars);
  ASSERT_EQ(ct_ad(s.high.data(), s.low.data(), s.close.data(), s.volume.data(),
                  full.data(), kBars),
            CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_ad_stream_create(h.out()), CT_STATUS_SUCCESS);
    std::vector<float> got(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_ad_stream_append(h.get(), s.high.data() + off, s.low.data() + off,
                                 s.close.data() + off, s.volume.data() + off, n,
                                 got.data() + off);
    });
    expect_stream_parity("AD chunk=" + std::to_string(chunk), full, got, 0, 1e-4);
  }
}

// ---------------------------------------------------------------------------
// Parity: multi-output families
// ---------------------------------------------------------------------------

TEST(TacudaStream, MacdMatchesFullSeries) {
  const auto &s = series();
  const int fast = 12, slow = 26, signal = 9;
  std::vector<float> fullMacd(kBars), fullSignal(kBars), fullHist(kBars);
  ASSERT_EQ(ct_macd(s.close.data(), fullMacd.data(), fullSignal.data(),
                    fullHist.data(), kBars, fast, slow, signal, CT_MA_EMA),
            CT_STATUS_SUCCESS);
  const int lookback = slow - 1 + signal - 1;

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_macd_stream_create(fast, slow, signal, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> m(kBars, 0.0f), sig(kBars, 0.0f), hist(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_macd_stream_append(h.get(), s.close.data() + off, n, m.data() + off,
                                   sig.data() + off, hist.data() + off);
    });
    const std::string tag = " chunk=" + std::to_string(chunk);
    expect_stream_parity("MACD.line" + tag, fullMacd, m, lookback, 1e-5);
    expect_stream_parity("MACD.signal" + tag, fullSignal, sig, lookback, 1e-5);
    expect_stream_parity("MACD.hist" + tag, fullHist, hist, lookback, 1e-5);
  }
}

TEST(TacudaStream, BbandsMatchesFullSeries) {
  const auto &s = series();
  const int period = 20;
  const float up = 2.0f, dn = 2.0f;
  std::vector<float> fullU(kBars), fullM(kBars), fullL(kBars);
  ASSERT_EQ(ct_bbands(s.close.data(), fullU.data(), fullM.data(), fullL.data(),
                      kBars, period, up, dn),
            CT_STATUS_SUCCESS);

  for (int chunk : kChunks) {
    StreamHandle h;
    ASSERT_EQ(ct_bbands_stream_create(period, up, dn, h.out()), CT_STATUS_SUCCESS);
    std::vector<float> u(kBars, 0.0f), m(kBars, 0.0f), l(kBars, 0.0f);
    feed(h.get(), kBars, chunk, [&](int off, int n) {
      return ct_bbands_stream_append(h.get(), s.close.data() + off, n, u.data() + off,
                                     m.data() + off, l.data() + off);
    });
    const std::string tag = " chunk=" + std::to_string(chunk);
    expect_stream_parity("BBANDS.upper" + tag, fullU, u, period - 1, 1e-5);
    expect_stream_parity("BBANDS.middle" + tag, fullM, m, period - 1, 1e-6);
    expect_stream_parity("BBANDS.lower" + tag, fullL, l, period - 1, 1e-5);
  }
}

// ---------------------------------------------------------------------------
// Semantics
// ---------------------------------------------------------------------------

TEST(TacudaStream, LookbackMatchesFirstNonNan) {
  struct Case {
    const char *name;
    ctStream_t *handle;
    int expected;
  };

  StreamHandle sma, ema, rsi, atr, macd, obv;
  ASSERT_EQ(ct_sma_stream_create(30, sma.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_ema_stream_create(21, ema.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_rsi_stream_create(14, rsi.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_atr_stream_create(14, atr.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_macd_stream_create(12, 26, 9, macd.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_obv_stream_create(obv.out()), CT_STATUS_SUCCESS);

  const Case cases[] = {{"SMA", sma.get(), 29},  {"EMA", ema.get(), 20},
                        {"RSI", rsi.get(), 14},  {"ATR", atr.get(), 14},
                        {"MACD", macd.get(), 33}, {"OBV", obv.get(), 0}};

  for (const auto &c : cases) {
    int lb = -1;
    ASSERT_EQ(ct_stream_lookback(c.handle, &lb), CT_STATUS_SUCCESS) << c.name;
    EXPECT_EQ(lb, c.expected) << c.name;
  }

  int outs = 0;
  ASSERT_EQ(ct_stream_outputs(macd.get(), &outs), CT_STATUS_SUCCESS);
  EXPECT_EQ(outs, 3);
  ASSERT_EQ(ct_stream_outputs(sma.get(), &outs), CT_STATUS_SUCCESS);
  EXPECT_EQ(outs, 1);
}

TEST(TacudaStream, ResetEqualsFreshStream) {
  const auto &s = series();
  const int period = 14;

  StreamHandle reused;
  ASSERT_EQ(ct_rsi_stream_create(period, reused.out()), CT_STATUS_SUCCESS);

  std::vector<float> firstRun(256, 0.0f);
  ASSERT_EQ(ct_rsi_stream_append(reused.get(), s.close.data(), 256, firstRun.data()),
            CT_STATUS_SUCCESS);

  long long bars = -1;
  ASSERT_EQ(ct_stream_bars(reused.get(), &bars), CT_STATUS_SUCCESS);
  EXPECT_EQ(bars, 256);

  ASSERT_EQ(ct_stream_reset(reused.get()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stream_bars(reused.get(), &bars), CT_STATUS_SUCCESS);
  EXPECT_EQ(bars, 0);

  std::vector<float> afterReset(256, 0.0f);
  ASSERT_EQ(ct_rsi_stream_append(reused.get(), s.close.data(), 256, afterReset.data()),
            CT_STATUS_SUCCESS);

  for (int i = 0; i < 256; ++i) {
    if (std::isnan(firstRun[i])) {
      EXPECT_TRUE(std::isnan(afterReset[i])) << "index " << i;
    } else {
      EXPECT_FLOAT_EQ(firstRun[i], afterReset[i]) << "index " << i;
    }
  }
}

TEST(TacudaStream, WarmupReplayThenTickByTick) {
  // The intended live-trading usage: replay history in one call, then append
  // each closed bar. Must equal a single bulk append of the same data.
  const auto &s = series();
  const int period = 14;
  const int history = 1000;

  StreamHandle bulk;
  ASSERT_EQ(ct_rsi_stream_create(period, bulk.out()), CT_STATUS_SUCCESS);
  std::vector<float> bulkOut(kBars, 0.0f);
  ASSERT_EQ(ct_rsi_stream_append(bulk.get(), s.close.data(), kBars, bulkOut.data()),
            CT_STATUS_SUCCESS);

  StreamHandle live;
  ASSERT_EQ(ct_rsi_stream_create(period, live.out()), CT_STATUS_SUCCESS);
  std::vector<float> liveOut(kBars, 0.0f);
  ASSERT_EQ(ct_rsi_stream_append(live.get(), s.close.data(), history, liveOut.data()),
            CT_STATUS_SUCCESS);
  for (int i = history; i < kBars; ++i) {
    ASSERT_EQ(ct_rsi_stream_append(live.get(), s.close.data() + i, 1, liveOut.data() + i),
              CT_STATUS_SUCCESS);
  }

  for (int i = 0; i < kBars; ++i) {
    if (std::isnan(bulkOut[i])) {
      ASSERT_TRUE(std::isnan(liveOut[i])) << "index " << i;
    } else {
      ASSERT_FLOAT_EQ(bulkOut[i], liveOut[i]) << "index " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// Lifetime and error contract
// ---------------------------------------------------------------------------

TEST(TacudaStream, DestroyIsNullSafeAndRejectsReuse) {
  EXPECT_EQ(ct_stream_destroy(nullptr), CT_STATUS_SUCCESS);

  ctStream_t *raw = nullptr;
  ASSERT_EQ(ct_sma_stream_create(10, &raw), CT_STATUS_SUCCESS);
  ASSERT_NE(raw, nullptr);
  ASSERT_EQ(ct_stream_destroy(raw), CT_STATUS_SUCCESS);

  // The handle is now dangling: every entry point must reject it instead of
  // dereferencing freed state.
  float in = 1.0f, out = 0.0f;
  int lb = 0;
  EXPECT_EQ(ct_sma_stream_append(raw, &in, 1, &out), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_reset(raw), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_lookback(raw, &lb), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_destroy(raw), CT_STATUS_INVALID_ARGUMENT);
}

TEST(TacudaStream, RejectsWrongFamilyAppend) {
  StreamHandle atr;
  ASSERT_EQ(ct_atr_stream_create(14, atr.out()), CT_STATUS_SUCCESS);

  float in = 1.0f, out = 0.0f;
  EXPECT_EQ(ct_sma_stream_append(atr.get(), &in, 1, &out), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(std::string(ct_get_last_error()).find("different indicator"),
            std::string::npos);
}

TEST(TacudaStream, RetiredHandleCannotAliasNewStream) {
  ctStream_t *retired = nullptr;
  ASSERT_EQ(ct_sma_stream_create(3, &retired), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stream_destroy(retired), CT_STATUS_SUCCESS);
  for (int i = 0; i < 256; ++i) {
    StreamHandle live;
    ASSERT_EQ(ct_sma_stream_create(3, live.out()), CT_STATUS_SUCCESS);
    ASSERT_NE(live.get(), retired);
    float in[] = {1, 2, 3}, out[3] = {};
    ASSERT_EQ(ct_sma_stream_append(live.get(), in, 3, out), CT_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(out[2], 2);
    long long bars = -1;
    int outputs = -1;
    EXPECT_EQ(ct_stream_bars(retired, &bars), CT_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ct_stream_outputs(retired, &outputs), CT_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ct_stream_reset(retired), CT_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ct_stream_destroy(retired), CT_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(bars, -1);
    EXPECT_EQ(outputs, -1);
    ASSERT_EQ(ct_stream_bars(live.get(), &bars), CT_STATUS_SUCCESS);
    EXPECT_EQ(bars, 3);
  }
}

TEST(TacudaStream, RejectsWrongHandleKindWithoutDereference) {
  ctStreamBatch_t *batch = nullptr;
  ASSERT_EQ(ct_sma_stream_batch_create(3, 2, &batch), CT_STATUS_SUCCESS);
  auto *wrong = reinterpret_cast<ctStream_t *>(batch);
  EXPECT_EQ(ct_stream_reset(wrong), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_destroy(wrong), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_batch_destroy(batch), CT_STATUS_SUCCESS);
}

TEST(TacudaStream, RejectsInvalidArguments) {
  ctStream_t *h = nullptr;
  EXPECT_EQ(ct_sma_stream_create(1, &h), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_create(100001, &h), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_create(10, nullptr), CT_STATUS_INVALID_ARGUMENT);

  StreamHandle sma;
  ASSERT_EQ(ct_sma_stream_create(10, sma.out()), CT_STATUS_SUCCESS);
  float in = 1.0f, out = 0.0f;
  EXPECT_EQ(ct_sma_stream_append(sma.get(), nullptr, 1, &out),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_append(sma.get(), &in, 1, nullptr),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_append(sma.get(), &in, 0, &out), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_append(sma.get(), &in, -5, &out), CT_STATUS_INVALID_ARGUMENT);

  // ATR degenerates to TRANGE at period 1, which TA-Lib accepts.
  StreamHandle atr;
  EXPECT_EQ(ct_atr_stream_create(1, atr.out()), CT_STATUS_SUCCESS);
}

TEST(TacudaStream, HandlesAreIndependent) {
  const auto &s = series();
  StreamHandle a, b;
  ASSERT_EQ(ct_ema_stream_create(10, a.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_ema_stream_create(10, b.out()), CT_STATUS_SUCCESS);

  std::vector<float> outA(200, 0.0f), outB(200, 0.0f);
  // Interleave the two streams over the same data; both must agree.
  for (int i = 0; i < 200; ++i) {
    ASSERT_EQ(ct_ema_stream_append(a.get(), s.close.data() + i, 1, outA.data() + i),
              CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_ema_stream_append(b.get(), s.close.data() + i, 1, outB.data() + i),
              CT_STATUS_SUCCESS);
  }
  for (int i = 0; i < 200; ++i) {
    if (std::isnan(outA[i])) {
      EXPECT_TRUE(std::isnan(outB[i])) << "index " << i;
    } else {
      EXPECT_FLOAT_EQ(outA[i], outB[i]) << "index " << i;
    }
  }
}

TEST(TacudaStream, NanInputPoisonsWilderRecursion) {
  // TA-Lib semantics: a NaN price poisons the Wilder averages instead of
  // being treated as a zero change. The stream must behave like the
  // full-series path, which the corpus already pins down.
  std::vector<float> close(200);
  for (int i = 0; i < 200; ++i) close[i] = 100.0f + 0.5f * static_cast<float>(i % 7);
  close[100] = std::numeric_limits<float>::quiet_NaN();

  std::vector<float> full(200);
  ASSERT_EQ(ct_rsi(close.data(), full.data(), 200, 14), CT_STATUS_SUCCESS);

  StreamHandle h;
  ASSERT_EQ(ct_rsi_stream_create(14, h.out()), CT_STATUS_SUCCESS);
  std::vector<float> got(200, 0.0f);
  for (int i = 0; i < 200; ++i) {
    ASSERT_EQ(ct_rsi_stream_append(h.get(), close.data() + i, 1, got.data() + i),
              CT_STATUS_SUCCESS);
  }

  for (int i = 0; i < 200; ++i) {
    EXPECT_EQ(std::isnan(full[i]), std::isnan(got[i]))
        << "NaN mask differs at " << i;
  }
}
