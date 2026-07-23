/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda/Context.h>
#include <tacuda.h>

#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
void generate_ohlcv(std::vector<float> &open, std::vector<float> &high,
                    std::vector<float> &low, std::vector<float> &close,
                    std::vector<float> &volume) {
  const size_t n = open.size();
  high.resize(n);
  low.resize(n);
  close.resize(n);
  volume.resize(n);

  for (size_t i = 0; i < n; ++i) {
    float t = static_cast<float>(i);

    // Deterministic synthetic price series (no RNG, stable across platforms).
    float base = 100.0f + 2.0f * std::sin(0.01f * t) + 0.5f * std::cos(0.005f * t) +
                 0.0001f * t;

    float o = base + 0.1f * std::sin(0.03f * t);
    float c = base + 0.1f * std::cos(0.02f * t);

    float h = std::max(o, c) + 0.2f + 0.3f * std::fabs(std::sin(0.01f * t));
    float l = std::min(o, c) - 0.2f - 0.3f * std::fabs(std::cos(0.01f * t));

    open[i] = o;
    high[i] = h;
    low[i] = l;
    close[i] = c;

    // Positive, varying volume.
    volume[i] = 100000.0f + static_cast<float>((i * 37) % 1000) * 10.0f;
  }
}
} // namespace

TEST(Tacuda, ContextPipeline_MultiIndicatorMatchesHostApi) {
  const int N = 4096;

  std::vector<float> open(N), high, low, close, volume;
  generate_ohlcv(open, high, low, close, volume);

  // --------------------------------------------------------------------------
  // Reference (Host API): each call does host->device->host.
  // --------------------------------------------------------------------------
  std::vector<float> ref_rsi(N), ref_atr(N), ref_adx(N), ref_obv(N), ref_mfi(N);
  std::vector<float> ref_ema_fast(N), ref_ema_slow(N), ref_trend(N);

  std::vector<float> ref_macd(N), ref_signal(N), ref_hist(N);
  std::vector<float> ref_stoch_k(N), ref_stoch_d(N);
  std::vector<float> ref_bb_u(N), ref_bb_m(N), ref_bb_l(N);

  ASSERT_EQ(ct_rsi(close.data(), ref_rsi.data(), N, 14), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_atr(high.data(), low.data(), close.data(), ref_atr.data(), N, 14, 0.0f),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_adx(high.data(), low.data(), close.data(), ref_adx.data(), N, 14),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_obv(close.data(), volume.data(), ref_obv.data(), N), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_mfi(high.data(), low.data(), close.data(), volume.data(), ref_mfi.data(), N, 14),
            CT_STATUS_SUCCESS);

  ASSERT_EQ(ct_ema(close.data(), ref_ema_fast.data(), N, 12), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_ema(close.data(), ref_ema_slow.data(), N, 26), CT_STATUS_SUCCESS);
  for (int i = 0; i < N; ++i) {
    ref_trend[i] = ref_ema_fast[i] - ref_ema_slow[i];
  }

  ASSERT_EQ(ct_macd(close.data(), ref_macd.data(), ref_signal.data(), ref_hist.data(), N, 12, 26, 9,
                    CT_MA_EMA),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stochastic(high.data(), low.data(), close.data(), ref_stoch_k.data(),
                          ref_stoch_d.data(), N, 14, 3),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_bbands(close.data(), ref_bb_u.data(), ref_bb_m.data(), ref_bb_l.data(), N, 20, 2.0f,
                      2.0f),
            CT_STATUS_SUCCESS);

  // --------------------------------------------------------------------------
  // Pipeline (Context): upload once, compute multiple indicators on-device,
  // then download results.
  // --------------------------------------------------------------------------
  tacuda::Context ctx;
  auto ohlcv = ctx.uploadOHLCV(open.data(), high.data(), low.data(), close.data(), volume.data(), N);

  auto rsi = ctx.rsi(ohlcv.close, 14);
  auto atr = ctx.atr(ohlcv.high, ohlcv.low, ohlcv.close, 14);
  auto adx = ctx.adx(ohlcv.high, ohlcv.low, ohlcv.close, 14);
  auto obv = ctx.obv(ohlcv.close, ohlcv.volume);
  auto mfi = ctx.mfi(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume, 14);

  auto ema_fast = ctx.ema(ohlcv.close, 12);
  auto ema_slow = ctx.ema(ohlcv.close, 26);
  auto trend = ctx.subtract(ema_fast, ema_slow);

  auto macd = ctx.macd(ohlcv.close, 12, 26, 9);
  auto stoch = ctx.stoch(ohlcv.high, ohlcv.low, ohlcv.close, 14, 3, 3);
  auto bb = ctx.bbands(ohlcv.close, 20, 2.0, 2.0, 0);

  std::vector<float> got_rsi(N), got_atr(N), got_adx(N), got_obv(N), got_mfi(N);
  std::vector<float> got_ema_fast(N), got_ema_slow(N), got_trend(N);
  std::vector<float> got_macd(N), got_signal(N), got_hist(N);
  std::vector<float> got_stoch_k(N), got_stoch_d(N);
  std::vector<float> got_bb_u(N), got_bb_m(N), got_bb_l(N);

  // Queue all downloads on the same stream and synchronize once.
  rsi.download(got_rsi.data(), ctx.stream());
  atr.download(got_atr.data(), ctx.stream());
  adx.download(got_adx.data(), ctx.stream());
  obv.download(got_obv.data(), ctx.stream());
  mfi.download(got_mfi.data(), ctx.stream());

  ema_fast.download(got_ema_fast.data(), ctx.stream());
  ema_slow.download(got_ema_slow.data(), ctx.stream());
  trend.download(got_trend.data(), ctx.stream());

  macd.macd.download(got_macd.data(), ctx.stream());
  macd.signal.download(got_signal.data(), ctx.stream());
  macd.hist.download(got_hist.data(), ctx.stream());

  stoch.slowK.download(got_stoch_k.data(), ctx.stream());
  stoch.slowD.download(got_stoch_d.data(), ctx.stream());

  bb.upper.download(got_bb_u.data(), ctx.stream());
  bb.middle.download(got_bb_m.data(), ctx.stream());
  bb.lower.download(got_bb_l.data(), ctx.stream());

  ctx.synchronize();

  // --------------------------------------------------------------------------
  // Compare Context pipeline results to Host API results.
  // --------------------------------------------------------------------------
  expect_approx_equal(got_rsi, ref_rsi);
  expect_approx_equal(got_atr, ref_atr);
  expect_approx_equal(got_adx, ref_adx);
  expect_approx_equal(got_obv, ref_obv);
  expect_approx_equal(got_mfi, ref_mfi);

  expect_approx_equal(got_ema_fast, ref_ema_fast);
  expect_approx_equal(got_ema_slow, ref_ema_slow);
  expect_approx_equal(got_trend, ref_trend);

  expect_approx_equal(got_macd, ref_macd);
  expect_approx_equal(got_signal, ref_signal);
  expect_approx_equal(got_hist, ref_hist);

  expect_approx_equal(got_stoch_k, ref_stoch_k);
  expect_approx_equal(got_stoch_d, ref_stoch_d);

  expect_approx_equal(got_bb_u, ref_bb_u);
  expect_approx_equal(got_bb_m, ref_bb_m);
  expect_approx_equal(got_bb_l, ref_bb_l);
}

TEST(Tacuda, ContextMacd_RespectsSignalPeriod) {
  const int N = 512;

  std::vector<float> x(N);
  for (int i = 0; i < N; ++i) {
    x[i] = 10.0f + 0.5f * std::sin(0.03f * i) + 0.2f * std::cos(0.01f * i);
  }

  std::vector<float> ref_macd7(N), ref_sig7(N), ref_hist7(N);
  std::vector<float> ref_macd9(N), ref_sig9(N), ref_hist9(N);

  ASSERT_EQ(ct_macd(x.data(), ref_macd7.data(), ref_sig7.data(), ref_hist7.data(), N, 12, 26, 7,
                    CT_MA_EMA),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_macd(x.data(), ref_macd9.data(), ref_sig9.data(), ref_hist9.data(), N, 12, 26, 9,
                    CT_MA_EMA),
            CT_STATUS_SUCCESS);

  tacuda::Context ctx;
  auto dx = ctx.upload(x.data(), N);

  auto r7 = ctx.macd(dx, 12, 26, 7);
  auto r9 = ctx.macd(dx, 12, 26, 9);

  std::vector<float> got_sig7(N), got_sig9(N);
  r7.signal.download(got_sig7.data(), ctx.stream());
  r9.signal.download(got_sig9.data(), ctx.stream());
  ctx.synchronize();

  expect_approx_equal(got_sig7, ref_sig7);
  expect_approx_equal(got_sig9, ref_sig9);

  // Guard against silently ignoring signalPeriod.
  bool any_diff = false;
  for (int i = 0; i < N; ++i) {
    if (std::isnan(ref_sig7[i]) || std::isnan(ref_sig9[i])) continue;
    if (std::fabs(ref_sig7[i] - ref_sig9[i]) > 1e-6f) {
      any_diff = true;
      break;
    }
  }
  EXPECT_TRUE(any_diff) << "signalPeriod appears to have no effect";
}

// ============================================================================
// DeviceBuffer::view() slicing
// ============================================================================
TEST(ContextPipeline, DeviceBufferView) {
    const int N = 100;
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i) data[i] = static_cast<float>(i);

    tacuda::Context ctx;
    auto buf = ctx.upload(data.data(), N);
    ASSERT_EQ(buf.size(), static_cast<size_t>(N));

    // Create a view of elements [10..49]
    auto slice = buf.view(10, 40);
    ASSERT_EQ(slice.size(), 40u);
    EXPECT_EQ(slice.data(), buf.data() + 10);

    // Download and verify the slice
    std::vector<float> downloaded(40);
    slice.download(downloaded.data(), ctx.stream());
    ctx.synchronize();

    for (int i = 0; i < 40; ++i) {
        EXPECT_FLOAT_EQ(downloaded[i], static_cast<float>(i + 10))
            << "View slice mismatch at " << i;
    }
}

// ============================================================================
// DeviceOHLCV move semantics
// ============================================================================
TEST(ContextPipeline, DeviceOHLCVMoveSemantics) {
    const int N = 256;
    std::vector<float> open(N), high(N), low(N), close(N), volume(N);
    for (int i = 0; i < N; ++i) {
        float v = 100.0f + static_cast<float>(i) * 0.01f;
        open[i] = v;
        high[i] = v + 1.0f;
        low[i] = v - 1.0f;
        close[i] = v + 0.5f;
        volume[i] = 10000.0f;
    }

    tacuda::Context ctx;
    auto ohlcv = ctx.uploadOHLCV(open.data(), high.data(), low.data(),
                                  close.data(), volume.data(), N);
    EXPECT_EQ(ohlcv.size, static_cast<size_t>(N));
    EXPECT_NE(ohlcv.close.data(), nullptr);

    // Move to another variable
    auto ohlcv2 = std::move(ohlcv);
    EXPECT_EQ(ohlcv2.size, static_cast<size_t>(N));
    EXPECT_NE(ohlcv2.close.data(), nullptr);
    // Moved-from DeviceBuffers have shared_ptr owner nulled,
    // but raw data_ pointer is not zeroed by default move.
    // Just verify the destination is correct.
}

// ============================================================================
// Math operations
// ============================================================================
TEST(ContextPipeline, MathOperations) {
    const int N = 100;
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i) data[i] = static_cast<float>(i + 1);

    tacuda::Context ctx;
    auto a = ctx.upload(data.data(), N);
    auto b = ctx.upload(data.data(), N);

    // add
    auto sum = ctx.add(a, b);
    std::vector<float> sumResult(N);
    sum.download(sumResult.data(), ctx.stream());
    ctx.synchronize();

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(sumResult[i], 2.0f * (i + 1), 1e-4f);
    }

    // scale
    auto scaled = ctx.scale(a, 3.0);
    std::vector<float> scaledResult(N);
    scaled.download(scaledResult.data(), ctx.stream());
    ctx.synchronize();

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(scaledResult[i], 3.0f * (i + 1), 1e-4f);
    }
}

// ============================================================================
// Error handling: invalid period
// ============================================================================
TEST(ContextPipeline, InvalidPeriodThrows) {
    const int N = 100;
    std::vector<float> data(N, 1.0f);

    tacuda::Context ctx;
    auto buf = ctx.upload(data.data(), N);

    // Non-positive periods are invalid arguments and throw.
    EXPECT_THROW(ctx.rsi(buf, 0), std::invalid_argument);
    EXPECT_THROW(ctx.sma(buf, -3), std::invalid_argument);

    // Since 0.6.0 a period larger than the input succeeds with an all-NaN
    // line (TA-Lib returns success with zero output elements).
    auto rsiOut = ctx.rsi(buf, N + 1);
    auto smaOut = ctx.sma(buf, N + 1);
    std::vector<float> host(N);
    rsiOut.download(host.data(), ctx.stream());
    ctx.synchronize();
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(std::isnan(host[i])) << "rsi short input at " << i;
    }
    smaOut.download(host.data(), ctx.stream());
    ctx.synchronize();
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(std::isnan(host[i])) << "sma short input at " << i;
    }
}

