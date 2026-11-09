/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// A window indicator must not change meaning when its implementation does.
//
// SMA switches from per-window sums to prefix sums at period 96. A non-finite
// input may affect only the windows containing it, on both sides of that
// switch and in streaming.

#include <tacuda.h>
#include <tacuda/Context.h>
#include <tacuda_device.h>
#include <tacuda_stream.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

std::vector<float> series(int n) {
  std::mt19937 rng(11);
  std::normal_distribution<float> step(0.0f, 0.5f);
  std::vector<float> out(static_cast<std::size_t>(n));
  float price = 100.0f;
  for (float &v : out) {
    price += step(rng);
    v = price;
  }
  return out;
}

std::vector<double> window_mean(const std::vector<float> &x, int period) {
  std::vector<double> out(x.size(), std::numeric_limits<double>::quiet_NaN());
  for (int i = period - 1; i < static_cast<int>(x.size()); ++i) {
    double sum = 0.0;
    for (int k = i - period + 1; k <= i; ++k) sum += static_cast<double>(x[static_cast<std::size_t>(k)]);
    out[static_cast<std::size_t>(i)] = sum / period;
  }
  return out;
}

void expect_window_semantics(const std::string &label, const std::vector<float> &got,
                             const std::vector<double> &want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isfinite(want[i])) {
      ASSERT_NEAR(got[i], want[i], 1e-2) << label << " at " << i;
    } else if (std::isnan(want[i])) {
      ASSERT_TRUE(std::isnan(got[i])) << label << " at " << i << " got " << got[i];
    } else {
      ASSERT_EQ(got[i], static_cast<float>(want[i])) << label << " at " << i;
    }
  }
}

std::vector<float> sma(const std::vector<float> &x, int period) {
  std::vector<float> out(x.size());
  EXPECT_EQ(ct_sma(x.data(), out.data(), static_cast<int>(x.size()), period, nullptr), CT_STATUS_SUCCESS);
  return out;
}

std::vector<float> sma_stream(const std::vector<float> &x, int period) {
  ctStream_t *h = nullptr;
  EXPECT_EQ(ct_sma_stream_create(period, &h), CT_STATUS_SUCCESS);
  std::vector<float> out(x.size());
  EXPECT_EQ(ct_sma_stream_append(h, x.data(), static_cast<int>(x.size()), out.data()), CT_STATUS_SUCCESS);
  EXPECT_EQ(ct_stream_destroy(h), CT_STATUS_SUCCESS);
  return out;
}

}  // namespace

TEST(SmaWindowSemantics, InteriorNanLeavesOnlyItsOwnWindows) {
  for (int period : {20, 95, 96, 97, 150}) {
    std::vector<float> x = series(400);
    x[120] = kNaN;
    const std::string label = "SMA(" + std::to_string(period) + ")";
    expect_window_semantics(label, sma(x, period), window_mean(x, period));
    EXPECT_TRUE(std::isfinite(sma(x, period)[120 + period])) << label << " must recover";
  }
}

TEST(SmaWindowSemantics, LeadingNansOnlyDelayTheFirstValue) {
  for (int period : {95, 96, 97, 150}) {
    std::vector<float> x = series(400);
    for (int i = 0; i < 25; ++i) x[static_cast<std::size_t>(i)] = kNaN;
    expect_window_semantics("leading " + std::to_string(period), sma(x, period), window_mean(x, period));
  }
}

TEST(SmaWindowSemantics, InfinityIsScopedLikeNan) {
  for (int period : {95, 96, 97}) {
    std::vector<float> x = series(400);
    x[120] = kInf;
    expect_window_semantics("+inf " + std::to_string(period), sma(x, period), window_mean(x, period));
    x[140] = -kInf;
    expect_window_semantics("+-inf " + std::to_string(period), sma(x, period), window_mean(x, period));
  }
}

TEST(SmaWindowSemantics, BulkAndStreamingMasksAgree) {
  for (int period : {95, 96, 97, 150}) {
    std::vector<float> x = series(400);
    x[120] = kNaN;
    x[260] = kNaN;
    const std::vector<float> bulk = sma(x, period);
    const std::vector<float> incremental = sma_stream(x, period);
    for (std::size_t i = 0; i < x.size(); ++i) {
      ASSERT_EQ(std::isnan(bulk[i]), std::isnan(incremental[i])) << "period " << period << " at " << i;
    }
  }
}

TEST(SmaWindowSemantics, ChainOverWarmupNansProducesValues) {
  const std::vector<float> x = series(500);
  const int n = static_cast<int>(x.size());
  float *dIn = ct_device_alloc(n), *dEma = ct_device_alloc(n), *dSma = ct_device_alloc(n);
  ASSERT_TRUE(dIn && dEma && dSma);
  ASSERT_EQ(ct_device_upload(dIn, x.data(), n, nullptr), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_ema_d(dIn, dEma, n, 20, nullptr), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_sma_d(dEma, dSma, n, 150, nullptr), CT_STATUS_SUCCESS);
  std::vector<float> ema(x.size()), out(x.size());
  ASSERT_EQ(ct_device_download(ema.data(), dEma, n, nullptr), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_device_download(out.data(), dSma, n, nullptr), CT_STATUS_SUCCESS);
  ct_device_sync();
  ct_device_free(dIn);
  ct_device_free(dEma);
  ct_device_free(dSma);
  expect_window_semantics("EMA->SMA", out, window_mean(ema, 150));
  EXPECT_TRUE(std::isfinite(out[19 + 149]));
}

TEST(ContextParameters, StddevMultiplierScalesTheResult) {
  std::vector<float> c = series(200), open = c, high = c, low = c, volume(c.size(), 1.0f);
  tacuda::Context ctx;
  auto ohlcv = ctx.uploadOHLCV(open.data(), high.data(), low.data(), c.data(), volume.data(), c.size());
  auto one = ctx.stddev(ohlcv.close, 20, 1.0);
  auto two = ctx.stddev(ohlcv.close, 20, 2.0);
  std::vector<float> a(c.size()), b(c.size());
  one.download(a.data(), ctx.stream());
  two.download(b.data(), ctx.stream());
  ctx.synchronize();
  for (std::size_t i = 0; i < c.size(); ++i) {
    ASSERT_EQ(std::isnan(a[i]), std::isnan(b[i])) << i;
    if (!std::isnan(a[i])) ASSERT_NEAR(b[i], 2.0f * a[i], 1e-4f) << i;
  }
}

TEST(ContextParameters, UnsupportedInputsAreRejected) {
  std::vector<float> c = series(64), open = c, high = c, low = c, volume(c.size(), 1.0f);
  tacuda::Context ctx;
  auto ohlcv = ctx.uploadOHLCV(open.data(), high.data(), low.data(), c.data(), volume.data(), c.size());
  EXPECT_THROW(ctx.bbands(ohlcv.close, 20, 2.0, 2.0, 1), std::invalid_argument);
  EXPECT_THROW(ctx.stddev(ohlcv.close, 20, std::nan("")), std::invalid_argument);
}

namespace {

template <typename Bulk, typename Stream>
void expect_masks_agree(const char *name, const std::vector<float> &x, Bulk bulk, Stream stream) {
  const std::vector<float> a = bulk(x);
  const std::vector<float> b = stream(x);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(std::isnan(a[i]), std::isnan(b[i])) << name << " at " << i;
    if (!std::isnan(a[i])) ASSERT_NEAR(a[i], b[i], 0.02f) << name << " at " << i;
  }
}

}  // namespace

TEST(WmaWindowSemantics, BulkAndStreamingAgreeAcrossAlgorithmSwitches) {
  for (int period : {20, 64, 65, 256, 257, 300}) {
    std::vector<float> x = series(900);
    x[300] = kNaN;
    const auto bulk = [&](const std::vector<float> &in) {
      std::vector<float> out(in.size());
      EXPECT_EQ(ct_wma(in.data(), out.data(), static_cast<int>(in.size()), period, nullptr), CT_STATUS_SUCCESS);
      return out;
    };
    const auto stream = [&](const std::vector<float> &in) {
      ctStream_t *h = nullptr;
      EXPECT_EQ(ct_wma_stream_create(period, &h), CT_STATUS_SUCCESS);
      std::vector<float> out(in.size());
      EXPECT_EQ(ct_wma_stream_append(h, in.data(), static_cast<int>(in.size()), out.data()), CT_STATUS_SUCCESS);
      EXPECT_EQ(ct_stream_destroy(h), CT_STATUS_SUCCESS);
      return out;
    };
    expect_masks_agree(("WMA " + std::to_string(period)).c_str(), x, bulk, stream);
    EXPECT_TRUE(std::isfinite(bulk(x)[300 + period])) << "WMA " << period << " must recover";
  }
}

TEST(BbandsWindowSemantics, BulkAndStreamingAgree) {
  for (int period : {20, 150}) {
    std::vector<float> x = series(500);
    x[200] = kNaN;
    const auto bulk = [&](const std::vector<float> &in) {
      std::vector<float> up(in.size()), mid(in.size()), lo(in.size());
      EXPECT_EQ(ct_bbands(in.data(), up.data(), mid.data(), lo.data(), static_cast<int>(in.size()), period, 2.0f, 2.0f, nullptr), CT_STATUS_SUCCESS);
      return up;
    };
    const auto stream = [&](const std::vector<float> &in) {
      ctStream_t *h = nullptr;
      EXPECT_EQ(ct_bbands_stream_create(period, 2.0f, 2.0f, &h), CT_STATUS_SUCCESS);
      std::vector<float> up(in.size()), mid(in.size()), lo(in.size());
      EXPECT_EQ(ct_bbands_stream_append(h, in.data(), static_cast<int>(in.size()), up.data(), mid.data(), lo.data()), CT_STATUS_SUCCESS);
      EXPECT_EQ(ct_stream_destroy(h), CT_STATUS_SUCCESS);
      return up;
    };
    expect_masks_agree(("BBANDS " + std::to_string(period)).c_str(), x, bulk, stream);
    EXPECT_TRUE(std::isfinite(bulk(x)[200 + period])) << "BBANDS " << period << " must recover";
  }
}
