/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include <tacuda_device.h>
#include "test_utils.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
class FusedMode {
 public:
  explicit FusedMode(const char* mode) {
    const char* previous = std::getenv("TACUDA_BATCH_FUSED");
    hadValue_ = previous != nullptr;
    if (previous) previous_ = previous;
    set(mode);
  }
  ~FusedMode() {
    if (hadValue_) set(previous_.c_str());
    else {
#ifdef _WIN32
      _putenv_s("TACUDA_BATCH_FUSED", "");
#else
      unsetenv("TACUDA_BATCH_FUSED");
#endif
    }
  }
 private:
  static void set(const char* value) {
#ifdef _WIN32
    _putenv_s("TACUDA_BATCH_FUSED", value);
#else
    setenv("TACUDA_BATCH_FUSED", value, 1);
#endif
  }
  bool hadValue_;
  std::string previous_;
};
}

TEST(IndicatorAudit, StochasticMinimumLength) {
  for (auto periods : {std::pair<int,int>{1,1}, {5,3}, {65,3}, {14,33}}) {
    const int kPeriod = periods.first, dPeriod = periods.second;
    const int minimum = kPeriod + 2 * (dPeriod - 1);
    for (int n : {std::max(1, minimum - 1), minimum, minimum + 1}) {
      std::vector<float> close(n), high(n), low(n), k(n), d(n);
      for (int i = 0; i < n; ++i) {
        close[i] = static_cast<float>(i + 1);
        high[i] = close[i] + 1;
        low[i] = close[i] - 1;
      }
      const float expected = 100.0f * kPeriod / (kPeriod + 1);
      auto check = [&] {
        for (int i = 0; i < n; ++i) {
          SCOPED_TRACE(::testing::Message() << "K=" << kPeriod << " D=" << dPeriod
                       << " n=" << n << " i=" << i);
          if (i < minimum - 1) {
            EXPECT_TRUE(std::isnan(k[i]));
            EXPECT_TRUE(std::isnan(d[i]));
          } else {
            EXPECT_NEAR(k[i], expected, 1e-3f);
            EXPECT_NEAR(d[i], expected, 1e-3f);
          }
        }
      };
      ASSERT_EQ(ct_stochastic(high.data(), low.data(), close.data(), k.data(),
                             d.data(), n, kPeriod, dPeriod), CT_STATUS_SUCCESS);
      check();
      for (const char* mode : {"auto", "0", "1"}) {
        FusedMode selected(mode);
        ASSERT_EQ(ct_stochastic_batch(high.data(), low.data(), close.data(),
                    k.data(), d.data(), n, 1, kPeriod, dPeriod, 4), CT_STATUS_SUCCESS);
        check();
      }
    }
  }
}

TEST(IndicatorAudit, CorrelationLowVarianceAtLargeOffset) {
  // Exact affine relationships; a raw-power cutoff must not call these
  // nonconstant windows degenerate just because their price level is large.
  const int n = 1048577;
  std::vector<float> x(n), y(n), out(n);
  for (int i = 0; i < n; ++i) {
    x[i] = 1000000.0f + ((i % 17) - 8) * 0.0625f;
    y[i] = 2 * x[i] + 3;
  }
  for (int period : {1, 14, 65, 257}) {
    ASSERT_EQ(ct_correl(x.data(), y.data(), out.data(), n, period), CT_STATUS_SUCCESS);
    for (int i = 0; i < period - 1; ++i) ASSERT_TRUE(std::isnan(out[i]));
    const float expected = period == 1 ? 0.0f : 1.0f;
    for (int i = period - 1; i < n; ++i) {
      ASSERT_NEAR(out[i], expected, 1e-6f) << "period=" << period << " bar=" << i;
    }
  }
}

TEST(IndicatorAudit, CorrelationWindowContracts) {
  const int n = 4097, period = 65;
  std::vector<float> x(n), y(n), out(n), prefix(1024);
  for (int i = 0; i < n; ++i) {
    x[i] = 100.0f + (i % 17) * 0.125f;
    y[i] = -2 * x[i] + 3;
  }
  ASSERT_EQ(ct_correl(x.data(), y.data(), out.data(), n, period), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_correl(x.data(), y.data(), prefix.data(), static_cast<int>(prefix.size()), period), CT_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(out.data(), prefix.data(), prefix.size() * sizeof(float)), 0);
  for (int i = period - 1; i < n; ++i) ASSERT_NEAR(out[i], -1.0f, 1e-6f);
  y[100] = std::numeric_limits<float>::quiet_NaN();
  ASSERT_EQ(ct_correl(x.data(), y.data(), out.data(), n, period), CT_STATUS_SUCCESS);
  for (int i = 100; i < n; ++i) ASSERT_EQ(out[i], 0.0f);
  std::fill(y.begin(), y.end(), 7.0f);
  ASSERT_EQ(ct_correl(x.data(), y.data(), out.data(), n, period), CT_STATUS_SUCCESS);
  for (int i = period - 1; i < n; ++i) ASSERT_EQ(out[i], 0.0f);
  ASSERT_EQ(ct_correl(x.data(), x.data(), out.data(), n, n), CT_STATUS_SUCCESS);
  ASSERT_NEAR(out.back(), 1.0f, 1e-6f);
}

TEST(IndicatorAudit, PvoEmaSeedAndWarmup) {
  std::vector<float> volume(80, 100.0f), out(80);
  volume[40] = 10000.0f;
  for (auto periods : {std::pair<int,int>{1,2}, {2,3}, {12,26}}) {
    const int fast = periods.first, slow = periods.second;
    auto ema = [&](int period) {
      std::vector<double> result(volume.size(), std::numeric_limits<double>::quiet_NaN());
      double value = 0.0;
      for (int i = 0; i < period; ++i) value += volume[i];
      value /= period;
      result[period - 1] = value;
      for (size_t i = period; i < volume.size(); ++i) {
        value += (static_cast<double>(volume[i]) - value) * (2.0 / (period + 1));
        result[i] = value;
      }
      return result;
    };
    const auto f = ema(fast), s = ema(slow);
    ASSERT_EQ(ct_pvo(volume.data(), out.data(), static_cast<int>(volume.size()),
                     fast, slow), CT_STATUS_SUCCESS);
    for (int i = 0; i < slow - 1; ++i) ASSERT_TRUE(std::isnan(out[i]));
    for (size_t i = slow - 1; i < volume.size(); ++i) {
      const float expected = static_cast<float>(100 * (f[i] - s[i]) / s[i]);
      ASSERT_NEAR(out[i], expected, 1e-4f) << "fast=" << fast << " slow=" << slow << " i=" << i;
    }
  }
  for (float level : {0.0f, 1e-20f}) {
    std::fill(volume.begin(), volume.end(), level);
    for (int length : {25, 26, 80}) {
      ASSERT_EQ(ct_pvo(volume.data(), out.data(), length, 12, 26), CT_STATUS_SUCCESS);
      for (int i = 0; i < length; ++i) {
        if (i < 25) ASSERT_TRUE(std::isnan(out[i]));
        else ASSERT_EQ(out[i], 0.0f);
      }
    }
  }
}

TEST(IndicatorAudit, VolumeIndicesLongDoubleRecurrence) {
  const int n = 1048577;
  std::vector<float> close(n), volume(n), out(n);
  for (int i = 0; i < n; ++i) {
    close[i] = 100.0f + ((i % 17) - 8) * 0.0001f;
    volume[i] = (i % 2) ? 200.0f : 100.0f;
  }
  for (bool positive : {false, true}) {
    auto call = [&] {
      return positive ? ct_pvi(close.data(), volume.data(), out.data(), n)
                      : ct_nvi(close.data(), volume.data(), out.data(), n);
    };
    ASSERT_EQ(call(), CT_STATUS_SUCCESS);
    double expected = 1000.0, maxScaled = 0.0;
    for (int i = 0; i < n; ++i) {
      if (i > 0 && (positive ? volume[i] > volume[i-1] : volume[i] < volume[i-1])) {
        expected *= static_cast<double>(close[i]) / close[i-1];
      }
      maxScaled = std::max(maxScaled, std::abs(static_cast<double>(out[i]) - expected)
                                      / std::max(1.0, std::abs(expected)));
    }
    EXPECT_LE(maxScaled, 1e-6) << (positive ? "PVI" : "NVI");
    const auto first = out;
    for (int repeat = 0; repeat < 4; ++repeat) {
      ASSERT_EQ(call(), CT_STATUS_SUCCESS);
      EXPECT_EQ(std::memcmp(out.data(), first.data(), n * sizeof(float)), 0)
          << (positive ? "PVI" : "NVI");
    }
  }
}

TEST(IndicatorAudit, HilbertDeviceDegeneratePhase) {
  const int n = 512;
  auto allocate = [&] {
    float* pointer = nullptr;
    const auto status = cudaMalloc(&pointer, n * sizeof(float));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return std::unique_ptr<float, decltype(&cudaFree)>(pointer, cudaFree);
  };
  auto input = allocate(), first = allocate(), second = allocate();
  std::vector<float> values(n), expected(n), expectedLead(n), actual(n), actualLead(n);
  for (float level : {0.0f, 1.0f, 100.0f, 123.45f, 1000000.0f}) {
    for (bool alternating : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "level=" << level << " alternating=" << alternating);
      for (int i = 0; i < n; ++i) values[i] = level + (alternating ? (i % 2 ? .1f : -.1f) : 0.0f);
      ASSERT_EQ(cudaMemcpy(input.get(), values.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(ct_ht_dcphase(values.data(), expected.data(), n), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ht_dcphase_d(input.get(), first.get(), n), CT_STATUS_SUCCESS);
      ASSERT_EQ(cudaMemcpy(actual.data(), first.get(), n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      expect_approx_equal(actual, expected, 1e-4f);
      ASSERT_EQ(ct_ht_sine(values.data(), expected.data(), expectedLead.data(), n), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ht_sine_d(input.get(), first.get(), second.get(), n), CT_STATUS_SUCCESS);
      ASSERT_EQ(cudaMemcpy(actual.data(), first.get(), n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(actualLead.data(), second.get(), n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      expect_approx_equal(actual, expected, 1e-6f);
      expect_approx_equal(actualLead, expectedLead, 1e-6f);
      ASSERT_EQ(ct_ht_trendmode(values.data(), expected.data(), n), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ht_trendmode_d(input.get(), first.get(), n), CT_STATUS_SUCCESS);
      ASSERT_EQ(cudaMemcpy(actual.data(), first.get(), n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(actual, expected);
    }
  }
}
