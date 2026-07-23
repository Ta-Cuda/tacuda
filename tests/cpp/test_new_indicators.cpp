/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

TEST(Tacuda, AvgDev) {
  const int N = 6;
  const int period = 3;
  std::vector<float> data{1.0f, 2.0f, 4.0f, 7.0f, 11.0f, 16.0f};
  std::vector<float> out(N, 0.0f);
  ASSERT_EQ(ct_avgdev(data.data(), out.data(), N, period), CT_STATUS_SUCCESS);
  std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
  // First period-1 values should be NaN (warmup period)
  for (int i = 0; i < period - 1; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
  // Compute reference values starting from index period-1
  for (int i = 0; i <= N - period; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < period; ++j) sum += data[i + j];
    float mean = sum / period;
    float dev = 0.0f;
    for (int j = 0; j < period; ++j) dev += std::fabs(data[i + j] - mean);
    ref[i + period - 1] = dev / period;  // Write at index i + period - 1
  }
  expect_approx_equal(out, ref);
}

TEST(Tacuda, MAVP) {
  {
    const int N = 20;
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i) data[i] = static_cast<float>(i + 1);
    std::vector<float> periods(N);
    for (int i = 0; i < N; ++i) periods[i] = 2.0f + (i % 2);  // Alternating 2 and 3
    const int minPeriod = 2;
    const int maxPeriod = 3;  // T3 requires size >= 6*period-5, so maxPeriod=3 needs size>=13
    // Compare against TA-Lib MAVP for each supported MA type.
    // Note: MAMA is excluded (period is not applicable and TA-Lib returns NaNs).
    std::array<ctMaType_t, 8> types = {CT_MA_SMA,  CT_MA_EMA,  CT_MA_WMA,
                                       CT_MA_DEMA, CT_MA_TEMA, CT_MA_TRIMA,
                                       CT_MA_KAMA, CT_MA_T3};

    for (ctMaType_t type : types) {
      std::vector<float> out(N, 0.0f);
      ASSERT_EQ(ct_mavp(data.data(), periods.data(), out.data(), N, minPeriod,
                        maxPeriod, type),
                CT_STATUS_SUCCESS);
      auto ref = mavp_ref(data, periods, minPeriod, maxPeriod,
                          static_cast<int>(type));
      expect_approx_equal(out, ref);
      for (int i = 0; i < N; ++i) {
        if (std::isnan(ref[i])) {
          EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
        }
      }
    }
  }

  {
    const int N = 5;
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> periods{2.0f, 2.5f, 3.0f, 3.0f, 2.0f};
    std::vector<float> out(N, 0.0f);
    ASSERT_EQ(ct_mavp(data.data(), periods.data(), out.data(), N, 2, 3,
                      CT_MA_SMA),
              CT_STATUS_SUCCESS);
    auto ref = mavp_ref(data, periods, 2, 3, static_cast<int>(CT_MA_SMA));
    expect_approx_equal(out, ref);
    for (int i = 0; i < N; ++i) {
      if (std::isnan(ref[i])) {
        EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
      }
    }
  }
}

TEST(Tacuda, NVI) {
  std::vector<float> close{1.0f, 2.0f, 1.5f, 1.8f};
  std::vector<float> volume{100.0f, 90.0f, 95.0f, 80.0f};
  std::vector<float> out(close.size(), 0.0f);
  ASSERT_EQ(ct_nvi(close.data(), volume.data(), out.data(), close.size()),
            CT_STATUS_SUCCESS);
  std::vector<float> ref{1000.0f, 2000.0f, 2000.0f,
                         2000.0f * (1.0f + (1.8f - 1.5f) / 1.5f)};
  expect_approx_equal(out, ref);
}

TEST(Tacuda, PVI) {
  std::vector<float> close{1.0f, 1.2f, 1.1f, 1.3f};
  std::vector<float> volume{100.0f, 120.0f, 110.0f, 150.0f};
  std::vector<float> out(close.size(), 0.0f);
  ASSERT_EQ(ct_pvi(close.data(), volume.data(), out.data(), close.size()),
            CT_STATUS_SUCCESS);
  std::vector<float> ref(close.size(), 0.0f);
  ref[0] = 1000.0f;
  ref[1] = 1000.0f * (1.0f + (1.2f - 1.0f) / 1.0f);
  ref[2] = ref[1];
  ref[3] = ref[2] * (1.0f + (1.3f - 1.1f) / 1.1f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, IMI) {
  const int N = 5;
  std::vector<float> open{1.0f, 2.0f, 1.0f, 1.5f, 1.2f};
  std::vector<float> close{1.5f, 1.5f, 0.8f, 1.6f, 1.0f};
  std::vector<float> out(N, 0.0f);
  ASSERT_EQ(ct_imi(open.data(), close.data(), out.data(), N, 3), CT_STATUS_SUCCESS);
  // TA-Lib-style alignment: window [end-p+1, end] reported at `end`,
  // leading warm-up of p-1 NaNs.
  std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
  ref[2] = 100.0f * (0.5f / 1.2f);
  ref[3] = 100.0f * (0.1f / 0.8f);
  ref[4] = 100.0f * (0.1f / 0.5f);
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

TEST(Tacuda, ACCBANDS) {
  const int N = 6;
  const int period = 3;
  std::vector<float> high{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float> low{9.0f, 9.5f, 10.0f, 11.0f, 12.0f, 13.0f};
  std::vector<float> close{9.5f, 10.5f, 11.0f, 12.5f, 13.0f, 14.0f};
  std::vector<float> upper(N), middle(N), lower(N);
  ASSERT_EQ(ct_accbands(high.data(), low.data(), close.data(), upper.data(),
                        middle.data(), lower.data(), N, period), CT_STATUS_SUCCESS);
  std::vector<float> rawUpper(N), rawLower(N);
  for (int i = 0; i < N; ++i) {
    float h = high[i];
    float l = low[i];
    float sum = h + l;
    if (sum != 0.0f) {
      float factor = 4.0f * (h - l) / sum;
      rawUpper[i] = h * (1.0f + factor);
      rawLower[i] = l * (1.0f - factor);
    } else {
      rawUpper[i] = h;
      rawLower[i] = l;
    }
  }
  std::vector<float> refUpper(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refMiddle(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refLower(N, std::numeric_limits<float>::quiet_NaN());
  // First period-1 values should be NaN (warmup period)
  for (int i = 0; i < period - 1; i++) {
    EXPECT_TRUE(std::isnan(upper[i])) << "expected NaN at warmup " << i;
    EXPECT_TRUE(std::isnan(middle[i])) << "expected NaN at warmup " << i;
    EXPECT_TRUE(std::isnan(lower[i])) << "expected NaN at warmup " << i;
  }
  // Compute reference values starting from index period-1
  for (int i = 0; i <= N - period; ++i) {
    float sumU = 0.0f, sumM = 0.0f, sumL = 0.0f;
    for (int j = 0; j < period; ++j) {
      sumU += rawUpper[i + j];
      sumM += close[i + j];
      sumL += rawLower[i + j];
    }
    // Write at index i + period - 1 to match TA-Lib behavior
    refUpper[i + period - 1] = sumU / period;
    refMiddle[i + period - 1] = sumM / period;
    refLower[i + period - 1] = sumL / period;
  }
  expect_approx_equal(upper, refUpper);
  expect_approx_equal(middle, refMiddle);
  expect_approx_equal(lower, refLower);
}

TEST(Tacuda, CDLThreeOutside) {
  const int N = 5;
  std::vector<float> open{10.0f, 8.5f, 10.4f, 11.5f, 9.4f};
  std::vector<float> high{10.5f, 11.0f, 11.5f, 11.8f, 9.6f};
  std::vector<float> low{9.5f, 8.3f, 10.2f, 9.0f, 8.5f};
  std::vector<float> close{9.0f, 10.6f, 11.2f, 9.5f, 8.8f};
  std::vector<float> out(N, 0.0f);
  ASSERT_EQ(ct_cdl_three_outside(open.data(), high.data(), low.data(), close.data(),
                                out.data(), N, 0), CT_STATUS_SUCCESS);
  auto ref = cdl_ref(open, high, low, close, "CDL3OUTSIDE");
  expect_approx_equal(out, ref);
}
