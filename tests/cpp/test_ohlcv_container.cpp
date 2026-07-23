/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda/OHLCVSeries.h>
#include <tacuda.h>

#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(OHLCVSeries, ColumnMajorLayout) {
  tacuda::OHLCVSeries series(3);
  series.set_row(0, 1.0f, 2.0f, 0.5f, 1.5f, 10.0f);
  series.set_row(1, 1.1f, 2.1f, 0.6f, 1.4f, 11.0f);
  series.set_row(2, 0.9f, 1.9f, 0.4f, 1.2f, 12.0f);

  auto packed = series.column_major();
  std::vector<float> expected{
      1.0f, 1.1f, 0.9f, // open
      2.0f, 2.1f, 1.9f, // high
      0.5f, 0.6f, 0.4f, // low
      1.5f, 1.4f, 1.2f, // close
      10.0f, 11.0f, 12.0f // volume
  };
  expect_approx_equal(packed, expected);

  auto truncated = series.column_major(false);
  ASSERT_EQ(truncated.size(), 4 * series.size());
  std::vector<float> truncated_expected{
      1.0f, 1.1f, 0.9f, 2.0f, 2.1f, 1.9f, 0.5f, 0.6f, 0.4f, 1.5f, 1.4f, 1.2f};
  expect_approx_equal(truncated, truncated_expected);
}

TEST(OHLCVSeries, ConstructFromVectors) {
  std::vector<float> open{1.0f, 2.0f};
  std::vector<float> high{1.5f, 2.5f};
  std::vector<float> low{0.5f, 1.5f};
  std::vector<float> close{1.2f, 2.2f};
  tacuda::OHLCVSeries series(open, high, low, close);
  EXPECT_EQ(series.size(), 2U);
  EXPECT_FLOAT_EQ(series.volume()[0], 0.0f);
  EXPECT_FLOAT_EQ(series.volume()[1], 0.0f);
}

TEST(OHLCVSeries, ConstructFromColumnMajor) {
  std::vector<float> packed{
      1.0f, 2.0f, 3.0f,
      1.5f, 2.5f, 3.5f,
      0.5f, 1.5f, 2.5f,
      1.2f, 2.2f, 3.2f,
      10.0f, 11.0f, 12.0f};
  tacuda::OHLCVSeries series(packed.data(), 3, true);
  EXPECT_FLOAT_EQ(series.high()[2], 3.5f);
  EXPECT_FLOAT_EQ(series.volume()[1], 11.0f);

  tacuda::OHLCVSeries truncated(packed.data(), 3, false);
  EXPECT_FLOAT_EQ(truncated.volume()[2], 0.0f);
}

TEST(OHLCVSeries, FromRows) {
  const float rows[] = {
      1.0f, 1.5f, 0.5f, 1.2f, 10.0f,
      2.0f, 2.5f, 1.5f, 2.2f, 11.0f,
  };
  auto series = tacuda::OHLCVSeries::from_rows(rows, 2);
  EXPECT_FLOAT_EQ(series.open()[1], 2.0f);
  EXPECT_FLOAT_EQ(series.volume()[0], 10.0f);

  const float ohlc_rows[] = {
      1.0f, 1.5f, 0.5f, 1.2f,
      2.0f, 2.5f, 1.5f, 2.2f,
  };
  auto no_volume = tacuda::OHLCVSeries::from_rows(ohlc_rows, 2, 4);
  EXPECT_FLOAT_EQ(no_volume.volume()[0], 0.0f);
}

TEST(OHLCVSeries, WorksWithCImi) {
  const int N = 5;
  std::vector<float> open{1.0f, 2.0f, 1.0f, 1.5f, 1.2f};
  std::vector<float> high{1.5f, 2.5f, 1.5f, 1.8f, 1.4f};
  std::vector<float> low{0.8f, 1.8f, 0.7f, 1.2f, 0.9f};
  std::vector<float> close{1.5f, 1.5f, 0.8f, 1.6f, 1.0f};

  tacuda::OHLCVSeries series(open, high, low, close);

  std::vector<float> expected(N, 0.0f);
  ASSERT_EQ(ct_imi(open.data(), close.data(), expected.data(), N, 3),
            CT_STATUS_SUCCESS);

  std::vector<float> actual(N, 0.0f);
  ASSERT_EQ(ct_imi(series.open_data(), series.close_data(), actual.data(), N, 3),
            CT_STATUS_SUCCESS);

  expect_approx_equal(actual, expected);
}
