/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, NATR) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                            48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                            49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close = {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                              49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                              49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);

  int p = 14;
  ctStatus_t rc =
      ct_natr(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_natr failed";

  auto ref = natr_ref(high, low, close, p);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, NatrPeriodOneReturnsUnnormalizedTrueRange) {
  const std::vector<float> high = {10.0f, 12.0f, NAN, 13.0f};
  const std::vector<float> low = {8.0f, 10.0f, 12.0f, 11.0f};
  const std::vector<float> close = {9.0f, 11.0f, 13.0f, 12.0f};
  std::vector<float> out(4, -1.0f);
  ASSERT_EQ(ct_natr(high.data(), low.data(), close.data(), out.data(), 4, 1),
            CT_STATUS_SUCCESS);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_EQ(out[1], 3.0f);
  EXPECT_TRUE(std::isnan(out[2]));
  EXPECT_EQ(out[3], 2.0f);
}
