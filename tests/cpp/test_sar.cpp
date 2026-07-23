/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, SARTrending) {
  std::vector<float> high = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sar(high.data(), low.data(), out.data(), N, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sar failed";
  auto ref = sar_talib_ref(high, low, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, SARRanging) {
  std::vector<float> high = {5.f, 6.f, 5.5f, 6.2f, 5.8f, 6.4f, 5.9f, 6.5f};
  std::vector<float> low = {4.f, 5.f, 4.5f, 5.2f, 4.8f, 5.4f, 4.9f, 5.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sar(high.data(), low.data(), out.data(), N, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sar failed";
  auto ref = sar_talib_ref(high, low, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}
