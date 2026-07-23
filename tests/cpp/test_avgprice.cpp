/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, AvgPrice) {
  std::vector<float> open = {10.f, 11.f, 12.f};
  std::vector<float> high = {12.f, 13.f, 14.f};
  std::vector<float> low = {9.f, 10.f, 11.f};
  std::vector<float> close = {11.f, 12.f, 13.f};
  const int N = open.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_avgprice(open.data(), high.data(), low.data(),
                              close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_avgprice failed";
  auto ref = avgprice_ref(open, high, low, close);
  expect_approx_equal(out, ref);
}
