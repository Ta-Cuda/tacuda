/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, AD) {
  std::vector<float> high = {12.f, 12.5f, 13.f, 13.5f};
  std::vector<float> low = {11.f, 11.5f, 12.f, 12.5f};
  std::vector<float> close = {11.5f, 12.f, 12.5f, 13.f};
  std::vector<float> volume = {100.f, 110.f, 120.f, 130.f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc =
      ct_ad(high.data(), low.data(), close.data(), volume.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ad failed";
  auto ref = ad_ref(high, low, close, volume);
  expect_approx_equal(out, ref);
}
