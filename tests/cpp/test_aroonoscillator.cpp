/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, AroonOscillator) {
  std::vector<float> high = {1.f, 2.f, 3.f, 2.f, 3.f, 4.f, 5.f, 4.f, 6.f, 7.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 1.5f, 2.5f,
                            3.5f, 4.5f, 3.5f, 5.5f, 6.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int p = 5;
  ctStatus_t rc =
      ct_aroonosc(high.data(), low.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_aroonosc failed";
  auto ref = aroonosc_ref(high, low, p);
  expect_approx_equal(out, ref);
  for (int i = 0; i < p; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
