/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, MedPrice) {
  std::vector<float> high = {12.f, 13.f, 14.f};
  std::vector<float> low = {9.f, 10.f, 11.f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_medprice(high.data(), low.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_medprice failed";
  std::vector<float> ref(N, 0.0f);
  for (int i = 0; i < N; ++i)
    ref[i] = 0.5f * (high[i] + low[i]);
  expect_approx_equal(out, ref);
}
