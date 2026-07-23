/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, HT_TRENDLINE) {
  const int N = 50;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_ht_trendline(x.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  auto ref = ht_trendline_ref(x);
  expect_approx_equal(out, ref, 1e-2f);
}
