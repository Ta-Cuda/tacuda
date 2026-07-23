/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, SUM) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc = ct_sum(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sum failed";
  // TA-Lib convention: sum over [end-p+1, end] reported at `end`, leading
  // warm-up of p-1 NaNs.
  for (int end = p - 1; end < N; ++end) {
    float s = 0.0f;
    for (int k = 0; k < p; ++k)
      s += x[end - k];
    ref[end] = s;
  }
  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN warm-up at " << i;
  }
}
