/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, LINEARREG_INTERCEPT) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.01f * i) + 0.5f;
  std::vector<float> out(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_linearreg_intercept(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_linearreg_intercept failed";
  auto ref = linearreg_intercept_ref(x, p);
  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
}
