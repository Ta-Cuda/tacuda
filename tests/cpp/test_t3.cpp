/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, T3) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  float v = 0.7f;
  ctStatus_t rc = ct_t3(x.data(), out.data(), N, p, v);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_t3 failed";

  ref = t3_ref(x, p, v);
  expect_approx_equal(out, ref);
}
