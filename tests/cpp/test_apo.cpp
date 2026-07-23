/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, APO) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = 100.0f + std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f);
  int fastP = 12, slowP = 26;
  ctStatus_t rc = ct_apo(x.data(), out.data(), N, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_apo failed";
  auto ref = apo_ref(x, fastP, slowP);
  expect_approx_equal(out, ref, 1e-1f);
  for (int i = 0; i < slowP - 1; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
