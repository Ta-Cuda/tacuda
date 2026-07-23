/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, BBANDS) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> upper(N, 0.0f), middle(N, 0.0f), lower(N, 0.0f);

  int p = 20;
  float up = 2.0f, down = 2.0f;
  ctStatus_t rc = ct_bbands(x.data(), upper.data(), middle.data(), lower.data(),
                            N, p, up, down);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_bbands failed";

  auto [refU, refM, refL] = bbands_ref(x, p, up, down);
  expect_approx_equal(upper, refU);
  expect_approx_equal(middle, refM);
  expect_approx_equal(lower, refL);
}
