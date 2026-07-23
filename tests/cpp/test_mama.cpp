/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, MAMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> mama(N, 0.0f), fama(N, 0.0f);
  float fast = 0.5f, slow = 0.05f;
  ctStatus_t rc = ct_mama(x.data(), mama.data(), fama.data(), N, fast, slow);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  auto [mamaRef, famaRef] = mama_ref(x, fast, slow);
  expect_approx_equal(mama, mamaRef);
  expect_approx_equal(fama, famaRef);
}
