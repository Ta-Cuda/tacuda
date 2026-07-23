/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, HT_PHASOR) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i) x[i] = std::sin(0.05f * i);
  std::vector<float> inphase(N, 0.0f), quadrature(N, 0.0f);
  ctStatus_t rc = ct_ht_phasor(x.data(), inphase.data(), quadrature.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ht_phasor failed";
  auto ref = ht_phasor_ref(x);
  expect_approx_equal(inphase, ref.first);
  expect_approx_equal(quadrature, ref.second);
}
