/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, Beta) {
  const int N = 128;
  std::vector<float> x(N), y(N);
  for (int i = 0; i < N; ++i) {
    x[i] = std::sin(0.01f * i);
    y[i] = std::cos(0.02f * i);
  }
  std::vector<float> out(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_beta(x.data(), y.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_beta failed";

  auto ref = beta_ref(x, y, p);
  // BETA can be ill-conditioned (denominator close to 0), so allow slightly
  // larger absolute tolerance vs TA-Lib for float inputs.
  expect_approx_equal(out, ref, 1e-2f);
}
