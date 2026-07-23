/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, HT_SINE) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i) x[i] = std::sin(0.05f * i);
  std::vector<float> sine(N, 0.0f), lead(N, 0.0f);
  ctStatus_t rc = ct_ht_sine(x.data(), sine.data(), lead.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ht_sine failed";
  auto ref = ht_sine_ref(x);
  expect_approx_equal(sine, ref.first);
  expect_approx_equal(lead, ref.second);
}
