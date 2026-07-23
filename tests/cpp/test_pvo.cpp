/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, PVO) {
  const int N = 128;
  std::vector<float> v(N);
  for (int i = 0; i < N; ++i)
    v[i] = 100.0f + std::sin(0.1f * i);
  std::vector<float> out(N, 0.0f);
  int fastP = 12, slowP = 26;
  ctStatus_t rc = ct_pvo(v.data(), out.data(), N, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_pvo failed";
  auto ref = pvo_ref(v, fastP, slowP);
  for (int i = 0; i < slowP && i < N; ++i) {
    ref[i] = out[i];
  }
  expect_approx_equal(out, ref, 2.5e-1f);
  for (int i = 0; i < slowP - 1; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
