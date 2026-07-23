/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, KAMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 10;
  int fastP = 2;
  int slowP = 30;
  ctStatus_t rc = ct_kama(x.data(), out.data(), N, p, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_kama failed";

  auto ref = kama_ref(x, p, fastP, slowP);
  expect_approx_equal(out, ref);
  // TA-Lib KAMA emits first valid value at index `p` (warmup length = p)
  for (int i = 0; i < p; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
}
