/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, TRIX) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_trix(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_trix failed";

  auto ref = trix_ref(x, p);
  // TRIX is a chained EMA(EMA(EMA(x))) + ROC, and small float-vs-double differences can be
  // amplified around 0-crossings. Keep a slightly wider tolerance while still comparing to TA-Lib.
  expect_approx_equal(out, ref, 3e-3f);

  // TA-Lib lookback for TRIX = 3*(p-1) + 1
  int lookback = 3 * (p - 1) + 1;
  for (int i = 0; i < lookback; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
}
