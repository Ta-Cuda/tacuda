/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, BOP) {
  const int N = 64;
  std::vector<float> open(N), high(N), low(N), close(N);
  for (int i = 0; i < N; ++i) {
    open[i] = 10.0f + 0.1f * i;
    high[i] = open[i] + 1.0f;
    low[i] = open[i] - 1.0f;
    close[i] = open[i] + std::sin(0.1f * i);
  }
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  ctStatus_t rc = ct_bop(open.data(), high.data(), low.data(), close.data(),
                         out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_bop failed";
  for (int i = 0; i < N; ++i) {
    float denom = high[i] - low[i];
    ref[i] = (denom == 0.0f) ? 0.0f : (close[i] - open[i]) / denom;
  }
  expect_approx_equal(out, ref);
}
