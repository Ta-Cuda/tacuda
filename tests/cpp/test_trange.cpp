/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, TRANGE) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f};
  std::vector<float> low = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                            48.64f, 48.94f, 48.86f, 49.50f, 49.87f};
  std::vector<float> close = {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                              49.03f, 49.07f, 49.32f, 49.91f, 49.91f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  ctStatus_t rc =
      ct_trange(high.data(), low.data(), close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_trange failed";
  // TA-Lib lookback is 1: the first bar has no previous close -> NaN.
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  for (int i = 1; i < N; ++i) {
    float tr1 = high[i] - low[i];
    float tr2 = std::fabs(high[i] - close[i - 1]);
    float tr3 = std::fabs(low[i] - close[i - 1]);
    ref[i] = std::max(tr1, std::max(tr2, tr3));
  }
  expect_approx_equal(out, ref);
}
