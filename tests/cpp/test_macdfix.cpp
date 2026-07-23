/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, MACDFIX) {
  const int N = 100;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> macd(N), signal(N), hist(N);
  int signalP = 9;
  ctStatus_t rc = ct_macdfix(x.data(), macd.data(), signal.data(), hist.data(), N, signalP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  auto ref = macdfix_ref(x, signalP);
  expect_approx_equal(macd, std::get<0>(ref));
  expect_approx_equal(signal, std::get<1>(ref));
  expect_approx_equal(hist, std::get<2>(ref));
}
