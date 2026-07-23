/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, OBV) {
  std::vector<float> price = {1.0f, 2.0f, 2.0f, 1.0f, 3.0f};
  std::vector<float> volume = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  const int N = price.size();
  std::vector<float> out(N, 0.0f), ref = {10.0f, 30.0f, 30.0f, -10.0f, 40.0f};

  ctStatus_t rc = ct_obv(price.data(), volume.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_obv failed";
  expect_approx_equal(out, ref);
}
