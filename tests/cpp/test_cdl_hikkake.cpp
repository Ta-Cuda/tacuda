/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLHikkake) {
  const int N = 5;
  std::vector<float> open{10.5f, 10.6f, 10.5f, 10.7f, 10.8f};
  std::vector<float> high{11.0f, 10.8f, 11.2f, 10.9f, 11.0f};
  std::vector<float> low{10.0f, 10.2f, 10.4f, 10.5f, 10.6f};
  std::vector<float> close{10.6f, 10.4f, 11.0f, 10.8f, 10.9f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_hikkake(open.data(), high.data(), low.data(),
                                 close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_hikkake failed";
  auto ref = cdl_ref(open, high, low, close, "CDLHIKKAKE");
  expect_approx_equal(out, ref);
}

