/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThreeWhiteSoldiers) {
  const int N = 5;
  std::vector<float> open{10.0f, 10.6f, 11.2f, 11.5f, 11.6f};
  std::vector<float> high{10.8f, 11.4f, 12.1f, 11.6f, 11.7f};
  std::vector<float> low{9.9f, 10.5f, 11.1f, 11.4f, 11.5f};
  std::vector<float> close{10.7f, 11.3f, 12.0f, 11.55f, 11.55f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_three_white_soldiers(open.data(), high.data(), low.data(),
                                              close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_three_white_soldiers failed";
  auto ref = cdl_ref(open, high, low, close, "CDL3WHITESOLDIERS");
  expect_approx_equal(out, ref);
}

