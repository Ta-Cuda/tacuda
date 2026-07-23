/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, MFI) {
  const int N = 30;
  const int p = 14;
  std::vector<float> high = {
      103.1797f, 103.1869f, 103.6822f, 106.0394f, 108.2978f, 106.3645f,
      108.2244f, 108.0788f, 107.2852f, 107.5736f, 107.9974f, 109.5241f,
      110.5949f, 110.5192f, 111.7877f, 110.7919f, 112.4462f, 112.1697f,
      113.2205f, 111.7666f, 109.5332f, 109.854f,  110.5902f, 109.7752f,
      112.8638f, 110.6323f, 110.7656f, 110.8366f, 113.0478f, 113.4313f};
  std::vector<float> low = {
      100.5071f, 102.0201f, 101.6783f, 104.6809f, 105.7863f, 105.3669f,
      106.1153f, 107.0141f, 106.5454f, 107.1999f, 107.0801f, 108.8005f,
      109.2626f, 109.2399f, 110.2089f, 109.6001f, 111.283f,  111.5296f,
      111.4559f, 111.2458f, 107.9698f, 108.0934f, 109.8739f, 108.6085f,
      111.6816f, 109.3505f, 110.0366f, 110.0088f, 110.9366f, 113.2555f};
  std::vector<float> close = {
      101.1815f, 102.6146f, 103.3758f, 104.6157f, 107.9955f, 107.2221f,
      107.8136f, 106.9829f, 106.4343f, 107.9075f, 107.3227f, 109.5898f,
      109.8437f, 110.3496f, 110.4833f, 110.9921f, 112.1381f, 112.8207f,
      112.3042f, 111.5877f, 109.7753f, 108.8134f, 109.7165f, 110.0943f,
      111.2928f, 111.3968f, 110.2639f, 109.9098f, 112.7778f, 114.026f};
  std::vector<float> volume = {
      835.f, 986.f, 912.f, 316.f, 481.f, 124.f, 167.f, 871.f, 334.f, 816.f,
      391.f, 826.f, 801.f, 809.f, 827.f, 655.f, 132.f, 111.f, 716.f, 312.f,
      238.f, 794.f, 521.f, 737.f, 768.f, 723.f, 588.f, 870.f, 639.f, 317.f};
  std::vector<float> out(N, 0.0f);

  ctStatus_t rc = ct_mfi(high.data(), low.data(), close.data(), volume.data(),
                         out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_mfi failed";

  std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
  float expected[] = {80.1529f, 71.4794f, 68.7325f, 67.9212f,
                      69.0350f, 67.2809f, 64.5633f, 65.0960f,
                      70.0452f, 60.4856f, 67.0099f, 57.7525f,
                      49.2357f, 38.7765f, 37.3642f, 43.3576f};
  for (int i = 0; i < 16; ++i) {
    ref[p + i] = expected[i];
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < p; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
