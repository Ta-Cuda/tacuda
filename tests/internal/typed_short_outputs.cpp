/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// The templated C++ indicators are not exported from the shared library, so
// this test links the library's objects directly instead of living in
// test_basic. A short input must initialize every requested element in the
// caller's type: filling sizeof(float) per element leaves half of a double
// array untouched and tears the last element of an odd-length one.

#include <indicators/EMA.h>
#include <indicators/RSI.h>
#include <indicators/SMA.h>
#include <tacuda.h>
#include <tacuda_device.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

template <typename Indicator>
void expect_all_nan(int size, int period) {
  const std::vector<double> sentinel(static_cast<std::size_t>(size), 12345.0);
  void *dIn = ct_device_alloc_ex(size, CT_DTYPE_F64);
  void *dOut = ct_device_alloc_ex(size, CT_DTYPE_F64);
  ASSERT_TRUE(dIn && dOut);
  ASSERT_EQ(ct_device_upload_ex(dIn, sentinel.data(), size, CT_DTYPE_F64, nullptr), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_device_upload_ex(dOut, sentinel.data(), size, CT_DTYPE_F64, nullptr), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_device_sync(nullptr), CT_STATUS_SUCCESS);

  Indicator(period).calculate(static_cast<const double *>(dIn), static_cast<double *>(dOut), size, nullptr);
  ASSERT_EQ(ct_device_sync(nullptr), CT_STATUS_SUCCESS);

  std::vector<double> out(static_cast<std::size_t>(size));
  ASSERT_EQ(ct_device_download_ex(out.data(), dOut, size, CT_DTYPE_F64, nullptr), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_device_sync(nullptr), CT_STATUS_SUCCESS);
  ct_device_free_ex(dIn);
  ct_device_free_ex(dOut);

  for (int i = 0; i < size; ++i) {
    EXPECT_TRUE(std::isnan(out[static_cast<std::size_t>(i)])) << "size " << size << " at " << i;
  }
}

}  // namespace

TEST(TypedShortOutputs, EveryDoubleElementIsInitialized) {
  for (int size : {1, 2, 3, 4}) {
    expect_all_nan<tacuda::SMAT<double>>(size, 5);
    expect_all_nan<tacuda::EMAT<double>>(size, 5);
    expect_all_nan<tacuda::RSIT<double>>(size, 5);
  }
}
