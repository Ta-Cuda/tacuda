/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <limits>

#include <tacuda.h>
#include <gtest/gtest.h>
#include <vector>

TEST(DeviceBufferPool, ReusesBuffersInIndicators) {
  const int N = 128;
  std::vector<float> high(N, 2.0f), low(N, 1.0f), out(N, 0.0f);
  ct_device_buffer_pool_cleanup();

  ctStatus_t rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_first = ct_device_buffer_pool_allocation_count();
  EXPECT_GT(after_first, 0u);

  rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_second = ct_device_buffer_pool_allocation_count();
  EXPECT_EQ(after_first, after_second);

  ct_device_buffer_pool_cleanup();
}

TEST(DeviceBufferPool, ReusesBuffersInMultiInputWrappers) {
  const int N = 64;
  std::vector<float> high(N, 2.0f), low(N, 1.0f), out(N, 0.0f);
  ct_device_buffer_pool_cleanup();

  ctStatus_t rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_first = ct_device_buffer_pool_allocation_count();
  EXPECT_GT(after_first, 0u);

  rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  size_t after_second = ct_device_buffer_pool_allocation_count();
  EXPECT_EQ(after_first, after_second);

  ct_device_buffer_pool_cleanup();
}

TEST(DeviceBufferPool, RespectsCacheLimit) {
  ct_device_buffer_pool_cleanup();

  // Allocate a buffer using PlusDM (internally uses pool).
  const int N = 128;
  std::vector<float> high(N, 2.0f), low(N, 1.0f), out(N, 0.0f);
  ctStatus_t rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  EXPECT_GT(ct_device_buffer_pool_allocation_count(), 0u);
  size_t first_count = ct_device_buffer_pool_allocation_count();

  // Set cache to 0 - forces new allocation
  ct_device_buffer_pool_set_max_cache_bytes(0);
  rc = ct_plus_dm(high.data(), low.data(), out.data(), N, 5);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  EXPECT_GT(ct_device_buffer_pool_allocation_count(), first_count);

  ct_device_buffer_pool_cleanup();
  ct_device_buffer_pool_set_max_cache_bytes(std::numeric_limits<size_t>::max());
}
