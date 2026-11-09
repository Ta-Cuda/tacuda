/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Failure contract of GPU batch streams, driven through injected faults.
//
// A failure before the kernel leaves the state untouched, so the same append
// can be retried. From the kernel onwards the device may have advanced past
// what the caller received: the handle then refuses work until reset instead
// of returning success on stale state.

#ifdef TACUDA_STREAM_FAULT_INJECTION

#include <tacuda_stream.h>
#include <utils/StreamBatchFaults.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

constexpr int kSeries = 4;
constexpr int kBars = 12;
constexpr int kPeriod = 5;

class Batch {
public:
  Batch() { EXPECT_EQ(ct_sma_stream_batch_create(kPeriod, kSeries, &handle_), CT_STATUS_SUCCESS); }
  ~Batch() { ct_stream_batch_destroy(handle_); }
  ctStreamBatch_t *get() { return handle_; }

  ctStatus_t exec(ctStreamExec_t mode) { return ct_stream_batch_set_exec(handle_, mode); }
  ctStatus_t append(const std::vector<float> &in, int bars, std::vector<float> &out) {
    out.assign(static_cast<std::size_t>(kSeries) * bars, -1.0f);
    return ct_sma_stream_batch_append(handle_, in.data(), bars, out.data());
  }
  long long bars() {
    long long value = -1;
    EXPECT_EQ(ct_stream_batch_bars(handle_, &value), CT_STATUS_SUCCESS);
    return value;
  }

private:
  ctStreamBatch_t *handle_ = nullptr;
};

std::vector<float> matrix(int bars) {
  std::vector<float> out(static_cast<std::size_t>(kSeries) * bars);
  for (int s = 0; s < kSeries; ++s) {
    for (int b = 0; b < bars; ++b) {
      out[static_cast<std::size_t>(s) * bars + b] = 100.0f + static_cast<float>(s) * 10.0f + static_cast<float>(b);
    }
  }
  return out;
}

void expect_same(const std::vector<float> &a, const std::vector<float> &b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(std::isnan(a[i]), std::isnan(b[i])) << i;
    if (!std::isnan(a[i])) ASSERT_EQ(a[i], b[i]) << i;
  }
}

std::vector<float> uninterrupted(int bars) {
  Batch b;
  EXPECT_EQ(b.exec(CT_STREAM_EXEC_GPU), CT_STATUS_SUCCESS);
  std::vector<float> out;
  EXPECT_EQ(b.append(matrix(bars), bars, out), CT_STATUS_SUCCESS);
  return out;
}

}  // namespace

TEST(StreamBatchFaults, FailuresBeforeTheKernelCanBeRetried) {
  const ctStreamBatchFault_t sites[] = {CT_STREAM_BATCH_FAULT_STATE_UPLOAD, CT_STREAM_BATCH_FAULT_IO_ALLOC,
                                        CT_STREAM_BATCH_FAULT_INPUT_UPLOAD};
  const std::vector<float> in = matrix(kBars);
  const std::vector<float> expected = uninterrupted(kBars);
  for (ctStreamBatchFault_t site : sites) {
    Batch b;
    ASSERT_EQ(b.exec(CT_STREAM_EXEC_GPU), CT_STATUS_SUCCESS);
    std::vector<float> out;
    ct_test_stream_batch_fail_next(site);
    EXPECT_NE(b.append(in, kBars, out), CT_STATUS_SUCCESS) << "site " << site;
    EXPECT_EQ(b.bars(), 0) << "site " << site;
    ASSERT_EQ(b.append(in, kBars, out), CT_STATUS_SUCCESS) << "site " << site;
    EXPECT_EQ(b.bars(), kBars);
    expect_same(out, expected);
  }
}

TEST(StreamBatchFaults, FailuresAfterTheKernelRefuseFurtherWork) {
  const ctStreamBatchFault_t sites[] = {CT_STREAM_BATCH_FAULT_KERNEL, CT_STREAM_BATCH_FAULT_OUTPUT_DOWNLOAD};
  const std::vector<float> in = matrix(kBars);
  const std::vector<float> expected = uninterrupted(kBars);
  for (ctStreamBatchFault_t site : sites) {
    Batch b;
    ASSERT_EQ(b.exec(CT_STREAM_EXEC_GPU), CT_STATUS_SUCCESS);
    std::vector<float> out;
    ASSERT_EQ(b.append(in, kBars, out), CT_STATUS_SUCCESS);

    ct_test_stream_batch_fail_next(site);
    EXPECT_NE(b.append(in, kBars, out), CT_STATUS_SUCCESS) << "site " << site;
    EXPECT_EQ(b.bars(), kBars) << "a failed append must not count";

    EXPECT_EQ(b.append(in, kBars, out), CT_STATUS_DEVICE_ERROR) << "site " << site;
    ASSERT_EQ(b.exec(CT_STREAM_EXEC_CPU), CT_STATUS_SUCCESS);
    EXPECT_EQ(b.append(in, kBars, out), CT_STATUS_DEVICE_ERROR) << "site " << site;
    EXPECT_EQ(b.bars(), kBars);

    ASSERT_EQ(ct_stream_batch_reset(b.get()), CT_STATUS_SUCCESS);
    ASSERT_EQ(b.exec(CT_STREAM_EXEC_GPU), CT_STATUS_SUCCESS);
    ASSERT_EQ(b.append(in, kBars, out), CT_STATUS_SUCCESS);
    expect_same(out, expected);
  }
}

TEST(StreamBatchFaults, FailedStateDownloadDoesNotContinueOnCpu) {
  const std::vector<float> in = matrix(kBars);
  Batch b;
  ASSERT_EQ(b.exec(CT_STREAM_EXEC_GPU), CT_STATUS_SUCCESS);
  std::vector<float> out;
  ASSERT_EQ(b.append(in, kBars, out), CT_STATUS_SUCCESS);

  ASSERT_EQ(b.exec(CT_STREAM_EXEC_CPU), CT_STATUS_SUCCESS);
  ct_test_stream_batch_fail_next(CT_STREAM_BATCH_FAULT_STATE_DOWNLOAD);
  EXPECT_EQ(b.append(in, kBars, out), CT_STATUS_COPY_FAILED);
  EXPECT_EQ(b.append(in, kBars, out), CT_STATUS_DEVICE_ERROR);
  EXPECT_EQ(b.bars(), kBars);

  ASSERT_EQ(ct_stream_batch_reset(b.get()), CT_STATUS_SUCCESS);
  ASSERT_EQ(b.append(in, kBars, out), CT_STATUS_SUCCESS);
  EXPECT_EQ(b.bars(), kBars);
  expect_same(out, uninterrupted(kBars));
}

#endif  // TACUDA_STREAM_FAULT_INJECTION
