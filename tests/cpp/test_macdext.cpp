/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <random>
#include <tacuda.h>
#include "test_utils.hpp"

namespace {

std::vector<float> make_walk(int n, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> stepDist(0.0f, 1.0f);
  std::vector<float> v(n);
  float price = 100.0f;
  for (int i = 0; i < n; ++i) {
    price += stepDist(gen);
    v[i] = price;
  }
  return v;
}

void run_macdext_case(const std::vector<float>& in, int fastP, ctMaType_t fastT,
                      int slowP, ctMaType_t slowT, int sigP, ctMaType_t sigT,
                      float eps = 1e-2f) {
  const int n = static_cast<int>(in.size());
  std::vector<float> macd(n, 0.0f), signal(n, 0.0f), hist(n, 0.0f);
  ctStatus_t rc = ct_macdext(in.data(), macd.data(), signal.data(), hist.data(),
                             n, fastP, fastT, slowP, slowT, sigP, sigT);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_macdext failed";
  auto [refMacd, refSignal, refHist] = macdext_talib_ref(
      in, fastP, static_cast<int>(fastT), slowP, static_cast<int>(slowT), sigP,
      static_cast<int>(sigT));
  expect_approx_equal(macd, refMacd, eps);
  expect_approx_equal(signal, refSignal, eps);
  expect_approx_equal(hist, refHist, eps);
}

} // namespace

TEST(TacudaMacdExt, AllSma) {
  auto x = make_walk(256, 1);
  run_macdext_case(x, 12, CT_MA_SMA, 26, CT_MA_SMA, 9, CT_MA_SMA);
}

TEST(TacudaMacdExt, AllWma) {
  auto x = make_walk(256, 2);
  run_macdext_case(x, 12, CT_MA_WMA, 26, CT_MA_WMA, 9, CT_MA_WMA);
}

TEST(TacudaMacdExt, AllEma) {
  auto x = make_walk(256, 3);
  run_macdext_case(x, 12, CT_MA_EMA, 26, CT_MA_EMA, 9, CT_MA_EMA);
}

// Window-based fast/slow lines with an EMA signal: the signal EMA is seeded
// on the first valid MACD values exactly like TA-Lib.
TEST(TacudaMacdExt, MixedSmaWmaEmaSignal) {
  auto x = make_walk(256, 4);
  run_macdext_case(x, 12, CT_MA_SMA, 26, CT_MA_WMA, 9, CT_MA_EMA);
}

TEST(TacudaMacdExt, DifferentPeriodsSma) {
  auto x = make_walk(512, 5);
  run_macdext_case(x, 5, CT_MA_SMA, 35, CT_MA_SMA, 5, CT_MA_SMA);
}

// TA-Lib swaps fast/slow (period AND type) when fastPeriod > slowPeriod.
TEST(TacudaMacdExt, FastSlowSwap) {
  auto x = make_walk(256, 6);
  run_macdext_case(x, 26, CT_MA_SMA, 12, CT_MA_WMA, 9, CT_MA_SMA);
}

TEST(TacudaMacdExt, InvalidArguments) {
  std::vector<float> x(64, 1.0f);
  std::vector<float> m(64), s(64), h(64);
  EXPECT_EQ(ct_macdext(nullptr, m.data(), s.data(), h.data(), 64, 12, CT_MA_SMA,
                       26, CT_MA_SMA, 9, CT_MA_SMA),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_macdext(x.data(), m.data(), s.data(), h.data(), 0, 12, CT_MA_SMA,
                       26, CT_MA_SMA, 9, CT_MA_SMA),
            CT_STATUS_INVALID_ARGUMENT);
}
