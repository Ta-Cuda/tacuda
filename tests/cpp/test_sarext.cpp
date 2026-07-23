/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <random>
#include <tacuda.h>
#include "test_utils.hpp"

namespace {

// Deterministic OHLC-like random walk for regression testing.
void make_walk(int n, std::vector<float>& high, std::vector<float>& low,
               unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> stepDist(0.0f, 1.0f);
  std::uniform_real_distribution<float> rangeDist(0.1f, 1.5f);
  high.resize(n);
  low.resize(n);
  float price = 100.0f;
  for (int i = 0; i < n; ++i) {
    price += stepDist(gen);
    float halfRange = rangeDist(gen) * 0.5f;
    high[i] = price + halfRange;
    low[i] = price - halfRange;
  }
}

void run_sarext_case(const std::vector<float>& high,
                     const std::vector<float>& low, float startValue,
                     float offset, float accInitLong, float accLong,
                     float accMaxLong, float accInitShort, float accShort,
                     float accMaxShort, float eps = 1e-2f) {
  const int n = static_cast<int>(high.size());
  std::vector<float> out(n, 0.0f);
  ctStatus_t rc = ct_sarext(high.data(), low.data(), out.data(), n, startValue,
                            offset, accInitLong, accLong, accMaxLong,
                            accInitShort, accShort, accMaxShort);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sarext failed";
  auto ref = sarext_talib_ref(high, low, startValue, offset, accInitLong,
                              accLong, accMaxLong, accInitShort, accShort,
                              accMaxShort);
  expect_approx_equal(out, ref, eps);
}

} // namespace

TEST(Tacuda, SAREXTTrending) {
  std::vector<float> high = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
  run_sarext_case(high, low, 0.0f, 0.0f, 0.02f, 0.02f, 0.2f, 0.02f, 0.02f, 0.2f,
                  1e-3f);
}

TEST(Tacuda, SAREXTRanging) {
  std::vector<float> high = {5.f, 6.f, 5.5f, 6.2f, 5.8f, 6.4f, 5.9f, 6.5f};
  std::vector<float> low = {4.f, 5.f, 4.5f, 5.2f, 4.8f, 5.4f, 4.9f, 5.5f};
  run_sarext_case(high, low, 0.0f, 0.0f, 0.02f, 0.02f, 0.2f, 0.02f, 0.02f, 0.2f,
                  1e-3f);
}

// Downtrend start: TA-Lib must pick the short direction from the first bars.
TEST(Tacuda, SAREXTInitialShortDirection) {
  std::vector<float> high = {8.f, 7.f, 6.5f, 6.f, 5.2f, 4.8f, 4.f, 3.5f};
  std::vector<float> low = {7.5f, 6.5f, 6.f, 5.5f, 4.7f, 4.3f, 3.5f, 3.f};
  run_sarext_case(high, low, 0.0f, 0.0f, 0.02f, 0.02f, 0.2f, 0.02f, 0.02f, 0.2f,
                  1e-3f);
}

TEST(Tacuda, SAREXTStartValueLong) {
  std::vector<float> high, low;
  make_walk(128, high, low, 7);
  run_sarext_case(high, low, 95.0f, 0.0f, 0.02f, 0.02f, 0.2f, 0.02f, 0.02f,
                  0.2f);
}

TEST(Tacuda, SAREXTStartValueShort) {
  std::vector<float> high, low;
  make_walk(128, high, low, 8);
  run_sarext_case(high, low, -105.0f, 0.0f, 0.02f, 0.02f, 0.2f, 0.02f, 0.02f,
                  0.2f);
}

TEST(Tacuda, SAREXTOffsetOnReverse) {
  std::vector<float> high, low;
  make_walk(128, high, low, 9);
  run_sarext_case(high, low, 0.0f, 0.01f, 0.02f, 0.02f, 0.2f, 0.02f, 0.02f,
                  0.2f);
}

TEST(Tacuda, SAREXTAsymmetricAcceleration) {
  std::vector<float> high, low;
  make_walk(256, high, low, 10);
  run_sarext_case(high, low, 0.0f, 0.0f, 0.02f, 0.03f, 0.25f, 0.01f, 0.02f,
                  0.15f);
}

TEST(Tacuda, SAREXTLongRandomWalk) {
  std::vector<float> high, low;
  make_walk(512, high, low, 11);
  run_sarext_case(high, low, 0.0f, 0.0f, 0.02f, 0.02f, 0.2f, 0.02f, 0.02f,
                  0.2f);
}
