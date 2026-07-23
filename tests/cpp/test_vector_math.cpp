/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <functional>
#include <random>
#include <tacuda.h>
#include "test_utils.hpp"

namespace {

std::vector<float> make_uniform(int n, float lo, float hi, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

// TA-Lib computes these in double; mirror that for the reference values.
void run_unary_case(ctStatus_t (*fn)(const float*, float*, int, cudaStream_t),
                    double (*ref)(double), const std::vector<float>& in,
                    float eps = 1e-4f) {
  const int n = static_cast<int>(in.size());
  std::vector<float> out(n, 0.0f);
  ASSERT_EQ(fn(in.data(), out.data(), n, 0), CT_STATUS_SUCCESS);
  std::vector<float> expected(n);
  for (int i = 0; i < n; ++i) {
    expected[i] = static_cast<float>(ref(static_cast<double>(in[i])));
  }
  expect_approx_equal(out, expected, eps);
}

} // namespace

TEST(TacudaVectorMath, Trigonometric) {
  auto x = make_uniform(4096, -0.99f, 0.99f, 1);
  run_unary_case(&ct_acos, [](double v) { return std::acos(v); }, x);
  run_unary_case(&ct_asin, [](double v) { return std::asin(v); }, x);

  auto wide = make_uniform(4096, -50.f, 50.f, 2);
  run_unary_case(&ct_atan, [](double v) { return std::atan(v); }, wide);
  run_unary_case(&ct_sin, [](double v) { return std::sin(v); }, wide, 1e-3f);
  run_unary_case(&ct_cos, [](double v) { return std::cos(v); }, wide, 1e-3f);
  run_unary_case(&ct_tanh, [](double v) { return std::tanh(v); }, wide);
}

TEST(TacudaVectorMath, TanNearZero) {
  auto x = make_uniform(4096, -1.5f, 1.5f, 3);
  run_unary_case(&ct_tan, [](double v) { return std::tan(v); }, x, 1e-3f);
}

TEST(TacudaVectorMath, HyperbolicAndExp) {
  auto x = make_uniform(4096, -5.f, 5.f, 4);
  run_unary_case(&ct_sinh, [](double v) { return std::sinh(v); }, x, 1e-2f);
  run_unary_case(&ct_cosh, [](double v) { return std::cosh(v); }, x, 1e-2f);
  run_unary_case(&ct_exp, [](double v) { return std::exp(v); }, x, 1e-2f);
}

TEST(TacudaVectorMath, LogarithmsAndSqrt) {
  auto x = make_uniform(4096, 0.01f, 1000.f, 5);
  run_unary_case(&ct_ln, [](double v) { return std::log(v); }, x);
  run_unary_case(&ct_log10, [](double v) { return std::log10(v); }, x);
  run_unary_case(&ct_sqrt, [](double v) { return std::sqrt(v); }, x, 1e-3f);
}

TEST(TacudaVectorMath, Rounding) {
  auto x = make_uniform(4096, -100.f, 100.f, 6);
  run_unary_case(&ct_ceil, [](double v) { return std::ceil(v); }, x, 0.0f);
  run_unary_case(&ct_floor, [](double v) { return std::floor(v); }, x, 0.0f);
}

TEST(TacudaVectorMath, NanPropagation) {
  std::vector<float> in = {0.5f, std::nanf(""), -0.25f, std::nanf("")};
  std::vector<float> out(in.size(), 0.0f);
  ASSERT_EQ(ct_sin(in.data(), out.data(), static_cast<int>(in.size()), 0),
            CT_STATUS_SUCCESS);
  EXPECT_FALSE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
  EXPECT_FALSE(std::isnan(out[2]));
  EXPECT_TRUE(std::isnan(out[3]));
}

TEST(TacudaVectorMath, BinaryOperators) {
  const int n = 4096;
  auto a = make_uniform(n, -100.f, 100.f, 7);
  auto b = make_uniform(n, 0.5f, 100.f, 8);

  std::vector<float> out(n, 0.0f);
  std::vector<float> expected(n);

  ASSERT_EQ(ct_add(a.data(), b.data(), out.data(), n, 0), CT_STATUS_SUCCESS);
  for (int i = 0; i < n; ++i) expected[i] = a[i] + b[i];
  expect_approx_equal(out, expected, 1e-4f);

  ASSERT_EQ(ct_sub(a.data(), b.data(), out.data(), n, 0), CT_STATUS_SUCCESS);
  for (int i = 0; i < n; ++i) expected[i] = a[i] - b[i];
  expect_approx_equal(out, expected, 1e-4f);

  ASSERT_EQ(ct_mult(a.data(), b.data(), out.data(), n, 0), CT_STATUS_SUCCESS);
  for (int i = 0; i < n; ++i) expected[i] = a[i] * b[i];
  expect_approx_equal(out, expected, 1e-1f);

  ASSERT_EQ(ct_div(a.data(), b.data(), out.data(), n, 0), CT_STATUS_SUCCESS);
  for (int i = 0; i < n; ++i) expected[i] = a[i] / b[i];
  expect_approx_equal(out, expected, 1e-3f);
}

TEST(TacudaVectorMath, DivByZeroGivesInf) {
  std::vector<float> a = {1.0f, -2.0f, 0.0f};
  std::vector<float> b = {0.0f, 0.0f, 1.0f};
  std::vector<float> out(3, 0.0f);
  ASSERT_EQ(ct_div(a.data(), b.data(), out.data(), 3, 0), CT_STATUS_SUCCESS);
  EXPECT_TRUE(std::isinf(out[0]) && out[0] > 0);
  EXPECT_TRUE(std::isinf(out[1]) && out[1] < 0);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
}

TEST(TacudaVectorMath, InvalidArguments) {
  std::vector<float> x = {1.f, 2.f};
  std::vector<float> out(2, 0.0f);
  EXPECT_EQ(ct_sin(nullptr, out.data(), 2, 0), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sin(x.data(), nullptr, 2, 0), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sin(x.data(), out.data(), 0, 0), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_add(x.data(), nullptr, out.data(), 2, 0),
            CT_STATUS_INVALID_ARGUMENT);
}
