/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * @file test_edge_cases.cpp
 * @brief Edge case tests for indicators: boundary sizes, special values.
 */

#include <tacuda.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

namespace {
// Helper: generate constant-value series
std::vector<float> constant_series(int n, float val) {
    return std::vector<float>(n, val);
}

// Helper: check that valid output elements are not NaN
int count_valid(const std::vector<float>& v) {
    int c = 0;
    for (auto x : v) if (!std::isnan(x)) ++c;
    return c;
}
} // namespace

// ============================================================================
// SMA Edge Cases
// ============================================================================

TEST(EdgeCases, SMA_PeriodEqualsSize) {
    const int N = 20;
    const int period = 20;
    std::vector<float> input(N);
    for (int i = 0; i < N; ++i) input[i] = static_cast<float>(i + 1);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_sma(input.data(), output.data(), N, period), CT_STATUS_SUCCESS);

    // Only the last value should be valid: average of 1..20 = 10.5
    EXPECT_NEAR(output[N - 1], 10.5f, 1e-3f);
    // All before should be NaN
    for (int i = 0; i < N - 1; ++i) {
        EXPECT_TRUE(std::isnan(output[i])) << "Expected NaN at index " << i;
    }
}

TEST(EdgeCases, SMA_Period1_IsPassthrough) {
    const int N = 50;
    std::vector<float> input(N);
    for (int i = 0; i < N; ++i) input[i] = static_cast<float>(i);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_sma(input.data(), output.data(), N, 1), CT_STATUS_SUCCESS);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]) << "SMA(1) should pass through at index " << i;
    }
}

TEST(EdgeCases, SMA_AllSameValues) {
    const int N = 100;
    const int period = 14;
    auto input = constant_series(N, 42.0f);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_sma(input.data(), output.data(), N, period), CT_STATUS_SUCCESS);

    for (int i = period - 1; i < N; ++i) {
        EXPECT_NEAR(output[i], 42.0f, 1e-4f) << "SMA of constant should equal constant at " << i;
    }
}

// ============================================================================
// EMA Edge Cases
// ============================================================================

TEST(EdgeCases, EMA_Period1_IsPassthrough) {
    const int N = 50;
    std::vector<float> input(N);
    for (int i = 0; i < N; ++i) input[i] = static_cast<float>(i * 0.5f);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_ema(input.data(), output.data(), N, 1), CT_STATUS_SUCCESS);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]) << "EMA(1) should pass through at index " << i;
    }
}

TEST(EdgeCases, EMA_AllSameValues) {
    const int N = 100;
    const int period = 12;
    auto input = constant_series(N, 50.0f);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_ema(input.data(), output.data(), N, period), CT_STATUS_SUCCESS);

    for (int i = period - 1; i < N; ++i) {
        EXPECT_NEAR(output[i], 50.0f, 1e-3f) << "EMA of constant should converge at " << i;
    }
}

TEST(EdgeCases, EMA_PeriodEqualsSize) {
    const int N = 14;
    std::vector<float> input(N);
    for (int i = 0; i < N; ++i) input[i] = static_cast<float>(i + 1);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_ema(input.data(), output.data(), N, N), CT_STATUS_SUCCESS);

    // First valid value at index N-1 = SMA of all values = (1+14)/2 = 7.5
    EXPECT_NEAR(output[N - 1], 7.5f, 1e-3f);
}

// ============================================================================
// RSI Edge Cases
// ============================================================================

TEST(EdgeCases, RSI_AllSameValues) {
    const int N = 100;
    const int period = 14;
    auto input = constant_series(N, 100.0f);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_rsi(input.data(), output.data(), N, period), CT_STATUS_SUCCESS);

    // When all values are the same, no gains or losses -> RSI is typically 0 (or undefined)
    for (int i = period; i < N; ++i) {
        if (!std::isnan(output[i])) {
            EXPECT_GE(output[i], 0.0f);
            EXPECT_LE(output[i], 100.0f);
        }
    }
}

TEST(EdgeCases, RSI_AllZeroValues) {
    const int N = 100;
    const int period = 14;
    auto input = constant_series(N, 0.0f);

    std::vector<float> output(N, -1.0f);
    ASSERT_EQ(ct_rsi(input.data(), output.data(), N, period), CT_STATUS_SUCCESS);

    // With zero changes, RSI should be 0 or NaN
    for (int i = period; i < N; ++i) {
        if (!std::isnan(output[i])) {
            EXPECT_GE(output[i], 0.0f);
            EXPECT_LE(output[i], 100.0f);
        }
    }
}

// ============================================================================
// ATR Edge Cases
// ============================================================================

TEST(EdgeCases, ATR_AllSameValues) {
    const int N = 100;
    const int period = 14;
    auto high = constant_series(N, 105.0f);
    auto low = constant_series(N, 95.0f);
    auto close = constant_series(N, 100.0f);

    std::vector<float> output(N, 0.0f);
    ASSERT_EQ(ct_atr(high.data(), low.data(), close.data(), output.data(), N, period, 0.0f),
              CT_STATUS_SUCCESS);

    // ATR of constant H-L range = 10.0
    for (int i = period; i < N; ++i) {
        if (!std::isnan(output[i])) {
            EXPECT_NEAR(output[i], 10.0f, 0.5f) << "ATR should be ~10 at " << i;
        }
    }
}

// ============================================================================
// BBands Edge Cases
// ============================================================================

TEST(EdgeCases, BBands_AllSameValues) {
    const int N = 100;
    const int period = 20;
    auto input = constant_series(N, 50.0f);

    std::vector<float> upper(N, 0.0f), middle(N, 0.0f), lower(N, 0.0f);
    ASSERT_EQ(ct_bbands(input.data(), upper.data(), middle.data(), lower.data(),
                        N, period, 2.0f, 2.0f),
              CT_STATUS_SUCCESS);

    for (int i = period - 1; i < N; ++i) {
        if (!std::isnan(middle[i])) {
            EXPECT_NEAR(middle[i], 50.0f, 1e-3f);
            // StdDev = 0, so upper == middle == lower
            EXPECT_NEAR(upper[i], 50.0f, 1e-3f);
            EXPECT_NEAR(lower[i], 50.0f, 1e-3f);
        }
    }
}

// ============================================================================
// MACD Edge Cases
// ============================================================================

TEST(EdgeCases, MACD_AllSameValues) {
    const int N = 100;
    auto input = constant_series(N, 100.0f);

    std::vector<float> macd_out(N, 0.0f), signal_out(N, 0.0f), hist_out(N, 0.0f);
    ASSERT_EQ(ct_macd(input.data(), macd_out.data(), signal_out.data(), hist_out.data(),
                      N, 12, 26, 9, CT_MA_EMA),
              CT_STATUS_SUCCESS);

    // MACD of constant = 0 (fast EMA - slow EMA = 0)
    for (int i = 25; i < N; ++i) {
        if (!std::isnan(macd_out[i])) {
            EXPECT_NEAR(macd_out[i], 0.0f, 1e-3f) << "MACD of constant should be 0 at " << i;
        }
    }
}
