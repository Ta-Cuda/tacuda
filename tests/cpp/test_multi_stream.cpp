/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * @file test_multi_stream.cpp
 * @brief Tests for concurrent CUDA stream execution.
 */

#include <tacuda.h>
#include <tacuda_device.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

class MultiStreamTest : public ::testing::Test {
protected:
    static constexpr int N = 4096;
    std::vector<float> hostInput;
    float* d_input = nullptr;
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;

    void SetUp() override {
        hostInput.resize(N);
        for (int i = 0; i < N; ++i) {
            hostInput[i] = 100.0f + 2.0f * std::sin(0.03f * i) + 0.05f * i;
        }
        d_input = ct_device_alloc(N);
        ASSERT_NE(d_input, nullptr);
        ASSERT_EQ(ct_device_upload(d_input, hostInput.data(), N, 0), CT_STATUS_SUCCESS);
        cudaDeviceSynchronize();

        ASSERT_EQ(cudaStreamCreate(&stream1), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&stream2), cudaSuccess);
    }

    void TearDown() override {
        if (stream1) cudaStreamDestroy(stream1);
        if (stream2) cudaStreamDestroy(stream2);
        if (d_input) ct_device_free(d_input);
    }
};

TEST_F(MultiStreamTest, TwoIndicatorsConcurrently) {
    // Run SMA on stream1, EMA on stream2 concurrently
    float* d_sma = ct_device_alloc(N);
    float* d_ema = ct_device_alloc(N);
    ASSERT_NE(d_sma, nullptr);
    ASSERT_NE(d_ema, nullptr);

    ASSERT_EQ(ct_sma_d(d_input, d_sma, N, 20, stream1), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_ema_d(d_input, d_ema, N, 12, stream2), CT_STATUS_SUCCESS);

    // Synchronize each stream independently
    ASSERT_EQ(cudaStreamSynchronize(stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream2), cudaSuccess);

    // Download and verify against host API
    std::vector<float> resultSma(N), resultEma(N);
    ASSERT_EQ(ct_device_download(resultSma.data(), d_sma, N, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_device_download(resultEma.data(), d_ema, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> refSma(N), refEma(N);
    ASSERT_EQ(ct_sma(hostInput.data(), refSma.data(), N, 20), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_ema(hostInput.data(), refEma.data(), N, 12), CT_STATUS_SUCCESS);

    for (int i = 19; i < N; ++i) {
        EXPECT_NEAR(resultSma[i], refSma[i], 1e-3f)
            << "SMA stream mismatch at " << i;
    }
    for (int i = 11; i < N; ++i) {
        EXPECT_NEAR(resultEma[i], refEma[i], 1e-3f)
            << "EMA stream mismatch at " << i;
    }

    ct_device_free(d_sma);
    ct_device_free(d_ema);
}

TEST_F(MultiStreamTest, RSI_TwoStreamsSameData) {
    // Compute RSI on two different streams, verify identical results
    float* d_rsi1 = ct_device_alloc(N);
    float* d_rsi2 = ct_device_alloc(N);
    ASSERT_NE(d_rsi1, nullptr);
    ASSERT_NE(d_rsi2, nullptr);

    ASSERT_EQ(ct_rsi_d(d_input, d_rsi1, N, 14, stream1), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_rsi_d(d_input, d_rsi2, N, 14, stream2), CT_STATUS_SUCCESS);

    ASSERT_EQ(cudaStreamSynchronize(stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream2), cudaSuccess);

    std::vector<float> result1(N), result2(N);
    ASSERT_EQ(ct_device_download(result1.data(), d_rsi1, N, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_device_download(result2.data(), d_rsi2, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    for (int i = 14; i < N; ++i) {
        if (!std::isnan(result1[i]) && !std::isnan(result2[i])) {
            EXPECT_NEAR(result1[i], result2[i], 1e-5f)
                << "RSI results on different streams should match at " << i;
        }
    }

    ct_device_free(d_rsi1);
    ct_device_free(d_rsi2);
}

TEST_F(MultiStreamTest, DifferentIndicatorsOnStreams) {
    // Run RSI on stream1, SMA on stream1, EMA on stream2 — mixed usage
    float* d_rsi = ct_device_alloc(N);
    float* d_sma = ct_device_alloc(N);
    float* d_ema = ct_device_alloc(N);

    ASSERT_EQ(ct_rsi_d(d_input, d_rsi, N, 14, stream1), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_sma_d(d_input, d_sma, N, 20, stream1), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_ema_d(d_input, d_ema, N, 26, stream2), CT_STATUS_SUCCESS);

    ASSERT_EQ(cudaStreamSynchronize(stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream2), cudaSuccess);

    // Just verify no crashes and some valid outputs
    std::vector<float> resultRsi(N), resultSma(N), resultEma(N);
    ASSERT_EQ(ct_device_download(resultRsi.data(), d_rsi, N, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_device_download(resultSma.data(), d_sma, N, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_device_download(resultEma.data(), d_ema, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    // At least some valid values
    int validRsi = 0, validSma = 0, validEma = 0;
    for (int i = 0; i < N; ++i) {
        if (!std::isnan(resultRsi[i])) ++validRsi;
        if (!std::isnan(resultSma[i])) ++validSma;
        if (!std::isnan(resultEma[i])) ++validEma;
    }
    EXPECT_GT(validRsi, N / 2);
    EXPECT_GT(validSma, N / 2);
    EXPECT_GT(validEma, N / 2);

    ct_device_free(d_rsi);
    ct_device_free(d_sma);
    ct_device_free(d_ema);
}
