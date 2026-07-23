/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * @file test_device_api.cpp
 * @brief Tests for the Device API (ct_*_d functions) that work with GPU pointers.
 */

#include <tacuda.h>
#include <tacuda_device.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

class DeviceApiTest : public ::testing::Test {
protected:
    static constexpr int N = 512;
    std::vector<float> hostInput;
    float* d_input = nullptr;
    float* d_output = nullptr;

    void SetUp() override {
        hostInput.resize(N);
        for (int i = 0; i < N; ++i) {
            hostInput[i] = 100.0f + 2.0f * std::sin(0.05f * i) + 0.1f * i;
        }
        d_input = ct_device_alloc(N);
        ASSERT_NE(d_input, nullptr);
        d_output = ct_device_alloc(N);
        ASSERT_NE(d_output, nullptr);
        ASSERT_EQ(ct_device_upload(d_input, hostInput.data(), N, 0), CT_STATUS_SUCCESS);
    }

    void TearDown() override {
        if (d_input) ct_device_free(d_input);
        if (d_output) ct_device_free(d_output);
    }
};

// ============================================================================
// Memory management tests
// ============================================================================

TEST(DeviceApiMemory, AllocAndFree) {
    float* p = ct_device_alloc(1024);
    ASSERT_NE(p, nullptr);
    ct_device_free(p);
}

TEST(DeviceApiMemory, FreeNull) {
    // Should not crash
    ct_device_free(nullptr);
}

TEST(DeviceApiMemory, UploadDownloadRoundtrip) {
    const int N = 256;
    std::vector<float> src(N);
    for (int i = 0; i < N; ++i) src[i] = static_cast<float>(i);

    float* d_buf = ct_device_alloc(N);
    ASSERT_NE(d_buf, nullptr);
    ASSERT_EQ(ct_device_upload(d_buf, src.data(), N, 0), CT_STATUS_SUCCESS);

    std::vector<float> dst(N, 0.0f);
    ASSERT_EQ(ct_device_download(dst.data(), d_buf, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(src[i], dst[i]) << "at index " << i;
    }
    ct_device_free(d_buf);
}

// ============================================================================
// Device API vs Host API comparison tests
// ============================================================================

TEST_F(DeviceApiTest, SMA_MatchesHostApi) {
    const int period = 20;
    // Device API
    ASSERT_EQ(ct_sma_d(d_input, d_output, N, period, 0), CT_STATUS_SUCCESS);
    std::vector<float> deviceResult(N);
    ASSERT_EQ(ct_device_download(deviceResult.data(), d_output, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    // Host API
    std::vector<float> hostResult(N);
    ASSERT_EQ(ct_sma(hostInput.data(), hostResult.data(), N, period), CT_STATUS_SUCCESS);

    for (int i = period - 1; i < N; ++i) {
        EXPECT_NEAR(deviceResult[i], hostResult[i], 1e-4f) << "SMA mismatch at index " << i;
    }
}

TEST_F(DeviceApiTest, EMA_MatchesHostApi) {
    const int period = 12;
    ASSERT_EQ(ct_ema_d(d_input, d_output, N, period, 0), CT_STATUS_SUCCESS);
    std::vector<float> deviceResult(N);
    ASSERT_EQ(ct_device_download(deviceResult.data(), d_output, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> hostResult(N);
    ASSERT_EQ(ct_ema(hostInput.data(), hostResult.data(), N, period), CT_STATUS_SUCCESS);

    for (int i = period - 1; i < N; ++i) {
        EXPECT_NEAR(deviceResult[i], hostResult[i], 1e-3f) << "EMA mismatch at index " << i;
    }
}

TEST_F(DeviceApiTest, RSI_MatchesHostApi) {
    const int period = 14;
    ASSERT_EQ(ct_rsi_d(d_input, d_output, N, period, 0), CT_STATUS_SUCCESS);
    std::vector<float> deviceResult(N);
    ASSERT_EQ(ct_device_download(deviceResult.data(), d_output, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> hostResult(N);
    ASSERT_EQ(ct_rsi(hostInput.data(), hostResult.data(), N, period), CT_STATUS_SUCCESS);

    for (int i = period; i < N; ++i) {
        if (!std::isnan(hostResult[i]) && !std::isnan(deviceResult[i])) {
            EXPECT_NEAR(deviceResult[i], hostResult[i], 1e-3f)
                << "RSI mismatch at index " << i;
        }
    }
}

// ============================================================================
// Pipeline test: multiple operations on same device data
// ============================================================================

TEST_F(DeviceApiTest, PipelineNoTransfers) {
    float* d_sma = ct_device_alloc(N);
    float* d_ema = ct_device_alloc(N);
    float* d_rsi = ct_device_alloc(N);
    ASSERT_NE(d_sma, nullptr);
    ASSERT_NE(d_ema, nullptr);
    ASSERT_NE(d_rsi, nullptr);

    // Upload once, compute three indicators
    ASSERT_EQ(ct_sma_d(d_input, d_sma, N, 20, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_ema_d(d_input, d_ema, N, 12, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_rsi_d(d_input, d_rsi, N, 14, 0), CT_STATUS_SUCCESS);

    // Verify by downloading and comparing with host API
    std::vector<float> resultSma(N), resultEma(N), resultRsi(N);
    ASSERT_EQ(ct_device_download(resultSma.data(), d_sma, N, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_device_download(resultEma.data(), d_ema, N, 0), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_device_download(resultRsi.data(), d_rsi, N, 0), CT_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> refSma(N), refEma(N), refRsi(N);
    ASSERT_EQ(ct_sma(hostInput.data(), refSma.data(), N, 20), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_ema(hostInput.data(), refEma.data(), N, 12), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_rsi(hostInput.data(), refRsi.data(), N, 14), CT_STATUS_SUCCESS);

    for (int i = 19; i < N; ++i)
        EXPECT_NEAR(resultSma[i], refSma[i], 1e-3f);
    for (int i = 11; i < N; ++i)
        EXPECT_NEAR(resultEma[i], refEma[i], 1e-3f);
    for (int i = 14; i < N; ++i) {
        if (!std::isnan(refRsi[i]))
            EXPECT_NEAR(resultRsi[i], refRsi[i], 1e-3f);
    }

    ct_device_free(d_sma);
    ct_device_free(d_ema);
    ct_device_free(d_rsi);
}
