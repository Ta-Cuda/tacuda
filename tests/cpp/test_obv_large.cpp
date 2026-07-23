/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

static std::vector<float> obv_cpu_ref(const std::vector<float>& price,
                                      const std::vector<float>& volume) {
    const int n = static_cast<int>(price.size());
    std::vector<float> out(n, 0.0f);
    if (n == 0) return out;
    double obv = static_cast<double>(volume[0]);
    out[0] = static_cast<float>(obv);
    for (int i = 1; i < n; ++i) {
        const double p = static_cast<double>(price[i]);
        const double prev = static_cast<double>(price[i - 1]);
        const double v = static_cast<double>(volume[i]);
        if (p > prev) obv += v;
        else if (p < prev) obv -= v;
        out[i] = static_cast<float>(obv);
    }
    return out;
}

TEST(Tacuda, OBV_LargeRandom) {
    const int N = 200000;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dprice(-0.01f, 0.01f);
    std::uniform_real_distribution<float> dvol(100.0f, 10000.0f);

    std::vector<float> price(N, 0.0f), volume(N, 0.0f);
    // Build price as random walk to avoid degenerate equalities dominating
    float p = 100.0f;
    for (int i = 0; i < N; ++i) {
        p += dprice(rng);
        price[i] = p;
        volume[i] = dvol(rng);
    }

    std::vector<float> out(N, 0.0f), ref = obv_cpu_ref(price, volume);

    ASSERT_EQ(ct_obv(price.data(), volume.data(), out.data(), N), CT_STATUS_SUCCESS);

    // Check max absolute difference
    double max_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(out[i] - ref[i])));
    }
    ASSERT_LE(max_diff, 1e-4) << "Max diff exceeded tolerance";
}

TEST(Tacuda, OBV_EdgeEqual) {
    const int N = 1024;
    std::vector<float> price(N, 100.0f), volume(N, 1000.0f);
    // Create segments: equal, up, equal, down, equal
    for (int i = 256; i < 512; ++i) price[i] = 101.0f;
    for (int i = 512; i < 768; ++i) price[i] = 101.0f; // equal segment
    for (int i = 768; i < 1024; ++i) price[i] = 100.0f; // down segment

    std::vector<float> out(N, 0.0f), ref = obv_cpu_ref(price, volume);

    ASSERT_EQ(ct_obv(price.data(), volume.data(), out.data(), N), CT_STATUS_SUCCESS);

    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(out[i], ref[i], 1e-4f);
    }
}
