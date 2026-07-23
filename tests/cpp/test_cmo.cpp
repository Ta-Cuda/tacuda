/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

static std::vector<float> cmo_ref(const std::vector<float>& in, int p) {
    const int n = static_cast<int>(in.size());
    std::vector<float> out(n, std::numeric_limits<float>::quiet_NaN());
    if (p <= 0 || n <= p) return out;
    double sumGain = 0.0, sumLoss = 0.0;
    for (int i = 1; i <= p; ++i) {
        double d = static_cast<double>(in[i]) - static_cast<double>(in[i-1]);
        if (d > 0) sumGain += d; else sumLoss -= d;
    }
    double avgGain = sumGain / p;
    double avgLoss = sumLoss / p;
    double denom = avgGain + avgLoss;
    out[p] = (denom == 0.0) ? 0.0f : static_cast<float>(100.0 * (avgGain - avgLoss) / denom);
    for (int i = p + 1; i < n; ++i) {
        double d = static_cast<double>(in[i]) - static_cast<double>(in[i-1]);
        double g = d > 0 ? d : 0.0;
        double l = d < 0 ? -d : 0.0;
        avgGain = (avgGain * (p - 1) + g) / p;
        avgLoss = (avgLoss * (p - 1) + l) / p;
        denom = avgGain + avgLoss;
        out[i] = (denom == 0.0) ? 0.0f : static_cast<float>(100.0 * (avgGain - avgLoss) / denom);
    }
    return out;
}

TEST(Tacuda, CMO_Simple) {
    const int N = 1000, P = 14;
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i) in[i] = std::sin(0.01f * i) + 0.1f * std::cos(0.03f * i);
    std::vector<float> out(N, 0.0f), ref = cmo_ref(in, P);
    ASSERT_EQ(ct_cmo(in.data(), out.data(), N, P), CT_STATUS_SUCCESS);
    for (int i = 0; i < P; ++i) ASSERT_TRUE(std::isnan(out[i]));
    double max_diff = 0.0;
    for (int i = P; i < N; ++i) max_diff = std::max(max_diff, static_cast<double>(std::abs(out[i] - ref[i])));
    ASSERT_LE(max_diff, 1e-4);
}

TEST(Tacuda, CMO_EdgeFlat) {
    const int N = 256, P = 10;
    std::vector<float> in(N, 100.0f), out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
    // For flat series, CMO must be 0 after warmup
    ref = cmo_ref(in, P);
    ASSERT_EQ(ct_cmo(in.data(), out.data(), N, P), CT_STATUS_SUCCESS);
    for (int i = P; i < N; ++i) ASSERT_NEAR(out[i], 0.0f, 1e-6f);
}
