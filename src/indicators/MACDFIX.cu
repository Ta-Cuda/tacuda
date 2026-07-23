/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MACDFIX.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

namespace {

// TA-Lib–compatible MACDFIX single-pass sequential kernel.
// Replicates TA-Lib's INT_MACD flow exactly:
//   1) Both fast & slow INT_EMA called with startIdx = macdLookback (= slowPeriod-1).
//   2) Fast EMA: SMA seed over input[macdLookback - fastPeriod + 1 .. macdLookback].
//   3) Slow EMA: SMA seed over input[0 .. slowPeriod - 1].
//   4) MACDFIX uses hardcoded k-values: k_fast=0.15, k_slow=0.075 (NOT 2/(P+1)).
//   5) Signal EMA uses standard k = 2/(signalPeriod+1), SMA-seeded from first
//      signalPeriod MACD values.
//   6) All outputs NaN before signalLookback = slowPeriod + signalPeriod - 2.
__global__ void macdfix_sequential_kernel(const float* __restrict__ input,
                                           float* __restrict__ macd,
                                           float* __restrict__ signal,
                                           float* __restrict__ hist,
                                           int size,
                                           int fastPeriod,
                                           int slowPeriod,
                                           int signalPeriod)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (size <= 0) return;

    const int macdLookback   = slowPeriod - 1;                      // 25
    const int signalLookback = macdLookback + signalPeriod - 1;     // 33

    // TA-Lib MACDFIX uses fixed k-values, NOT 2/(period+1):
    const double kFast = 0.15;                              // fixed for period 12
    const double kSlow = 0.075;                             // fixed for period 26
    const double kSig  = 2.0 / (signalPeriod + 1.0);       // standard for signal

    if (signalLookback >= size) {
        for (int i = 0; i < size; ++i) {
            macd[i] = nanf(""); signal[i] = nanf(""); hist[i] = nanf("");
        }
        return;
    }

    // Fill NaN for all outputs before signalLookback
    for (int i = 0; i < signalLookback; ++i) {
        macd[i]   = nanf("");
        signal[i] = nanf("");
        hist[i]   = nanf("");
    }

    // --- Fast EMA seed: SMA over input[macdLookback - fastPeriod + 1 .. macdLookback] ---
    // TA-Lib's INT_EMA is called with startIdx=macdLookback, so it seeds from
    // startIdx - (fastPeriod-1) = macdLookback - fastPeriod + 1.
    double ef = 0.0;
    {
        const int seedStart = macdLookback - fastPeriod + 1;  // e.g. 25-12+1=14
        double sum = 0.0;
        for (int i = seedStart; i <= macdLookback; ++i)
            sum += (double)input[i];
        ef = sum / (double)fastPeriod;
    }

    // --- Slow EMA seed: SMA over input[0 .. slowPeriod-1] ---
    // TA-Lib's INT_EMA called with startIdx=macdLookback, lookback=slowPeriod-1,
    // so seed starts at macdLookback - (slowPeriod-1) = 0.
    double es = 0.0;
    {
        double sum = 0.0;
        for (int i = 0; i < slowPeriod; ++i)
            sum += (double)input[i];
        es = sum / (double)slowPeriod;
    }

    // At index macdLookback: both EMAs are at their SMA seed values.
    // Compute MACD values from macdLookback through signalLookback,
    // accumulating for signal SMA seed.
    double signalSMASum = 0.0;
    double macdVal = ef - es;   // MACD at macdLookback
    signalSMASum += macdVal;

    for (int i = macdLookback + 1; i <= signalLookback; ++i) {
        ef = ((double)input[i] - ef) * kFast + ef;
        es = ((double)input[i] - es) * kSlow + es;
        macdVal = ef - es;
        signalSMASum += macdVal;
    }

    // Signal seed = SMA of MACD[macdLookback .. signalLookback] (signalPeriod values)
    double sig = signalSMASum / (double)signalPeriod;

    // First valid output at signalLookback
    macd[signalLookback]   = (float)macdVal;
    signal[signalLookback] = (float)sig;
    hist[signalLookback]   = (float)(macdVal - sig);

    // Continue recursion for remaining indices
    for (int i = signalLookback + 1; i < size; ++i) {
        ef = ((double)input[i] - ef) * kFast + ef;
        es = ((double)input[i] - es) * kSlow + es;
        double m = ef - es;
        sig = ((m - sig) * kSig) + sig;
        macd[i]   = (float)m;
        signal[i] = (float)sig;
        hist[i]   = (float)(m - sig);
    }
}

} // namespace

namespace tacuda {

MACDFIX::MACDFIX(int signalPeriod) : signalPeriod(signalPeriod) {}

void MACDFIX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (signalPeriod <= 0) {
        throw std::invalid_argument("MACDFIX: invalid signalPeriod");
    }

    const int fastPeriod = 12;
    const int slowPeriod = 26;

    float* macdOut   = output;
    float* signalOut = output + size;
    float* histOut   = output + 2 * size;

    macdfix_sequential_kernel<<<1, 1, 0, stream>>>(
        input, macdOut, signalOut, histOut, size,
        fastPeriod, slowPeriod, signalPeriod);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda
