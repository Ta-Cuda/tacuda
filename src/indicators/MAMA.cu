/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// MAMA (CUDA, sm_86-tuned) — API preserved
#include <indicators/MAMA.h>
#include <utils/CudaUtils.h>
#include <utils/TacudaConfig.h>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <algorithm>

// -------------------- internal --------------------
namespace {

constexpr double HT_A     = 0.0962;
constexpr double HT_B     = 0.5769;
constexpr double RAD2DEG  = 180.0 / 3.14159265358979323846;
constexpr int    LOOKBACK = 32;

// Literal double-precision port of TA-Lib's TA_MAMA (ta_MAMA.c). The
// whole pipeline — WMA(4) price smoothing, Hilbert transforms, phase and
// period — runs in double, and the phase/period use plain atan() like
// TA-Lib (atan2 differs by +-180 deg for I1 < 0, which locks the adaptive
// alpha onto a different trajectory). FP64 throughput is poor on consumer
// GPUs, but this kernel is a sequential per-series state machine anyway.

// ==== WMA(4) smoothing (double, mirrors DO_PRICE_WMA) ====
__device__ __forceinline__
void doPriceWMA(double newPrice,
                double& sub, double& sum,
                double& trailing, const float* __restrict__ in,
                int& trailingIdx, int stride,
                double& smooth)
{
    sub += newPrice;
    sub -= trailing;
    sum += newPrice * 4.0;
    trailing = static_cast<double>(in[(size_t)trailingIdx * stride]);
    ++trailingIdx;
    smooth = sum * 0.1;      // /10
    sum   -= sub;
}

// ==== Full computation of one series ====
__device__ __forceinline__
void compute_mama_one_series(const float* __restrict__ in,
                             float* __restrict__ outMama,
                             float* __restrict__ outFama,
                             int n, int stride,
                             float fastLimit, float slowLimit)
{
    const float qnan = nanf("");
    const int nInit  = (n < LOOKBACK) ? n : LOOKBACK;

    #pragma unroll
    for (int t = 0; t < nInit; ++t) {
        outMama[(size_t)t * stride] = qnan;
        outFama[(size_t)t * stride] = qnan;
    }
    if (n <= LOOKBACK) return;

    const int startIdx = LOOKBACK;
    const int endIdx   = n - 1;

    // ---- WMA(4) init (double) ----
    int trailingIdx = startIdx - LOOKBACK; // 0
    int today       = trailingIdx;

    double sub = static_cast<double>(in[(size_t)today++ * stride]);
    double sum = sub;

    double tmp = static_cast<double>(in[(size_t)today++ * stride]);
    sub += tmp; sum += tmp * 2.0;

    tmp = static_cast<double>(in[(size_t)today++ * stride]);
    sub += tmp; sum += tmp * 3.0;

    double trailing = 0.0, smoothed = 0.0;

    #pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        tmp = static_cast<double>(in[(size_t)today++ * stride]);
        doPriceWMA(tmp, sub, sum, trailing, in, trailingIdx, stride, smoothed);
    }

    // ---- Hilbert state in double ----
    double period = 0.0, prevI2 = 0.0, prevQ2 = 0.0, Re = 0.0, Im = 0.0;
    double I1Odd3 = 0.0, I1Odd2 = 0.0;
    double I1Even3 = 0.0, I1Even2 = 0.0;
    double prevPhase = 0.0;

    double detOdd[3]  = {0.0,0.0,0.0}, detEven[3] = {0.0,0.0,0.0};
    double q1Odd[3]   = {0.0,0.0,0.0}, q1Even[3]  = {0.0,0.0,0.0};
    double jiOdd[3]   = {0.0,0.0,0.0}, jiEven[3]  = {0.0,0.0,0.0};
    double jqOdd[3]   = {0.0,0.0,0.0}, jqEven[3]  = {0.0,0.0,0.0};

    double prevDetOdd = 0.0, prevDetEven = 0.0;
    double prevQ1Odd  = 0.0, prevQ1Even  = 0.0;
    double prevJiOdd  = 0.0, prevJiEven  = 0.0;
    double prevJqOdd  = 0.0, prevJqEven  = 0.0;

    double prevDetInOdd = 0.0, prevDetInEven = 0.0;
    double prevQ1InOdd  = 0.0, prevQ1InEven  = 0.0;
    double prevJiInOdd  = 0.0, prevJiInEven  = 0.0;
    double prevJqInOdd  = 0.0, prevJqInEven  = 0.0;

    int hilbertIdx = 0;

    // TA-Lib seeds both lines at zero and lets the adaptive EMA pull them in.
    double dmama = 0.0;
    double dfama = 0.0;

    while (today <= endIdx) {
        const double adj = fma(0.075, period, 0.54);     // (0.075*period + 0.54)
        const double x   = static_cast<double>(in[(size_t)today * stride]);

        // smooth the price of the current bar (TA-Lib order)
        doPriceWMA(x, sub, sum, trailing, in, trailingIdx, stride, smoothed);
        const double smooth = smoothed;

        double htr, det, Q1, jI, jQ;
        if ((today & 1) == 0) {
            // even
            htr = HT_A * smooth;
            det = -detEven[hilbertIdx];
            detEven[hilbertIdx] = htr;
            det += htr - prevDetEven;
            prevDetEven = HT_B * prevDetInEven;
            det += prevDetEven;
            prevDetInEven = smooth;
            det *= adj;

            htr = HT_A * det;
            Q1  = -q1Even[hilbertIdx];
            q1Even[hilbertIdx] = htr;
            Q1 += htr - prevQ1Even;
            prevQ1Even = HT_B * prevQ1InEven;
            Q1 += prevQ1Even;
            prevQ1InEven = det;
            Q1 *= adj;

            htr = HT_A * I1Even3;
            jI  = -jiEven[hilbertIdx];
            jiEven[hilbertIdx] = htr;
            jI += htr - prevJiEven;
            prevJiEven = HT_B * prevJiInEven;
            jI += prevJiEven;
            prevJiInEven = I1Even3;
            jI *= adj;

            htr = HT_A * Q1;
            jQ  = -jqEven[hilbertIdx];
            jqEven[hilbertIdx] = htr;
            jQ += htr - prevJqEven;
            prevJqEven = HT_B * prevJqInEven;
            jQ += prevJqEven;
            prevJqInEven = Q1;
            jQ *= adj;

            if (++hilbertIdx == 3) hilbertIdx = 0;

            const double Q2 = fma(0.2, (Q1 + jI), 0.8 * prevQ2);
            const double I2 = fma(0.2, (I1Even3 - jQ), 0.8 * prevI2);
            I1Odd3 = I1Odd2;
            I1Odd2 = det;

            // TA-Lib: plain atan(Q1/I1), 0 when I1 == 0
            const double phase = (I1Even3 != 0.0) ? atan(Q1 / I1Even3) * RAD2DEG : 0.0;
            double dphi = prevPhase - phase;
            if (dphi < 1.0) dphi = 1.0;

            double alpha = static_cast<double>(fastLimit) / dphi;
            if (alpha < static_cast<double>(slowLimit)) alpha = static_cast<double>(slowLimit);

            dmama = fma(alpha, (x - dmama), dmama);
            const double a2 = 0.5 * alpha;
            dfama = fma(a2, (dmama - dfama), dfama);

            Re = fma(0.2, (I2 * prevI2 + Q2 * prevQ2), 0.8 * Re);
            Im = fma(0.2, (I2 * prevQ2 - Q2 * prevI2), 0.8 * Im);
            prevQ2 = Q2; prevI2 = I2;

            double ptmp = period;
            // TA-Lib: plain atan(Im/Re)
            if (Im != 0.0 && Re != 0.0) period = 360.0 / (atan(Im / Re) * RAD2DEG);
            period = fmin(period, 1.5 * ptmp);
            period = fmax(period, 0.67 * ptmp);
            period = fmin(fmax(period, 6.0), 50.0);
            period = fma(0.2, period, 0.8 * ptmp);

            if (today >= startIdx) {
                outMama[(size_t)today * stride] = static_cast<float>(dmama);
                outFama[(size_t)today * stride] = static_cast<float>(dfama);
            }
            prevPhase = phase;
        } else {
            // odd
            htr = HT_A * smooth;
            det = -detOdd[hilbertIdx];
            detOdd[hilbertIdx] = htr;
            det += htr - prevDetOdd;
            prevDetOdd = HT_B * prevDetInOdd;
            det += prevDetOdd;
            prevDetInOdd = smooth;
            det *= adj;

            htr = HT_A * det;
            Q1  = -q1Odd[hilbertIdx];
            q1Odd[hilbertIdx] = htr;
            Q1 += htr - prevQ1Odd;
            prevQ1Odd = HT_B * prevQ1InOdd;
            Q1 += prevQ1Odd;
            prevQ1InOdd = det;
            Q1 *= adj;

            htr = HT_A * I1Odd3;
            jI  = -jiOdd[hilbertIdx];
            jiOdd[hilbertIdx] = htr;
            jI += htr - prevJiOdd;
            prevJiOdd = HT_B * prevJiInOdd;
            jI += prevJiOdd;
            prevJiInOdd = I1Odd3;
            jI *= adj;

            htr = HT_A * Q1;
            jQ  = -jqOdd[hilbertIdx];
            jqOdd[hilbertIdx] = htr;
            jQ += htr - prevJqOdd;
            prevJqOdd = HT_B * prevJqInOdd;
            jQ += prevJqOdd;
            prevJqInOdd = Q1;
            jQ *= adj;

            if (++hilbertIdx == 3) hilbertIdx = 0;

            const double Q2 = fma(0.2, (Q1 + jI), 0.8 * prevQ2);
            const double I2 = fma(0.2, (I1Odd3 - jQ), 0.8 * prevI2);
            I1Even3 = I1Even2;
            I1Even2 = det;

            const double phase = (I1Odd3 != 0.0) ? atan(Q1 / I1Odd3) * RAD2DEG : 0.0;
            double dphi = prevPhase - phase;
            if (dphi < 1.0) dphi = 1.0;

            double alpha = static_cast<double>(fastLimit) / dphi;
            if (alpha < static_cast<double>(slowLimit)) alpha = static_cast<double>(slowLimit);

            dmama = fma(alpha, (x - dmama), dmama);
            const double a2 = 0.5 * alpha;
            dfama = fma(a2, (dmama - dfama), dfama);

            Re = fma(0.2, (I2 * prevI2 + Q2 * prevQ2), 0.8 * Re);
            Im = fma(0.2, (I2 * prevQ2 - Q2 * prevI2), 0.8 * Im);
            prevQ2 = Q2; prevI2 = I2;

            double ptmp = period;
            if (Im != 0.0 && Re != 0.0) period = 360.0 / (atan(Im / Re) * RAD2DEG);
            period = fmin(period, 1.5 * ptmp);
            period = fmax(period, 0.67 * ptmp);
            period = fmin(fmax(period, 6.0), 50.0);
            period = fma(0.2, period, 0.8 * ptmp);

            if (today >= startIdx) {
                outMama[(size_t)today * stride] = static_cast<float>(dmama);
                outFama[(size_t)today * stride] = static_cast<float>(dfama);
            }
            prevPhase = phase;
        }
        ++today;
    }
}

#ifndef MAMA_LAUNCH_BOUNDS
#define MAMA_LAUNCH_BOUNDS __launch_bounds__(256, 2)
#endif

// grid-stride: one thread = one series
__global__ MAMA_LAUNCH_BOUNDS
void mamaManySeriesKernel(const float* __restrict__ input,
                           float* __restrict__ mama,
                           float* __restrict__ fama,
                          int seriesLength,
                          int batchCount,
                          float fastLimit, float slowLimit)
{
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < batchCount;
         s += blockDim.x * gridDim.x)
    {
    #ifdef TACUDA_MAMA_TIME_MAJOR
        const int   stride = batchCount;           // [t][batch]
        const float* in    = input + s;
        float* outM        = mama  + s;
        float* outF        = fama  + s;
    #else
        const int   stride = 1;                    // [series][t]
        const float* in    = input + (size_t)s * seriesLength;
        float* outM        = mama  + (size_t)s * seriesLength;
        float* outF        = fama  + (size_t)s * seriesLength;
    #endif
        compute_mama_one_series(in, outM, outF, seriesLength, stride, fastLimit, slowLimit);
    }
}

// env-config (kept for backward compatibility)
enum class MAMAAlgo { Auto = 0, Sequential = 1, Batch = 2 };
struct MAMAConfig { MAMAAlgo algo = MAMAAlgo::Auto; int batchCount = 1; int seriesLength = 0; };

static MAMAConfig getMAMAConfig() {
    static MAMAConfig cfg = []() {
        auto& gcfg = tacuda::Config::instance();
        MAMAConfig c;
        int v = gcfg.mama.algo;
        if (v >= 0 && v <= 2) c.algo = static_cast<MAMAAlgo>(v);
        c.batchCount   = gcfg.batch.count;
        c.seriesLength = gcfg.batch.seriesLen;
        return c;
    }();
    return cfg;
}

} // namespace

// -------------------- public API --------------------
namespace tacuda {

MAMA::MAMA(float fastLimit, float slowLimit) : fastLimit(fastLimit), slowLimit(slowLimit) {}

void MAMA::calculate(const float* input, float* output, int size, cudaStream_t stream) {
    if (size <= 0) throw std::invalid_argument("MAMA: invalid size");

    MAMAConfig cfg = getMAMAConfig();
    int batchCount   = 1;
    int seriesLength = size;
    if (cfg.batchCount > 1 && cfg.seriesLength > 0 &&
        cfg.batchCount * cfg.seriesLength == size)
    {
        batchCount   = cfg.batchCount;
        seriesLength = cfg.seriesLength;
    }

    float* mama = output;
    float* fama = output + size;

    // Occupancy-driven configuration
    int minGrid = 0, block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGrid, &block, mamaManySeriesKernel, 0, 0));        // NVIDIA recommendations for auto-tuning

    block = std::min(std::max(128, ((block + 31) / 32) * 32), 256);
    int grid = (batchCount + block - 1) / block;

    int dev = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    grid = std::min(grid, sms * 4);                             // don't overbloat the grid

    // For kernel without shared memory, prefer L1 carveout
#if CUDART_VERSION >= 11000
    (void)cudaFuncSetAttribute(
        mamaManySeriesKernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxL1);
#endif

    mamaManySeriesKernel<<<grid, block, 0, stream>>>(
        input, mama, fama, seriesLength, batchCount, fastLimit, slowLimit);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda
