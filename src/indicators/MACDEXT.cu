/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <stdexcept>

#include <indicators/MACDEXT.h>
#include <utils/AffineScan.cuh>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace {

using AffinePair = tacuda::detail::AffineTransform1<double>;
using AffineCompose = tacuda::detail::AffineCompose1<double>;

struct ToDouble {
    __host__ __device__ double operator()(float v) const { return static_cast<double>(v); }
};

__global__ void buildEmaPairsFromIndexKernel(const float* __restrict__ input,
                                    AffinePair* __restrict__ pairs,
                                    int firstIdx,
                                    int nPairs,
                                    double a,
                                    double alpha) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    int i = firstIdx + k;
    pairs[k] = AffinePair{a, alpha * static_cast<double>(input[i])};
}

__global__ void finalizeSeededEmaKernel(const AffinePair* __restrict__ pref,
                                  double* __restrict__ output,
                                  int firstIdx,
                                  int nPairs,
                                  double seed) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    AffinePair p = pref[k];
    output[firstIdx + k] = p.A * seed + p.b;
}

__global__ void finalizeSeededEmaToFloatKernel(const AffinePair* __restrict__ pref,
                                               float* __restrict__ output,
                                               int firstIdx,
                                               int nPairs,
                                               double seed) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    AffinePair p = pref[k];
    output[firstIdx + k] = static_cast<float>(p.A * seed + p.b);
}

__global__ void computeMacdFromEmaKernel(const double* __restrict__ fastEma,
                                         const double* __restrict__ slowEma,
                                         float* __restrict__ macd,
                                         int size,
                                         int macdStart) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (i < macdStart) {
        macd[i] = nanf("");
    } else {
        macd[i] = static_cast<float>(fastEma[i] - slowEma[i]);
    }
}

__global__ void computeMacdFromMaKernel(const float* __restrict__ fastMa,
                                        const float* __restrict__ slowMa,
                                        float* __restrict__ macd,
                                        int size,
                                        int macdStart) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (i < macdStart) {
        macd[i] = nanf("");
    } else {
        float f = fastMa[i];
        float s = slowMa[i];
        macd[i] = (isnan(f) || isnan(s)) ? nanf("") : (f - s);
    }
}

__global__ void finalizeMacdOutputsKernel(float* __restrict__ macd,
                                          float* __restrict__ signal,
                                          float* __restrict__ hist,
                                          int size,
                                          int signalStart) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (i < signalStart) {
        macd[i] = nanf("");
        signal[i] = nanf("");
        hist[i] = nanf("");
        return;
    }
    float m = macd[i];
    float s = signal[i];
    hist[i] = (isnan(m) || isnan(s)) ? nanf("") : (m - s);
}

static void computeSeededEmaFrom(const float* input,
                                 double* output,
                                 int size,
                                 int firstIdx,
                                 double seed,
                                 int period,
                                 cudaStream_t stream) {
    if (firstIdx >= size) return;
    const double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    const double a = 1.0 - alpha;
    const int nPairs = size - firstIdx;
    auto pairs = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);
    auto pref = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);

    dim3 block = defaultBlock();
    dim3 gridPairs = defaultGrid(nPairs);
    buildEmaPairsFromIndexKernel<<<gridPairs, block, 0, stream>>>(
        input, pairs.get(), firstIdx, nPairs, a, alpha);
    CUDA_CHECK(cudaGetLastError());

    tacuda::detail::affine_inclusive_scan<AffinePair, AffineCompose>(
        pairs.get(), pref.get(), nPairs, stream);

    finalizeSeededEmaKernel<<<gridPairs, block, 0, stream>>>(
        pref.get(), output, firstIdx, nPairs, seed);
    CUDA_CHECK(cudaGetLastError());
}

static void computeSeededEmaFromToFloat(const float* input,
                                        float* output,
                                        int size,
                                        int firstIdx,
                                        double seed,
                                        int period,
                                        cudaStream_t stream) {
    if (firstIdx >= size) return;
    const double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    const double a = 1.0 - alpha;
    const int nPairs = size - firstIdx;
    auto pairs = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);
    auto pref = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);

    dim3 block = defaultBlock();
    dim3 gridPairs = defaultGrid(nPairs);
    buildEmaPairsFromIndexKernel<<<gridPairs, block, 0, stream>>>(
        input, pairs.get(), firstIdx, nPairs, a, alpha);
    CUDA_CHECK(cudaGetLastError());

    tacuda::detail::affine_inclusive_scan<AffinePair, AffineCompose>(
        pairs.get(), pref.get(), nPairs, stream);

    finalizeSeededEmaToFloatKernel<<<gridPairs, block, 0, stream>>>(
        pref.get(), output, firstIdx, nPairs, seed);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace tacuda {

MACDEXT::MACDEXT(int fastPeriod, int slowPeriod, int signalPeriod, MAType type)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), signalPeriod(signalPeriod),
      fastType(type), slowType(type), signalType(type) {}

MACDEXT::MACDEXT(int fastPeriod, MAType fastType, int slowPeriod, MAType slowType,
                 int signalPeriod, MAType signalType)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), signalPeriod(signalPeriod),
      fastType(fastType), slowType(slowType), signalType(signalType) {
    // TA-Lib swaps the fast/slow definitions (period AND type) when the
    // "fast" one is actually slower.
    if (this->fastPeriod > this->slowPeriod) {
        int p = this->fastPeriod;
        this->fastPeriod = this->slowPeriod;
        this->slowPeriod = p;
        MAType t = this->fastType;
        this->fastType = this->slowType;
        this->slowType = t;
    }
}

int MACDEXT::maTypeLookback(MAType type, int period) {
    switch (type) {
    case MAType::SMA:
    case MAType::EMA:
    case MAType::WMA:
    case MAType::TRIMA:
        return period - 1;
    case MAType::DEMA:
        return 2 * (period - 1);
    case MAType::TEMA:
        return 3 * (period - 1);
    case MAType::KAMA:
        return period;
    case MAType::MAMA:
        return 32;
    case MAType::T3:
        return 6 * (period - 1);
    default:
        return period - 1;
    }
}

void MACDEXT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    calculate(input, output, output + size, output + 2 * size, size, stream);
}

void MACDEXT::calculate(const float* input, float* macd, float* signal, float* hist,
                        int size, cudaStream_t stream) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    // TA-Lib MACDEXT normalizes reversed periods (slow = max) and swaps the
    // corresponding MA types together with the periods.
    if (slowPeriod < fastPeriod) {
        int tp = fastPeriod;
        fastPeriod = slowPeriod;
        slowPeriod = tp;
        MAType tt = fastType;
        fastType = slowType;
        slowType = tt;
    }
    if (size <= 0) {
        return;
    }

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    CUDA_CHECK(cudaMemsetAsync(signal, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(hist, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(macd, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));

    const bool allEma = fastType == MAType::EMA && slowType == MAType::EMA &&
                        signalType == MAType::EMA;
    if (allEma) {
        // TA-Lib MACDEXT parity:
        // - EMA seeds are computed at macdLookback=(slowPeriod-1), not at each EMA's own lookback.
        // - Signal seed is SMA over MACD[macdLookback..signalLookback].
        const int macdLookback = slowPeriod - 1;
        const int signalLookback = macdLookback + signalPeriod - 1;
        if (signalLookback >= size) {
            return;
        }

        auto fastEma = acquireDeviceBufferAsync<double>(size, stream);
        auto slowEma = acquireDeviceBufferAsync<double>(size, stream);
        CUDA_CHECK(cudaMemsetAsync(fastEma.get(), 0xFF, static_cast<size_t>(size) * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(slowEma.get(), 0xFF, static_cast<size_t>(size) * sizeof(double), stream));

        const int fastSeedStart = macdLookback - fastPeriod + 1;
        double fastSeedSum = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(input) + fastSeedStart,
            thrust::device_pointer_cast(input) + macdLookback + 1,
            ToDouble{}, 0.0, thrust::plus<double>{});
        double fastSeed = fastSeedSum / static_cast<double>(fastPeriod);

        double slowSeedSum = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(input),
            thrust::device_pointer_cast(input) + slowPeriod,
            ToDouble{}, 0.0, thrust::plus<double>{});
        double slowSeed = slowSeedSum / static_cast<double>(slowPeriod);

        CUDA_CHECK(cudaMemcpyAsync(fastEma.get() + macdLookback, &fastSeed, sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(slowEma.get() + macdLookback, &slowSeed, sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

        computeSeededEmaFrom(input, fastEma.get(), size, slowPeriod, fastSeed, fastPeriod, stream);
        computeSeededEmaFrom(input, slowEma.get(), size, slowPeriod, slowSeed, slowPeriod, stream);

        computeMacdFromEmaKernel<<<grid, block, 0, stream>>>(
            fastEma.get(), slowEma.get(), macd, size, macdLookback);
        CUDA_CHECK(cudaGetLastError());

        double signalSeedSum = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(macd) + macdLookback,
            thrust::device_pointer_cast(macd) + signalLookback + 1,
            ToDouble{}, 0.0, thrust::plus<double>{});
        double signalSeed = signalSeedSum / static_cast<double>(signalPeriod);
        float signalSeedF = static_cast<float>(signalSeed);
        CUDA_CHECK(cudaMemcpyAsync(signal + signalLookback, &signalSeedF, sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        computeSeededEmaFromToFloat(macd, signal, size, signalLookback + 1, signalSeed, signalPeriod, stream);

        finalizeMacdOutputsKernel<<<grid, block, 0, stream>>>(macd, signal, hist, size, signalLookback);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Generic composition with independent MA types per line. Window-based
        // types (SMA/WMA/TRIMA) match TA-Lib exactly; recursive types seed at
        // their own lookback, so the earliest bars may differ slightly from
        // TA-Lib's MACDEXT (documented in docs/TALIB_COMPATIBILITY_ISSUES.md).
        const int fastLb = maTypeLookback(fastType, fastPeriod);
        const int slowLb = maTypeLookback(slowType, slowPeriod);
        const int signalLb = maTypeLookback(signalType, signalPeriod);
        const int macdStart = fastLb > slowLb ? fastLb : slowLb;
        if (macdStart + 1 > size) {
            return; // not enough data: outputs stay NaN
        }

        auto fastMa = acquireDeviceBufferAsync<float>(size, stream);
        auto slowMa = acquireDeviceBufferAsync<float>(size, stream);
        MA fast(fastPeriod, fastType);
        MA slow(slowPeriod, slowType);
        fast.calculate(input, fastMa.get(), size, stream);
        slow.calculate(input, slowMa.get(), size, stream);
        computeMacdFromMaKernel<<<grid, block, 0, stream>>>(
            fastMa.get(), slowMa.get(), macd, size, macdStart);
        CUDA_CHECK(cudaGetLastError());

        const int tailLen = size - macdStart;
        auto macdTail = acquireDeviceBufferAsync<float>(size, stream);
        auto signalTail = acquireDeviceBufferAsync<float>(size, stream);
        if (tailLen >= signalLb + 1) {
            CUDA_CHECK(cudaMemcpyAsync(macdTail.get(), macd + macdStart,
                                       static_cast<size_t>(tailLen) * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            MA signalMa(signalPeriod, signalType);
            signalMa.calculate(macdTail.get(), signalTail.get(), tailLen, stream);
            CUDA_CHECK(cudaMemcpyAsync(signal + macdStart, signalTail.get(),
                                       static_cast<size_t>(tailLen) * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        const int signalStart = macdStart + signalLb;
        finalizeMacdOutputsKernel<<<grid, block, 0, stream>>>(macd, signal, hist, size, signalStart);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace tacuda
