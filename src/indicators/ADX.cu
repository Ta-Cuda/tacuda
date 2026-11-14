/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/ADX.h>
#include <indicators/DX.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <utils/DeviceSeed.cuh>
#include <stdexcept>
#include <cmath>

namespace {

using AffinePair = tacuda::detail::AffineTransform1<double>;
using AffineCompose = tacuda::detail::AffineCompose1<double>;

__global__ void fillPairsKernel(const float* __restrict__ x,
                                AffinePair* __restrict__ pairs,
                                int offset, int nPairs,
                                double a, double invP)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    int i = offset + k;
    AffinePair t; t.A = a; t.b = static_cast<double>(x[i]) * invP;
    pairs[k] = t;
}

__global__ void finalizeSmoothKernel(const AffinePair* __restrict__ pref,
                                     float* __restrict__ y,
                                     const double* __restrict__ y0p,
                                     int baseIdx,
                                     int nPairs)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    AffinePair p = pref[k];
    double val = p.A * (*y0p) + p.b;
    y[baseIdx + 1 + k] = static_cast<float>(val);
}

// Wilder smoothing for ADX with input data offset
// offset - index at which valid data begins in input array x
static void wilderSmoothDeviceWithOffset(const float* x, float* y,
                                         int n, int period, int offset,
                                         cudaStream_t stream)
{
    // Check if enough data for smoothing
    int availableData = n - offset;
    if (availableData < period) return;

    // TA-Lib style for ADX: seed at index (offset + period - 1) with
    // mean(x[offset .. offset+period-1]). Computed and kept on the device.
    int seedIdx = offset + period - 1;
    auto dSeed = acquireDeviceBufferAsync<double>(1, stream);
    tacuda::detail::launch_seed_sum(x, offset, period, static_cast<double>(period),
                                    dSeed.get(), y, seedIdx, stream);

    // Recurrence smoothing starts at seedIdx+1
    const int firstIdx = seedIdx + 1;
    int nPairs = n - firstIdx;
    if (nPairs <= 0) return;

    const double a    = (static_cast<double>(period) - 1.0) / static_cast<double>(period);
    const double invP = 1.0 / static_cast<double>(period);

    auto dPairs = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);
    auto dPref  = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);

    {
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(nPairs);
        fillPairsKernel<<<g, b, 0, stream>>>(x, dPairs.get(), firstIdx, nPairs, a, invP);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        tacuda::detail::affine_inclusive_scan<AffinePair, AffineCompose>(
            dPairs.get(), dPref.get(), nPairs, stream);
    }
    {
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(nPairs);
        finalizeSmoothKernel<<<g, b, 0, stream>>>(dPref.get(), y, dSeed.get(), seedIdx, nPairs);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace


namespace tacuda {

ADX::ADX(int period) : period(period) {}

void ADX::calculate(const float* high, const float* low, const float* close,
                    float* output, int size, cudaStream_t stream) noexcept(false)
{
    if (period <= 0)
        throw std::invalid_argument("ADX: invalid period");
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    
    // ADX lookback = 2*period - 2
    int lookback = 2 * period - 2;
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    if (size <= lookback) return;

    // 1) Compute DX
    auto dDX = acquireDeviceBufferAsync<float>(size, stream);
    CUDA_CHECK(cudaMemsetAsync(dDX.get(), 0xFF, size * sizeof(float), stream));

    DX dx(period);
    dx.calculate(high, low, close, dDX.get(), size, stream);

    // 2) Apply Wilder smoothing to DX to obtain ADX
    // DX starts at index 'period' (first period values = NaN)
    // ADX = Wilder smooth(DX, period), starts at index 2*period
    wilderSmoothDeviceWithOffset(dDX.get(), output, size, period, period, stream);
}

void ADX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false)
{
    const float* high  = input;
    const float* low   = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}

} // namespace tacuda
