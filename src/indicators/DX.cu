/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
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

// -------------------- TR, +DM, -DM --------------------
__global__ void buildTrDmKernel(const float* __restrict__ high,
                               const float* __restrict__ low,
                               const float* __restrict__ close,
                                float* __restrict__ tr,
                                float* __restrict__ pdm,
                                float* __restrict__ mdm,
                                int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i == 0) {
        tr[0]  = high[0] - low[0];
        pdm[0] = 0.0f;
        mdm[0] = 0.0f;
        return;
    }

    float up   = high[i] - high[i - 1];
    float down = low[i - 1] - low[i];

    float plusDM  = (up   > 0.0f && up   > down) ? up   : 0.0f;
    float minusDM = (down > 0.0f && down > up  ) ? down : 0.0f;

    float prevClose = close[i - 1];
    float tr0 = high[i] - low[i];
    float tr1 = fabsf(high[i] - prevClose);
    float tr2 = fabsf(low[i]  - prevClose);
    float trueRange = fmaxf(tr0, fmaxf(tr1, tr2));

    tr[i]  = trueRange;
    pdm[i] = plusDM;
    mdm[i] = minusDM;
}

__global__ void fillPairsKernel(const float* __restrict__ x,
                                AffinePair* __restrict__ pairs,
                                int offset, int nPairs,
                                double a)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    int i = offset + k;
    // Wilder smoothing: y[i] = a*y[i-1] + x[i], where a=(p-1)/p
    AffinePair t; t.A = a; t.b = static_cast<double>(x[i]);
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

// Wilder smoothing for DX
static void wilderSmoothDevice(const float* x, float* y,
                               int n, int period,
                               cudaStream_t stream)
{
    if (n <= period) return;

    // TA-Lib Wilder smoothing seed:
    // y[period-1] = sum(x[0..period-1]), computed and kept on the device.
    const int seedIdx = period - 1;
    auto dSeed = acquireDeviceBufferAsync<double>(1, stream);
    tacuda::detail::launch_seed_sum(x, 0, period, 1.0, dSeed.get(), y, seedIdx, stream);

    const int firstIdx = seedIdx + 1; // == period
    int nPairs = n - firstIdx;
    if (nPairs <= 0) return;

    const double a = (static_cast<double>(period) - 1.0) / static_cast<double>(period);

    auto dPairs = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);
    auto dPref  = acquireDeviceBufferAsync<AffinePair>(nPairs, stream);

    {
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(nPairs);
        fillPairsKernel<<<g, b, 0, stream>>>(x, dPairs.get(), firstIdx, nPairs, a);
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

// DX from smoothed TR/+DM/-DM
__global__ void dxFromSmoothedKernel(const float* __restrict__ trS,
                                     const float* __restrict__ pdmS,
                                     const float* __restrict__ mdmS,
                                     float* __restrict__ dx,
                                     int period, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < period || i >= n) return;

    float trv = trS[i];
    float diPlus  = (trv > 0.f) ? (100.f * pdmS[i] / trv) : 0.f;
    float diMinus = (trv > 0.f) ? (100.f * mdmS[i] / trv) : 0.f;

    float denom = diPlus + diMinus;
    float val = (denom > 0.f) ? (100.f * fabsf(diPlus - diMinus) / denom) : 0.f;
    dx[i] = val;
}

} // namespace


namespace tacuda {

DX::DX(int period) : period(period) {}

void DX::calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream) noexcept(false)
{
    if (period <= 0)
        throw std::invalid_argument("DX: invalid period");
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    
    // First 'period' values are NaN
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // 1) TR, +DM, -DM
    auto dTR  = acquireDeviceBufferAsync<float>(size, stream);
    auto dPDM = acquireDeviceBufferAsync<float>(size, stream);
    auto dMDM = acquireDeviceBufferAsync<float>(size, stream);

    {
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(size);
        buildTrDmKernel<<<g, b, 0, stream>>>(high, low, close, dTR.get(), dPDM.get(), dMDM.get(), size);
    CUDA_CHECK(cudaGetLastError());
}

    // 2) Wilder smoothing
    auto dTRs  = acquireDeviceBufferAsync<float>(size, stream);
    auto dPDMs = acquireDeviceBufferAsync<float>(size, stream);
    auto dMDMs = acquireDeviceBufferAsync<float>(size, stream);

    CUDA_CHECK(cudaMemsetAsync(dTRs.get(),  0xFF, size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dPDMs.get(), 0xFF, size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dMDMs.get(), 0xFF, size * sizeof(float), stream));

    wilderSmoothDevice(dTR.get(),  dTRs.get(),  size, period, stream);
    wilderSmoothDevice(dPDM.get(), dPDMs.get(), size, period, stream);
    wilderSmoothDevice(dMDM.get(), dMDMs.get(), size, period, stream);

    // 3) DX
    {
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(size);
        dxFromSmoothedKernel<<<g, b, 0, stream>>>(dTRs.get(), dPDMs.get(), dMDMs.get(), output, period, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

void DX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false)
{
    const float* high  = input;
    const float* low   = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}

} // namespace tacuda
