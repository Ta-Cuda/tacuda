/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/CMO.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <utils/LeadingNaN.cuh>
#include <stdexcept>

namespace {

using Affine2 = tacuda::detail::AffineTransform2<float>;
using AffineCompose2 = tacuda::detail::AffineCompose2<float>;

struct CmoSeed {
    int index;
    float gain;
    float loss;
};

__global__ void cmoSeedKernel(const float* input, const float* gains, const float* losses,
                              CmoSeed* seed, int period, int size) {
    const int first = tacuda::detail::leading_nan_count(input, size);
    if (period >= size - first) {
        *seed = {size, 0.0f, 0.0f};
        return;
    }
    double gain = 0.0, loss = 0.0;
    for (int i = first + 1; i <= first + period; ++i) {
        gain += gains[i];
        loss += losses[i];
    }
    *seed = {first + period, static_cast<float>(gain / period), static_cast<float>(loss / period)};
}

// Build per-step gains/losses in float
__global__ void cmoBuildGainLoss(const float* __restrict__ input,
                                 float* __restrict__ gains,
                                 float* __restrict__ losses,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx == 0) {
        gains[0] = 0.0f;
        losses[0] = 0.0f;
        return;
    }
    float diff = input[idx] - input[idx - 1];
    gains[idx]  = diff > 0.0f ? diff : 0.0f;
    losses[idx] = diff < 0.0f ? -diff : 0.0f;
}

// Build affine transforms for two-channel Wilder smoothing (same as RSI pattern)
__global__ void cmoBuildTransformsKernel(const float* __restrict__ gains,
                                          const float* __restrict__ losses,
                                          Affine2* __restrict__ transforms,
                                          int period, int size, const CmoSeed* seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float invP = 1.0f / static_cast<float>(period);
    Affine2 v;
    if (idx <= seed->index) {
        // Identity before and at seed index
        v.A = 1.0f; v.bG = 0.0f; v.bL = 0.0f;
    } else {
        float a = (static_cast<float>(period) - 1.0f) * invP;
        v.A  = a;
        v.bG = gains[idx] * invP;
        v.bL = losses[idx] * invP;
    }
    transforms[idx] = v;
}

__global__ void cmoFinalizeKernel(const Affine2* __restrict__ prefix,
                                   const CmoSeed* __restrict__ seed,
                                   float* __restrict__ output,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (idx < seed->index) {
        output[idx] = nanf("");
    } else {
        const float avgG0 = seed->gain;
        const float avgL0 = seed->loss;
        float avgG, avgL;
        if (idx == seed->index) {
            avgG = avgG0;
            avgL = avgL0;
        } else {
            Affine2 a = prefix[idx];
            avgG = a.A * avgG0 + a.bG;
            avgL = a.A * avgL0 + a.bL;
        }
        float denom = avgG + avgL;
        output[idx] = (denom == 0.0f) ? 0.0f : 100.0f * (avgG - avgL) / denom;
    }
}

} // namespace

tacuda::CMO::CMO(int period) : period(period) {}

void tacuda::CMO::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("CMO: invalid period");
    }
    if (period >= size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    auto gains = acquireDeviceBufferAsync<float>(size, stream);
    auto losses = acquireDeviceBufferAsync<float>(size, stream);
    auto seed = acquireDeviceBufferAsync<CmoSeed>(1, stream);
    auto transforms = acquireDeviceBufferAsync<Affine2>(size, stream);
    const dim3 block = defaultBlock();
    const dim3 grid = defaultGrid(size);
    cmoBuildGainLoss<<<grid, block, 0, stream>>>(input, gains.get(), losses.get(), size);
    CUDA_CHECK(cudaGetLastError());
    cmoSeedKernel<<<1, 1, 0, stream>>>(input, gains.get(), losses.get(), seed.get(), period, size);
    CUDA_CHECK(cudaGetLastError());
    cmoBuildTransformsKernel<<<grid, block, 0, stream>>>(
        gains.get(), losses.get(), transforms.get(), period, size, seed.get());
    CUDA_CHECK(cudaGetLastError());
    tacuda::detail::affine_inclusive_scan<Affine2, AffineCompose2>(
        transforms.get(), transforms.get(), size, stream);
    cmoFinalizeKernel<<<grid, block, 0, stream>>>(transforms.get(), seed.get(), output, size);
    CUDA_CHECK(cudaGetLastError());
}
