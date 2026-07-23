/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/CMO.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <stdexcept>

namespace {

using Affine2 = tacuda::detail::AffineTransform2<float>;
using AffineCompose2 = tacuda::detail::AffineCompose2<float>;

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
                                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float invP = 1.0f / static_cast<float>(period);
    Affine2 v;
    if (idx <= period) {
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

// Compute seed averages from prefix sums and produce final CMO values
__global__ void cmoFinalizeKernel(const Affine2* __restrict__ prefix,
                                   const float* __restrict__ gainPrefix,
                                   const float* __restrict__ lossPrefix,
                                   float* __restrict__ output,
                                   int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (idx < period) {
        output[idx] = nanf("");
    } else {
        // Seed = average of gains/losses from indices [1..period]
        float avgG0 = (gainPrefix[period] - gainPrefix[0]) / static_cast<float>(period);
        float avgL0 = (lossPrefix[period] - lossPrefix[0]) / static_cast<float>(period);

        float avgG, avgL;
        if (idx == period) {
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
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // 1) Build gains/losses arrays
    auto gains = acquireDeviceBufferAsync<float>(size, stream);
    auto losses = acquireDeviceBufferAsync<float>(size, stream);
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        cmoBuildGainLoss<<<grid, block, 0, stream>>>(input, gains.get(), losses.get(), size);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2) Inclusive prefix sums for seed computation
    {
        thrust::device_ptr<float> gPtr(gains.get());
        thrust::device_ptr<float> lPtr(losses.get());
        // Make copies for prefix sums since we need originals for transforms
        auto gainPfx = acquireDeviceBufferAsync<float>(size, stream);
        auto lossPfx = acquireDeviceBufferAsync<float>(size, stream);
        thrust::inclusive_scan(thrust::cuda::par.on(stream), gPtr, gPtr + size,
                               thrust::device_pointer_cast(gainPfx.get()));
        thrust::inclusive_scan(thrust::cuda::par.on(stream), lPtr, lPtr + size,
                               thrust::device_pointer_cast(lossPfx.get()));

        // 3) Build affine transforms
        auto transforms = acquireDeviceBufferAsync<Affine2>(size, stream);
        auto prefix = acquireDeviceBufferAsync<Affine2>(size, stream);
        {
            dim3 block = defaultBlock();
            dim3 grid = defaultGrid(size);
            cmoBuildTransformsKernel<<<grid, block, 0, stream>>>(
                gains.get(), losses.get(), transforms.get(), period, size);
            CUDA_CHECK(cudaGetLastError());
        }

        // 4) Parallel affine scan
        tacuda::detail::affine_inclusive_scan<Affine2, AffineCompose2>(
            transforms.get(), prefix.get(), size, stream);

        // 5) Finalize CMO values
        {
            dim3 block = defaultBlock();
            dim3 grid = defaultGrid(size);
            cmoFinalizeKernel<<<grid, block, 0, stream>>>(
                prefix.get(), gainPfx.get(), lossPfx.get(), output, period, size);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}
