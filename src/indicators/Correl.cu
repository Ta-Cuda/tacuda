/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/Correl.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <algorithm>
#include <stdexcept>
#include <math.h>

namespace {
__global__ void findInvalidSuffix(const float* x, const float* y, int* suffix, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (!isfinite(x[i]) || !isfinite(y[i]))) atomicMax(suffix, n - i);
}

// Rebase sums at the first sample of each short run of windows. No sum
// contains the entire history, and no arithmetic order depends on scheduling.
// Reseeding also bounds roundoff from sliding updates. For large periods a
// thread emits more windows, amortizing the initial O(period) accumulation.
__global__ void correlWindows(const float* x, const float* y, float* output,
                               const int* invalidSuffix, int n, int period,
                               int windowsPerThread) {
    const size_t first = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x)
                         * windowsPerThread + period - 1;
    if (first >= static_cast<size_t>(n)) return;
    const int begin = static_cast<int>(first);
    const int end = begin + min(windowsPerThread, n - begin);
    const int firstInvalid = n - *invalidSuffix;
    // Preserve the original path's non-finite prefix behavior: subsequent
    // published correlations are zero, without contaminating another row.
    if (begin >= firstInvalid) {
        for (int i = begin; i < end; ++i) output[i] = 0.0f;
        return;
    }
    const double originX = x[begin - period + 1];
    const double originY = y[begin - period + 1];
    double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = begin - period + 1; i <= begin; ++i) {
        const double dx = static_cast<double>(x[i]) - originX;
        const double dy = static_cast<double>(y[i]) - originY;
        sx += dx; sy += dy;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    const double p = static_cast<double>(period);
    for (int i = begin; i < end; ++i) {
        if (i >= firstInvalid) {
            output[i] = 0.0f;
            continue;
        }
        const double cov = sxy - sx * sy / p;
        const double vx = sxx - sx * sx / p;
        const double vy = syy - sy * sy / p;
        const double product = vx * vy;
        output[i] = vx > 0.0 && vy > 0.0 && product > 1e-14
                      ? static_cast<float>(cov / sqrt(product)) : 0.0f;
        if (i + 1 < end && i + 1 < firstInvalid) {
            const double oldX = static_cast<double>(x[i - period + 1]) - originX;
            const double oldY = static_cast<double>(y[i - period + 1]) - originY;
            const double newX = static_cast<double>(x[i + 1]) - originX;
            const double newY = static_cast<double>(y[i + 1]) - originY;
            sx += newX - oldX; sy += newY - oldY;
            sxx += newX * newX - oldX * oldX;
            syy += newY * newY - oldY * oldY;
            sxy += newX * newY - oldX * oldY;
        }
    }
}
}

tacuda::Correl::Correl(int period) : period(period) {}

void tacuda::Correl::calculate(const float* x, const float* y, float* output,
                               int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) throw std::invalid_argument("Correl: invalid period");
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
    if (period > size) return;
    auto invalidSuffix = acquireDeviceBufferAsync<int>(1, stream);
    CUDA_CHECK(cudaMemsetAsync(invalidSuffix.get(), 0, sizeof(int), stream));
    findInvalidSuffix<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        x, y, invalidSuffix.get(), size);
    CUDA_CHECK(cudaGetLastError());
    // Depend on the period only: appending future bars must not change the
    // summation order of an existing window.
    const int run = std::max(1, std::min(4096, period / 16));
    const int windows = size - period + 1;
    const int threads = static_cast<int>((static_cast<size_t>(windows) + run - 1) / run);
    correlWindows<<<defaultGrid(threads), defaultBlock(), 0, stream>>>(
        x, y, output, invalidSuffix.get(), size, period, run);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::Correl::calculate(const float* input, float* output, int size,
                               cudaStream_t stream) noexcept(false) {
    calculate(input, input + size, output, size, stream);
}
