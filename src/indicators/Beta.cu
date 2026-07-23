/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/Beta.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <stdexcept>
#include <math.h>

namespace {
__global__ void betaReturnsKernel(const float* __restrict__ x,
                                  const float* __restrict__ y,
                                  double* __restrict__ retX,
                                  double* __restrict__ retY,
                                  double* __restrict__ retXX,
                                  double* __restrict__ retXY,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    constexpr double eps = 1.0e-14;
    if (idx == 0) {
        retX[0] = 0.0;
        retY[0] = 0.0;
        retXX[0] = 0.0;
        retXY[0] = 0.0;
        return;
    }

    double prevX = static_cast<double>(x[idx - 1]);
    double prevY = static_cast<double>(y[idx - 1]);
    double curX = static_cast<double>(x[idx]);
    double curY = static_cast<double>(y[idx]);

    double dx = (fabs(prevX) < eps) ? 0.0 : (curX - prevX) / prevX;
    double dy = (fabs(prevY) < eps) ? 0.0 : (curY - prevY) / prevY;

    retX[idx] = dx;
    retY[idx] = dy;
    retXX[idx] = dx * dx;
    retXY[idx] = dx * dy;
}

__global__ void betaFromPrefixesKernel(const double* __restrict__ prefixX,
                                       const double* __restrict__ prefixY,
                                       const double* __restrict__ prefixXX,
                                       const double* __restrict__ prefixXY,
                                       float* __restrict__ output,
                                       int period,
                                       int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx < period) return; // First valid value at index == period

    int start = idx - period + 1;
    double prev = (start > 0) ? prefixX[start - 1] : 0.0;
    double sumX = prefixX[idx] - prev;

    prev = (start > 0) ? prefixY[start - 1] : 0.0;
    double sumY = prefixY[idx] - prev;

    prev = (start > 0) ? prefixXX[start - 1] : 0.0;
    double sumXX = prefixXX[idx] - prev;

    prev = (start > 0) ? prefixXY[start - 1] : 0.0;
    double sumXY = prefixXY[idx] - prev;

    const double n = static_cast<double>(period);
    const double denom = n * sumXX - sumX * sumX;
    const double numer = n * sumXY - sumX * sumY;
    output[idx] = static_cast<float>((fabs(denom) < 1.0e-14) ? 0.0 : (numer / denom));
}
} // namespace

tacuda::Beta::Beta(int period) : period(period) {}

void tacuda::Beta::calculate(const float* x, const float* y, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("Beta: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);

    auto retX = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);
    auto retY = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);
    auto retXX = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);
    auto retXY = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);

    betaReturnsKernel<<<grid, block, 0, stream>>>(x, y, retX.get(), retY.get(), retXX.get(), retXY.get(), size);
    CUDA_CHECK(cudaGetLastError());

    auto scanInPlace = [stream](double* ptr, int n) {
        thrust::device_ptr<double> dev(ptr);
        thrust::inclusive_scan(thrust::cuda::par.on(stream), dev, dev + n, dev);
    };

    scanInPlace(retX.get(), size);
    scanInPlace(retY.get(), size);
    scanInPlace(retXX.get(), size);
    scanInPlace(retXY.get(), size);

    betaFromPrefixesKernel<<<grid, block, 0, stream>>>(
        retX.get(), retY.get(), retXX.get(), retXY.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::Beta::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* x = input;
    const float* y = input + size;
    calculate(x, y, output, size, stream);
}

