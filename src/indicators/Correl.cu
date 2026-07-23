/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/Correl.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <stdexcept>
#include <math.h>

// Build double-precision helper arrays
__global__ void makeDouble(const float* __restrict__ in, double* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = static_cast<double>(in[idx]);
}

__global__ void elementwiseMulDouble(const double* __restrict__ a,
                                     const double* __restrict__ b,
                                     double* __restrict__ out,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void squareDouble(const double* __restrict__ in,
                             double* __restrict__ out,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double v = in[idx];
        out[idx] = v * v;
    }
}

// Compute rolling Pearson correlation using double-precision prefix sums
__global__ void correlFromPrefixesD(const double* __restrict__ preX,
                                    const double* __restrict__ preY,
                                    const double* __restrict__ preX2,
                                    const double* __restrict__ preY2,
                                    const double* __restrict__ preXY,
                                    float* __restrict__ out,
                                    int period, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (idx < period - 1) return; // warmup stays NaN
    const int start = idx - period + 1;
    const double sumX  = preX[idx]  - (start > 0 ? preX[start - 1]  : 0.0);
    const double sumY  = preY[idx]  - (start > 0 ? preY[start - 1]  : 0.0);
    const double sumX2 = preX2[idx] - (start > 0 ? preX2[start - 1] : 0.0);
    const double sumY2 = preY2[idx] - (start > 0 ? preY2[start - 1] : 0.0);
    const double sumXY = preXY[idx] - (start > 0 ? preXY[start - 1] : 0.0);
    const double p = static_cast<double>(period);
    const double meanX = sumX / p;
    const double meanY = sumY / p;
    const double cov = sumXY - p * meanX * meanY;
    const double varX = sumX2 - p * meanX * meanX;
    const double varY = sumY2 - p * meanY * meanY;
    // TA-Lib emits 0 for a degenerate window (TA_IS_ZERO_OR_NEG on the
    // variance product). Prefix-sum cancellation can leave a tiny positive
    // residue on constant windows, so variances below 1e-10 of the raw
    // power are treated as zero as well.
    const bool degenerate = !(varX * varY > 1e-14) ||
                            varX <= 1e-10 * sumX2 || varY <= 1e-10 * sumY2;
    out[idx] = degenerate ? 0.0f : static_cast<float>(cov / sqrt(varX * varY));
}

tacuda::Correl::Correl(int period) : period(period) {}

void tacuda::Correl::calculate(const float* x, const float* y, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("Correl: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // Allocate double buffers and prefixes
    auto xd = acquireDeviceBufferAsync<double>(size, stream);
    auto yd = acquireDeviceBufferAsync<double>(size, stream);
    auto x2 = acquireDeviceBufferAsync<double>(size, stream);
    auto y2 = acquireDeviceBufferAsync<double>(size, stream);
    auto xy = acquireDeviceBufferAsync<double>(size, stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    makeDouble<<<grid, block, 0, stream>>>(x, xd.get(), size);
    CUDA_CHECK(cudaGetLastError());
    makeDouble<<<grid, block, 0, stream>>>(y, yd.get(), size);
    CUDA_CHECK(cudaGetLastError());
    squareDouble<<<grid, block, 0, stream>>>(xd.get(), x2.get(), size);
    CUDA_CHECK(cudaGetLastError());
    squareDouble<<<grid, block, 0, stream>>>(yd.get(), y2.get(), size);
    CUDA_CHECK(cudaGetLastError());
    elementwiseMulDouble<<<grid, block, 0, stream>>>(xd.get(), yd.get(), xy.get(), size);
    CUDA_CHECK(cudaGetLastError());

    // Inclusive scans to get prefixes
    thrust::device_ptr<double> px(xd.get());
    thrust::device_ptr<double> py(yd.get());
    thrust::device_ptr<double> px2(x2.get());
    thrust::device_ptr<double> py2(y2.get());
    thrust::device_ptr<double> pxy(xy.get());
    thrust::inclusive_scan(thrust::cuda::par.on(stream), px, px + size, px);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), py, py + size, py);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), px2, px2 + size, px2);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), py2, py2 + size, py2);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), pxy, pxy + size, pxy);

    // Final rolling correlation
    correlFromPrefixesD<<<grid, block, 0, stream>>>(px.get(), py.get(), px2.get(), py2.get(), pxy.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::Correl::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* x = input;
    const float* y = input + size;
    calculate(x, y, output, size, stream);
}

