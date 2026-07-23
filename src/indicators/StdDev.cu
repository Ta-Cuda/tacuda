/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/StdDev.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
/* keep thrust/scan include order aligned with other files */
#include <thrust/scan.h>
#include <stdexcept>
#include <math.h>

// Copy floats to doubles for accurate prefix sums
__global__ void stddevCastToDoubleKernel(const float* __restrict__ input,
                                         double* __restrict__ out,
                                         int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = static_cast<double>(input[idx]);
    }
}

// Build squares in double (unique name to avoid duplicate device symbols)
__global__ void stddevSquareToDoubleKernel(const float* __restrict__ input,
                                           double* __restrict__ outSq,
                                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const double v = static_cast<double>(input[idx]);
        outSq[idx] = v * v;
    }
}

// Compute rolling standard deviation using double-precision prefix sums
// (population stddev): sqrt(E[x^2] - (E[x])^2)
__global__ void stddevFromPrefixesD(const double* __restrict__ prefix,
                                    const double* __restrict__ prefixSq,
                                    float* __restrict__ output,
                                    int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx < period - 1) return; // keep NaN warmup
    const double prevSum = (idx == period - 1) ? 0.0 : prefix[idx - period];
    const double sum = prefix[idx] - prevSum;
    const double prevSumSq = (idx == period - 1) ? 0.0 : prefixSq[idx - period];
    const double sumSq = prefixSq[idx] - prevSumSq;
    const double p = static_cast<double>(period);
    const double mean = sum / p;
    double variance = (sumSq / p) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    output[idx] = static_cast<float>(sqrt(variance));
}

tacuda::StdDev::StdDev(int period) : period(period) {}

void tacuda::StdDev::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("StdDev: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    // Initialize warmup region as NaN
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // Prepare double prefixes for sum and sum of squares
    auto prefix = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);
    auto prefixSq = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);

    // Inclusive scan of input (float -> double)
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        stddevCastToDoubleKernel<<<grid, block, 0, stream>>>(input, prefix.get(), size);
        CUDA_CHECK(cudaGetLastError());
        thrust::device_ptr<double> prePtr(prefix.get());
        thrust::inclusive_scan(thrust::cuda::par.on(stream), prePtr, prePtr + size, prePtr);
    }

    // Build squares in double then inclusive scan in-place
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        stddevSquareToDoubleKernel<<<grid, block, 0, stream>>>(input, prefixSq.get(), size);
        CUDA_CHECK(cudaGetLastError());
        thrust::device_ptr<double> preSqPtr(prefixSq.get());
        thrust::inclusive_scan(thrust::cuda::par.on(stream), preSqPtr, preSqPtr + size, preSqPtr);
    }

    // Compute rolling stddev from prefixes
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        stddevFromPrefixesD<<<grid, block, 0, stream>>>(prefix.get(), prefixSq.get(), output, period, size);
        CUDA_CHECK(cudaGetLastError());
    }
}
