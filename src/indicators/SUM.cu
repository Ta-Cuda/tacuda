/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/SUM.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/DeterministicScan.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <cfloat>
#include <math_constants.h>
#include <stdexcept>

namespace {

struct WindowSum {
    double sum;
    int nans;
    int positiveInf;
    int negativeInf;
};

struct ToWindowSum {
    __host__ __device__ WindowSum operator()(float x) const {
        if (x != x) return {0.0, 1, 0, 0};
        if (x > FLT_MAX) return {0.0, 0, 1, 0};
        if (x < -FLT_MAX) return {0.0, 0, 0, 1};
        return {x, 0, 0, 0};
    }
};

struct AddWindowSum {
    __host__ __device__ WindowSum operator()(const WindowSum& a, const WindowSum& b) const {
        return {a.sum + b.sum, a.nans + b.nans, a.positiveInf + b.positiveInf,
                a.negativeInf + b.negativeInf};
    }
};

__global__ void finishSum(const WindowSum* prefix, float* output, int period, int size) {
    const size_t end = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (end >= static_cast<size_t>(size)) return;
    if (end < static_cast<size_t>(period - 1)) {
        output[end] = __int_as_float(0xFFFFFFFF);
        return;
    }
    const WindowSum a = prefix[end];
    const WindowSum b = end >= static_cast<size_t>(period) ? prefix[end - period] : WindowSum{};
    const bool positive = a.positiveInf != b.positiveInf;
    const bool negative = a.negativeInf != b.negativeInf;
    if (a.nans != b.nans || (positive && negative)) output[end] = nanf("");
    else if (positive) output[end] = CUDART_INF_F;
    else if (negative) output[end] = -CUDART_INF_F;
    else output[end] = static_cast<float>(a.sum - b.sum);
}

}  // namespace

tacuda::SUM::SUM(int period) : period(period) {}

void tacuda::SUM::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) throw std::invalid_argument("SUM: invalid period");
    if (period > size) {
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    auto prefix = acquireDeviceBufferAsync<WindowSum>(size, stream);
    auto values = thrust::make_transform_iterator(input, ToWindowSum{});
    detail::deterministic_inclusive_scan<WindowSum>(
        values, prefix.get(), size, stream, AddWindowSum{}, WindowSum{});
    finishSum<<<defaultGrid(size), defaultBlock(), 0, stream>>>(prefix.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
