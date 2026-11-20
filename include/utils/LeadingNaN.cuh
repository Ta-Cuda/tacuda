/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <cuda_runtime.h>
#include <utils/CudaUtils.h>

namespace tacuda::detail {

template <typename T>
__device__ inline int leading_nan_count(const T* input, int length, int stride = 1) {
    int first = 0;
    while (first < length && isnan(input[static_cast<size_t>(first) * stride])) ++first;
    return first;
}

// Called only when the first sample of an otherwise usable extrema window
// is NaN. Preserve the existing extrema behavior after the series has started.
__device__ inline bool in_leading_nan_prefix(const float* input, int index) {
    if (!isnan(input[0])) return false;
    for (int i = index - 1; i > 0; --i) {
        if (!isnan(input[i])) return false;
    }
    return true;
}

template <typename T>
__global__ void leading_nan_count_kernel(const T* input, int length, int* result) {
    *result = leading_nan_count(input, length);
}

template <typename T>
inline void launch_leading_nan_count(const T* input, int length, int* result, cudaStream_t stream) {
    leading_nan_count_kernel<<<1, 1, 0, stream>>>(input, length, result);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace tacuda::detail
