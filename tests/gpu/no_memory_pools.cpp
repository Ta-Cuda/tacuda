/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Linux-only CUDA capability shim for end-to-end compatibility tests.
// The GPU still executes kernels. Only stream-ordered allocation support is
// removed; accidental use of those APIs fails as it would on a legacy device.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

namespace {
thread_local cudaError_t pending = cudaSuccess;

template <typename T>
T real(const char* name) {
    void* p = dlsym(RTLD_NEXT, name);
    if (!p) {
        std::fprintf(stderr, "no-memory-pools shim: missing %s\n", name);
        std::abort();
    }
    return reinterpret_cast<T>(p);
}

cudaError_t unsupported(const char* api) {
    std::fprintf(stderr, "no-memory-pools shim: unexpected %s\n", api);
    pending = cudaErrorNotSupported;
    return pending;
}
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    static auto fn = real<decltype(&cudaDeviceGetAttribute)>("cudaDeviceGetAttribute");
    const cudaError_t status = fn(value, attr, device);
    if (status == cudaSuccess && attr == cudaDevAttrMemoryPoolsSupported) *value = 0;
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void**, size_t, cudaStream_t) {
    return unsupported("cudaMallocAsync");
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void**, size_t, cudaMemPool_t, cudaStream_t) {
    return unsupported("cudaMallocFromPoolAsync");
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
    if (pending != cudaSuccess) {
        const cudaError_t error = pending;
        pending = cudaSuccess;
        return error;
    }
    static auto fn = real<decltype(&cudaGetLastError)>("cudaGetLastError");
    return fn();
}

extern "C" cudaError_t CUDARTAPI cudaPeekAtLastError() {
    if (pending != cudaSuccess) return pending;
    static auto fn = real<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    return fn();
}
