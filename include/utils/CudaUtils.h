/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
  #define CTAPI_EXPORT __declspec(dllexport)
#else
  #define CTAPI_EXPORT __attribute__((visibility("default")))
#endif

#define CUDA_CHECK(stmt)                                                         \
    do {                                                                         \
        cudaError_t err__ = (stmt);                                              \
        if (err__ != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(err__) +                 \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                        \
    } while (0)

inline dim3 defaultBlock() { return dim3(256,1,1); }
inline dim3 defaultGrid(int n) { return dim3((n + defaultBlock().x - 1) / defaultBlock().x,1,1); }

#endif
