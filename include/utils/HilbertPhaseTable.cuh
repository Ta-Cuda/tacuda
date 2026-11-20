/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once
#include <cuda_runtime.h>

namespace tacuda { namespace hilbert {
// Triangular layout: period N starts at N*(N-1)/2, with N samples.
// smoothPeriod is bounded by 50; zero/negative periods never enter the loop.
// The build generates both copies from the same hexfloat literals, so the host
// and the device read identical bits. Host results then do not depend on which
// sin/cos variant the runtime math library dispatches to (glibc selects FMA
// code on capable CPUs; the MSVC CRT has _set_FMA3_enable).
extern __device__ __constant__ const double phaseSin[1275];
extern __device__ __constant__ const double phaseCos[1275];
extern const double phaseSinHost[1275];
extern const double phaseCosHost[1275];

__host__ __device__ __forceinline__ double phase_sin(int weight) {
#if defined(__CUDA_ARCH__)
    return phaseSin[weight];
#else
    return phaseSinHost[weight];
#endif
}

__host__ __device__ __forceinline__ double phase_cos(int weight) {
#if defined(__CUDA_ARCH__)
    return phaseCos[weight];
#else
    return phaseCosHost[weight];
#endif
}
}}
