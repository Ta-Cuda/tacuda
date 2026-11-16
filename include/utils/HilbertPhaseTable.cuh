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
extern __device__ __constant__ const double phaseSin[1275];
extern __device__ __constant__ const double phaseCos[1275];
}}
