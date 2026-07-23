/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <cuda_runtime.h>

namespace tacuda::detail {

/**
 * @brief Computes Wilder's smoothing for a given input.
 * 
 * Wilder's smoothing is a type of EMA where alpha = 1 / period.
 * This implementation uses a parallel prefix scan (O(log N)) to achieve high performance on GPU.
 * 
 * @param input   Input data pointer (device)
 * @param output  Output data pointer (device)
 * @param size    Number of elements in the input
 * @param period  The period for smoothing
 * @param stream  CUDA stream to use
 */
void computeWildersSmoothing(const float* input,
                             float* output,
                             int size,
                             int period,
                             cudaStream_t stream);

/**
 * @brief Computes TA-Lib-compatible DM smoothing (used by PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI).
 * 
 * Unlike standard Wilder EMA, TA-Lib's DM smoothing:
 *   - Seeds with SUM of input[1..period-1] at index (period-1)
 *   - Recursion: smoothed[i] = smoothed[i-1] * (period-1)/period + input[i]
 *     (note: input[i] is NOT divided by period)
 *   - NaN for indices 0..period-2
 * 
 * @param input   Raw DM values (device), with input[0] = 0 by convention
 * @param output  Output data pointer (device), caller must pre-fill with NaN
 * @param size    Number of elements
 * @param period  The period for smoothing
 * @param stream  CUDA stream to use
 */
void computeDMSmoothing(const float* input,
                        float* output,
                        int size,
                        int period,
                        cudaStream_t stream);

} // namespace tacuda::detail

