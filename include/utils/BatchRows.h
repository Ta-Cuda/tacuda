/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_BATCH_ROWS_H
#define TACUDA_BATCH_ROWS_H

/**
 * @file BatchRows.h
 * @brief Internal: whole-matrix launchers for legacy batch families that keep
 *        the family's single-series arithmetic.
 *
 * Every function processes `series` rows of `length` floats (series-major,
 * rows contiguous) and reproduces, bit for bit, what the family's
 * single-series device entry point writes for each row, in a fixed number of
 * launches instead of one launch sequence per row. Callers check the
 * parameters first: values these launchers do not accept keep the
 * per-series dispatcher, which owns the error reporting.
 *
 * The functions throw (CUDA_CHECK / std::invalid_argument) instead of
 * returning a status; the batch runner classifies the exception.
 */

#include <cuda_runtime.h>

#include <algorithm>

namespace tacuda {
namespace detail {

/// Launch shape of the one-thread-per-row kernels with uniform control flow
/// (the same steps for every row): 64 rows per block, rows grid-strided so
/// the grid is capped without limiting the number of rows.
constexpr int kRowThreads = 64;
inline int row_blocks(int series) {
    return std::max(1, std::min((series + kRowThreads - 1) / kRowThreads, 65535));
}

/// Launch shape of the one-thread-per-row kernels whose rows take
/// data-dependent branches (trend flips, window rescans). Rows sharing a warp
/// run their branches one after another, so while rows are few each row gets
/// a warp of its own (`spread` 32: only lane 0 takes a row); beyond that they
/// pack densely (`spread` 1). A warp with one active lane still occupies the
/// FP64 pipeline for a full warp, so state machines that compute in double
/// (`fp64`) stop at 6 rows per SM, where SAR measured even; the others go up
/// to the device's resident warps.
struct RowLaunch {
    int blocks;
    int threads;
    int spread;
};
RowLaunch divergent_row_launch(int series, bool fp64);

void midpoint_rows(const float* input, float* output, int length, int series,
                   int period, cudaStream_t stream);

#if defined(__CUDACC__)
/// First row and row step of the calling thread under `spread`; false when
/// the thread takes no row.
__device__ __forceinline__ bool row_slot(int spread, size_t& first, size_t& step) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid % static_cast<size_t>(spread) != 0) return false;
    first = tid / static_cast<size_t>(spread);
    step = static_cast<size_t>(gridDim.x) * blockDim.x / static_cast<size_t>(spread);
    return true;
}
#endif

/// Copies the first `count` elements of row 0 of every output matrix into
/// rows 1..series-1. Used after one single-series call over a flattened
/// matrix: those elements of the later rows were computed from the tail of
/// the previous row, while row 0 holds the family's data-independent warm-up
/// values (NaN with the family's own bit pattern, or 0).
void copy_row_prefix(float* const* outs, int nOut, int length, int series,
                     int count, cudaStream_t stream);

/// True when tacuda::SMA computes `period` with a direct window kernel
/// rather than a prefix-sum scan. Only then does one call over a flattened
/// matrix reproduce every row exactly (a scan would run across rows).
bool sma_direct_window(int period);

// One GPU thread per row runs the family's single-series state machine on
// its own row; outputs are prepared exactly like the single-series path.
void sar_rows(const float* high, const float* low, float* out, int length,
              int series, float acceleration, float maximum,
              cudaStream_t stream);
void sarext_rows(const float* high, const float* low, float* out, int length,
                 int series, float startValue, float offsetOnReverse,
                 float accInitLong, float accLong, float accMaxLong,
                 float accInitShort, float accShort, float accMaxShort,
                 cudaStream_t stream);
void sar_el_rows(const float* high, const float* low, const float* close,
                 float* out, int length, int series, float accFact,
                 float maxAccel, cudaStream_t stream);
void max_index_rows(const float* in, float* out, int length, int series,
                    int period, cudaStream_t stream);
void min_index_rows(const float* in, float* out, int length, int series,
                    int period, cudaStream_t stream);
void min_max_index_rows(const float* in, float* minOut, float* maxOut,
                        int length, int series, int period,
                        cudaStream_t stream);
void ht_dcperiod_rows(const float* in, float* out, int length, int series,
                      cudaStream_t stream);
void ht_dcphase_rows(const float* in, float* out, int length, int series,
                     cudaStream_t stream);
void ht_phasor_rows(const float* in, float* inPhase, float* quadrature,
                    int length, int series, cudaStream_t stream);
void ht_sine_rows(const float* in, float* sine, float* leadSine, int length,
                  int series, cudaStream_t stream);
void ht_trendline_rows(const float* in, float* out, int length, int series,
                       cudaStream_t stream);
void ht_trendmode_rows(const float* in, float* out, int length, int series,
                       cudaStream_t stream);
void mama_rows(const float* in, float* mama, float* fama, int length,
               int series, float fastLimit, float slowLimit,
               cudaStream_t stream);
void macdfix_rows(const float* in, float* macd, float* signal, float* hist,
                  int length, int series, int signalPeriod,
                  cudaStream_t stream);

}  // namespace detail
}  // namespace tacuda

#endif  // TACUDA_BATCH_ROWS_H
