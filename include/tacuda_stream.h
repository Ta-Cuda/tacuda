/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_STREAM_H
#define TACUDA_STREAM_H

/*
 * Streaming (incremental update) API — see docs/DESIGN_STREAMING_API.md.
 *
 * A stream carries the indicator state across calls, so appending a newly
 * closed bar costs O(1) (or O(period) for window families) instead of
 * recomputing the whole history. Appending bars one at a time reproduces
 * the tail of a full-series recompute: same values within the documented
 * tolerance, same NaN warm-up, same TA-Lib-aligned semantics.
 *
 * Single-series streams run on the CPU with double-precision state. A
 * one-bar update is a handful of floating-point operations; a GPU roundtrip
 * would pay two PCIe latencies plus a kernel launch to do the same work.
 * The GPU pays off for whole-portfolio appends (batch streams, planned for
 * the next milestone). Because the state is double while the full-series
 * kernels accumulate some recursions in float, stream values are at least
 * as accurate as the full-series API — they are not bit-identical to it.
 *
 * Handles are NOT thread-safe: use one stream per producer thread, or
 * synchronize externally. Distinct handles are fully independent.
 */

#include "tacuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle holding the incremental state of one indicator over
 * one series. Created by a `ct_<indicator>_stream_create` call, released by
 * ct_stream_destroy().
 */
typedef struct ctStream ctStream_t;

/* ---------------------------------------------------------------------------
 * Lifetime and introspection (valid for every stream kind)
 * ------------------------------------------------------------------------ */

/**
 * @brief Release a stream. Passing NULL is a no-op and returns success.
 * Using a handle after destroying it is rejected with
 * CT_STATUS_INVALID_ARGUMENT rather than causing undefined behavior.
 */
CTAPI_EXPORT ctStatus_t ct_stream_destroy(ctStream_t *stream);

/**
 * @brief Drop all accumulated state, keeping the parameters — equivalent to
 * destroying the handle and creating it again with the same arguments.
 */
CTAPI_EXPORT ctStatus_t ct_stream_reset(ctStream_t *stream);

/**
 * @brief Number of bars that must be appended before the first non-NaN
 * output appears, i.e. the TA-Lib lookback of the configured indicator.
 * The output for the bar at index `lookback` is the first real value.
 */
CTAPI_EXPORT ctStatus_t ct_stream_lookback(const ctStream_t *stream,
                                           int *outLookback);

/**
 * @brief Total number of bars appended since creation (or the last reset).
 */
CTAPI_EXPORT ctStatus_t ct_stream_bars(const ctStream_t *stream,
                                       long long *outBars);

/**
 * @brief Number of output lines the matching append call writes (1 for most
 * families, 3 for MACD and BBANDS).
 */
CTAPI_EXPORT ctStatus_t ct_stream_outputs(const ctStream_t *stream,
                                          int *outCount);

/* ---------------------------------------------------------------------------
 * Creation
 * ------------------------------------------------------------------------ */

CTAPI_EXPORT ctStatus_t ct_sma_stream_create(int period, ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_ema_stream_create(int period, ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_wma_stream_create(int period, ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_rsi_stream_create(int period, ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_atr_stream_create(int period, ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_natr_stream_create(int period, ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_obv_stream_create(ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_ad_stream_create(ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_macd_stream_create(int fastPeriod, int slowPeriod,
                                              int signalPeriod,
                                              ctStream_t **outStream);
CTAPI_EXPORT ctStatus_t ct_bbands_stream_create(int period, float upperMul,
                                                float lowerMul,
                                                ctStream_t **outStream);

/* ---------------------------------------------------------------------------
 * Append
 *
 * Every append consumes `n` closed bars (n >= 1) and writes exactly `n`
 * values per output line, aligned one-to-one with the appended bars. Bars
 * still inside the warm-up window produce NaN, exactly like the leading
 * NaNs of a full-series call. Warm-up "replay" is nothing special: append
 * the history in one call, then append each new bar as it closes.
 *
 * Inputs and outputs are host float32 buffers owned by the caller; nothing
 * is retained after the call returns.
 * ------------------------------------------------------------------------ */

CTAPI_EXPORT ctStatus_t ct_sma_stream_append(ctStream_t *stream,
                                             const float *close, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_ema_stream_append(ctStream_t *stream,
                                             const float *close, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_wma_stream_append(ctStream_t *stream,
                                             const float *close, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_rsi_stream_append(ctStream_t *stream,
                                             const float *close, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_atr_stream_append(ctStream_t *stream,
                                             const float *high, const float *low,
                                             const float *close, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_natr_stream_append(ctStream_t *stream,
                                              const float *high, const float *low,
                                              const float *close, int n,
                                              float *out);
CTAPI_EXPORT ctStatus_t ct_obv_stream_append(ctStream_t *stream,
                                             const float *close,
                                             const float *volume, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_ad_stream_append(ctStream_t *stream,
                                            const float *high, const float *low,
                                            const float *close,
                                            const float *volume, int n,
                                            float *out);
CTAPI_EXPORT ctStatus_t ct_macd_stream_append(ctStream_t *stream,
                                              const float *close, int n,
                                              float *outMacd, float *outSignal,
                                              float *outHist);
CTAPI_EXPORT ctStatus_t ct_bbands_stream_append(ctStream_t *stream,
                                                const float *close, int n,
                                                float *outUpper,
                                                float *outMiddle,
                                                float *outLower);

/* ---------------------------------------------------------------------------
 * Batch (portfolio) streams
 *
 * One handle carries the state of `numSeries` independent series of one
 * indicator, and one append call advances all of them. Numerically identical
 * to keeping numSeries separate streams — it is the same state machine — so
 * the choice is about call overhead and execution policy, not about results.
 *
 * Matrices are series-major, like the batch API: series `s`'s `n` bars live
 * at `[s * n .. s * n + n - 1]`, and outputs land at the same offsets.
 * ------------------------------------------------------------------------ */

typedef struct ctStreamBatch ctStreamBatch_t;

/**
 * @brief Where a batch append runs. The crossover is measured, not guessed —
 * see docs/STREAMING_API.md. A state update is tens of nanoseconds, so small
 * portfolios finish on one core long before a kernel launch plus two
 * transfers would return; large ones amortize that fixed cost and win on the
 * GPU. AUTO switches at the measured crossover (~2048 series).
 *
 * There is deliberately no library-managed thread pool: series are fully
 * independent, so a caller who wants CPU parallelism can shard the portfolio
 * across several batch handles and drive them from its own threads.
 */
typedef enum ctStreamExec {
  CT_STREAM_EXEC_AUTO = 0,
  CT_STREAM_EXEC_CPU = 1,
  CT_STREAM_EXEC_GPU = 2,
} ctStreamExec_t;

CTAPI_EXPORT ctStatus_t ct_stream_batch_destroy(ctStreamBatch_t *batch);
CTAPI_EXPORT ctStatus_t ct_stream_batch_reset(ctStreamBatch_t *batch);
CTAPI_EXPORT ctStatus_t ct_stream_batch_lookback(const ctStreamBatch_t *batch,
                                                 int *outLookback);
CTAPI_EXPORT ctStatus_t ct_stream_batch_series(const ctStreamBatch_t *batch,
                                               int *outSeries);
CTAPI_EXPORT ctStatus_t ct_stream_batch_outputs(const ctStreamBatch_t *batch,
                                                int *outCount);
CTAPI_EXPORT ctStatus_t ct_stream_batch_bars(const ctStreamBatch_t *batch,
                                             long long *outBars);
CTAPI_EXPORT ctStatus_t ct_stream_batch_set_exec(ctStreamBatch_t *batch,
                                                 ctStreamExec_t exec);

CTAPI_EXPORT ctStatus_t ct_sma_stream_batch_create(int period, int numSeries,
                                                   ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_ema_stream_batch_create(int period, int numSeries,
                                                   ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_wma_stream_batch_create(int period, int numSeries,
                                                   ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_rsi_stream_batch_create(int period, int numSeries,
                                                   ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_atr_stream_batch_create(int period, int numSeries,
                                                   ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_natr_stream_batch_create(int period, int numSeries,
                                                    ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_obv_stream_batch_create(int numSeries,
                                                   ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_ad_stream_batch_create(int numSeries,
                                                  ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_macd_stream_batch_create(int fastPeriod, int slowPeriod,
                                                    int signalPeriod, int numSeries,
                                                    ctStreamBatch_t **outBatch);
CTAPI_EXPORT ctStatus_t ct_bbands_stream_batch_create(int period, float upperMul,
                                                      float lowerMul, int numSeries,
                                                      ctStreamBatch_t **outBatch);

CTAPI_EXPORT ctStatus_t ct_sma_stream_batch_append(ctStreamBatch_t *batch,
                                                   const float *closes, int n,
                                                   float *outs);
CTAPI_EXPORT ctStatus_t ct_ema_stream_batch_append(ctStreamBatch_t *batch,
                                                   const float *closes, int n,
                                                   float *outs);
CTAPI_EXPORT ctStatus_t ct_wma_stream_batch_append(ctStreamBatch_t *batch,
                                                   const float *closes, int n,
                                                   float *outs);
CTAPI_EXPORT ctStatus_t ct_rsi_stream_batch_append(ctStreamBatch_t *batch,
                                                   const float *closes, int n,
                                                   float *outs);
CTAPI_EXPORT ctStatus_t ct_atr_stream_batch_append(ctStreamBatch_t *batch,
                                                   const float *highs,
                                                   const float *lows,
                                                   const float *closes, int n,
                                                   float *outs);
CTAPI_EXPORT ctStatus_t ct_natr_stream_batch_append(ctStreamBatch_t *batch,
                                                    const float *highs,
                                                    const float *lows,
                                                    const float *closes, int n,
                                                    float *outs);
CTAPI_EXPORT ctStatus_t ct_obv_stream_batch_append(ctStreamBatch_t *batch,
                                                   const float *closes,
                                                   const float *volumes, int n,
                                                   float *outs);
CTAPI_EXPORT ctStatus_t ct_ad_stream_batch_append(ctStreamBatch_t *batch,
                                                  const float *highs,
                                                  const float *lows,
                                                  const float *closes,
                                                  const float *volumes, int n,
                                                  float *outs);
CTAPI_EXPORT ctStatus_t ct_macd_stream_batch_append(ctStreamBatch_t *batch,
                                                    const float *closes, int n,
                                                    float *outMacd,
                                                    float *outSignal,
                                                    float *outHist);
CTAPI_EXPORT ctStatus_t ct_bbands_stream_batch_append(ctStreamBatch_t *batch,
                                                      const float *closes, int n,
                                                      float *outUpper,
                                                      float *outMiddle,
                                                      float *outLower);

#ifdef __cplusplus
}
#endif

#endif /* TACUDA_STREAM_H */
