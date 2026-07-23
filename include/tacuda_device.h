/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_DEVICE_H
#define TACUDA_DEVICE_H

/**
 * @file tacuda_device.h
 * @brief Device-pointer API for tacuda indicators
 * 
 * These functions work directly with GPU memory pointers, eliminating
 * host<->device transfer overhead. Use these when:
 * 
 * 1. Data is already on the GPU
 * 2. Computing multiple indicators on the same data
 * 3. Building indicator pipelines
 * 
 * Memory Management:
 * - Use ct_device_alloc() to allocate GPU buffers
 * - Use ct_device_free() to release GPU buffers
 * - Use ct_device_upload() to copy host->device
 * - Use ct_device_download() to copy device->host
 * 
 * Example:
 * @code
 *   float* d_close = ct_device_alloc(size);
 *   float* d_sma = ct_device_alloc(size);
 *   float* d_ema = ct_device_alloc(size);
 *   float* d_rsi = ct_device_alloc(size);
 *   
 *   ct_device_upload(d_close, host_close, size, stream);
 *   
 *   // All calculations on GPU - no intermediate transfers!
 *   ct_sma_d(d_close, d_sma, size, 20, stream);
 *   ct_ema_d(d_close, d_ema, size, 12, stream);
 *   ct_rsi_d(d_close, d_rsi, size, 14, stream);
 *   
 *   // Download only what you need
 *   ct_device_download(host_rsi, d_rsi, size, stream);
 *   
 *   ct_device_free(d_close);
 *   ct_device_free(d_sma);
 *   ct_device_free(d_ema);
 *   ct_device_free(d_rsi);
 * @endcode
 */

#include <cuda_runtime.h>
#include "tacuda.h"

// tacuda.h #undefs CTAPI_STREAM_DEFAULT at end-of-file, so re-define it here.
// NVCC can reject default arguments on extern "C" declarations, so strip the
// default when compiling CUDA translation units while leaving it for C++.
#ifndef CTAPI_STREAM_DEFAULT
#  ifdef __cplusplus
#    if !defined(__CUDACC__)
#      define CTAPI_STREAM_DEFAULT = 0
#    else
#      define CTAPI_STREAM_DEFAULT
#    endif
#  else
#    define CTAPI_STREAM_DEFAULT
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Memory Management
// =============================================================================

/**
 * @brief Allocate GPU memory buffer
 * @param count Number of float elements
 * @return Device pointer or NULL on failure
 */
CTAPI_EXPORT float* ct_device_alloc(int count);
CTAPI_EXPORT void* ct_device_alloc_ex(int count, ctDataType_t storageType);

/**
 * @brief Free GPU memory buffer
 * @param d_ptr Device pointer (safe to pass NULL)
 */
CTAPI_EXPORT void ct_device_free(float* d_ptr);
CTAPI_EXPORT void ct_device_free_ex(void* d_ptr);

/**
 * @brief Copy data from host to device
 * @param d_dst Device destination pointer
 * @param h_src Host source pointer
 * @param count Number of float elements
 * @param stream CUDA stream (0 for default)
 * @return CT_STATUS_SUCCESS or error code
 */
CTAPI_EXPORT ctStatus_t ct_device_upload(float* d_dst, const float* h_src,
                                         int count, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_device_upload_ex(void* d_dst, const void* h_src,
                                            int count, ctDataType_t storageType,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);

/**
 * @brief Copy data from device to host
 * @param h_dst Host destination pointer
 * @param d_src Device source pointer
 * @param count Number of float elements
 * @param stream CUDA stream (0 for default)
 * @return CT_STATUS_SUCCESS or error code
 */
CTAPI_EXPORT ctStatus_t ct_device_download(float* h_dst, const float* d_src,
                                           int count, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_device_download_ex(void* h_dst, const void* d_src,
                                              int count, ctDataType_t storageType,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);

/**
 * @brief Synchronize CUDA stream
 * @param stream CUDA stream (0 for default)
 * @return CT_STATUS_SUCCESS or error code
 */
CTAPI_EXPORT ctStatus_t ct_device_sync(cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Moving Averages (Device API) - suffix _d means device pointers
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_sma_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sma_ex_d(const void* d_input, void* d_output,
                                    int size, int period, ctDataType_t storageType,
                                    ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_ema_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ema_ex_d(const void* d_input, void* d_output,
                                    int size, int period, ctDataType_t storageType,
                                    ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_wma_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_dema_d(const float* d_input, float* d_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_tema_d(const float* d_input, float* d_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_trima_d(const float* d_input, float* d_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_kama_d(const float* d_input, float* d_output,
                                   int size, int period, int fastPeriod,
                                   int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_t3_d(const float* d_input, float* d_output,
                                 int size, int period, float vFactor, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Momentum Indicators (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_rsi_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rsi_ex_d(const void* d_input, void* d_output,
                                    int size, int period, ctDataType_t storageType,
                                    ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_momentum_d(const float* d_input, float* d_output,
                                       int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_roc_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_rocp_d(const float* d_input, float* d_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_rocr_d(const float* d_input, float* d_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_rocr100_d(const float* d_input, float* d_output,
                                      int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_cmo_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Volatility Indicators (Device API) - HLC input
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_atr_d(const float* d_high, const float* d_low,
                                  const float* d_close, float* d_output,
                                  int size, int period, float percentile,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_atr_ex_d(const void* d_high, const void* d_low,
                                    const void* d_close, void* d_output,
                                    int size, int period, float percentile,
                                    ctDataType_t storageType, ctDataType_t computeType,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_natr_d(const float* d_high, const float* d_low,
                                   const float* d_close, float* d_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_trange_d(const float* d_high, const float* d_low,
                                     const float* d_close, float* d_output,
                                     int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Trend Indicators (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_adx_d(const float* d_high, const float* d_low,
                                  const float* d_close, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_adxr_d(const float* d_high, const float* d_low,
                                   const float* d_close, float* d_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_dx_d(const float* d_high, const float* d_low,
                                 const float* d_close, float* d_output,
                                 int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_plus_di_d(const float* d_high, const float* d_low,
                                      const float* d_close, float* d_output,
                                      int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_minus_di_d(const float* d_high, const float* d_low,
                                       const float* d_close, float* d_output,
                                       int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Oscillators (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_macd_d(const float* d_input, float* d_macd,
                                   float* d_signal, float* d_hist,
                                   int size, int fastPeriod, int slowPeriod,
                                   int signalPeriod, int maType,
                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_ex_d(const void* d_input, void* d_macd,
                                     void* d_signal, void* d_hist,
                                     int size, int fastPeriod, int slowPeriod,
                                     int signalPeriod, int maType,
                                     ctDataType_t storageType, ctDataType_t computeType,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_stochastic_d(const float* d_high, const float* d_low,
                                         const float* d_close, float* d_slowK,
                                         float* d_slowD, int size, int kPeriod,
                                         int dPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_stochastic_fast_d(const float* d_high, const float* d_low,
                                              const float* d_close, float* d_fastK,
                                              float* d_fastD, int size, int kPeriod,
                                              int dPeriod, int dMaType,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_willr_d(const float* d_high, const float* d_low,
                                    const float* d_close, float* d_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_cci_d(const float* d_high, const float* d_low,
                                  const float* d_close, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_ultosc_d(const float* d_high, const float* d_low,
                                     const float* d_close, float* d_output,
                                     int size, int period1, int period2, int period3,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Bands (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_bbands_d(const float* d_input, float* d_upper,
                                     float* d_middle, float* d_lower,
                                     int size, int period, float nbDevUp,
                                     float nbDevDn, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands_ex_d(const void* d_input, void* d_upper,
                                       void* d_middle, void* d_lower,
                                       int size, int period, float nbDevUp,
                                       float nbDevDn, ctDataType_t storageType,
                                       ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_accbands_d(const float* d_high, const float* d_low,
                                       const float* d_close, float* d_upper,
                                       float* d_middle, float* d_lower,
                                       int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Volume Indicators (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_ad_d(const float* d_high, const float* d_low,
                                 const float* d_close, const float* d_volume,
                                 float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_adosc_d(const float* d_high, const float* d_low,
                                    const float* d_close, const float* d_volume,
                                    float* d_output, int size, int fastPeriod,
                                    int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_obv_d(const float* d_close, const float* d_volume,
                                  float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_mfi_d(const float* d_high, const float* d_low,
                                  const float* d_close, const float* d_volume,
                                  float* d_output, int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Statistical (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_stddev_d(const float* d_input, float* d_output,
                                     int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_var_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_sum_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_max_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_min_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Price Transform (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_avgprice_d(const float* d_open, const float* d_high,
                                       const float* d_low, const float* d_close,
                                       float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_medprice_d(const float* d_high, const float* d_low,
                                       float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_typprice_d(const float* d_high, const float* d_low,
                                       const float* d_close, float* d_output,
                                       int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_wclprice_d(const float* d_high, const float* d_low,
                                       const float* d_close, float* d_output,
                                       int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Linear Regression (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_linearreg_d(const float* d_input, float* d_output,
                                        int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_linearreg_slope_d(const float* d_input, float* d_output,
                                              int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_linearreg_intercept_d(const float* d_input, float* d_output,
                                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_linearreg_angle_d(const float* d_input, float* d_output,
                                              int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_tsf_d(const float* d_input, float* d_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Aroon (Device API)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_aroon_d(const float* d_high, const float* d_low,
                                    float* d_aroonDown, float* d_aroonUp,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);

CTAPI_EXPORT ctStatus_t ct_aroonosc_d(const float* d_high, const float* d_low,
                                       float* d_output, int size, int period,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Oscillators (Device API, continued)
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_macdext_d(const float* d_input, float* d_macd,
                                     float* d_signal, float* d_hist,
                                     int size, int fastPeriod, int fastType,
                                     int slowPeriod, int slowType,
                                     int signalPeriod, int signalType,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Vector Math (Device API) - element-wise, no lookback
// =============================================================================

CTAPI_EXPORT ctStatus_t ct_acos_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asin_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_atan_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ceil_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cos_d(const float* d_input, float* d_output,
                                 int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cosh_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_exp_d(const float* d_input, float* d_output,
                                 int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_floor_d(const float* d_input, float* d_output,
                                   int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ln_d(const float* d_input, float* d_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_log10_d(const float* d_input, float* d_output,
                                   int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sin_d(const float* d_input, float* d_output,
                                 int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sinh_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sqrt_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tan_d(const float* d_input, float* d_output,
                                 int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tanh_d(const float* d_input, float* d_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_add_d(const float* d_input1, const float* d_input2,
                                 float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sub_d(const float* d_input1, const float* d_input2,
                                 float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mult_d(const float* d_input1, const float* d_input2,
                                  float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_div_d(const float* d_input1, const float* d_input2,
                                 float* d_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

/* --- BEGIN GENERATED DEVICE API (scripts/generate_batch_api.py) --- */
/* Generated device entry points (thin wrappers over the CUDA indicator
 * classes; see scripts/generate_batch_api.py). Do not edit by hand. */
CTAPI_EXPORT ctStatus_t ct_trix_d(const float* d_input, float* d_output,
                                  int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_change_d(const float* d_input, float* d_output,
                                    int size, int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midpoint_d(const float* d_input, float* d_output,
                                      int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_avgdev_d(const float* d_input, float* d_output,
                                    int size, int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_maxindex_d(const float* d_input, float* d_output,
                                      int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minindex_d(const float* d_input, float* d_output,
                                      int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_apo_d(const float* d_input, float* d_output,
                                 int size, int fastPeriod, int slowPeriod,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo_d(const float* d_input, float* d_output,
                                 int size, int fastPeriod, int slowPeriod,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo_d(const float* d_volume, float* d_output,
                                 int size, int fastPeriod, int slowPeriod,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ma_d(const float* d_input, float* d_output,
                                int size, int period, int maType,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcperiod_d(const float* d_input, float* d_output,
                                         int size,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcphase_d(const float* d_input, float* d_output,
                                        int size,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendline_d(const float* d_input,
                                          float* d_output, int size,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendmode_d(const float* d_input,
                                          float* d_output, int size,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochrsi_d(const float* d_input, float* d_fast_k,
                                      float* d_fast_d, int size, int rsiPeriod,
                                      int kPeriod, int dPeriod,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macdfix_d(const float* d_input, float* d_macd,
                                     float* d_signal, float* d_hist, int size,
                                     int signalPeriod,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mama_d(const float* d_input, float* d_mama,
                                  float* d_fama, int size, float fastLimit,
                                  float slowLimit,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmax_d(const float* d_input, float* d_min,
                                    float* d_max, int size, int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmaxindex_d(const float* d_input, float* d_minidx,
                                         float* d_maxidx, int size, int period,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_phasor_d(const float* d_input, float* d_inphase,
                                       float* d_quadrature, int size,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_sine_d(const float* d_input, float* d_sine,
                                     float* d_leadsine, int size,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_line_d(const float* d_input, float* d_output,
                                       int size, int fastPeriod,
                                       int slowPeriod,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midprice_d(const float* d_high, const float* d_low,
                                      float* d_output, int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_plus_dm_d(const float* d_high, const float* d_low,
                                     float* d_output, int size, int period,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minus_dm_d(const float* d_high, const float* d_low,
                                      float* d_output, int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar_d(const float* d_high, const float* d_low,
                                 float* d_output, int size, float step,
                                 float maxAcceleration,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sarext_d(const float* d_high, const float* d_low,
                                    float* d_output, int size,
                                    float startValue, float offsetOnReverse,
                                    float accInitLong, float accLong,
                                    float accMaxLong, float accInitShort,
                                    float accShort, float accMaxShort,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_imi_d(const float* d_open, const float* d_close,
                                 float* d_output, int size, int period,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_nvi_d(const float* d_close, const float* d_volume,
                                 float* d_output, int size,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvi_d(const float* d_close, const float* d_volume,
                                 float* d_output, int size,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_correl_d(const float* d_x, const float* d_y,
                                    float* d_output, int size, int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_beta_d(const float* d_x, const float* d_y,
                                  float* d_output, int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mavp_d(const float* d_values,
                                  const float* d_periods, float* d_output,
                                  int size, int minPeriod, int maxPeriod,
                                  int maType,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bop_d(const float* d_open, const float* d_high,
                                 const float* d_low, const float* d_close,
                                 float* d_output, int size,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_d(const float* d_open, const float* d_high,
                                      const float* d_low, const float* d_close,
                                      float* d_output, int size,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hammer_d(const float* d_open,
                                        const float* d_high,
                                        const float* d_low,
                                        const float* d_close, float* d_output,
                                        int size,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_inverted_hammer_d(const float* d_open,
                                                 const float* d_high,
                                                 const float* d_low,
                                                 const float* d_close,
                                                 float* d_output, int size,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bullish_engulfing_d(const float* d_open,
                                                   const float* d_high,
                                                   const float* d_low,
                                                   const float* d_close,
                                                   float* d_output, int size,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bearish_engulfing_d(const float* d_open,
                                                   const float* d_high,
                                                   const float* d_low,
                                                   const float* d_close,
                                                   float* d_output, int size,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_white_soldiers_d(const float* d_open,
                                                      const float* d_high,
                                                      const float* d_low,
                                                      const float* d_close,
                                                      float* d_output,
                                                      int size,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_abandoned_baby_d(const float* d_open,
                                                const float* d_high,
                                                const float* d_low,
                                                const float* d_close,
                                                float* d_output, int size,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_advance_block_d(const float* d_open,
                                               const float* d_high,
                                               const float* d_low,
                                               const float* d_close,
                                               float* d_output, int size,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_belt_hold_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_breakaway_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_two_crows_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_black_crows_d(const float* d_open,
                                                   const float* d_high,
                                                   const float* d_low,
                                                   const float* d_close,
                                                   float* d_output, int size,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_inside_d(const float* d_open,
                                              const float* d_high,
                                              const float* d_low,
                                              const float* d_close,
                                              float* d_output, int size,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_outside_d(const float* d_open,
                                               const float* d_high,
                                               const float* d_low,
                                               const float* d_close,
                                               float* d_output, int size,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_line_strike_d(const float* d_open,
                                                   const float* d_high,
                                                   const float* d_low,
                                                   const float* d_close,
                                                   float* d_output, int size,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_stars_in_south_d(const float* d_open,
                                                      const float* d_high,
                                                      const float* d_low,
                                                      const float* d_close,
                                                      float* d_output,
                                                      int size,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_closing_marubozu_d(const float* d_open,
                                                  const float* d_high,
                                                  const float* d_low,
                                                  const float* d_close,
                                                  float* d_output, int size,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_conceal_baby_swallow_d(const float* d_open,
                                                      const float* d_high,
                                                      const float* d_low,
                                                      const float* d_close,
                                                      float* d_output,
                                                      int size,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_counterattack_d(const float* d_open,
                                               const float* d_high,
                                               const float* d_low,
                                               const float* d_close,
                                               float* d_output, int size,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dark_cloud_cover_d(const float* d_open,
                                                  const float* d_high,
                                                  const float* d_low,
                                                  const float* d_close,
                                                  float* d_output, int size,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_star_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dragonfly_doji_d(const float* d_open,
                                                const float* d_high,
                                                const float* d_low,
                                                const float* d_close,
                                                float* d_output, int size,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_engulfing_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_doji_star_d(const float* d_open,
                                                   const float* d_high,
                                                   const float* d_low,
                                                   const float* d_close,
                                                   float* d_output, int size,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_star_d(const float* d_open,
                                              const float* d_high,
                                              const float* d_low,
                                              const float* d_close,
                                              float* d_output, int size,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gap_side_side_white_d(const float* d_open,
                                                     const float* d_high,
                                                     const float* d_low,
                                                     const float* d_close,
                                                     float* d_output, int size,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gravestone_doji_d(const float* d_open,
                                                 const float* d_high,
                                                 const float* d_low,
                                                 const float* d_close,
                                                 float* d_output, int size,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hanging_man_d(const float* d_open,
                                             const float* d_high,
                                             const float* d_low,
                                             const float* d_close,
                                             float* d_output, int size,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_d(const float* d_open,
                                        const float* d_high,
                                        const float* d_low,
                                        const float* d_close, float* d_output,
                                        int size,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_cross_d(const float* d_open,
                                              const float* d_high,
                                              const float* d_low,
                                              const float* d_close,
                                              float* d_output, int size,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_high_wave_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_d(const float* d_open,
                                         const float* d_high,
                                         const float* d_low,
                                         const float* d_close, float* d_output,
                                         int size,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_mod_d(const float* d_open,
                                             const float* d_high,
                                             const float* d_low,
                                             const float* d_close,
                                             float* d_output, int size,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_homing_pigeon_d(const float* d_open,
                                               const float* d_high,
                                               const float* d_low,
                                               const float* d_close,
                                               float* d_output, int size,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_identical_three_crows_d(const float* d_open,
                                                       const float* d_high,
                                                       const float* d_low,
                                                       const float* d_close,
                                                       float* d_output,
                                                       int size,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_in_neck_d(const float* d_open,
                                         const float* d_high,
                                         const float* d_low,
                                         const float* d_close, float* d_output,
                                         int size,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_d(const float* d_open,
                                         const float* d_high,
                                         const float* d_low,
                                         const float* d_close, float* d_output,
                                         int size,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_by_length_d(const float* d_open,
                                                   const float* d_high,
                                                   const float* d_low,
                                                   const float* d_close,
                                                   float* d_output, int size,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_ladder_bottom_d(const float* d_open,
                                               const float* d_high,
                                               const float* d_low,
                                               const float* d_close,
                                               float* d_output, int size,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_legged_doji_d(const float* d_open,
                                                  const float* d_high,
                                                  const float* d_low,
                                                  const float* d_close,
                                                  float* d_output, int size,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_line_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_marubozu_d(const float* d_open,
                                          const float* d_high,
                                          const float* d_low,
                                          const float* d_close,
                                          float* d_output, int size,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_matching_low_d(const float* d_open,
                                              const float* d_high,
                                              const float* d_low,
                                              const float* d_close,
                                              float* d_output, int size,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_mat_hold_d(const float* d_open,
                                          const float* d_high,
                                          const float* d_low,
                                          const float* d_close,
                                          float* d_output, int size,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_doji_star_d(const float* d_open,
                                                   const float* d_high,
                                                   const float* d_low,
                                                   const float* d_close,
                                                   float* d_output, int size,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_star_d(const float* d_open,
                                              const float* d_high,
                                              const float* d_low,
                                              const float* d_close,
                                              float* d_output, int size,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_on_neck_d(const float* d_open,
                                         const float* d_high,
                                         const float* d_low,
                                         const float* d_close, float* d_output,
                                         int size,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_piercing_d(const float* d_open,
                                          const float* d_high,
                                          const float* d_low,
                                          const float* d_close,
                                          float* d_output, int size,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rickshaw_man_d(const float* d_open,
                                              const float* d_high,
                                              const float* d_low,
                                              const float* d_close,
                                              float* d_output, int size,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rise_fall3_methods_d(const float* d_open,
                                                    const float* d_high,
                                                    const float* d_low,
                                                    const float* d_close,
                                                    float* d_output, int size,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_separating_lines_d(const float* d_open,
                                                  const float* d_high,
                                                  const float* d_low,
                                                  const float* d_close,
                                                  float* d_output, int size,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_shooting_star_d(const float* d_open,
                                               const float* d_high,
                                               const float* d_low,
                                               const float* d_close,
                                               float* d_output, int size,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_short_line_d(const float* d_open,
                                            const float* d_high,
                                            const float* d_low,
                                            const float* d_close,
                                            float* d_output, int size,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_spinning_top_d(const float* d_open,
                                              const float* d_high,
                                              const float* d_low,
                                              const float* d_close,
                                              float* d_output, int size,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stalled_pattern_d(const float* d_open,
                                                 const float* d_high,
                                                 const float* d_low,
                                                 const float* d_close,
                                                 float* d_output, int size,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stick_sandwich_d(const float* d_open,
                                                const float* d_high,
                                                const float* d_low,
                                                const float* d_close,
                                                float* d_output, int size,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_takuri_d(const float* d_open,
                                        const float* d_high,
                                        const float* d_low,
                                        const float* d_close, float* d_output,
                                        int size,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tasuki_gap_d(const float* d_open,
                                            const float* d_high,
                                            const float* d_low,
                                            const float* d_close,
                                            float* d_output, int size,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_thrusting_d(const float* d_open,
                                           const float* d_high,
                                           const float* d_low,
                                           const float* d_close,
                                           float* d_output, int size,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tristar_d(const float* d_open,
                                         const float* d_high,
                                         const float* d_low,
                                         const float* d_close, float* d_output,
                                         int size,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_unique_3_river_d(const float* d_open,
                                                const float* d_high,
                                                const float* d_low,
                                                const float* d_close,
                                                float* d_output, int size,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_upside_gap_2_crows_d(const float* d_open,
                                                    const float* d_high,
                                                    const float* d_low,
                                                    const float* d_close,
                                                    float* d_output, int size,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_xside_gap_3_methods_d(const float* d_open,
                                                     const float* d_high,
                                                     const float* d_low,
                                                     const float* d_close,
                                                     float* d_output, int size,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asi_d(const float* d_open, const float* d_high,
                                 const float* d_low, const float* d_close,
                                 float* d_output, int size,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar_el_d(const float* d_high, const float* d_low,
                                    const float* d_close, float* d_output,
                                    int size, float accFact, float maxAccel,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_obtr_d(const float* d_high, const float* d_low,
                                  const float* d_close, float* d_output,
                                  int size,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wobv_d(const float* d_open, const float* d_high,
                                  const float* d_low, const float* d_close,
                                  const float* d_volume, float* d_output,
                                  int size,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_hpi_d(const float* d_high, const float* d_low,
                                 const float* d_volume, float* d_output,
                                 int size, float factor,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_market_strength_d(const float* d_input,
                                             float* d_output, int size,
                                             int period,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trendscore_d(const float* d_input, float* d_output,
                                        int size, int lookback,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_refdev_d(const float* d_input, float* d_output,
                                    int size, int smaPeriod, int sumPeriod,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_skew_d(const float* d_input, float* d_output,
                                  int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kurtosis_d(const float* d_input, float* d_output,
                                      int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_high_d(const float* d_input, float* d_output,
                                        int size, int strength,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_low_d(const float* d_input, float* d_output,
                                       int size, int strength,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_xaverage_d(const float* d_input, float* d_output,
                                      int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_mean_d(const float* d_input,
                                          float* d_output, int size,
                                          int period,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_max_d(const float* d_input, float* d_output,
                                         int size, int period,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_min_d(const float* d_input, float* d_output,
                                         int size, int period,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_std_d(const float* d_input, float* d_output,
                                         int size, int period,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
/* --- END GENERATED DEVICE API --- */

#ifdef __cplusplus
}
#endif

#undef CTAPI_STREAM_DEFAULT

#endif // TACUDA_DEVICE_H

