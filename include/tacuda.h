/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_H
#define TACUDA_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>

#if defined(_WIN32)
#define CTAPI_EXPORT __declspec(dllexport)
#else
#define CTAPI_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
// Default stream is supported for C++ callers; NVCC (__CUDACC__) struggles
// with default arguments in some compilation modes, so disable there.
#if !defined(__CUDACC__)
#define CTAPI_STREAM_DEFAULT = 0
#else
#define CTAPI_STREAM_DEFAULT
#endif
extern "C" {
#else
#define CTAPI_STREAM_DEFAULT
#endif

typedef enum ctStatus {
  CT_STATUS_SUCCESS = 0,
  CT_STATUS_ALLOC_FAILED = 1,
  CT_STATUS_COPY_FAILED = 2,
  CT_STATUS_KERNEL_FAILED = 3,
  CT_STATUS_INVALID_ARGUMENT = 4,
  CT_STATUS_DEVICE_ERROR = 5,
  CT_STATUS_UNSUPPORTED = 6,
} ctStatus_t;

/**
 * @brief Return a human-readable string for the given status code.
 */
CTAPI_EXPORT const char* ct_status_string(ctStatus_t status);

/**
 * @brief Return the error message from the most recent failed API call
 *        on the calling thread.
 *
 * Thread-safe (uses thread-local storage). Returns "" when the last
 * call succeeded or no error has been recorded yet.
 */
CTAPI_EXPORT const char* ct_get_last_error(void);

/*
 * Library version. The C ABI is append-only within a MAJOR version:
 * existing symbols never change signature or semantics; new capabilities
 * are added as new symbols. See the ABI stability policy in
 * docs/CONTRIBUTING.md.
 */
#define CT_VERSION_MAJOR 0
#define CT_VERSION_MINOR 9
#define CT_VERSION_PATCH 3
#define CT_VERSION_STRING "0.9.3"

/**
 * @brief Return the library version as "MAJOR.MINOR.PATCH".
 */
CTAPI_EXPORT const char* ct_version(void);

typedef enum ctMaType {
  CT_MA_SMA = 0,
  CT_MA_EMA = 1,
  CT_MA_WMA = 2,
  CT_MA_DEMA = 3,
  CT_MA_TEMA = 4,
  CT_MA_TRIMA = 5,
  CT_MA_KAMA = 6,
  CT_MA_MAMA = 7,
  CT_MA_T3 = 8,
  CT_MA_HMA = 9,
  CT_MA_DISABLED = 10,
  CT_MA_DEFAULT = 11,
  CT_MA_ZLEMA = 12,
  CT_MA_RMA = 13,
  CT_MA_VIDYA = 14,
  CT_MA_ALMA = 15,
} ctMaType_t;

typedef enum ctDataType {
  CT_DTYPE_F16 = 0,
  CT_DTYPE_F32 = 1,
  CT_DTYPE_F64 = 2,
} ctDataType_t;

// All APIs copy host->device->host internally for ease of binding.
CTAPI_EXPORT ctStatus_t ct_sma(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sma_ex(const void *host_input, void *host_output,
                                  int size, int period, ctDataType_t storageType,
                                  ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ma(const float *host_input, float *host_output,
                              int size, int period, ctMaType_t type, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mavp(const float *host_input,
                                const float *host_periods, float *host_output,
                                int size, int minPeriod, int maxPeriod,
                                ctMaType_t type, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wma(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_momentum(const float *host_input, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_change(const float *host_input, float *host_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_roc(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocp(const float *host_input, float *host_output,
                                int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr(const float *host_input, float *host_output,
                                int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr100(const float *host_input, float *host_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ema(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ema_ex(const void *host_input, void *host_output,
                                  int size, int period, ctDataType_t storageType,
                                  ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dema(const float *host_input, float *host_output,
                                int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tema(const float *host_input, float *host_output,
                                int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_t3(const float *host_input, float *host_output,
                              int size, int period, float vFactor, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trima(const float *host_input, float *host_output,
                                 int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trix(const float *host_input, float *host_output,
                                int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_max(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_min(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_maxindex(const float *host_input, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minindex(const float *host_input, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmax(const float *host_input, float *host_min,
                                  float *host_max, int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmaxindex(const float *host_input,
                                       float *host_minidx, float *host_maxidx,
                                       int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stddev(const float *host_input, float *host_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_avgdev(const float *host_input, float *host_output,
                                  int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_var(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sum(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rsi(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rsi_ex(const void *host_input, void *host_output,
                                  int size, int period, ctDataType_t storageType,
                                  ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kama(const float *host_input, float *host_output,
                                int size, int period, int fastPeriod,
                                int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
// MACD line only (EMA_fast - EMA_slow)
CTAPI_EXPORT ctStatus_t ct_macd_line(const float *host_input,
                                     float *host_output, int size,
                                     int fastPeriod, int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd(const float *host_input, float *host_macd,
                                float *host_signal, float *host_hist, int size,
                                int fastPeriod, int slowPeriod,
                                int signalPeriod, ctMaType_t type, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_ex(const void *host_input, void *host_macd,
                                   void *host_signal, void *host_hist, int size,
                                   int fastPeriod, int slowPeriod, int signalPeriod,
                                   ctMaType_t type, ctDataType_t storageType,
                                   ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macdfix(const float *host_input, float *host_macd,
                                   float *host_signal, float *host_hist,
                                   int size, int signalPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mama(const float *host_input, float *host_mama,
                                float *host_fama, int size, float fastLimit,
                                float slowLimit, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_apo(const float *host_input, float *host_output,
                               int size, int fastPeriod, int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo(const float *host_input, float *host_output,
                               int size, int fastPeriod, int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
/* TA-Lib-complete variants with a selectable MA type (TA_APO/TA_PPO
 * optInMAType). The plain ct_apo/ct_ppo retain EMA, also the default in
 * TA-Lib 0.8.3 (older TA-Lib used SMA). Appended in 0.6.0, ABI append-only. */
CTAPI_EXPORT ctStatus_t ct_apo_ma(const float *host_input, float *host_output,
                                  int size, int fastPeriod, int slowPeriod,
                                  ctMaType_t type, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo_ma(const float *host_input, float *host_output,
                                  int size, int fastPeriod, int slowPeriod,
                                  ctMaType_t type, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo(const float *host_volume, float *host_output,
                               int size, int fastPeriod, int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
// Device-pointer variants that operate directly on GPU memory.
CTAPI_EXPORT ctStatus_t ct_sma_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ma_device(const float *device_input,
                                     float *device_output, int size,
                                     int period, ctMaType_t type,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wma_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_momentum_device(const float *device_input,
                                           float *device_output, int size,
                                           int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_change_device(const float *device_input,
                                         float *device_output, int size,
                                         int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_roc_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocp_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr100_device(const float *device_input,
                                          float *device_output, int size,
                                          int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ema_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dema_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tema_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_t3_device(const float *device_input,
                                     float *device_output, int size,
                                     int period, float vFactor,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trima_device(const float *device_input,
                                        float *device_output, int size,
                                        int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trix_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_max_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_min_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_maxindex_device(const float *device_input,
                                           float *device_output, int size,
                                           int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minindex_device(const float *device_input,
                                           float *device_output, int size,
                                           int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stddev_device(const float *device_input,
                                         float *device_output, int size,
                                         int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_var_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sum_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rsi_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kama_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, int fastPeriod,
                                       int slowPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_line_device(const float *device_input,
                                            float *device_output, int size,
                                            int fastPeriod, int slowPeriod,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_apo_device(const float *device_input,
                                      float *device_output, int size,
                                      int fastPeriod, int slowPeriod,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo_device(const float *device_input,
                                      float *device_output, int size,
                                      int fastPeriod, int slowPeriod,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo_device(const float *device_volume,
                                      float *device_output, int size,
                                      int fastPeriod, int slowPeriod,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_accbands(const float *host_high,
                                    const float *host_low,
                                    const float *host_close,
                                    float *host_upper, float *host_middle,
                                    float *host_lower, int size, int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands(const float *host_input, float *host_upper,
                                  float *host_middle, float *host_lower,
                                  int size, int period, float upperMul,
                                  float lowerMul, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands_ex(const void *host_input, void *host_upper,
                                     void *host_middle, void *host_lower,
                                     int size, int period, float upperMul,
                                     float lowerMul, ctDataType_t storageType,
                                     ctDataType_t computeType, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_atr(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size, int period, float initial,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_atr_ex(const void *host_high, const void *host_low,
                                  const void *host_close, void *host_output,
                                  int size, int period, float initial,
                                  ctDataType_t storageType, ctDataType_t computeType,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_natr(const float *host_high, const float *host_low,
                                const float *host_close, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trange(const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochastic(const float *host_high,
                                      const float *host_low,
                                      const float *host_close, float *host_k,
                                      float *host_d, int size, int kPeriod,
                                      int dPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochf(const float *host_high, const float *host_low,
                                  const float *host_close, float *host_k,
                                  float *host_d, int size, int kPeriod,
                                  int dPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochrsi(const float *host_input, float *host_k,
                                    float *host_d, int size, int rsiPeriod,
                                    int kPeriod, int dPeriod,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cci(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adx(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adxr(const float *host_high, const float *host_low,
                                const float *host_close, float *host_output,
                                int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_plus_dm(const float *host_high,
                                   const float *host_low, float *host_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minus_dm(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_plus_di(const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minus_di(const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mfi(const float *host_high, const float *host_low,
                               const float *host_close,
                               const float *host_volume, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_imi(const float *host_open, const float *host_close,
                               float *host_output, int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_obv(const float *host_price,
                               const float *host_volume, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_nvi(const float *host_close,
                               const float *host_volume, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvi(const float *host_close,
                               const float *host_volume, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar(const float *host_high, const float *host_low,
                               float *host_output, int size, float step,
                               float maxAcceleration, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sarext(const float *host_high, const float *host_low,
                                  float *host_output, int size,
                                  float startValue, float offsetOnReverse,
                                  float accInitLong, float accLong,
                                  float accMaxLong, float accInitShort,
                                  float accShort, float accMaxShort, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_aroon(const float *host_high, const float *host_low,
                                 float *host_up, float *host_down,
                                 float *host_osc, int size, int upPeriod,
                                 int downPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_aroonosc(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adosc(const float *host_high, const float *host_low,
                                 const float *host_close,
                                 const float *host_volume, float *host_output,
                                 int size, int shortPeriod, int longPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ultosc(const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, int shortPeriod, int mediumPeriod,
                                  int longPeriod, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ad(const float *host_high, const float *host_low,
                              const float *host_close, const float *host_volume,
                              float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_avgprice(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_medprice(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_typprice(const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wclprice(const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_willr(const float *host_high, const float *host_low,
                                 const float *host_close, float *host_output,
                                 int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midpoint(const float *host_input, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midprice(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_beta(const float *host_x, const float *host_y,
                                float *host_output, int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bop(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hammer(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_inverted_hammer(const float *host_open,
                                               const float *host_high,
                                               const float *host_low,
                                               const float *host_close,
                                               float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bullish_engulfing(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bearish_engulfing(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_white_soldiers(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_abandoned_baby(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_advance_block(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_belt_hold(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_breakaway(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_two_crows(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_black_crows(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_inside(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_outside(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_line_strike(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_stars_in_south(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_closing_marubozu(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_conceal_baby_swallow(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_counterattack(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dark_cloud_cover(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_star(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dragonfly_doji(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_engulfing(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_doji_star(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_star(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gap_side_side_white(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gravestone_doji(const float *host_open,
                                               const float *host_high,
                                               const float *host_low,
                                               const float *host_close,
                                               float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hanging_man(const float *host_open,
                                           const float *host_high,
                                           const float *host_low,
                                           const float *host_close,
                                           float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_cross(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_high_wave(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_mod(const float *host_open,
                                           const float *host_high,
                                           const float *host_low,
                                           const float *host_close,
                                           float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_homing_pigeon(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_identical_three_crows(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_in_neck(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_by_length(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_ladder_bottom(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_legged_doji(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_line(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_marubozu(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_matching_low(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_mat_hold(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_doji_star(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_star(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_on_neck(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_piercing(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rickshaw_man(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rise_fall3_methods(const float *host_open,
                                                  const float *host_high,
                                                  const float *host_low,
                                                  const float *host_close,
                                                  float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_separating_lines(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_shooting_star(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_short_line(const float *host_open,
                                          const float *host_high,
                                          const float *host_low,
                                          const float *host_close,
                                          float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_spinning_top(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stalled_pattern(const float *host_open,
                                               const float *host_high,
                                               const float *host_low,
                                               const float *host_close,
                                               float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stick_sandwich(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_takuri(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tasuki_gap(const float *host_open,
                                          const float *host_high,
                                          const float *host_low,
                                          const float *host_close,
                                          float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_thrusting(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tristar(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_unique_3_river(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_upside_gap_2_crows(const float *host_open,
                                                  const float *host_high,
                                                  const float *host_low,
                                                  const float *host_close,
                                                  float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_xside_gap_3_methods(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmo(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_correl(const float *host_x, const float *host_y,
                                  float *host_output, int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dx(const float *host_high, const float *host_low,
                              const float *host_close, float *host_output,
                              int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg(const float *host_input,
                                     float *host_output, int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_slope(const float *host_input,
                                           float *host_output, int size,
                                           int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_intercept(const float *host_input,
                                               float *host_output, int size,
                                               int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_angle(const float *host_input,
                                           float *host_output, int size,
                                           int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tsf(const float *host_input, float *host_output,
                               int size, int period, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcperiod(const float *host_input,
                                       float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcphase(const float *host_input,
                                      float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_phasor(const float *host_input,
                                     float *host_inphase,
                                     float *host_quadrature, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_sine(const float *host_input, float *host_sine,
                                   float *host_leadsine, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendline(const float *host_input,
                                        float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendmode(const float *host_input,
                                        float *host_output, int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// MACDEXT with independent MA types per line (TA-Lib MACDEXT).
CTAPI_EXPORT ctStatus_t ct_macdext(const float *host_input, float *host_macd,
                                   float *host_signal, float *host_hist,
                                   int size, int fastPeriod, ctMaType_t fastType,
                                   int slowPeriod, ctMaType_t slowType,
                                   int signalPeriod, ctMaType_t signalType,
                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);

// =============================================================================
// Vector math (element-wise, TA-Lib Math Transform / Math Operators)
// No lookback: output[i] = f(input[i]). Optimised for large arrays.
// =============================================================================
CTAPI_EXPORT ctStatus_t ct_acos(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asin(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_atan(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ceil(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cos(const float *host_input, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cosh(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_exp(const float *host_input, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_floor(const float *host_input, float *host_output,
                                 int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ln(const float *host_input, float *host_output,
                              int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_log10(const float *host_input, float *host_output,
                                 int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sin(const float *host_input, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sinh(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sqrt(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tan(const float *host_input, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tanh(const float *host_input, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_add(const float *host_input1,
                               const float *host_input2, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sub(const float *host_input1,
                               const float *host_input2, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mult(const float *host_input1,
                                const float *host_input2, float *host_output,
                                int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_div(const float *host_input1,
                               const float *host_input2, float *host_output,
                               int size, cudaStream_t stream CTAPI_STREAM_DEFAULT);

// DeviceBufferPool management functions
CTAPI_EXPORT void ct_device_buffer_pool_cleanup(void);
CTAPI_EXPORT size_t ct_device_buffer_pool_allocation_count(void);
CTAPI_EXPORT size_t ct_device_buffer_pool_set_max_cache_bytes(size_t bytes);

/* =========================================================================
 * Batched (multi-symbol / portfolio) API
 * =========================================================================
 * Inputs and outputs are dense series-major matrices: numSeries rows of
 * seriesLen elements each (row i = series i, contiguous). Every series is
 * processed independently with the same parameters.
 *
 * Supported fused workloads launch one kernel on the supplied stream and
 * ignore maxConcurrency. TACUDA_BATCH_FUSED=auto|0|1 controls dispatch; see
 * docs/BATCH_API.md for the family/shape limits and numerical contract.
 * Otherwise work is distributed round-robin over an internal pool of CUDA
 * streams so host->device transfers of one series overlap kernels and
 * device->host transfers of others. maxConcurrency limits the pool size
 * (0 = auto heuristic, currently min(numSeries, 4)). When a non-null
 * `stream` is passed, everything runs ordered on that stream instead and
 * maxConcurrency is ignored. Host matrices are temporarily page-locked via
 * cudaHostRegister when large enough for the registration cost to pay off;
 * if registration fails the call still succeeds with reduced overlap.
 */
CTAPI_EXPORT ctStatus_t ct_sma_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ema_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rsi_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_atr_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period, float initial,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_batch(const float *host_inputs,
                                      float *host_macds, float *host_signals,
                                      float *host_hists, int seriesLen,
                                      int numSeries, int fastPeriod,
                                      int slowPeriod, int signalPeriod,
                                      int maType, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands_batch(const float *host_inputs,
                                        float *host_uppers,
                                        float *host_middles,
                                        float *host_lowers, int seriesLen,
                                        int numSeries, int period,
                                        float upperMul, float lowerMul,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochastic_batch(const float *host_highs,
                                            const float *host_lows,
                                            const float *host_closes,
                                            float *host_ks, float *host_ds,
                                            int seriesLen, int numSeries,
                                            int kPeriod, int dPeriod,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);

/*
 * Device-pointer batch variants: matrices already resident on the GPU
 * (numSeries * seriesLen floats). No host transfers are performed. Supported
 * workloads use one fused kernel; others use the internal stream pool.
 * stream == 0 is the legacy default stream (also PyTorch's and CuPy's): the
 * work waits for preceding operations on it and the call returns once
 * the batch is done. A non-null stream orders the batch on it, asynchronously:
 * the call only enqueues work (no host synchronization) and can be captured
 * into a CUDA graph, in any capture mode. See docs/BATCH_API.md.
 */
CTAPI_EXPORT ctStatus_t ct_sma_batch_d(const float *d_inputs,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ema_batch_d(const float *d_inputs,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rsi_batch_d(const float *d_inputs,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_atr_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       float initial, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_batch_d(const float *d_inputs, float *d_macds,
                                        float *d_signals, float *d_hists,
                                        int seriesLen, int numSeries,
                                        int fastPeriod, int slowPeriod,
                                        int signalPeriod, int maType,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands_batch_d(const float *d_inputs,
                                          float *d_uppers, float *d_middles,
                                          float *d_lowers, int seriesLen,
                                          int numSeries, int period,
                                          float upperMul, float lowerMul,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochastic_batch_d(const float *d_highs,
                                              const float *d_lows,
                                              const float *d_closes,
                                              float *d_ks, float *d_ds,
                                              int seriesLen, int numSeries,
                                              int kPeriod, int dPeriod,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);

/* --- BEGIN GENERATED BATCH API (scripts/generate_batch_api.py) --- */
/* Generated batch entry points: same matrix layout and scheduling as the
 * hand-written ct_*_batch functions above. Club-pack families also get a
 * plain single-series host entry point. Do not edit by hand. */
CTAPI_EXPORT ctStatus_t ct_wma_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wma_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dema_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dema_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tema_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tema_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trima_batch(const float *host_inputs,
                                       float *host_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trima_batch_d(const float *d_inputs,
                                         float *d_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_momentum_batch(const float *host_inputs,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_momentum_batch_d(const float *d_inputs,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_roc_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_roc_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocp_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocp_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr100_batch(const float *host_inputs,
                                         float *host_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rocr100_batch_d(const float *d_inputs,
                                           float *d_outputs, int seriesLen,
                                           int numSeries, int period,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmo_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmo_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stddev_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stddev_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_var_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_var_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sum_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sum_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_max_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_max_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_min_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_min_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_batch(const float *host_inputs,
                                           float *host_outputs, int seriesLen,
                                           int numSeries, int period,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_batch_d(const float *d_inputs,
                                             float *d_outputs, int seriesLen,
                                             int numSeries, int period,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_slope_batch(const float *host_inputs,
                                                 float *host_outputs,
                                                 int seriesLen, int numSeries,
                                                 int period,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_slope_batch_d(const float *d_inputs,
                                                   float *d_outputs,
                                                   int seriesLen,
                                                   int numSeries, int period,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_intercept_batch(const float *host_inputs,
                                                     float *host_outputs,
                                                     int seriesLen,
                                                     int numSeries, int period,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_intercept_batch_d(const float *d_inputs,
                                                       float *d_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int period,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_angle_batch(const float *host_inputs,
                                                 float *host_outputs,
                                                 int seriesLen, int numSeries,
                                                 int period,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_linearreg_angle_batch_d(const float *d_inputs,
                                                   float *d_outputs,
                                                   int seriesLen,
                                                   int numSeries, int period,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tsf_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tsf_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kama_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int fastPeriod, int slowPeriod,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kama_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int fastPeriod, int slowPeriod,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_t3_batch(const float *host_inputs,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int period, float vFactor,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_t3_batch_d(const float *d_inputs, float *d_outputs,
                                      int seriesLen, int numSeries, int period,
                                      float vFactor, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_natr_batch(const float *host_highs,
                                      const float *host_lows,
                                      const float *host_closes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_natr_batch_d(const float *d_highs,
                                        const float *d_lows,
                                        const float *d_closes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adx_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adx_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adxr_batch(const float *host_highs,
                                      const float *host_lows,
                                      const float *host_closes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adxr_batch_d(const float *d_highs,
                                        const float *d_lows,
                                        const float *d_closes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dx_batch(const float *host_highs,
                                    const float *host_lows,
                                    const float *host_closes,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int period,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dx_batch_d(const float *d_highs,
                                      const float *d_lows,
                                      const float *d_closes, float *d_outputs,
                                      int seriesLen, int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_plus_di_batch(const float *host_highs,
                                         const float *host_lows,
                                         const float *host_closes,
                                         float *host_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_plus_di_batch_d(const float *d_highs,
                                           const float *d_lows,
                                           const float *d_closes,
                                           float *d_outputs, int seriesLen,
                                           int numSeries, int period,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minus_di_batch(const float *host_highs,
                                          const float *host_lows,
                                          const float *host_closes,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minus_di_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            const float *d_closes,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_willr_batch(const float *host_highs,
                                       const float *host_lows,
                                       const float *host_closes,
                                       float *host_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_willr_batch_d(const float *d_highs,
                                         const float *d_lows,
                                         const float *d_closes,
                                         float *d_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cci_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cci_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trange_batch(const float *host_highs,
                                        const float *host_lows,
                                        const float *host_closes,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trange_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          const float *d_closes,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_typprice_batch(const float *host_highs,
                                          const float *host_lows,
                                          const float *host_closes,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_typprice_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            const float *d_closes,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wclprice_batch(const float *host_highs,
                                          const float *host_lows,
                                          const float *host_closes,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wclprice_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            const float *d_closes,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ultosc_batch(const float *host_highs,
                                        const float *host_lows,
                                        const float *host_closes,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period1,
                                        int period2, int period3,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ultosc_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          const float *d_closes,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period1,
                                          int period2, int period3,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_medprice_batch(const float *host_highs,
                                          const float *host_lows,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_medprice_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_aroonosc_batch(const float *host_highs,
                                          const float *host_lows,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_aroonosc_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_aroon_batch(const float *host_highs,
                                       const float *host_lows,
                                       float *host_aroon_downs,
                                       float *host_aroon_ups, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_aroon_batch_d(const float *d_highs,
                                         const float *d_lows,
                                         float *d_aroon_downs,
                                         float *d_aroon_ups, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochf_batch(const float *host_highs,
                                        const float *host_lows,
                                        const float *host_closes,
                                        float *host_fast_ks,
                                        float *host_fast_ds, int seriesLen,
                                        int numSeries, int kPeriod,
                                        int dPeriod, int dMaType,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochf_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          const float *d_closes,
                                          float *d_fast_ks, float *d_fast_ds,
                                          int seriesLen, int numSeries,
                                          int kPeriod, int dPeriod,
                                          int dMaType, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macdext_batch(const float *host_inputs,
                                         float *host_macds,
                                         float *host_signals,
                                         float *host_hists, int seriesLen,
                                         int numSeries, int fastPeriod,
                                         int fastType, int slowPeriod,
                                         int slowType, int signalPeriod,
                                         int signalType, int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macdext_batch_d(const float *d_inputs,
                                           float *d_macds, float *d_signals,
                                           float *d_hists, int seriesLen,
                                           int numSeries, int fastPeriod,
                                           int fastType, int slowPeriod,
                                           int slowType, int signalPeriod,
                                           int signalType, int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_accbands_batch(const float *host_highs,
                                          const float *host_lows,
                                          const float *host_closes,
                                          float *host_uppers,
                                          float *host_middles,
                                          float *host_lowers, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_accbands_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            const float *d_closes,
                                            float *d_uppers, float *d_middles,
                                            float *d_lowers, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_avgprice_batch(const float *host_opens,
                                          const float *host_highs,
                                          const float *host_lows,
                                          const float *host_closes,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_avgprice_batch_d(const float *d_opens,
                                            const float *d_highs,
                                            const float *d_lows,
                                            const float *d_closes,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_obv_batch(const float *host_closes,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_obv_batch_d(const float *d_closes,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mfi_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mfi_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ad_batch(const float *host_highs,
                                    const float *host_lows,
                                    const float *host_closes,
                                    const float *host_volumes,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ad_batch_d(const float *d_highs,
                                      const float *d_lows,
                                      const float *d_closes,
                                      const float *d_volumes, float *d_outputs,
                                      int seriesLen, int numSeries,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adosc_batch(const float *host_highs,
                                       const float *host_lows,
                                       const float *host_closes,
                                       const float *host_volumes,
                                       float *host_outputs, int seriesLen,
                                       int numSeries, int fastPeriod,
                                       int slowPeriod, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adosc_batch_d(const float *d_highs,
                                         const float *d_lows,
                                         const float *d_closes,
                                         const float *d_volumes,
                                         float *d_outputs, int seriesLen,
                                         int numSeries, int fastPeriod,
                                         int slowPeriod, int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trix_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trix_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_change_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_change_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midpoint_batch(const float *host_inputs,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midpoint_batch_d(const float *d_inputs,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_avgdev_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_avgdev_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_maxindex_batch(const float *host_inputs,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_maxindex_batch_d(const float *d_inputs,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minindex_batch(const float *host_inputs,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minindex_batch_d(const float *d_inputs,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_apo_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int fastPeriod,
                                     int slowPeriod, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_apo_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int fastPeriod, int slowPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int fastPeriod,
                                     int slowPeriod, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int fastPeriod, int slowPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo_batch(const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int fastPeriod,
                                     int slowPeriod, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo_batch_d(const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int fastPeriod,
                                       int slowPeriod, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ma_batch(const float *host_inputs,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int period, int maType,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ma_batch_d(const float *d_inputs, float *d_outputs,
                                      int seriesLen, int numSeries, int period,
                                      int maType, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcperiod_batch(const float *host_inputs,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcperiod_batch_d(const float *d_inputs,
                                               float *d_outputs, int seriesLen,
                                               int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcphase_batch(const float *host_inputs,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_dcphase_batch_d(const float *d_inputs,
                                              float *d_outputs, int seriesLen,
                                              int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendline_batch(const float *host_inputs,
                                              float *host_outputs,
                                              int seriesLen, int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendline_batch_d(const float *d_inputs,
                                                float *d_outputs,
                                                int seriesLen, int numSeries,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendmode_batch(const float *host_inputs,
                                              float *host_outputs,
                                              int seriesLen, int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_trendmode_batch_d(const float *d_inputs,
                                                float *d_outputs,
                                                int seriesLen, int numSeries,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochrsi_batch(const float *host_inputs,
                                          float *host_fast_ks,
                                          float *host_fast_ds, int seriesLen,
                                          int numSeries, int rsiPeriod,
                                          int kPeriod, int dPeriod,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochrsi_batch_d(const float *d_inputs,
                                            float *d_fast_ks, float *d_fast_ds,
                                            int seriesLen, int numSeries,
                                            int rsiPeriod, int kPeriod,
                                            int dPeriod, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macdfix_batch(const float *host_inputs,
                                         float *host_macds,
                                         float *host_signals,
                                         float *host_hists, int seriesLen,
                                         int numSeries, int signalPeriod,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macdfix_batch_d(const float *d_inputs,
                                           float *d_macds, float *d_signals,
                                           float *d_hists, int seriesLen,
                                           int numSeries, int signalPeriod,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mama_batch(const float *host_inputs,
                                      float *host_mamas, float *host_famas,
                                      int seriesLen, int numSeries,
                                      float fastLimit, float slowLimit,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mama_batch_d(const float *d_inputs, float *d_mamas,
                                        float *d_famas, int seriesLen,
                                        int numSeries, float fastLimit,
                                        float slowLimit, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmax_batch(const float *host_inputs,
                                        float *host_mins, float *host_maxs,
                                        int seriesLen, int numSeries,
                                        int period, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmax_batch_d(const float *d_inputs, float *d_mins,
                                          float *d_maxs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmaxindex_batch(const float *host_inputs,
                                             float *host_minidxs,
                                             float *host_maxidxs,
                                             int seriesLen, int numSeries,
                                             int period, int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minmaxindex_batch_d(const float *d_inputs,
                                               float *d_minidxs,
                                               float *d_maxidxs, int seriesLen,
                                               int numSeries, int period,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_phasor_batch(const float *host_inputs,
                                           float *host_inphases,
                                           float *host_quadratures,
                                           int seriesLen, int numSeries,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_phasor_batch_d(const float *d_inputs,
                                             float *d_inphases,
                                             float *d_quadratures,
                                             int seriesLen, int numSeries,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_sine_batch(const float *host_inputs,
                                         float *host_sines,
                                         float *host_leadsines, int seriesLen,
                                         int numSeries, int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ht_sine_batch_d(const float *d_inputs,
                                           float *d_sines, float *d_leadsines,
                                           int seriesLen, int numSeries,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_line_batch(const float *host_inputs,
                                           float *host_outputs, int seriesLen,
                                           int numSeries, int fastPeriod,
                                           int slowPeriod, int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_macd_line_batch_d(const float *d_inputs,
                                             float *d_outputs, int seriesLen,
                                             int numSeries, int fastPeriod,
                                             int slowPeriod,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midprice_batch(const float *host_highs,
                                          const float *host_lows,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_midprice_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_plus_dm_batch(const float *host_highs,
                                         const float *host_lows,
                                         float *host_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_plus_dm_batch_d(const float *d_highs,
                                           const float *d_lows,
                                           float *d_outputs, int seriesLen,
                                           int numSeries, int period,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minus_dm_batch(const float *host_highs,
                                          const float *host_lows,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_minus_dm_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar_batch(const float *host_highs,
                                     const float *host_lows,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, float step,
                                     float maxAcceleration, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar_batch_d(const float *d_highs,
                                       const float *d_lows, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       float step, float maxAcceleration,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sarext_batch(const float *host_highs,
                                        const float *host_lows,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, float startValue,
                                        float offsetOnReverse,
                                        float accInitLong, float accLong,
                                        float accMaxLong, float accInitShort,
                                        float accShort, float accMaxShort,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sarext_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, float startValue,
                                          float offsetOnReverse,
                                          float accInitLong, float accLong,
                                          float accMaxLong, float accInitShort,
                                          float accShort, float accMaxShort,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_imi_batch(const float *host_opens,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_imi_batch_d(const float *d_opens,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_nvi_batch(const float *host_closes,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_nvi_batch_d(const float *d_closes,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvi_batch(const float *host_closes,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvi_batch_d(const float *d_closes,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_correl_batch(const float *host_xs,
                                        const float *host_ys,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_correl_batch_d(const float *d_xs, const float *d_ys,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_beta_batch(const float *host_xs,
                                      const float *host_ys,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_beta_batch_d(const float *d_xs, const float *d_ys,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mavp_batch(const float *host_values,
                                      const float *host_periods,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int minPeriod,
                                      int maxPeriod, int maType,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mavp_batch_d(const float *d_values,
                                        const float *d_periods,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int minPeriod,
                                        int maxPeriod, int maType,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bop_batch(const float *host_opens,
                                     const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bop_batch_d(const float *d_opens,
                                       const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_batch(const float *host_opens,
                                          const float *host_highs,
                                          const float *host_lows,
                                          const float *host_closes,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_batch_d(const float *d_opens,
                                            const float *d_highs,
                                            const float *d_lows,
                                            const float *d_closes,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hammer_batch(const float *host_opens,
                                            const float *host_highs,
                                            const float *host_lows,
                                            const float *host_closes,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hammer_batch_d(const float *d_opens,
                                              const float *d_highs,
                                              const float *d_lows,
                                              const float *d_closes,
                                              float *d_outputs, int seriesLen,
                                              int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_inverted_hammer_batch(const float *host_opens,
                                                     const float *host_highs,
                                                     const float *host_lows,
                                                     const float *host_closes,
                                                     float *host_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_inverted_hammer_batch_d(const float *d_opens,
                                                       const float *d_highs,
                                                       const float *d_lows,
                                                       const float *d_closes,
                                                       float *d_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bullish_engulfing_batch(const float *host_opens,
                                                       const float *host_highs,
                                                       const float *host_lows,
                                                       const float *host_closes,
                                                       float *host_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bullish_engulfing_batch_d(const float *d_opens,
                                                         const float *d_highs,
                                                         const float *d_lows,
                                                         const float *d_closes,
                                                         float *d_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bearish_engulfing_batch(const float *host_opens,
                                                       const float *host_highs,
                                                       const float *host_lows,
                                                       const float *host_closes,
                                                       float *host_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_bearish_engulfing_batch_d(const float *d_opens,
                                                         const float *d_highs,
                                                         const float *d_lows,
                                                         const float *d_closes,
                                                         float *d_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_white_soldiers_batch(const float *host_opens,
                                                          const float *host_highs,
                                                          const float *host_lows,
                                                          const float *host_closes,
                                                          float *host_outputs,
                                                          int seriesLen,
                                                          int numSeries,
                                                          int maxConcurrency,
                                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_white_soldiers_batch_d(const float *d_opens,
                                                            const float *d_highs,
                                                            const float *d_lows,
                                                            const float *d_closes,
                                                            float *d_outputs,
                                                            int seriesLen,
                                                            int numSeries,
                                                            int maxConcurrency,
                                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_abandoned_baby_batch(const float *host_opens,
                                                    const float *host_highs,
                                                    const float *host_lows,
                                                    const float *host_closes,
                                                    float *host_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_abandoned_baby_batch_d(const float *d_opens,
                                                      const float *d_highs,
                                                      const float *d_lows,
                                                      const float *d_closes,
                                                      float *d_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_advance_block_batch(const float *host_opens,
                                                   const float *host_highs,
                                                   const float *host_lows,
                                                   const float *host_closes,
                                                   float *host_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_advance_block_batch_d(const float *d_opens,
                                                     const float *d_highs,
                                                     const float *d_lows,
                                                     const float *d_closes,
                                                     float *d_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_belt_hold_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_belt_hold_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_breakaway_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_breakaway_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_two_crows_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_two_crows_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_black_crows_batch(const float *host_opens,
                                                       const float *host_highs,
                                                       const float *host_lows,
                                                       const float *host_closes,
                                                       float *host_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_black_crows_batch_d(const float *d_opens,
                                                         const float *d_highs,
                                                         const float *d_lows,
                                                         const float *d_closes,
                                                         float *d_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_inside_batch(const float *host_opens,
                                                  const float *host_highs,
                                                  const float *host_lows,
                                                  const float *host_closes,
                                                  float *host_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_inside_batch_d(const float *d_opens,
                                                    const float *d_highs,
                                                    const float *d_lows,
                                                    const float *d_closes,
                                                    float *d_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_outside_batch(const float *host_opens,
                                                   const float *host_highs,
                                                   const float *host_lows,
                                                   const float *host_closes,
                                                   float *host_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_outside_batch_d(const float *d_opens,
                                                     const float *d_highs,
                                                     const float *d_lows,
                                                     const float *d_closes,
                                                     float *d_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_line_strike_batch(const float *host_opens,
                                                       const float *host_highs,
                                                       const float *host_lows,
                                                       const float *host_closes,
                                                       float *host_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_line_strike_batch_d(const float *d_opens,
                                                         const float *d_highs,
                                                         const float *d_lows,
                                                         const float *d_closes,
                                                         float *d_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_stars_in_south_batch(const float *host_opens,
                                                          const float *host_highs,
                                                          const float *host_lows,
                                                          const float *host_closes,
                                                          float *host_outputs,
                                                          int seriesLen,
                                                          int numSeries,
                                                          int maxConcurrency,
                                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_three_stars_in_south_batch_d(const float *d_opens,
                                                            const float *d_highs,
                                                            const float *d_lows,
                                                            const float *d_closes,
                                                            float *d_outputs,
                                                            int seriesLen,
                                                            int numSeries,
                                                            int maxConcurrency,
                                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_closing_marubozu_batch(const float *host_opens,
                                                      const float *host_highs,
                                                      const float *host_lows,
                                                      const float *host_closes,
                                                      float *host_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_closing_marubozu_batch_d(const float *d_opens,
                                                        const float *d_highs,
                                                        const float *d_lows,
                                                        const float *d_closes,
                                                        float *d_outputs,
                                                        int seriesLen,
                                                        int numSeries,
                                                        int maxConcurrency,
                                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_conceal_baby_swallow_batch(const float *host_opens,
                                                          const float *host_highs,
                                                          const float *host_lows,
                                                          const float *host_closes,
                                                          float *host_outputs,
                                                          int seriesLen,
                                                          int numSeries,
                                                          int maxConcurrency,
                                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_conceal_baby_swallow_batch_d(const float *d_opens,
                                                            const float *d_highs,
                                                            const float *d_lows,
                                                            const float *d_closes,
                                                            float *d_outputs,
                                                            int seriesLen,
                                                            int numSeries,
                                                            int maxConcurrency,
                                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_counterattack_batch(const float *host_opens,
                                                   const float *host_highs,
                                                   const float *host_lows,
                                                   const float *host_closes,
                                                   float *host_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_counterattack_batch_d(const float *d_opens,
                                                     const float *d_highs,
                                                     const float *d_lows,
                                                     const float *d_closes,
                                                     float *d_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dark_cloud_cover_batch(const float *host_opens,
                                                      const float *host_highs,
                                                      const float *host_lows,
                                                      const float *host_closes,
                                                      float *host_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dark_cloud_cover_batch_d(const float *d_opens,
                                                        const float *d_highs,
                                                        const float *d_lows,
                                                        const float *d_closes,
                                                        float *d_outputs,
                                                        int seriesLen,
                                                        int numSeries,
                                                        int maxConcurrency,
                                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_star_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_star_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dragonfly_doji_batch(const float *host_opens,
                                                    const float *host_highs,
                                                    const float *host_lows,
                                                    const float *host_closes,
                                                    float *host_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_dragonfly_doji_batch_d(const float *d_opens,
                                                      const float *d_highs,
                                                      const float *d_lows,
                                                      const float *d_closes,
                                                      float *d_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_engulfing_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_engulfing_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_doji_star_batch(const float *host_opens,
                                                       const float *host_highs,
                                                       const float *host_lows,
                                                       const float *host_closes,
                                                       float *host_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_doji_star_batch_d(const float *d_opens,
                                                         const float *d_highs,
                                                         const float *d_lows,
                                                         const float *d_closes,
                                                         float *d_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_star_batch(const float *host_opens,
                                                  const float *host_highs,
                                                  const float *host_lows,
                                                  const float *host_closes,
                                                  float *host_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_star_batch_d(const float *d_opens,
                                                    const float *d_highs,
                                                    const float *d_lows,
                                                    const float *d_closes,
                                                    float *d_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gap_side_side_white_batch(const float *host_opens,
                                                         const float *host_highs,
                                                         const float *host_lows,
                                                         const float *host_closes,
                                                         float *host_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gap_side_side_white_batch_d(const float *d_opens,
                                                           const float *d_highs,
                                                           const float *d_lows,
                                                           const float *d_closes,
                                                           float *d_outputs,
                                                           int seriesLen,
                                                           int numSeries,
                                                           int maxConcurrency,
                                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gravestone_doji_batch(const float *host_opens,
                                                     const float *host_highs,
                                                     const float *host_lows,
                                                     const float *host_closes,
                                                     float *host_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_gravestone_doji_batch_d(const float *d_opens,
                                                       const float *d_highs,
                                                       const float *d_lows,
                                                       const float *d_closes,
                                                       float *d_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hanging_man_batch(const float *host_opens,
                                                 const float *host_highs,
                                                 const float *host_lows,
                                                 const float *host_closes,
                                                 float *host_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hanging_man_batch_d(const float *d_opens,
                                                   const float *d_highs,
                                                   const float *d_lows,
                                                   const float *d_closes,
                                                   float *d_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_batch(const float *host_opens,
                                            const float *host_highs,
                                            const float *host_lows,
                                            const float *host_closes,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_batch_d(const float *d_opens,
                                              const float *d_highs,
                                              const float *d_lows,
                                              const float *d_closes,
                                              float *d_outputs, int seriesLen,
                                              int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_cross_batch(const float *host_opens,
                                                  const float *host_highs,
                                                  const float *host_lows,
                                                  const float *host_closes,
                                                  float *host_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_cross_batch_d(const float *d_opens,
                                                    const float *d_highs,
                                                    const float *d_lows,
                                                    const float *d_closes,
                                                    float *d_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_high_wave_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_high_wave_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_batch(const float *host_opens,
                                             const float *host_highs,
                                             const float *host_lows,
                                             const float *host_closes,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_batch_d(const float *d_opens,
                                               const float *d_highs,
                                               const float *d_lows,
                                               const float *d_closes,
                                               float *d_outputs, int seriesLen,
                                               int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_mod_batch(const float *host_opens,
                                                 const float *host_highs,
                                                 const float *host_lows,
                                                 const float *host_closes,
                                                 float *host_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_mod_batch_d(const float *d_opens,
                                                   const float *d_highs,
                                                   const float *d_lows,
                                                   const float *d_closes,
                                                   float *d_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_homing_pigeon_batch(const float *host_opens,
                                                   const float *host_highs,
                                                   const float *host_lows,
                                                   const float *host_closes,
                                                   float *host_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_homing_pigeon_batch_d(const float *d_opens,
                                                     const float *d_highs,
                                                     const float *d_lows,
                                                     const float *d_closes,
                                                     float *d_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_identical_three_crows_batch(const float *host_opens,
                                                           const float *host_highs,
                                                           const float *host_lows,
                                                           const float *host_closes,
                                                           float *host_outputs,
                                                           int seriesLen,
                                                           int numSeries,
                                                           int maxConcurrency,
                                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_identical_three_crows_batch_d(const float *d_opens,
                                                             const float *d_highs,
                                                             const float *d_lows,
                                                             const float *d_closes,
                                                             float *d_outputs,
                                                             int seriesLen,
                                                             int numSeries,
                                                             int maxConcurrency,
                                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_in_neck_batch(const float *host_opens,
                                             const float *host_highs,
                                             const float *host_lows,
                                             const float *host_closes,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_in_neck_batch_d(const float *d_opens,
                                               const float *d_highs,
                                               const float *d_lows,
                                               const float *d_closes,
                                               float *d_outputs, int seriesLen,
                                               int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_batch(const float *host_opens,
                                             const float *host_highs,
                                             const float *host_lows,
                                             const float *host_closes,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_batch_d(const float *d_opens,
                                               const float *d_highs,
                                               const float *d_lows,
                                               const float *d_closes,
                                               float *d_outputs, int seriesLen,
                                               int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_by_length_batch(const float *host_opens,
                                                       const float *host_highs,
                                                       const float *host_lows,
                                                       const float *host_closes,
                                                       float *host_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_by_length_batch_d(const float *d_opens,
                                                         const float *d_highs,
                                                         const float *d_lows,
                                                         const float *d_closes,
                                                         float *d_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_ladder_bottom_batch(const float *host_opens,
                                                   const float *host_highs,
                                                   const float *host_lows,
                                                   const float *host_closes,
                                                   float *host_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_ladder_bottom_batch_d(const float *d_opens,
                                                     const float *d_highs,
                                                     const float *d_lows,
                                                     const float *d_closes,
                                                     float *d_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_legged_doji_batch(const float *host_opens,
                                                      const float *host_highs,
                                                      const float *host_lows,
                                                      const float *host_closes,
                                                      float *host_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_legged_doji_batch_d(const float *d_opens,
                                                        const float *d_highs,
                                                        const float *d_lows,
                                                        const float *d_closes,
                                                        float *d_outputs,
                                                        int seriesLen,
                                                        int numSeries,
                                                        int maxConcurrency,
                                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_line_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_long_line_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_marubozu_batch(const float *host_opens,
                                              const float *host_highs,
                                              const float *host_lows,
                                              const float *host_closes,
                                              float *host_outputs,
                                              int seriesLen, int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_marubozu_batch_d(const float *d_opens,
                                                const float *d_highs,
                                                const float *d_lows,
                                                const float *d_closes,
                                                float *d_outputs,
                                                int seriesLen, int numSeries,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_matching_low_batch(const float *host_opens,
                                                  const float *host_highs,
                                                  const float *host_lows,
                                                  const float *host_closes,
                                                  float *host_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_matching_low_batch_d(const float *d_opens,
                                                    const float *d_highs,
                                                    const float *d_lows,
                                                    const float *d_closes,
                                                    float *d_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_mat_hold_batch(const float *host_opens,
                                              const float *host_highs,
                                              const float *host_lows,
                                              const float *host_closes,
                                              float *host_outputs,
                                              int seriesLen, int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_mat_hold_batch_d(const float *d_opens,
                                                const float *d_highs,
                                                const float *d_lows,
                                                const float *d_closes,
                                                float *d_outputs,
                                                int seriesLen, int numSeries,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_doji_star_batch(const float *host_opens,
                                                       const float *host_highs,
                                                       const float *host_lows,
                                                       const float *host_closes,
                                                       float *host_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_doji_star_batch_d(const float *d_opens,
                                                         const float *d_highs,
                                                         const float *d_lows,
                                                         const float *d_closes,
                                                         float *d_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_star_batch(const float *host_opens,
                                                  const float *host_highs,
                                                  const float *host_lows,
                                                  const float *host_closes,
                                                  float *host_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_star_batch_d(const float *d_opens,
                                                    const float *d_highs,
                                                    const float *d_lows,
                                                    const float *d_closes,
                                                    float *d_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_on_neck_batch(const float *host_opens,
                                             const float *host_highs,
                                             const float *host_lows,
                                             const float *host_closes,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_on_neck_batch_d(const float *d_opens,
                                               const float *d_highs,
                                               const float *d_lows,
                                               const float *d_closes,
                                               float *d_outputs, int seriesLen,
                                               int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_piercing_batch(const float *host_opens,
                                              const float *host_highs,
                                              const float *host_lows,
                                              const float *host_closes,
                                              float *host_outputs,
                                              int seriesLen, int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_piercing_batch_d(const float *d_opens,
                                                const float *d_highs,
                                                const float *d_lows,
                                                const float *d_closes,
                                                float *d_outputs,
                                                int seriesLen, int numSeries,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rickshaw_man_batch(const float *host_opens,
                                                  const float *host_highs,
                                                  const float *host_lows,
                                                  const float *host_closes,
                                                  float *host_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rickshaw_man_batch_d(const float *d_opens,
                                                    const float *d_highs,
                                                    const float *d_lows,
                                                    const float *d_closes,
                                                    float *d_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rise_fall3_methods_batch(const float *host_opens,
                                                        const float *host_highs,
                                                        const float *host_lows,
                                                        const float *host_closes,
                                                        float *host_outputs,
                                                        int seriesLen,
                                                        int numSeries,
                                                        int maxConcurrency,
                                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_rise_fall3_methods_batch_d(const float *d_opens,
                                                          const float *d_highs,
                                                          const float *d_lows,
                                                          const float *d_closes,
                                                          float *d_outputs,
                                                          int seriesLen,
                                                          int numSeries,
                                                          int maxConcurrency,
                                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_separating_lines_batch(const float *host_opens,
                                                      const float *host_highs,
                                                      const float *host_lows,
                                                      const float *host_closes,
                                                      float *host_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_separating_lines_batch_d(const float *d_opens,
                                                        const float *d_highs,
                                                        const float *d_lows,
                                                        const float *d_closes,
                                                        float *d_outputs,
                                                        int seriesLen,
                                                        int numSeries,
                                                        int maxConcurrency,
                                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_shooting_star_batch(const float *host_opens,
                                                   const float *host_highs,
                                                   const float *host_lows,
                                                   const float *host_closes,
                                                   float *host_outputs,
                                                   int seriesLen,
                                                   int numSeries,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_shooting_star_batch_d(const float *d_opens,
                                                     const float *d_highs,
                                                     const float *d_lows,
                                                     const float *d_closes,
                                                     float *d_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_short_line_batch(const float *host_opens,
                                                const float *host_highs,
                                                const float *host_lows,
                                                const float *host_closes,
                                                float *host_outputs,
                                                int seriesLen, int numSeries,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_short_line_batch_d(const float *d_opens,
                                                  const float *d_highs,
                                                  const float *d_lows,
                                                  const float *d_closes,
                                                  float *d_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_spinning_top_batch(const float *host_opens,
                                                  const float *host_highs,
                                                  const float *host_lows,
                                                  const float *host_closes,
                                                  float *host_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_spinning_top_batch_d(const float *d_opens,
                                                    const float *d_highs,
                                                    const float *d_lows,
                                                    const float *d_closes,
                                                    float *d_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stalled_pattern_batch(const float *host_opens,
                                                     const float *host_highs,
                                                     const float *host_lows,
                                                     const float *host_closes,
                                                     float *host_outputs,
                                                     int seriesLen,
                                                     int numSeries,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stalled_pattern_batch_d(const float *d_opens,
                                                       const float *d_highs,
                                                       const float *d_lows,
                                                       const float *d_closes,
                                                       float *d_outputs,
                                                       int seriesLen,
                                                       int numSeries,
                                                       int maxConcurrency,
                                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stick_sandwich_batch(const float *host_opens,
                                                    const float *host_highs,
                                                    const float *host_lows,
                                                    const float *host_closes,
                                                    float *host_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_stick_sandwich_batch_d(const float *d_opens,
                                                      const float *d_highs,
                                                      const float *d_lows,
                                                      const float *d_closes,
                                                      float *d_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_takuri_batch(const float *host_opens,
                                            const float *host_highs,
                                            const float *host_lows,
                                            const float *host_closes,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_takuri_batch_d(const float *d_opens,
                                              const float *d_highs,
                                              const float *d_lows,
                                              const float *d_closes,
                                              float *d_outputs, int seriesLen,
                                              int numSeries,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tasuki_gap_batch(const float *host_opens,
                                                const float *host_highs,
                                                const float *host_lows,
                                                const float *host_closes,
                                                float *host_outputs,
                                                int seriesLen, int numSeries,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tasuki_gap_batch_d(const float *d_opens,
                                                  const float *d_highs,
                                                  const float *d_lows,
                                                  const float *d_closes,
                                                  float *d_outputs,
                                                  int seriesLen, int numSeries,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_thrusting_batch(const float *host_opens,
                                               const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_outputs,
                                               int seriesLen, int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_thrusting_batch_d(const float *d_opens,
                                                 const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_outputs,
                                                 int seriesLen, int numSeries,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tristar_batch(const float *host_opens,
                                             const float *host_highs,
                                             const float *host_lows,
                                             const float *host_closes,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_tristar_batch_d(const float *d_opens,
                                               const float *d_highs,
                                               const float *d_lows,
                                               const float *d_closes,
                                               float *d_outputs, int seriesLen,
                                               int numSeries,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_unique_3_river_batch(const float *host_opens,
                                                    const float *host_highs,
                                                    const float *host_lows,
                                                    const float *host_closes,
                                                    float *host_outputs,
                                                    int seriesLen,
                                                    int numSeries,
                                                    int maxConcurrency,
                                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_unique_3_river_batch_d(const float *d_opens,
                                                      const float *d_highs,
                                                      const float *d_lows,
                                                      const float *d_closes,
                                                      float *d_outputs,
                                                      int seriesLen,
                                                      int numSeries,
                                                      int maxConcurrency,
                                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_upside_gap_2_crows_batch(const float *host_opens,
                                                        const float *host_highs,
                                                        const float *host_lows,
                                                        const float *host_closes,
                                                        float *host_outputs,
                                                        int seriesLen,
                                                        int numSeries,
                                                        int maxConcurrency,
                                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_upside_gap_2_crows_batch_d(const float *d_opens,
                                                          const float *d_highs,
                                                          const float *d_lows,
                                                          const float *d_closes,
                                                          float *d_outputs,
                                                          int seriesLen,
                                                          int numSeries,
                                                          int maxConcurrency,
                                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_xside_gap_3_methods_batch(const float *host_opens,
                                                         const float *host_highs,
                                                         const float *host_lows,
                                                         const float *host_closes,
                                                         float *host_outputs,
                                                         int seriesLen,
                                                         int numSeries,
                                                         int maxConcurrency,
                                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cdl_xside_gap_3_methods_batch_d(const float *d_opens,
                                                           const float *d_highs,
                                                           const float *d_lows,
                                                           const float *d_closes,
                                                           float *d_outputs,
                                                           int seriesLen,
                                                           int numSeries,
                                                           int maxConcurrency,
                                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asi(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asi_batch(const float *host_opens,
                                     const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asi_batch_d(const float *d_opens,
                                       const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar_el(const float *host_high,
                                  const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, float accFact, float maxAccel,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar_el_batch(const float *host_highs,
                                        const float *host_lows,
                                        const float *host_closes,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, float accFact,
                                        float maxAccel, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sar_el_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          const float *d_closes,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, float accFact,
                                          float maxAccel, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_obtr(const float *host_high, const float *host_low,
                                const float *host_close, float *host_output,
                                int size,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_obtr_batch(const float *host_highs,
                                      const float *host_lows,
                                      const float *host_closes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_obtr_batch_d(const float *d_highs,
                                        const float *d_lows,
                                        const float *d_closes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wobv(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                const float *host_volume, float *host_output,
                                int size,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wobv_batch(const float *host_opens,
                                      const float *host_highs,
                                      const float *host_lows,
                                      const float *host_closes,
                                      const float *host_volumes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wobv_batch_d(const float *d_opens,
                                        const float *d_highs,
                                        const float *d_lows,
                                        const float *d_closes,
                                        const float *d_volumes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_hpi(const float *host_high, const float *host_low,
                               const float *host_volume, float *host_output,
                               int size, float factor,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_hpi_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, float factor,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_hpi_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, float factor,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_market_strength(const float *host_input,
                                           float *host_output, int size,
                                           int period,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_market_strength_batch(const float *host_inputs,
                                                 float *host_outputs,
                                                 int seriesLen, int numSeries,
                                                 int period,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_market_strength_batch_d(const float *d_inputs,
                                                   float *d_outputs,
                                                   int seriesLen,
                                                   int numSeries, int period,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trendscore(const float *host_input,
                                      float *host_output, int size,
                                      int lookback,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trendscore_batch(const float *host_inputs,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int lookback,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_trendscore_batch_d(const float *d_inputs,
                                              float *d_outputs, int seriesLen,
                                              int numSeries, int lookback,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_refdev(const float *host_input, float *host_output,
                                  int size, int smaPeriod, int sumPeriod,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_refdev_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int smaPeriod,
                                        int sumPeriod, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_refdev_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int smaPeriod,
                                          int sumPeriod, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_skew(const float *host_input, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_skew_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_skew_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kurtosis(const float *host_input,
                                    float *host_output, int size, int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kurtosis_batch(const float *host_inputs,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kurtosis_batch_d(const float *d_inputs,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_high(const float *host_input,
                                      float *host_output, int size,
                                      int strength,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_high_batch(const float *host_inputs,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int strength,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_high_batch_d(const float *d_inputs,
                                              float *d_outputs, int seriesLen,
                                              int numSeries, int strength,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_low(const float *host_input,
                                     float *host_output, int size,
                                     int strength,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_low_batch(const float *host_inputs,
                                           float *host_outputs, int seriesLen,
                                           int numSeries, int strength,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pivot_low_batch_d(const float *d_inputs,
                                             float *d_outputs, int seriesLen,
                                             int numSeries, int strength,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_xaverage(const float *host_input,
                                    float *host_output, int size, int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_xaverage_batch(const float *host_inputs,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_xaverage_batch_d(const float *d_inputs,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_mean(const float *host_input,
                                        float *host_output, int size,
                                        int period,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_mean_batch(const float *host_inputs,
                                              float *host_outputs,
                                              int seriesLen, int numSeries,
                                              int period, int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_mean_batch_d(const float *d_inputs,
                                                float *d_outputs,
                                                int seriesLen, int numSeries,
                                                int period, int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_max(const float *host_input,
                                       float *host_output, int size,
                                       int period,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_max_batch(const float *host_inputs,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int period, int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_max_batch_d(const float *d_inputs,
                                               float *d_outputs, int seriesLen,
                                               int numSeries, int period,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_min(const float *host_input,
                                       float *host_output, int size,
                                       int period,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_min_batch(const float *host_inputs,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int period, int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_min_batch_d(const float *d_inputs,
                                               float *d_outputs, int seriesLen,
                                               int numSeries, int period,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_std(const float *host_input,
                                       float *host_output, int size,
                                       int period,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_std_batch(const float *host_inputs,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int period, int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rolling_std_batch_d(const float *d_inputs,
                                               float *d_outputs, int seriesLen,
                                               int numSeries, int period,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ac(const float *host_high, const float *host_low,
                              float *host_output, int size, int fastPeriod,
                              int slowPeriod, int signalPeriod,
                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ac_batch(const float *host_highs,
                                    const float *host_lows,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int fastPeriod,
                                    int slowPeriod, int signalPeriod,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ac_batch_d(const float *d_highs,
                                      const float *d_lows, float *d_outputs,
                                      int seriesLen, int numSeries,
                                      int fastPeriod, int slowPeriod,
                                      int signalPeriod, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adr(const float *host_high, const float *host_low,
                               float *host_output, int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adr_batch(const float *host_highs,
                                     const float *host_lows,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_adr_batch_d(const float *d_highs,
                                       const float *d_lows, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_alma(const float *host_input, float *host_output,
                                int size, int period, double sigma,
                                double offset,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_alma_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period, double sigma,
                                      double offset, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_alma_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        double sigma, double offset,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ao(const float *host_high, const float *host_low,
                              float *host_output, int size, int fastPeriod,
                              int slowPeriod,
                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ao_batch(const float *host_highs,
                                    const float *host_lows,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int fastPeriod,
                                    int slowPeriod, int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ao_batch_d(const float *d_highs,
                                      const float *d_lows, float *d_outputs,
                                      int seriesLen, int numSeries,
                                      int fastPeriod, int slowPeriod,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_apo_ma_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int fastPeriod,
                                        int slowPeriod, int maType,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_apo_ma_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int fastPeriod,
                                          int slowPeriod, int maType,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asi_wilder(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size,
                                      double limitMove,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asi_wilder_batch(const float *host_opens,
                                            const float *host_highs,
                                            const float *host_lows,
                                            const float *host_closes,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, double limitMove,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_asi_wilder_batch_d(const float *d_opens,
                                              const float *d_highs,
                                              const float *d_lows,
                                              const float *d_closes,
                                              float *d_outputs, int seriesLen,
                                              int numSeries, double limitMove,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands_ma(const float *host_input,
                                     float *host_realUpperBand,
                                     float *host_realMiddleBand,
                                     float *host_realLowerBand, int size,
                                     int period, double nbDevUp,
                                     double nbDevDn, int maType,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands_ma_batch(const float *host_inputs,
                                           float *host_realUpperBand,
                                           float *host_realMiddleBand,
                                           float *host_realLowerBand,
                                           int seriesLen, int numSeries,
                                           int period, double nbDevUp,
                                           double nbDevDn, int maType,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbands_ma_batch_d(const float *d_inputs,
                                             float *d_realUpperBand,
                                             float *d_realMiddleBand,
                                             float *d_realLowerBand,
                                             int seriesLen, int numSeries,
                                             int period, double nbDevUp,
                                             double nbDevDn, int maType,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbw(const float *host_input, float *host_output,
                               int size, int period, double nbDevUp,
                               double nbDevDn, int maType,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbw_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period, double nbDevUp,
                                     double nbDevDn, int maType,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_bbw_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, double nbDevUp,
                                       double nbDevDn, int maType,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cg(const float *host_input, float *host_output,
                              int size, int period,
                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cg_batch(const float *host_inputs,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int period,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cg_batch_d(const float *d_inputs, float *d_outputs,
                                      int seriesLen, int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_chop(const float *host_high, const float *host_low,
                                const float *host_close, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_chop_batch(const float *host_highs,
                                      const float *host_lows,
                                      const float *host_closes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_chop_batch_d(const float *d_highs,
                                        const float *d_lows,
                                        const float *d_closes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_choptr(const float *host_high,
                                  const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_choptr_batch(const float *host_highs,
                                        const float *host_lows,
                                        const float *host_closes,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_choptr_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          const float *d_closes,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cksp(const float *host_high, const float *host_low,
                                const float *host_close, float *host_highStop,
                                float *host_lowStop, int size, int period,
                                double multiplier, int stopPeriod,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cksp_batch(const float *host_highs,
                                      const float *host_lows,
                                      const float *host_closes,
                                      float *host_highStop,
                                      float *host_lowStop, int seriesLen,
                                      int numSeries, int period,
                                      double multiplier, int stopPeriod,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cksp_batch_d(const float *d_highs,
                                        const float *d_lows,
                                        const float *d_closes,
                                        float *d_highStop, float *d_lowStop,
                                        int seriesLen, int numSeries,
                                        int period, double multiplier,
                                        int stopPeriod, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmf(const float *host_high, const float *host_low,
                               const float *host_close,
                               const float *host_volume, float *host_output,
                               int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmf_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmf_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmou(const float *host_input, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmou_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cmou_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_coppock(const float *host_input, float *host_output,
                                   int size, int wmaPeriod, int roc1Period,
                                   int roc2Period,
                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_coppock_batch(const float *host_inputs,
                                         float *host_outputs, int seriesLen,
                                         int numSeries, int wmaPeriod,
                                         int roc1Period, int roc2Period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_coppock_batch_d(const float *d_inputs,
                                           float *d_outputs, int seriesLen,
                                           int numSeries, int wmaPeriod,
                                           int roc1Period, int roc2Period,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_crsi(const float *host_input, float *host_output,
                                int size, int period, int streakPeriod,
                                int rankPeriod,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_crsi_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int streakPeriod, int rankPeriod,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_crsi_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int streakPeriod, int rankPeriod,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cti(const float *host_input, float *host_output,
                               int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cti_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cti_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cumsum(const float *host_input, float *host_output,
                                  int size,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cumsum_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cumsum_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cvi(const float *host_high, const float *host_low,
                               float *host_output, int size, int period,
                               int rocPeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cvi_batch(const float *host_highs,
                                     const float *host_lows,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period, int rocPeriod,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_cvi_batch_d(const float *d_highs,
                                       const float *d_lows, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int rocPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_donchian(const float *host_high,
                                    const float *host_low,
                                    float *host_realUpperBand,
                                    float *host_realMiddleBand,
                                    float *host_realLowerBand, int size,
                                    int period,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_donchian_batch(const float *host_highs,
                                          const float *host_lows,
                                          float *host_realUpperBand,
                                          float *host_realMiddleBand,
                                          float *host_realLowerBand,
                                          int seriesLen, int numSeries,
                                          int period, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_donchian_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            float *d_realUpperBand,
                                            float *d_realMiddleBand,
                                            float *d_realLowerBand,
                                            int seriesLen, int numSeries,
                                            int period, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dosc(const float *host_input, float *host_output,
                                int size, int period, int firstPeriod,
                                int secondPeriod, int signalPeriod,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dosc_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int firstPeriod, int secondPeriod,
                                      int signalPeriod, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dosc_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int firstPeriod, int secondPeriod,
                                        int signalPeriod, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dpo(const float *host_input, float *host_output,
                               int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dpo_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_dpo_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_efi(const float *host_close,
                               const float *host_volume, float *host_output,
                               int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_efi_batch(const float *host_closes,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_efi_batch_d(const float *d_closes,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_emv(const float *host_high, const float *host_low,
                               const float *host_volume, float *host_output,
                               int size, int period, double volumeDivisor,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_emv_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     double volumeDivisor, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_emv_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int period,
                                       double volumeDivisor,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_er(const float *host_input, float *host_output,
                              int size, int period,
                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_er_batch(const float *host_inputs,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, int period,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_er_batch_d(const float *d_inputs, float *d_outputs,
                                      int seriesLen, int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_eri(const float *host_high, const float *host_low,
                               const float *host_close, float *host_bullPower,
                               float *host_bearPower, int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_eri_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_bullPower,
                                     float *host_bearPower, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_eri_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes,
                                       float *d_bullPower, float *d_bearPower,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fisher(const float *host_high,
                                  const float *host_low, float *host_fisher,
                                  float *host_trigger, int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fisher_batch(const float *host_highs,
                                        const float *host_lows,
                                        float *host_fisher,
                                        float *host_trigger, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fisher_batch_d(const float *d_highs,
                                          const float *d_lows, float *d_fisher,
                                          float *d_trigger, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fosc(const float *host_input, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fosc_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fosc_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fractal(const float *host_high,
                                   const float *host_low, int *host_swingHigh,
                                   int *host_swingLow, int size, int leftBars,
                                   int rightBars,
                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fractal_batch(const float *host_highs,
                                         const float *host_lows,
                                         int *host_swingHigh,
                                         int *host_swingLow, int seriesLen,
                                         int numSeries, int leftBars,
                                         int rightBars, int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_fractal_batch_d(const float *d_highs,
                                           const float *d_lows,
                                           int *d_swingHigh, int *d_swingLow,
                                           int seriesLen, int numSeries,
                                           int leftBars, int rightBars,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_frama(const float *host_high, const float *host_low,
                                 float *host_output, int size, int period,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_frama_batch(const float *host_highs,
                                       const float *host_lows,
                                       float *host_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_frama_batch_d(const float *d_highs,
                                         const float *d_lows, float *d_outputs,
                                         int seriesLen, int numSeries,
                                         int period, int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ha(const float *host_open, const float *host_high,
                              const float *host_low, const float *host_close,
                              float *host_haOpen, float *host_haHigh,
                              float *host_haLow, float *host_haClose, int size,
                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ha_batch(const float *host_opens,
                                    const float *host_highs,
                                    const float *host_lows,
                                    const float *host_closes,
                                    float *host_haOpen, float *host_haHigh,
                                    float *host_haLow, float *host_haClose,
                                    int seriesLen, int numSeries,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ha_batch_d(const float *d_opens,
                                      const float *d_highs,
                                      const float *d_lows,
                                      const float *d_closes, float *d_haOpen,
                                      float *d_haHigh, float *d_haLow,
                                      float *d_haClose, int seriesLen,
                                      int numSeries, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_hma(const float *host_input, float *host_output,
                               int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_hma_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_hma_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ibs(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ibs_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ibs_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ichimoku(const float *host_high,
                                    const float *host_low,
                                    float *host_tenkanSen,
                                    float *host_kijunSen,
                                    float *host_senkouSpanA,
                                    float *host_senkouSpanB, int size,
                                    int tenkanPeriod, int kijunPeriod,
                                    int senkouBPeriod,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ichimoku_batch(const float *host_highs,
                                          const float *host_lows,
                                          float *host_tenkanSen,
                                          float *host_kijunSen,
                                          float *host_senkouSpanA,
                                          float *host_senkouSpanB,
                                          int seriesLen, int numSeries,
                                          int tenkanPeriod, int kijunPeriod,
                                          int senkouBPeriod,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ichimoku_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            float *d_tenkanSen,
                                            float *d_kijunSen,
                                            float *d_senkouSpanA,
                                            float *d_senkouSpanB,
                                            int seriesLen, int numSeries,
                                            int tenkanPeriod, int kijunPeriod,
                                            int senkouBPeriod,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kc(const float *host_high, const float *host_low,
                              const float *host_close,
                              float *host_realUpperBand,
                              float *host_realMiddleBand,
                              float *host_realLowerBand, int size, int period,
                              int atrPeriod, double nbDev,
                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kc_batch(const float *host_highs,
                                    const float *host_lows,
                                    const float *host_closes,
                                    float *host_realUpperBand,
                                    float *host_realMiddleBand,
                                    float *host_realLowerBand, int seriesLen,
                                    int numSeries, int period, int atrPeriod,
                                    double nbDev, int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kc_batch_d(const float *d_highs,
                                      const float *d_lows,
                                      const float *d_closes,
                                      float *d_realUpperBand,
                                      float *d_realMiddleBand,
                                      float *d_realLowerBand, int seriesLen,
                                      int numSeries, int period, int atrPeriod,
                                      double nbDev, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kdj(const float *host_high, const float *host_low,
                               const float *host_close, float *host_k,
                               float *host_d, float *host_j, int size,
                               int fastKPeriod, int slowKPeriod,
                               int slowKMAType, int slowDPeriod,
                               int slowDMAType,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kdj_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes, float *host_k,
                                     float *host_d, float *host_j,
                                     int seriesLen, int numSeries,
                                     int fastKPeriod, int slowKPeriod,
                                     int slowKMAType, int slowDPeriod,
                                     int slowDMAType, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kdj_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_k,
                                       float *d_d, float *d_j, int seriesLen,
                                       int numSeries, int fastKPeriod,
                                       int slowKPeriod, int slowKMAType,
                                       int slowDPeriod, int slowDMAType,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kst(const float *host_input, float *host_kst,
                               float *host_kstSignal, int size, int roc1Period,
                               int roc2Period, int roc3Period, int roc4Period,
                               int sma1Period, int sma2Period, int sma3Period,
                               int sma4Period, int signalPeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kst_batch(const float *host_inputs, float *host_kst,
                                     float *host_kstSignal, int seriesLen,
                                     int numSeries, int roc1Period,
                                     int roc2Period, int roc3Period,
                                     int roc4Period, int sma1Period,
                                     int sma2Period, int sma3Period,
                                     int sma4Period, int signalPeriod,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kst_batch_d(const float *d_inputs, float *d_kst,
                                       float *d_kstSignal, int seriesLen,
                                       int numSeries, int roc1Period,
                                       int roc2Period, int roc3Period,
                                       int roc4Period, int sma1Period,
                                       int sma2Period, int sma3Period,
                                       int sma4Period, int signalPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kstext(const float *host_input, float *host_kst,
                                  float *host_kstSignal, int size,
                                  int roc1Period, int roc2Period,
                                  int roc3Period, int roc4Period,
                                  int ma1Period, int ma2Period, int ma3Period,
                                  int ma4Period, int signalPeriod,
                                  int rocmaType, int signalMAType,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kstext_batch(const float *host_inputs,
                                        float *host_kst, float *host_kstSignal,
                                        int seriesLen, int numSeries,
                                        int roc1Period, int roc2Period,
                                        int roc3Period, int roc4Period,
                                        int ma1Period, int ma2Period,
                                        int ma3Period, int ma4Period,
                                        int signalPeriod, int rocmaType,
                                        int signalMAType, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kstext_batch_d(const float *d_inputs, float *d_kst,
                                          float *d_kstSignal, int seriesLen,
                                          int numSeries, int roc1Period,
                                          int roc2Period, int roc3Period,
                                          int roc4Period, int ma1Period,
                                          int ma2Period, int ma3Period,
                                          int ma4Period, int signalPeriod,
                                          int rocmaType, int signalMAType,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kurtosis_unbiased(const float *host_input,
                                             float *host_output, int size,
                                             int period,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kurtosis_unbiased_batch(const float *host_inputs,
                                                   float *host_outputs,
                                                   int seriesLen,
                                                   int numSeries, int period,
                                                   int maxConcurrency,
                                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kurtosis_unbiased_batch_d(const float *d_inputs,
                                                     float *d_outputs,
                                                     int seriesLen,
                                                     int numSeries, int period,
                                                     int maxConcurrency,
                                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kvo(const float *host_high, const float *host_low,
                               const float *host_close,
                               const float *host_volume, float *host_kvo,
                               float *host_kvoSignal, int size, int fastPeriod,
                               int slowPeriod, int signalPeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kvo_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     const float *host_volumes,
                                     float *host_kvo, float *host_kvoSignal,
                                     int seriesLen, int numSeries,
                                     int fastPeriod, int slowPeriod,
                                     int signalPeriod, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_kvo_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes,
                                       const float *d_volumes, float *d_kvo,
                                       float *d_kvoSignal, int seriesLen,
                                       int numSeries, int fastPeriod,
                                       int slowPeriod, int signalPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_marketfi(const float *host_high,
                                    const float *host_low,
                                    const float *host_volume,
                                    float *host_output, int size,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_marketfi_batch(const float *host_highs,
                                          const float *host_lows,
                                          const float *host_volumes,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_marketfi_batch_d(const float *d_highs,
                                            const float *d_lows,
                                            const float *d_volumes,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_massi(const float *host_high, const float *host_low,
                                 float *host_output, int size, int fastPeriod,
                                 int slowPeriod,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_massi_batch(const float *host_highs,
                                       const float *host_lows,
                                       float *host_outputs, int seriesLen,
                                       int numSeries, int fastPeriod,
                                       int slowPeriod, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_massi_batch_d(const float *d_highs,
                                         const float *d_lows, float *d_outputs,
                                         int seriesLen, int numSeries,
                                         int fastPeriod, int slowPeriod,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mcgd(const float *host_input, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mcgd_batch(const float *host_inputs,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_mcgd_batch_d(const float *d_inputs,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_median(const float *host_input, float *host_output,
                                  int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_median_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_median_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentb(const float *host_input,
                                    float *host_output, int size, int period,
                                    double nbDevUp, double nbDevDn, int maType,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentb_batch(const float *host_inputs,
                                          float *host_outputs, int seriesLen,
                                          int numSeries, int period,
                                          double nbDevUp, double nbDevDn,
                                          int maType, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentb_batch_d(const float *d_inputs,
                                            float *d_outputs, int seriesLen,
                                            int numSeries, int period,
                                            double nbDevUp, double nbDevDn,
                                            int maType, int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentile(const float *host_input,
                                      float *host_output, int size, int period,
                                      double percentile,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentile_batch(const float *host_inputs,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int period,
                                            double percentile,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentile_batch_d(const float *d_inputs,
                                              float *d_outputs, int seriesLen,
                                              int numSeries, int period,
                                              double percentile,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentrank(const float *host_input,
                                       float *host_output, int size,
                                       int period,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentrank_batch(const float *host_inputs,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int period, int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_percentrank_batch_d(const float *d_inputs,
                                               float *d_outputs, int seriesLen,
                                               int numSeries, int period,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo_ma_batch(const float *host_inputs,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int fastPeriod,
                                        int slowPeriod, int maType,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ppo_ma_batch_d(const float *d_inputs,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int fastPeriod,
                                          int slowPeriod, int maType,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pso(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size, int fastKPeriod, int emaPeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pso_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int fastKPeriod,
                                     int emaPeriod, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pso_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int fastKPeriod, int emaPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo_ma(const float *host_volume, float *host_output,
                                  int size, int fastPeriod, int slowPeriod,
                                  int maType,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo_ma_batch(const float *host_volumes,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int fastPeriod,
                                        int slowPeriod, int maType,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvo_ma_batch_d(const float *d_volumes,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int fastPeriod,
                                          int slowPeriod, int maType,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvt(const float *host_close,
                               const float *host_volume, float *host_output,
                               int size,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvt_batch(const float *host_closes,
                                     const float *host_volumes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_pvt_batch_d(const float *d_closes,
                                       const float *d_volumes,
                                       float *d_outputs, int seriesLen,
                                       int numSeries, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_qstick(const float *host_open,
                                  const float *host_close, float *host_output,
                                  int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_qstick_batch(const float *host_opens,
                                        const float *host_closes,
                                        float *host_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_qstick_batch_d(const float *d_opens,
                                          const float *d_closes,
                                          float *d_outputs, int seriesLen,
                                          int numSeries, int period,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rma(const float *host_input, float *host_output,
                               int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rma_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rma_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rogerssatchell(const float *host_open,
                                          const float *host_high,
                                          const float *host_low,
                                          const float *host_close,
                                          float *host_output, int size,
                                          int period, double annualization,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rogerssatchell_batch(const float *host_opens,
                                                const float *host_highs,
                                                const float *host_lows,
                                                const float *host_closes,
                                                float *host_outputs,
                                                int seriesLen, int numSeries,
                                                int period,
                                                double annualization,
                                                int maxConcurrency,
                                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rogerssatchell_batch_d(const float *d_opens,
                                                  const float *d_highs,
                                                  const float *d_lows,
                                                  const float *d_closes,
                                                  float *d_outputs,
                                                  int seriesLen, int numSeries,
                                                  int period,
                                                  double annualization,
                                                  int maxConcurrency,
                                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvi(const float *host_input, float *host_output,
                               int size, int period, int stdDevPeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvi_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int stdDevPeriod, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvi_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int stdDevPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvir(const float *host_high, const float *host_low,
                                float *host_output, int size, int period,
                                int stdDevPeriod,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvir_batch(const float *host_highs,
                                      const float *host_lows,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int stdDevPeriod, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvir_batch_d(const float *d_highs,
                                        const float *d_lows, float *d_outputs,
                                        int seriesLen, int numSeries,
                                        int period, int stdDevPeriod,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvol(const float *host_volume, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvol_batch(const float *host_volumes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_rvol_batch_d(const float *d_volumes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_si(const float *host_open, const float *host_high,
                              const float *host_low, const float *host_close,
                              float *host_output, int size, double limitMove,
                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_si_batch(const float *host_opens,
                                    const float *host_highs,
                                    const float *host_lows,
                                    const float *host_closes,
                                    float *host_outputs, int seriesLen,
                                    int numSeries, double limitMove,
                                    int maxConcurrency,
                                    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_si_batch_d(const float *d_opens,
                                      const float *d_highs,
                                      const float *d_lows,
                                      const float *d_closes, float *d_outputs,
                                      int seriesLen, int numSeries,
                                      double limitMove, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_smi(const float *host_high, const float *host_low,
                               const float *host_close, float *host_smi,
                               float *host_smiSignal, int size, int period,
                               int fastPeriod, int slowPeriod,
                               int signalPeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_smi_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes, float *host_smi,
                                     float *host_smiSignal, int seriesLen,
                                     int numSeries, int period, int fastPeriod,
                                     int slowPeriod, int signalPeriod,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_smi_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_smi,
                                       float *d_smiSignal, int seriesLen,
                                       int numSeries, int period,
                                       int fastPeriod, int slowPeriod,
                                       int signalPeriod, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sqzmom(const float *host_high,
                                  const float *host_low,
                                  const float *host_close,
                                  float *host_momentum, int *host_squeeze,
                                  int size, int bbPeriod, double nbDev,
                                  int kcPeriod, double factorWide,
                                  double factorNormal, double factorNarrow,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sqzmom_batch(const float *host_highs,
                                        const float *host_lows,
                                        const float *host_closes,
                                        float *host_momentum,
                                        int *host_squeeze, int seriesLen,
                                        int numSeries, int bbPeriod,
                                        double nbDev, int kcPeriod,
                                        double factorWide, double factorNormal,
                                        double factorNarrow,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_sqzmom_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          const float *d_closes,
                                          float *d_momentum, int *d_squeeze,
                                          int seriesLen, int numSeries,
                                          int bbPeriod, double nbDev,
                                          int kcPeriod, double factorWide,
                                          double factorNormal,
                                          double factorNarrow,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stc(const float *host_input, float *host_output,
                               int size, int fastPeriod, int slowPeriod,
                               int cyclePeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stc_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int fastPeriod,
                                     int slowPeriod, int cyclePeriod,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stc_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int fastPeriod, int slowPeriod,
                                       int cyclePeriod, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochastic_ma(const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_slowK, float *host_slowD,
                                         int size, int fastKPeriod,
                                         int slowKPeriod, int slowKMAType,
                                         int slowDPeriod, int slowDMAType,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochastic_ma_batch(const float *host_highs,
                                               const float *host_lows,
                                               const float *host_closes,
                                               float *host_slowK,
                                               float *host_slowD,
                                               int seriesLen, int numSeries,
                                               int fastKPeriod,
                                               int slowKPeriod,
                                               int slowKMAType,
                                               int slowDPeriod,
                                               int slowDMAType,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochastic_ma_batch_d(const float *d_highs,
                                                 const float *d_lows,
                                                 const float *d_closes,
                                                 float *d_slowK,
                                                 float *d_slowD, int seriesLen,
                                                 int numSeries,
                                                 int fastKPeriod,
                                                 int slowKPeriod,
                                                 int slowKMAType,
                                                 int slowDPeriod,
                                                 int slowDMAType,
                                                 int maxConcurrency,
                                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochf_ma(const float *host_high,
                                     const float *host_low,
                                     const float *host_close,
                                     float *host_fastK, float *host_fastD,
                                     int size, int fastKPeriod,
                                     int fastDPeriod, int fastDMAType,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochf_ma_batch(const float *host_highs,
                                           const float *host_lows,
                                           const float *host_closes,
                                           float *host_fastK,
                                           float *host_fastD, int seriesLen,
                                           int numSeries, int fastKPeriod,
                                           int fastDPeriod, int fastDMAType,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochf_ma_batch_d(const float *d_highs,
                                             const float *d_lows,
                                             const float *d_closes,
                                             float *d_fastK, float *d_fastD,
                                             int seriesLen, int numSeries,
                                             int fastKPeriod, int fastDPeriod,
                                             int fastDMAType,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochrsi_ma(const float *host_input,
                                       float *host_fastK, float *host_fastD,
                                       int size, int period, int fastKPeriod,
                                       int fastDPeriod, int fastDMAType,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochrsi_ma_batch(const float *host_inputs,
                                             float *host_fastK,
                                             float *host_fastD, int seriesLen,
                                             int numSeries, int period,
                                             int fastKPeriod, int fastDPeriod,
                                             int fastDMAType,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_stochrsi_ma_batch_d(const float *d_inputs,
                                               float *d_fastK, float *d_fastD,
                                               int seriesLen, int numSeries,
                                               int period, int fastKPeriod,
                                               int fastDPeriod,
                                               int fastDMAType,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_supertrend(const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_supertrend, int *host_trend,
                                      int size, int period, double multiplier,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_supertrend_batch(const float *host_highs,
                                            const float *host_lows,
                                            const float *host_closes,
                                            float *host_supertrend,
                                            int *host_trend, int seriesLen,
                                            int numSeries, int period,
                                            double multiplier,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_supertrend_batch_d(const float *d_highs,
                                              const float *d_lows,
                                              const float *d_closes,
                                              float *d_supertrend,
                                              int *d_trend, int seriesLen,
                                              int numSeries, int period,
                                              double multiplier,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_2php(const float *host_input,
                                     float *host_output, int size, int period,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_2php_batch(const float *host_inputs,
                                           float *host_outputs, int seriesLen,
                                           int numSeries, int period,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_2php_batch_d(const float *d_inputs,
                                             float *d_outputs, int seriesLen,
                                             int numSeries, int period,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_bp(const float *host_input, float *host_output,
                                   int size, int period, double delta,
                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_bp_batch(const float *host_inputs,
                                         float *host_outputs, int seriesLen,
                                         int numSeries, int period,
                                         double delta, int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_bp_batch_d(const float *d_inputs,
                                           float *d_outputs, int seriesLen,
                                           int numSeries, int period,
                                           double delta, int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_butter(const float *host_input,
                                       float *host_output, int size,
                                       int period,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_butter_batch(const float *host_inputs,
                                             float *host_outputs,
                                             int seriesLen, int numSeries,
                                             int period, int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_butter_batch_d(const float *d_inputs,
                                               float *d_outputs, int seriesLen,
                                               int numSeries, int period,
                                               int maxConcurrency,
                                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_gauss(const float *host_input,
                                      float *host_output, int size, int period,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_gauss_batch(const float *host_inputs,
                                            float *host_outputs, int seriesLen,
                                            int numSeries, int period,
                                            int maxConcurrency,
                                            cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_gauss_batch_d(const float *d_inputs,
                                              float *d_outputs, int seriesLen,
                                              int numSeries, int period,
                                              int maxConcurrency,
                                              cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_hp(const float *host_input, float *host_output,
                                   int size, int period,
                                   cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_hp_batch(const float *host_inputs,
                                         float *host_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_swak_hp_batch_d(const float *d_inputs,
                                           float *d_outputs, int seriesLen,
                                           int numSeries, int period,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tsi(const float *host_input, float *host_output,
                               int size, int firstPeriod, int secondPeriod,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tsi_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int firstPeriod,
                                     int secondPeriod, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_tsi_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int firstPeriod, int secondPeriod,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vhf(const float *host_input, float *host_output,
                               int size, int period,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vhf_batch(const float *host_inputs,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int period,
                                     int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vhf_batch_d(const float *d_inputs, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int period, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vidya(const float *host_input, float *host_output,
                                 int size, int period, int cmoPeriod,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vidya_batch(const float *host_inputs,
                                       float *host_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int cmoPeriod, int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vidya_batch_d(const float *d_inputs,
                                         float *d_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int cmoPeriod, int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vortex(const float *host_high,
                                  const float *host_low,
                                  const float *host_close, float *host_plusVI,
                                  float *host_minusVI, int size, int period,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vortex_batch(const float *host_highs,
                                        const float *host_lows,
                                        const float *host_closes,
                                        float *host_plusVI,
                                        float *host_minusVI, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vortex_batch_d(const float *d_highs,
                                          const float *d_lows,
                                          const float *d_closes,
                                          float *d_plusVI, float *d_minusVI,
                                          int seriesLen, int numSeries,
                                          int period, int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vwap(const float *host_high, const float *host_low,
                                const float *host_close,
                                const float *host_volume, float *host_output,
                                int size,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vwap_batch(const float *host_highs,
                                      const float *host_lows,
                                      const float *host_closes,
                                      const float *host_volumes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vwap_batch_d(const float *d_highs,
                                        const float *d_lows,
                                        const float *d_closes,
                                        const float *d_volumes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vwma(const float *host_input,
                                const float *host_volume, float *host_output,
                                int size, int period,
                                cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vwma_batch(const float *host_inputs,
                                      const float *host_volumes,
                                      float *host_outputs, int seriesLen,
                                      int numSeries, int period,
                                      int maxConcurrency,
                                      cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_vwma_batch_d(const float *d_inputs,
                                        const float *d_volumes,
                                        float *d_outputs, int seriesLen,
                                        int numSeries, int period,
                                        int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wad(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size,
                               cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wad_batch(const float *host_highs,
                                     const float *host_lows,
                                     const float *host_closes,
                                     float *host_outputs, int seriesLen,
                                     int numSeries, int maxConcurrency,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wad_batch_d(const float *d_highs,
                                       const float *d_lows,
                                       const float *d_closes, float *d_outputs,
                                       int seriesLen, int numSeries,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wavetrend(const float *host_high,
                                     const float *host_low,
                                     const float *host_close, float *host_wT1,
                                     float *host_wT2, int size,
                                     int channelPeriod, int averagePeriod,
                                     int signalPeriod,
                                     cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wavetrend_batch(const float *host_highs,
                                           const float *host_lows,
                                           const float *host_closes,
                                           float *host_wT1, float *host_wT2,
                                           int seriesLen, int numSeries,
                                           int channelPeriod,
                                           int averagePeriod, int signalPeriod,
                                           int maxConcurrency,
                                           cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_wavetrend_batch_d(const float *d_highs,
                                             const float *d_lows,
                                             const float *d_closes,
                                             float *d_wT1, float *d_wT2,
                                             int seriesLen, int numSeries,
                                             int channelPeriod,
                                             int averagePeriod,
                                             int signalPeriod,
                                             int maxConcurrency,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_zigzag(const float *host_high,
                                  const float *host_low, float *host_zigZag,
                                  int *host_trend, int *host_pivotIdx,
                                  int size, double sensitivity,
                                  int minTrendLength,
                                  cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_zigzag_batch(const float *host_highs,
                                        const float *host_lows,
                                        float *host_zigZag, int *host_trend,
                                        int *host_pivotIdx, int seriesLen,
                                        int numSeries, double sensitivity,
                                        int minTrendLength, int maxConcurrency,
                                        cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_zigzag_batch_d(const float *d_highs,
                                          const float *d_lows, float *d_zigZag,
                                          int *d_trend, int *d_pivotIdx,
                                          int seriesLen, int numSeries,
                                          double sensitivity,
                                          int minTrendLength,
                                          int maxConcurrency,
                                          cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_zlema(const float *host_input, float *host_output,
                                 int size, int period,
                                 cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_zlema_batch(const float *host_inputs,
                                       float *host_outputs, int seriesLen,
                                       int numSeries, int period,
                                       int maxConcurrency,
                                       cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_zlema_batch_d(const float *d_inputs,
                                         float *d_outputs, int seriesLen,
                                         int numSeries, int period,
                                         int maxConcurrency,
                                         cudaStream_t stream CTAPI_STREAM_DEFAULT);
/* --- END GENERATED BATCH API --- */

#ifdef __cplusplus
} // extern "C"
#endif

#undef CTAPI_STREAM_DEFAULT

#endif // TACUDA_H
