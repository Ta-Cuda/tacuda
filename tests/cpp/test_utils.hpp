/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <tuple>
#include <vector>

void expect_approx_equal(const std::vector<float> &a,
                         const std::vector<float> &b, float eps = 1e-3f);

std::vector<float> dema_ref(const std::vector<float> &in, int period);
std::vector<float> tema_ref(const std::vector<float> &in, int period);
std::vector<float> trix_ref(const std::vector<float> &in, int period);
std::vector<float> kama_ref(const std::vector<float> &in, int period,
                            int fastPeriod, int slowPeriod);
std::vector<float> beta_ref(const std::vector<float> &x,
                            const std::vector<float> &y, int period);
std::vector<float> rsi_ref(const std::vector<float> &in, int period);
std::vector<float> dx_ref(const std::vector<float> &high,
                          const std::vector<float> &low,
                          const std::vector<float> &close, int period);
std::vector<float> adx_ref(const std::vector<float> &high,
                           const std::vector<float> &low,
                           const std::vector<float> &close, int period);
std::vector<float> adxr_talib_ref(const std::vector<float> &high,
                                  const std::vector<float> &low,
                                  const std::vector<float> &close, int period);
std::vector<float> natr_ref(const std::vector<float> &high,
                            const std::vector<float> &low,
                            const std::vector<float> &close, int period);
std::vector<float> atr_ref(const std::vector<float> &high,
                           const std::vector<float> &low,
                           const std::vector<float> &close, int period);
std::pair<std::vector<float>, std::vector<float>>
stoch_ref(const std::vector<float> &high, const std::vector<float> &low,
          const std::vector<float> &close, int fastkPeriod, int slowPeriod);
std::vector<float> t3_ref(const std::vector<float> &in, int period, float vFactor);
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
bbands_ref(const std::vector<float> &in, int period, float upperMul,
           float lowerMul);
std::pair<std::vector<float>, std::vector<float>>
mama_ref(const std::vector<float> &in, float fastLimit, float slowLimit);
std::vector<float> sar_talib_ref(const std::vector<float> &high,
                                 const std::vector<float> &low, float step,
                                 float maxAcc);
std::vector<float> sarext_talib_ref(const std::vector<float> &high,
                                    const std::vector<float> &low,
                                    float startValue, float offsetOnReverse,
                                    float accInitLong, float accLong,
                                    float accMaxLong, float accInitShort,
                                    float accShort, float accMaxShort);
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
macdext_talib_ref(const std::vector<float> &in, int fastPeriod, int fastType,
                  int slowPeriod, int slowType, int signalPeriod,
                  int signalType);
std::vector<float> cdl_ref(const std::vector<float> &open,
                           const std::vector<float> &high,
                           const std::vector<float> &low,
                           const std::vector<float> &close,
                           const char *talibFunc,
                           float penetration = std::numeric_limits<float>::quiet_NaN());
std::vector<float> mavp_ref(const std::vector<float> &in,
                            const std::vector<float> &periods, int minPeriod,
                            int maxPeriod, int maType);
std::vector<float> ht_dcperiod_ref(const std::vector<float> &in);
std::vector<float> ht_dcphase_ref(const std::vector<float> &in);
std::pair<std::vector<float>, std::vector<float>>
ht_phasor_ref(const std::vector<float> &in);
std::pair<std::vector<float>, std::vector<float>>
ht_sine_ref(const std::vector<float> &in);
std::vector<float> ht_trendmode_ref(const std::vector<float> &in);
std::vector<float> ht_trendline_ref(const std::vector<float> &in);
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
macdfix_ref(const std::vector<float> &in, int signalPeriod);
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
macd_ref(const std::vector<float> &in, int fastPeriod, int slowPeriod,
         int signalPeriod);
std::vector<float> macd_line_ref(const std::vector<float> &in, int fastPeriod,
                                 int slowPeriod);
std::vector<float> sar_ref(const std::vector<float> &high,
                           const std::vector<float> &low, float step,
                           float maxAcc);
std::vector<float> adosc_ref(const std::vector<float> &high,
                             const std::vector<float> &low,
                             const std::vector<float> &close,
                             const std::vector<float> &volume, int shortP,
                             int longP);
std::vector<float> ultosc_ref(const std::vector<float> &high,
                              const std::vector<float> &low,
                              const std::vector<float> &close, int shortP,
                              int medP, int longP);
std::vector<float> ad_ref(const std::vector<float> &high,
                          const std::vector<float> &low,
                          const std::vector<float> &close,
                          const std::vector<float> &volume);
std::vector<float> apo_ref(const std::vector<float> &in, int fastP, int slowP);
std::vector<float> ppo_ref(const std::vector<float> &in, int fastP, int slowP);
std::vector<float> pvo_ref(const std::vector<float> &in, int fastP, int slowP);
std::vector<float> aroonosc_ref(const std::vector<float> &high,
                                const std::vector<float> &low, int period);
std::vector<float> adxr_ref(const std::vector<float> &high,
                            const std::vector<float> &low,
                            const std::vector<float> &close, int p);
std::vector<float> avgprice_ref(const std::vector<float> &open,
                                const std::vector<float> &high,
                                const std::vector<float> &low,
                                const std::vector<float> &close);
std::vector<float> typprice_ref(const std::vector<float> &high,
                                const std::vector<float> &low,
                                const std::vector<float> &close);
std::vector<float> wclprice_ref(const std::vector<float> &high,
                                const std::vector<float> &low,
                                const std::vector<float> &close);
std::vector<float> linearreg_ref(const std::vector<float> &in, int period);
std::vector<float> linearreg_slope_ref(const std::vector<float> &in,
                                       int period);
std::vector<float> linearreg_intercept_ref(const std::vector<float> &in,
                                           int period);
std::vector<float> linearreg_angle_ref(const std::vector<float> &in,
                                       int period);
std::vector<float> tsf_ref(const std::vector<float> &in, int period);
std::vector<float> plus_dm_ref(const std::vector<float> &high,
                               const std::vector<float> &low, int period);
std::vector<float> minus_dm_ref(const std::vector<float> &high,
                                const std::vector<float> &low, int period);
std::vector<float> plus_di_ref(const std::vector<float> &high,
                               const std::vector<float> &low,
                               const std::vector<float> &close, int period);
std::vector<float> minus_di_ref(const std::vector<float> &high,
                                const std::vector<float> &low,
                                const std::vector<float> &close, int period);
std::pair<std::vector<float>, std::vector<float>>
stochrsi_ref(const std::vector<float> &in, int rsiPeriod, int kPeriod,
             int dPeriod);
