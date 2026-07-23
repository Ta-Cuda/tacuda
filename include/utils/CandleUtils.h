/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef CANDLEUTILS_H
#define CANDLEUTILS_H

#include <cmath>

enum class CandleRangeType {
  RealBody,
  HighLow,
  Shadows
};

enum class CandleSettingType {
  BodyLong,
  BodyVeryLong,
  BodyShort,
  BodyDoji,
  ShadowLong,
  ShadowVeryLong,
  ShadowShort,
  ShadowVeryShort,
  Near,
  Far,
  Equal
};

struct CandleSettingDef {
  CandleRangeType rangeType;
  int avgPeriod;
  float factor;
};

__host__ __device__ inline CandleSettingDef get_candle_setting(CandleSettingType type) {
  switch (type) {
    case CandleSettingType::BodyLong:
      return {CandleRangeType::RealBody, 10, 1.0f};
    case CandleSettingType::BodyVeryLong:
      return {CandleRangeType::RealBody, 10, 3.0f};
    case CandleSettingType::BodyShort:
      return {CandleRangeType::RealBody, 10, 1.0f};
    case CandleSettingType::BodyDoji:
      return {CandleRangeType::HighLow, 10, 0.1f};
    case CandleSettingType::ShadowLong:
      return {CandleRangeType::RealBody, 0, 1.0f};
    case CandleSettingType::ShadowVeryLong:
      return {CandleRangeType::RealBody, 0, 2.0f};
    case CandleSettingType::ShadowShort:
      return {CandleRangeType::Shadows, 10, 1.0f};
    case CandleSettingType::ShadowVeryShort:
      return {CandleRangeType::HighLow, 10, 0.1f};
    case CandleSettingType::Near:
      return {CandleRangeType::HighLow, 5, 0.2f};
    case CandleSettingType::Far:
      return {CandleRangeType::HighLow, 5, 0.6f};
    case CandleSettingType::Equal:
      return {CandleRangeType::HighLow, 5, 0.05f};
    default:
      return {CandleRangeType::RealBody, 0, 1.0f};
  }
}

__host__ __device__ inline float real_body(float open, float close) {
  return fabsf(close - open);
}

__host__ __device__ inline int candle_color(float open, float close) {
  return close >= open ? 1 : -1;
}

__host__ __device__ inline float upper_shadow(float high, float open,
                                              float close) {
  return high - fmaxf(open, close);
}

__host__ __device__ inline float lower_shadow(float low, float open,
                                              float close) {
  return fminf(open, close) - low;
}

// TA-Lib: Gap between candle BODIES (not shadows)
__host__ __device__ inline bool real_body_gap_down(float currentOpen, float currentClose,
                                                  float prevOpen, float prevClose) {
  return fmaxf(currentOpen, currentClose) < fminf(prevOpen, prevClose);
}

__host__ __device__ inline bool real_body_gap_up(float currentOpen, float currentClose,
                                                float prevOpen, float prevClose) {
  return fminf(currentOpen, currentClose) > fmaxf(prevOpen, prevClose);
}

__host__ __device__ inline float candle_range_value(CandleRangeType type,
                                                   float open, float high,
                                                   float low, float close) {
  switch (type) {
    case CandleRangeType::RealBody:
      return real_body(open, close);
    case CandleRangeType::HighLow:
      return high - low;
    case CandleRangeType::Shadows:
      return upper_shadow(high, open, close) + lower_shadow(low, open, close);
  }
  return 0.0f;
}

__host__ __device__ inline float candle_average(CandleSettingType type,
                                                const float* open,
                                                const float* high,
                                                const float* low,
                                                const float* close,
                                                int idx) {
  CandleSettingDef def = get_candle_setting(type);
  if (idx < 0) idx = 0;

  float divisor = (def.rangeType == CandleRangeType::Shadows) ? 2.0f : 1.0f;

  if (def.avgPeriod <= 0) {
    return def.factor *
           candle_range_value(def.rangeType,
                              open[idx], high[idx], low[idx], close[idx]) /
           divisor;
  }

  int start = idx - def.avgPeriod;
  if (start < 0) start = 0;
  int count = idx - start;
  if (count <= 0) {
    return def.factor *
           candle_range_value(def.rangeType,
                              open[idx], high[idx], low[idx], close[idx]) /
           divisor;
  }

  float sum = 0.0f;
  for (int i = start; i < idx; ++i) {
    sum += candle_range_value(def.rangeType, open[i], high[i], low[i], close[i]);
  }
  float avg = sum / static_cast<float>(count);
  return def.factor * avg / divisor;
}

__host__ __device__ inline int candle_setting_period(CandleSettingType type) {
  return get_candle_setting(type).avgPeriod;
}

__host__ __device__ inline float candle_setting_factor(CandleSettingType type) {
  return get_candle_setting(type).factor;
}

__host__ __device__ inline int morning_doji_star_lookback();

__host__ __device__ inline bool
is_doji(const float* open, const float* high, const float* low,
        const float* close, int idx, float factorScale = 1.0f) {
  if (idx < 0) return false;
  float threshold = candle_average(CandleSettingType::BodyDoji, open, high, low,
                                   close, idx);
  if (factorScale != 1.0f) {
    threshold *= factorScale;
  }
  float body = real_body(open[idx], close[idx]);
  return body <= threshold;
}

__host__ __device__ inline int doji_lookback() {
  return candle_setting_period(CandleSettingType::BodyDoji);
}

__host__ __device__ inline bool is_hammer(const float* open,
                                          const float* high,
                                          const float* low,
                                          const float* close,
                                          int idx) {
  if (idx <= 0) return false;

  float body = real_body(open[idx], close[idx]);
  float lower = lower_shadow(low[idx], open[idx], close[idx]);
  float upper = upper_shadow(high[idx], open[idx], close[idx]);

  bool smallBody = body < candle_average(CandleSettingType::BodyShort, open, high, low, close, idx);
  bool longLower = lower > candle_average(CandleSettingType::ShadowLong, open, high, low, close, idx);
  bool veryShortUpper = upper < candle_average(CandleSettingType::ShadowVeryShort, open, high, low, close, idx);
  float near = candle_average(CandleSettingType::Near, open, high, low, close, idx - 1);
  bool bodyNearPriorLow = fminf(open[idx], close[idx]) <= low[idx - 1] + near;

  return smallBody && longLower && veryShortUpper && bodyNearPriorLow;
}

__host__ __device__ inline int hammer_lookback() {
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int shadowLong = candle_setting_period(CandleSettingType::ShadowLong);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  int near = candle_setting_period(CandleSettingType::Near);
  int maxPeriod = bodyShort;
  if (shadowLong > maxPeriod) maxPeriod = shadowLong;
  if (shadowVeryShort > maxPeriod) maxPeriod = shadowVeryShort;
  if (near > maxPeriod) maxPeriod = near;
  return maxPeriod + 1;
}

__host__ __device__ inline bool is_inverted_hammer(const float* open,
                                                   const float* high,
                                                   const float* low,
                                                   const float* close,
                                                   int idx) {
  if (idx <= 0) return false;

  float body = real_body(open[idx], close[idx]);
  float upper = upper_shadow(high[idx], open[idx], close[idx]);
  float lower = lower_shadow(low[idx], open[idx], close[idx]);

  bool smallBody = body < candle_average(CandleSettingType::BodyShort, open, high, low, close, idx);
  bool longUpper = upper > candle_average(CandleSettingType::ShadowLong, open, high, low, close, idx);
  bool veryShortLower = lower < candle_average(CandleSettingType::ShadowVeryShort, open, high, low, close, idx);
  bool gapDown = real_body_gap_down(open[idx], close[idx], open[idx - 1], close[idx - 1]);

  return smallBody && longUpper && veryShortLower && gapDown;
}

__host__ __device__ inline int inverted_hammer_lookback() {
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int shadowLong = candle_setting_period(CandleSettingType::ShadowLong);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  int maxPeriod = bodyShort;
  if (shadowLong > maxPeriod) maxPeriod = shadowLong;
  if (shadowVeryShort > maxPeriod) maxPeriod = shadowVeryShort;
  return maxPeriod + 1;
}

__host__ __device__ inline float engulfing(const float* open,
                                           const float* close, int idx) {
  if (idx <= 0) return 0.0f;
  int prev = idx - 1;
  float prevOpen = open[prev];
  float prevClose = close[prev];
  float currOpen = open[idx];
  float currClose = close[idx];
  int currColor = candle_color(currOpen, currClose);
  int prevColor = candle_color(prevOpen, prevClose);

  bool bullish =
      (currColor == 1 && prevColor == -1) &&
      (((currClose >= prevOpen) && (currOpen < prevClose)) ||
       ((currClose > prevOpen) && (currOpen <= prevClose)));
  bool bearish =
      (currColor == -1 && prevColor == 1) &&
      (((currOpen >= prevClose) && (currClose < prevOpen)) ||
       ((currOpen > prevClose) && (currClose <= prevOpen)));

  if (bullish || bearish) {
    bool strict =
        (currOpen != prevClose) && (currClose != prevOpen);
    float magnitude = strict ? 100.0f : 80.0f;
    return static_cast<float>(currColor) * magnitude;
  }
  return 0.0f;
}

__host__ __device__ inline bool
is_three_white_soldiers(float o1, float h1, float l1, float c1, float o2,
                        float h2, float l2, float c2, float o3, float h3,
                        float l3, float c3) {
  // TA-Lib: three white candles with consecutively rising closes
  if (!(c1 > o1 && c2 > o2 && c3 > o3)) return false; // all bullish
  if (!(c2 > c1 && c3 > c2)) return false; // consecutive higher closes
  
  float body1 = c1 - o1;
  float body2 = c2 - o2;
  float body3 = c3 - o3;
  float range1 = h1 - l1;
  float range2 = h2 - l2;
  float range3 = h3 - l3;
  
  if (range1 <= 0.0f || range2 <= 0.0f || range3 <= 0.0f) return false;
  
  // TA-Lib: bodies not short (>= 60% of range for static approximation)
  float body_pct1 = body1 / range1;
  float body_pct2 = body2 / range2;
  float body_pct3 = body3 / range3;
  
  if (body_pct1 < 0.6f || body_pct2 < 0.6f || body_pct3 < 0.6f) return false;
  
  // TA-Lib: very short upper shadows (< 5% of range)
  float upper1 = (h1 - c1) / range1;
  float upper2 = (h2 - c2) / range2;
  float upper3 = (h3 - c3) / range3;
  
  if (upper1 > 0.05f || upper2 > 0.05f || upper3 > 0.05f) return false;
  
  // TA-Lib: each open within/near previous body
  // Near tolerance ≈ 20% of previous body
  const float nearFactor = 0.2f;
  if (!(o2 >= o1 && o2 <= c1 + body1 * nearFactor)) return false;
  if (!(o3 >= o2 && o3 <= c2 + body2 * nearFactor)) return false;
  
  // TA-Lib CRITICAL: each body "not far shorter" than previous
  // Far threshold ≈ 30% of previous body
  const float farFactor = 0.3f;
  if (body2 < body1 - body1 * farFactor) return false; // 2nd not far shorter than 1st
  if (body3 < body2 - body2 * farFactor) return false; // 3rd not far shorter than 2nd
  
  return true;
}

// TA-Lib: CDL3WHITESOLDIERS lookback = max(BodyShort, ShadowVeryShort) + 2
__host__ __device__ inline int three_white_soldiers_lookback() {
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  int maxPeriod = bodyShort > shadowVeryShort ? bodyShort : shadowVeryShort;
  return maxPeriod + 2;
}

__host__ __device__ inline bool is_abandoned_baby(const float* open,
                                                 const float* high,
                                                 const float* low,
                                                 const float* close, int idx) {
  if (idx < 2) return false;
  int first = idx - 2;
  int second = idx - 1;
  int third = idx;
  bool bearish1 = close[first] < open[first];
  bool doji2 = is_doji(open, high, low, close, second);
  bool bullish3 = close[third] > open[third];
  bool gapDown = high[second] < low[first];
  bool gapUp = low[third] > high[second];
  return bearish1 && doji2 && bullish3 && gapDown && gapUp;
}

// TA-Lib: CDLABANDONEDBABY lookback = max(BodyLong, BodyDoji) + 2
__host__ __device__ inline int abandoned_baby_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int bodyDoji = candle_setting_period(CandleSettingType::BodyDoji);
  int maxPeriod = bodyLong > bodyDoji ? bodyLong : bodyDoji;
  return maxPeriod + 2;
}

__host__ __device__ inline bool is_advance_block(float o1, float h1, float l1,
                                                 float c1, float o2, float h2,
                                                 float l2, float c2, float o3,
                                                 float h3, float l3, float c3) {
  if (!(c1 > o1 && c2 > o2 && c3 > o3))
    return false;
  if (!(o2 >= o1 && o2 <= c1 && o3 >= o2 && o3 <= c2))
    return false;
  float b1 = c1 - o1;
  float b2 = c2 - o2;
  float b3 = c3 - o3;
  if (!(b1 > b2 && b2 > b3))
    return false;
  return true;
}

// TA-Lib: CDLADVANCEBLOCK lookback = BodyLong + 2
__host__ __device__ inline int advance_block_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 2;
}

__host__ __device__ inline bool is_belt_hold(float open, float high, float low,
                                             float close) {
  float body = fabsf(close - open);
  float range = high - low;
  return open == low && close > open && body > range * 0.5f;
}

// TA-Lib: CDLBELTHOLD lookback = max(BodyLong, ShadowVeryShort, Near)
__host__ __device__ inline int belt_hold_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  int near = candle_setting_period(CandleSettingType::Near);
  int maxPeriod = bodyLong;
  if (shadowVeryShort > maxPeriod) maxPeriod = shadowVeryShort;
  if (near > maxPeriod) maxPeriod = near;
  return maxPeriod;
}

__host__ __device__ inline bool
is_breakaway(float o1, float h1, float l1, float c1, float o2, float h2,
             float l2, float c2, float o3, float h3, float l3, float c3,
             float o4, float h4, float l4, float c4, float o5, float h5,
             float l5, float c5) {
  if (!(c1 < o1 && c2 < o2 && c3 < o3 && c4 < o4 && c5 > o5))
    return false;
  if (!(o2 < c1 && o3 < c2 && o4 < c3))
    return false;
  if (!(c5 > o1))
    return false;
  return true;
}

// TA-Lib: CDLBREAKAWAY lookback = BodyLong + 4
__host__ __device__ inline int breakaway_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 4;
}

__host__ __device__ inline bool is_two_crows(float o1, float h1, float l1,
                                             float c1, float o2, float h2,
                                             float l2, float c2, float o3,
                                             float h3, float l3, float c3) {
  return c1 > o1 && c2 < o2 && c3 < o3 && o2 > c1 && c2 > c1 && o3 < o2 &&
         o3 > c2 && c3 < c2 && c3 > o1 && c3 < c1;
}

// TA-Lib: CDL2CROWS lookback = max(BodyLong, BodyShort) + 2
__host__ __device__ inline int two_crows_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int maxPeriod = bodyLong > bodyShort ? bodyLong : bodyShort;
  return maxPeriod + 2;
}

__host__ __device__ inline bool
is_three_black_crows(float o1, float h1, float l1, float c1, float o2, float h2,
                     float l2, float c2, float o3, float h3, float l3,
                     float c3) {
  // TA-Lib: three black candles with consecutively falling closes
  if (!(c1 < o1 && c2 < o2 && c3 < o3)) return false; // all bearish
  if (!(c2 < c1 && c3 < c2)) return false; // consecutive lower closes
  
  float body1 = o1 - c1;
  float body2 = o2 - c2;
  float body3 = o3 - c3;
  float range1 = h1 - l1;
  float range2 = h2 - l2;
  float range3 = h3 - l3;
  
  if (range1 <= 0.0f || range2 <= 0.0f || range3 <= 0.0f) return false;
  
  // TA-Lib: bodies not short (>= 60% of range for static approximation)
  float body_pct1 = body1 / range1;
  float body_pct2 = body2 / range2;
  float body_pct3 = body3 / range3;
  
  if (body_pct1 < 0.6f || body_pct2 < 0.6f || body_pct3 < 0.6f) return false;
  
  // TA-Lib: very short lower shadows (< 5% of range)
  float lower1 = (c1 - l1) / range1;
  float lower2 = (c2 - l2) / range2;
  float lower3 = (c3 - l3) / range3;
  
  if (lower1 > 0.05f || lower2 > 0.05f || lower3 > 0.05f) return false;
  
  // TA-Lib: each open within/near previous body
  // Near tolerance ≈ 20% of previous body
  const float nearFactor = 0.2f;
  if (!(o2 <= o1 && o2 >= c1 - body1 * nearFactor)) return false;
  if (!(o3 <= o2 && o3 >= c2 - body2 * nearFactor)) return false;
  
  // TA-Lib CRITICAL: each body "not far shorter" than previous
  // Far threshold ≈ 30% of previous body
  const float farFactor = 0.3f;
  if (body2 < body1 - body1 * farFactor) return false; // 2nd not far shorter than 1st
  if (body3 < body2 - body2 * farFactor) return false; // 3rd not far shorter than 2nd
  
  return true;
}

// TA-Lib: CDL3BLACKCROWS lookback = max(BodyShort, ShadowVeryShort) + 3
__host__ __device__ inline int three_black_crows_lookback() {
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  int maxPeriod = bodyShort > shadowVeryShort ? bodyShort : shadowVeryShort;
  return maxPeriod + 3;
}

__host__ __device__ inline bool is_three_inside(float o1, float h1, float l1,
                                                float c1, float o2, float h2,
                                                float l2, float c2, float o3,
                                                float h3, float l3, float c3) {
  return c1 > o1 && c2 < o2 && o2 >= o1 && c2 <= c1 && o2 <= c1 && c2 >= o1 &&
         c3 > o3 && c3 > c1;
}

// TA-Lib: CDL3INSIDE lookback = BodyLong + 2
__host__ __device__ inline int three_inside_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 2;
}

__host__ __device__ inline bool
is_three_line_strike(float o1, float h1, float l1, float c1, float o2, float h2,
                     float l2, float c2, float o3, float h3, float l3, float c3,
                     float o4, float h4, float l4, float c4) {
  bool firstThree = c1 > o1 && c2 > o2 && c3 > o3 && o2 >= o1 && o2 <= c1 &&
                    o3 >= o2 && o3 <= c2 && c1 < c2 && c2 < c3;
  bool fourth = c4 < o4 && o4 >= c3 && c4 < o1;
  return firstThree && fourth;
}

// TA-Lib: CDL3LINESTRIKE lookback = Near + 3
__host__ __device__ inline int three_line_strike_lookback() {
  return candle_setting_period(CandleSettingType::Near) + 3;
}

__host__ __device__ inline bool
is_three_stars_in_south(float o1, float h1, float l1, float c1, float o2,
                        float h2, float l2, float c2, float o3, float h3,
                        float l3, float c3) {
  float ls1 = lower_shadow(l1, o1, c1);
  float ls2 = lower_shadow(l2, o2, c2);
  float ls3 = lower_shadow(l3, o3, c3);
  return c1 < o1 && c2 < o2 && c3 < o3 && c1 > c2 && c2 > c3 && ls1 > ls2 &&
         ls2 > ls3;
}

// TA-Lib: CDL3STARSINSOUTH lookback = BodyLong + 2
__host__ __device__ inline int three_stars_in_south_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 2;
}

__host__ __device__ inline bool is_closing_marubozu(float open, float high,
                                                    float low, float close) {
  return (close == high && open > low) || (close == low && open < high);
}

// TA-Lib: CDLCLOSINGMARUBOZU lookback = max(BodyLong, ShadowVeryShort)
__host__ __device__ inline int closing_marubozu_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  return bodyLong > shadowVeryShort ? bodyLong : shadowVeryShort;
}

// TA-Lib: CDL3OUTSIDE lookback = 3
__host__ __device__ inline int three_outside_lookback() { return 3; }

__host__ __device__ inline bool
is_conceal_baby_swallow(float o1, float h1, float l1, float c1, float o2,
                        float h2, float l2, float c2, float o3, float h3,
                        float l3, float c3, float o4, float h4, float l4,
                        float c4) {
  return c1 < o1 && c2 < o2 && c3 < o3 && c4 < o4 && c1 > c2 && c2 > c3 &&
         c3 > c4;
}

// TA-Lib: CDLCONCEALBABYSWALLOW lookback = max(BodyLong, ShadowVeryShort) + 3
__host__ __device__ inline int conceal_baby_swallow_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  int maxPeriod = bodyLong > shadowVeryShort ? bodyLong : shadowVeryShort;
  return maxPeriod + 3;
}

__host__ __device__ inline bool is_counterattack(float o1, float h1, float l1,
                                                 float c1, float o2, float h2,
                                                 float l2, float c2) {
  bool bullBear = c1 > o1 && c2 < o2;
  bool bearBull = c1 < o1 && c2 > o2;
  return (bullBear || bearBull) && fabsf(c2 - c1) <= 1e-3f;
}

// TA-Lib: CDLCOUNTERATTACK lookback = max(BodyLong, Equal) + 1
__host__ __device__ inline int counterattack_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int equal = candle_setting_period(CandleSettingType::Equal);
  int maxPeriod = bodyLong > equal ? bodyLong : equal;
  return maxPeriod + 1;
}

__host__ __device__ inline bool is_dark_cloud_cover(const float* open,
                                                    const float* high,
                                                    const float* low,
                                                    const float* close,
                                                    int idx,
                                                    float penetration = 0.5f) {
  if (idx <= 0) return false;
  int prev = idx - 1;
  double prevOpen = static_cast<double>(open[prev]);
  double prevClose = static_cast<double>(close[prev]);
  double prevBody = static_cast<double>(real_body(open[prev], close[prev]));
  bool firstWhite =
      prevClose > prevOpen &&
      prevBody > static_cast<double>(candle_average(CandleSettingType::BodyLong, open, high, low, close, prev));
  bool secondBlack = close[idx] < open[idx];
  bool openAboveHigh = open[idx] > high[prev];
  bool closeWithinBody = close[idx] > open[prev];
  double penLimit = prevClose - prevBody * static_cast<double>(penetration);
  bool deepPenetration = static_cast<double>(close[idx]) < penLimit;

  return firstWhite && secondBlack && openAboveHigh && closeWithinBody && deepPenetration;
}

__host__ __device__ inline int dark_cloud_cover_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 1;
}

__host__ __device__ inline bool is_piercing(const float* open,
                                            const float* high,
                                            const float* low,
                                            const float* close,
                                            int idx) {
  if (idx <= 0) return false;
  int prev = idx - 1;

  float prevBody = real_body(open[prev], close[prev]);
  float currBody = real_body(open[idx], close[idx]);

  bool firstBlack =
      close[prev] < open[prev] &&
      prevBody > candle_average(CandleSettingType::BodyLong, open, high, low, close, prev);
  bool secondWhite =
      close[idx] > open[idx] &&
      currBody > candle_average(CandleSettingType::BodyLong, open, high, low, close, idx);
  bool openBelowPrevLow = open[idx] < low[prev];
  bool closeWithinBody = close[idx] < open[prev];
  bool closeAboveMid = close[idx] > close[prev] + prevBody * 0.5f;

  return firstBlack && secondWhite && openBelowPrevLow && closeWithinBody && closeAboveMid;
}

__host__ __device__ inline int piercing_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 1;
}

__host__ __device__ inline float doji_star_value(const float* open,
                                                 const float* high,
                                                 const float* low,
                                                 const float* close,
                                                 int idx) {
  if (idx <= 0) return 0.0f;
  int prev = idx - 1;

  float firstBody = real_body(open[prev], close[prev]);
  bool firstLong = firstBody > candle_average(CandleSettingType::BodyLong, open, high, low, close, prev);

  float secondBody = real_body(open[idx], close[idx]);
  bool secondDoji = secondBody <= candle_average(CandleSettingType::BodyDoji, open, high, low, close, idx);

  int color = candle_color(open[prev], close[prev]);
  bool gapUp = real_body_gap_up(open[idx], close[idx], open[prev], close[prev]);
  bool gapDown = real_body_gap_down(open[idx], close[idx], open[prev], close[prev]);
  bool gapCondition = (color == 1 && gapUp) || (color == -1 && gapDown);

  if (firstLong && secondDoji && gapCondition) {
    return -static_cast<float>(color) * 100.0f;
  }
  return 0.0f;
}

__host__ __device__ inline int doji_star_lookback() {
  int bodyDoji = candle_setting_period(CandleSettingType::BodyDoji);
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int maxPeriod = bodyDoji > bodyLong ? bodyDoji : bodyLong;
  return maxPeriod + 1;
}

__host__ __device__ inline bool is_dragonfly_doji(const float* open,
                                                  const float* high,
                                                  const float* low,
                                                  const float* close, int idx) {
  if (!is_doji(open, high, low, close, idx)) return false;
  float upper = upper_shadow(high[idx], open[idx], close[idx]);
  float lower = lower_shadow(low[idx], open[idx], close[idx]);
  float shadowAvg =
      candle_average(CandleSettingType::ShadowVeryShort, open, high, low, close,
                     idx);
  return upper < shadowAvg && lower > shadowAvg;
}

__host__ __device__ inline int dragonfly_doji_lookback() {
  int bodyDoji = candle_setting_period(CandleSettingType::BodyDoji);
  int shadowShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  return bodyDoji > shadowShort ? bodyDoji : shadowShort;
}

__host__ __device__ inline bool is_evening_star(const float* open,
                                                const float* high,
                                                const float* low,
                                                const float* close,
                                                int idx,
                                                float penetration = 0.3f) {
  if (idx < 2) return false;
  int first = idx - 2;
  int second = idx - 1;
  int third = idx;

  float firstBody = real_body(open[first], close[first]);
  bool firstLong = firstBody > candle_average(CandleSettingType::BodyLong, open, high, low, close, first);
  bool firstWhite = close[first] > open[first];

  bool secondShort = real_body(open[second], close[second]) <=
                     candle_average(CandleSettingType::BodyShort, open, high, low, close, second);
  bool gapUp = real_body_gap_up(open[second], close[second], open[first], close[first]);

  bool thirdLongerThanShort = real_body(open[third], close[third]) >
                              candle_average(CandleSettingType::BodyShort, open, high, low, close, third);
  bool thirdBlack = close[third] < open[third];
  bool closesWithin = close[third] < close[first] - firstBody * penetration;

  return firstLong && firstWhite && secondShort && gapUp && thirdLongerThanShort && thirdBlack && closesWithin;
}

__host__ __device__ inline int evening_star_lookback() {
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int maxPeriod = bodyShort > bodyLong ? bodyShort : bodyLong;
  return maxPeriod + 2;
}

__host__ __device__ inline bool
is_evening_doji_star(const float* open, const float* high, const float* low,
                     const float* close, int idx, float penetration = 0.3f) {
  if (idx < 2) return false;
  int first = idx - 2;
  int second = idx - 1;
  int third = idx;

  float firstBody = real_body(open[first], close[first]);
  bool firstLong = firstBody > candle_average(CandleSettingType::BodyLong, open, high, low, close, first);
  bool firstWhite = close[first] > open[first];

  bool secondDoji = real_body(open[second], close[second]) <=
                    candle_average(CandleSettingType::BodyDoji, open, high, low, close, second);
  bool gapUp = real_body_gap_up(open[second], close[second], open[first], close[first]);

  bool thirdLongerThanShort = real_body(open[third], close[third]) >
                              candle_average(CandleSettingType::BodyShort, open, high, low, close, third);
  bool thirdBlack = close[third] < open[third];
  bool closesWithin = close[third] < close[first] - firstBody * penetration;

  return firstLong && firstWhite && secondDoji && gapUp && thirdLongerThanShort && thirdBlack && closesWithin;
}

__host__ __device__ inline int evening_doji_star_lookback() {
  return morning_doji_star_lookback();
}

__host__ __device__ inline bool is_morning_star(const float* open,
                                                const float* high,
                                                const float* low,
                                                const float* close,
                                                int idx,
                                                float penetration = 0.3f) {
  if (idx < 2) return false;
  int first = idx - 2;
  int second = idx - 1;
  int third = idx;

  float firstBody = real_body(open[first], close[first]);
  bool firstLong = firstBody > candle_average(CandleSettingType::BodyLong, open, high, low, close, first);
  bool firstBlack = close[first] < open[first];

  bool secondShort = real_body(open[second], close[second]) <=
                     candle_average(CandleSettingType::BodyShort, open, high, low, close, second);
  bool gapDown = real_body_gap_down(open[second], close[second], open[first], close[first]);

  bool thirdLongerThanShort = real_body(open[third], close[third]) >
                              candle_average(CandleSettingType::BodyShort, open, high, low, close, third);
  bool thirdWhite = close[third] > open[third];
  bool closesWithin = close[third] > close[first] + firstBody * penetration;

  return firstLong && firstBlack && secondShort && gapDown && thirdLongerThanShort && thirdWhite && closesWithin;
}

__host__ __device__ inline int morning_star_lookback() {
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int maxPeriod = bodyShort > bodyLong ? bodyShort : bodyLong;
  return maxPeriod + 2;
}

__host__ __device__ inline bool
is_morning_doji_star(const float* open, const float* high, const float* low,
                     const float* close, int idx, float penetration = 0.3f) {
  if (idx < 2) return false;
  int first = idx - 2;
  int second = idx - 1;
  int third = idx;

  float firstBody = real_body(open[first], close[first]);
  bool firstLong = firstBody > candle_average(CandleSettingType::BodyLong, open, high, low, close, first);
  bool firstBlack = close[first] < open[first];

  bool secondDoji = real_body(open[second], close[second]) <=
                    candle_average(CandleSettingType::BodyDoji, open, high, low, close, second);
  bool gapDown = real_body_gap_down(open[second], close[second], open[first], close[first]);

  bool thirdLongerThanShort = real_body(open[third], close[third]) >
                              candle_average(CandleSettingType::BodyShort, open, high, low, close, third);
  bool thirdWhite = close[third] > open[third];
  bool closesWithin = close[third] > close[first] + firstBody * penetration;

  return firstLong && firstBlack && secondDoji && gapDown && thirdLongerThanShort && thirdWhite && closesWithin;
}

__host__ __device__ inline int morning_doji_star_lookback() {
  int bodyDoji = candle_setting_period(CandleSettingType::BodyDoji);
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int maxPeriod = bodyDoji;
  if (bodyLong > maxPeriod) maxPeriod = bodyLong;
  if (bodyShort > maxPeriod) maxPeriod = bodyShort;
  return maxPeriod + 2;
}

__host__ __device__ inline bool is_gravestone_doji(const float* open,
                                                   const float* high,
                                                   const float* low,
                                                   const float* close,
                                                   int idx) {
  if (!is_doji(open, high, low, close, idx)) return false;
  float upper = upper_shadow(high[idx], open[idx], close[idx]);
  float lower = lower_shadow(low[idx], open[idx], close[idx]);
  float shadowAvg =
      candle_average(CandleSettingType::ShadowVeryShort, open, high, low, close,
                     idx);
  return lower < shadowAvg && upper > shadowAvg;
}

__host__ __device__ inline int gravestone_doji_lookback() {
  int bodyDoji = candle_setting_period(CandleSettingType::BodyDoji);
  int shadowShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  return bodyDoji > shadowShort ? bodyDoji : shadowShort;
}

__host__ __device__ inline bool is_hanging_man(const float* open,
                                               const float* high,
                                               const float* low,
                                               const float* close,
                                               int idx) {
  if (idx <= 0) return false;

  float body = real_body(open[idx], close[idx]);
  float lower = lower_shadow(low[idx], open[idx], close[idx]);
  float upper = upper_shadow(high[idx], open[idx], close[idx]);

  bool smallBody = body < candle_average(CandleSettingType::BodyShort, open, high, low, close, idx);
  bool longLower = lower > candle_average(CandleSettingType::ShadowLong, open, high, low, close, idx);
  bool veryShortUpper = upper < candle_average(CandleSettingType::ShadowVeryShort, open, high, low, close, idx);
  float near = candle_average(CandleSettingType::Near, open, high, low, close, idx - 1);
  bool bodyNearPriorHigh = fminf(open[idx], close[idx]) >= high[idx - 1] - near;

  return smallBody && longLower && veryShortUpper && bodyNearPriorHigh;
}

__host__ __device__ inline int hanging_man_lookback() {
  return hammer_lookback();
}

__host__ __device__ inline bool
is_gap_side_side_white(float o1, float h1, float l1, float c1, float o2,
                       float h2, float l2, float c2, float o3, float h3,
                       float l3, float c3) {
  return c1 > o1 && c2 > o2 && c3 > o3 && l2 > h1 && l3 > h1 &&
         fabsf(o2 - o3) <= 1e-3f;
}

// TA-Lib: CDLGAPSIDESIDEWHITE lookback = Near + 2
__host__ __device__ inline int gap_side_side_white_lookback() {
  return candle_setting_period(CandleSettingType::Near) + 2;
}

__host__ __device__ inline float harami(const float* open, const float* high,
                                        const float* low, const float* close,
                                        int idx) {
  if (idx <= 0) return 0.0f;
  int first = idx - 1;
  int second = idx;
  float firstBody = real_body(open[first], close[first]);
  float secondBody = real_body(open[second], close[second]);
  bool firstLong = firstBody >
                   candle_average(CandleSettingType::BodyLong, open, high, low,
                                  close, first);
  bool secondShort =
      secondBody <= candle_average(CandleSettingType::BodyShort, open, high,
                                   low, close, second);
  if (!firstLong || !secondShort) return 0.0f;

  float firstMax = fmaxf(open[first], close[first]);
  float firstMin = fminf(open[first], close[first]);
  float secondMax = fmaxf(open[second], close[second]);
  float secondMin = fminf(open[second], close[second]);

  bool strictInside = secondMax < firstMax && secondMin > firstMin;
  bool inside = strictInside || (secondMax <= firstMax && secondMin >= firstMin);
  if (!inside) return 0.0f;

  float magnitude = strictInside ? 100.0f : 80.0f;
  return -static_cast<float>(candle_color(open[first], close[first])) *
         magnitude;
}

__host__ __device__ inline float harami_cross(const float* open,
                                              const float* high,
                                              const float* low,
                                              const float* close, int idx) {
  if (idx <= 0) return 0.0f;
  int first = idx - 1;
  int second = idx;
  float firstBody = real_body(open[first], close[first]);
  float secondBody = real_body(open[second], close[second]);
  bool firstLong = firstBody >
                   candle_average(CandleSettingType::BodyLong, open, high, low,
                                  close, first);
  bool secondDoji =
      secondBody <= candle_average(CandleSettingType::BodyDoji, open, high, low,
                                   close, second);
  if (!firstLong || !secondDoji) return 0.0f;

  float firstMax = fmaxf(open[first], close[first]);
  float firstMin = fminf(open[first], close[first]);
  float secondMax = fmaxf(open[second], close[second]);
  float secondMin = fminf(open[second], close[second]);

  bool strictInside = secondMax < firstMax && secondMin > firstMin;
  bool inside = strictInside || (secondMax <= firstMax && secondMin >= firstMin);
  if (!inside) return 0.0f;

  float magnitude = strictInside ? 100.0f : 80.0f;
  return -static_cast<float>(candle_color(open[first], close[first])) *
         magnitude;
}

__host__ __device__ inline int harami_lookback() {
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  return (bodyShort > bodyLong ? bodyShort : bodyLong) + 1;
}

__host__ __device__ inline int harami_cross_lookback() {
  int bodyDoji = candle_setting_period(CandleSettingType::BodyDoji);
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  return (bodyDoji > bodyLong ? bodyDoji : bodyLong) + 1;
}

__host__ __device__ inline bool is_high_wave(float open, float high, float low,
                                             float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return body <= range * 0.3f && upper >= range * 0.4f && lower >= range * 0.4f;
}

// TA-Lib: CDLHIGHWAVE lookback = BodyShort
__host__ __device__ inline int high_wave_lookback() {
  return candle_setting_period(CandleSettingType::BodyShort);
}

__host__ __device__ inline float hikkake(float o1, float h1, float l1, float c1,
                                         float o2, float h2, float l2, float c2,
                                         float o3, float h3, float l3,
                                         float c3) {
  // Inside bar: candle 2 is completely inside candle 1
  bool inside = h2 < h1 && l2 > l1;
  // Bullish: breakout above candle 1's high
  bool bullish = inside && (h3 > h1 || c3 > h1);
  // Bearish: breakout below candle 1's low  
  bool bearish = inside && (l3 < l1 || c3 < l1);
  if (bullish)
    return 100.0f;
  if (bearish)
    return -100.0f;
  return 0.0f;
}

// TA-Lib: CDLHIKKAKE lookback = 5
__host__ __device__ inline int hikkake_lookback() { return 5; }

__host__ __device__ inline float
hikkake_mod(float o1, float h1, float l1, float c1, float o2, float h2,
            float l2, float c2, float o3, float h3, float l3, float c3,
            float o4, float h4, float l4, float c4) {
  bool inside = h2 <= h1 && l2 >= l1;
  bool bullBreak = inside && h3 > h2 && l3 > l2 && c3 > o3 && c4 > h2;
  bool bearBreak = inside && h3 < h2 && l3 < l2 && c3 < o3 && c4 < l2;
  if (bullBreak)
    return 100.0f;
  if (bearBreak)
    return -100.0f;
  return 0.0f;
}

// TA-Lib: CDLHIKKAKEMOD lookback = 10
__host__ __device__ inline int hikkake_mod_lookback() { return 10; }

__host__ __device__ inline bool is_homing_pigeon(float o1, float h1, float l1,
                                                 float c1, float o2, float h2,
                                                 float l2, float c2) {
  return c1 < o1 && c2 < o2 && o2 >= c1 && o2 <= o1 && c2 >= c1 && c2 <= o1;
}

// TA-Lib: CDLHOMINGPIGEON lookback = BodyLong + 1
__host__ __device__ inline int homing_pigeon_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 1;
}

__host__ __device__ inline bool is_in_neck(float o1, float h1, float l1,
                                           float c1, float o2, float h2,
                                           float l2, float c2) {
  float body1 = real_body(o1, c1);
  return c1 < o1 && c2 > o2 && o2 < l1 && c2 >= c1 && c2 <= c1 + body1 * 0.1f;
}

// TA-Lib: CDLINNECK lookback = BodyLong + 1
__host__ __device__ inline int in_neck_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 1;
}

__host__ __device__ inline bool is_on_neck(float o1, float h1, float l1,
                                           float c1, float o2, float h2,
                                           float l2, float c2) {
  float body1 = real_body(o1, c1);
  return c1 < o1 && c2 > o2 && o2 < l1 && fabsf(c2 - l1) <= body1 * 0.1f;
}

// TA-Lib: CDLONNECK lookback = BodyLong + 1
__host__ __device__ inline int on_neck_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 1;
}

__host__ __device__ inline bool is_marubozu(float open, float high, float low,
                                            float close) {
  return upper_shadow(high, open, close) <= 1e-6f &&
         lower_shadow(low, open, close) <= 1e-6f;
}

// TA-Lib: CDLMARUBOZU lookback = ShadowVeryShort
__host__ __device__ inline int marubozu_lookback() {
  return candle_setting_period(CandleSettingType::ShadowVeryShort);
}

__host__ __device__ inline bool is_matching_low(float o1, float h1, float l1,
                                                float c1, float o2, float h2,
                                                float l2, float c2) {
  return c1 < o1 && c2 < o2 && fabsf(c2 - c1) <= 1e-6f;
}

// TA-Lib: CDLMATCHINGLOW lookback = Equal + 1
__host__ __device__ inline int matching_low_lookback() {
  return candle_setting_period(CandleSettingType::Equal) + 1;
}

__host__ __device__ inline bool
is_mat_hold(float o1, float h1, float l1, float c1, float o2, float h2,
            float l2, float c2, float o3, float h3, float l3, float c3,
            float o4, float h4, float l4, float c4, float o5, float h5,
            float l5, float c5) {
  bool firstBull = c1 > o1;
  bool gapUp = o2 > c1;
  bool smallBear2 = c2 < o2;
  bool down3 = c3 < c2;
  bool down4 = c4 < c3;
  bool aboveLow = l2 > l1 && l3 > l1 && l4 > l1;
  bool finalBull = c5 > o5 && c5 > h1;
  return firstBull && gapUp && smallBear2 && down3 && down4 && aboveLow &&
         finalBull;
}

// TA-Lib: CDLMATHOLD lookback = BodyLong + 4
__host__ __device__ inline int mat_hold_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 4;
}

__host__ __device__ inline bool
is_ladder_bottom(float o1, float h1, float l1, float c1, float o2, float h2,
                 float l2, float c2, float o3, float h3, float l3, float c3,
                 float o4, float h4, float l4, float c4, float o5, float h5,
                 float l5, float c5) {
  return c1 < o1 && c2 < o2 && c3 < o3 && c1 > c2 && c2 > c3 && c4 < o4 &&
         c4 > c3 && c5 > o5 && c5 > o4 && c5 > c4;
}

// TA-Lib: CDLLADDERBOTTOM lookback = BodyLong + 4
__host__ __device__ inline int ladder_bottom_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 4;
}

__host__ __device__ inline bool is_long_legged_doji(const float* open,
                                                    const float* high,
                                                    const float* low,
                                                    const float* close,
                                                    int idx) {
  if (!is_doji(open, high, low, close, idx))
    return false;
  float range = high[idx] - low[idx];
  if (range <= 0.0f)
    return false;
  float upper = upper_shadow(high[idx], open[idx], close[idx]);
  float lower = lower_shadow(low[idx], open[idx], close[idx]);
  return upper >= range * 0.4f && lower >= range * 0.4f;
}

__host__ __device__ inline bool is_long_legged_doji(float open, float high,
                                                    float low, float close) {
  float o[1] = {open};
  float h[1] = {high};
  float l[1] = {low};
  float c[1] = {close};
  return is_long_legged_doji(o, h, l, c, 0);
}

__host__ __device__ inline bool is_long_line(float open, float high, float low,
                                             float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return body >= range * 0.7f && upper <= range * 0.15f &&
         lower <= range * 0.15f;
}

// TA-Lib: CDLLONGLINE lookback = BodyLong
__host__ __device__ inline int long_line_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong);
}

__host__ __device__ inline bool is_kicking(float o1, float h1, float l1,
                                           float c1, float o2, float h2,
                                           float l2, float c2) {
  bool firstBull = c1 > o1;
  bool firstBear = c1 < o1;
  bool secondBull = c2 > o2;
  bool secondBear = c2 < o2;
  bool opposite = (firstBull && secondBear) || (firstBear && secondBull);
  if (!opposite)
    return false;
  if (!is_marubozu(o1, h1, l1, c1) || !is_marubozu(o2, h2, l2, c2))
    return false;
  if (firstBull)
    return o2 < l1;
  else
    return o2 > h1;
}

// TA-Lib: CDLKICKING lookback = max(BodyLong, ShadowVeryShort) + 1
__host__ __device__ inline int kicking_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int shadowVeryShort = candle_setting_period(CandleSettingType::ShadowVeryShort);
  int maxPeriod = bodyLong > shadowVeryShort ? bodyLong : shadowVeryShort;
  return maxPeriod + 1;
}

__host__ __device__ inline bool is_kicking_by_length(float o1, float h1,
                                                     float l1, float c1,
                                                     float o2, float h2,
                                                     float l2, float c2) {
  return is_kicking(o1, h1, l1, c1, o2, h2, l2, c2) &&
         real_body(o2, c2) > real_body(o1, c1);
}

// TA-Lib: CDLKICKINGBYLENGTH lookback = CDLKICKING lookback
__host__ __device__ inline int kicking_by_length_lookback() {
  return kicking_lookback();
}

__host__ __device__ inline bool is_short_line(float open, float high, float low,
                                              float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  // Short line: body and shadows are all relatively small (<= 40% of range each)
  return body <= range * 0.4f && upper <= range * 0.4f && lower <= range * 0.4f;
}

// TA-Lib: CDLSHORTLINE lookback = BodyShort
__host__ __device__ inline int short_line_lookback() {
  return candle_setting_period(CandleSettingType::BodyShort);
}

__host__ __device__ inline bool is_shooting_star(const float* open,
                                                const float* high,
                                                const float* low,
                                                const float* close,
                                                int idx) {
  if (idx <= 0) return false;

  float body = real_body(open[idx], close[idx]);
  float upper = upper_shadow(high[idx], open[idx], close[idx]);
  float lower = lower_shadow(low[idx], open[idx], close[idx]);

  bool smallBody = body < candle_average(CandleSettingType::BodyShort, open, high, low, close, idx);
  bool longUpper = upper > candle_average(CandleSettingType::ShadowLong, open, high, low, close, idx);
  bool veryShortLower = lower < candle_average(CandleSettingType::ShadowVeryShort, open, high, low, close, idx);
  bool gapUp = real_body_gap_up(open[idx], close[idx], open[idx - 1], close[idx - 1]);

  return smallBody && longUpper && veryShortLower && gapUp;
}

__host__ __device__ inline int shooting_star_lookback() {
  return inverted_hammer_lookback();
}

__host__ __device__ inline bool is_rickshaw_man(float open, float high,
                                                float low, float close) {
  if (!is_long_legged_doji(open, high, low, close))
    return false;
  float mid = (high + low) * 0.5f;
  float bodyMid = (open + close) * 0.5f;
  return fabsf(bodyMid - mid) <= (high - low) * 0.1f;
}

// TA-Lib: CDLRICKSHAWMAN lookback = BodyDoji
__host__ __device__ inline int rickshaw_man_lookback() {
  return candle_setting_period(CandleSettingType::BodyDoji);
}

__host__ __device__ inline float separating_lines(float o1, float h1, float l1,
                                                  float c1, float o2, float h2,
                                                  float l2, float c2) {
  bool bullish = c1 < o1 && c2 > o2 && fabsf(o2 - o1) <= 1e-3f && c2 > c1;
  bool bearish = c1 > o1 && c2 < o2 && fabsf(o2 - o1) <= 1e-3f && c2 < c1;
  if (bullish)
    return 100.0f;
  if (bearish)
    return -100.0f;
  return 0.0f;
}

// TA-Lib: CDLSEPARATINGLINES lookback = max(BodyLong, Equal) + 1
__host__ __device__ inline int separating_lines_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int equal = candle_setting_period(CandleSettingType::Equal);
  int maxPeriod = bodyLong > equal ? bodyLong : equal;
  return maxPeriod + 1;
}

__host__ __device__ inline bool is_spinning_top(float open, float high, float low,
                                                float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return body <= range * 0.3f && upper >= range * 0.1f && lower >= range * 0.1f;
}

// TA-Lib: CDLSPINNINGTOP lookback = BodyShort
__host__ __device__ inline int spinning_top_lookback() {
  return candle_setting_period(CandleSettingType::BodyShort);
}

__host__ __device__ inline bool is_stalled_pattern(float o1, float h1, float l1,
                                                  float c1, float o2, float h2,
                                                  float l2, float c2, float o3,
                                                  float h3, float l3, float c3) {
  float b2 = real_body(o2, c2);
  float b3 = real_body(o3, c3);
  return c1 > o1 && c2 > o2 && c3 > o3 && o2 > c1 && o3 > c2 && h2 > h1 &&
         h3 > h2 && b3 < b2 * 0.5f;
}

// TA-Lib: CDLSTALLEDPATTERN lookback = max(BodyLong, BodyShort) + 2
__host__ __device__ inline int stalled_pattern_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int maxPeriod = bodyLong > bodyShort ? bodyLong : bodyShort;
  return maxPeriod + 2;
}

__host__ __device__ inline bool is_stick_sandwich(float o1, float h1, float l1,
                                                  float c1, float o2, float h2,
                                                  float l2, float c2, float o3,
                                                  float h3, float l3, float c3) {
  return c1 < o1 && c2 > o2 && c3 < o3 && o2 > c1 &&
         fabsf(c3 - c1) <= 1e-3f;
}

// TA-Lib: CDLSTICKSANDWICH lookback = Equal + 2
__host__ __device__ inline int stick_sandwich_lookback() {
  return candle_setting_period(CandleSettingType::Equal) + 2;
}

__host__ __device__ inline bool is_takuri(float open, float high, float low,
                                          float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return body <= range * 0.1f && upper <= range * 0.1f && lower >= range * 0.6f;
}

// TA-Lib: CDLTAKURI lookback = BodyDoji
__host__ __device__ inline int takuri_lookback() {
  return candle_setting_period(CandleSettingType::BodyDoji);
}

__host__ __device__ inline bool is_thrusting(float o1, float h1, float l1,
                                            float c1, float o2, float h2,
                                            float l2, float c2) {
  float mid1 = (o1 + c1) * 0.5f;
  return c1 < o1 && c2 > o2 && o2 < l1 && c2 > c1 && c2 < mid1;
}

// TA-Lib: CDLTHRUSTING lookback = max(BodyLong, BodyShort) + 1
__host__ __device__ inline int thrusting_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int maxPeriod = bodyLong > bodyShort ? bodyLong : bodyShort;
  return maxPeriod + 1;
}

__host__ __device__ inline float
tristar(const float* open, const float* high, const float* low,
        const float* close, int idx) {
  if (idx < 2) return 0.0f;
  int first = idx - 2;
  int second = idx - 1;
  int third = idx;
  bool d1 = is_doji(open, high, low, close, first);
  bool d2 = is_doji(open, high, low, close, second);
  bool d3 = is_doji(open, high, low, close, third);
  if (!(d1 && d2 && d3))
    return 0.0f;
  bool gapUp =
      real_body_gap_up(open[second], close[second], open[first], close[first]);
  bool gapDown = real_body_gap_down(open[second], close[second], open[first],
                                    close[first]);
  if (gapUp &&
      fmaxf(open[third], close[third]) < fmaxf(open[second], close[second]))
    return -100.0f;
  if (gapDown &&
      fminf(open[third], close[third]) > fminf(open[second], close[second]))
    return 100.0f;
  return 0.0f;
}

__host__ __device__ inline bool
is_unique_3_river(float o1, float h1, float l1, float c1, float o2, float h2,
                  float l2, float c2, float o3, float h3, float l3, float c3) {
  bool firstBlack = c1 < o1;
  bool secondBlack = c2 < o2;
  bool harami = c2 > c1 && o2 <= o1;
  bool lowerLow = l2 < l1;
  bool thirdWhite = c3 > o3;
  bool openNotLower = o3 > l2;
  return firstBlack && secondBlack && harami && lowerLow && thirdWhite &&
         openNotLower;
}

// TA-Lib: CDLUNIQUE3RIVER lookback = max(BodyLong, BodyShort) + 2
__host__ __device__ inline int unique_3_river_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int maxPeriod = bodyLong > bodyShort ? bodyLong : bodyShort;
  return maxPeriod + 2;
}

__host__ __device__ inline bool is_upside_gap_2_crows(
    float o1, float h1, float l1, float c1, float o2, float h2, float l2,
    float c2, float o3, float h3, float l3, float c3) {
  bool firstWhite = c1 > o1;
  bool secondBlack = c2 < o2;
  bool gapUp = fminf(o2, c2) > fmaxf(o1, c1);
  bool thirdBlack = c3 < o3;
  bool engulf = o3 > o2 && c3 < c2;
  bool closeAboveFirst = c3 > c1;
  return firstWhite && secondBlack && gapUp && thirdBlack && engulf &&
         closeAboveFirst;
}

// TA-Lib: CDLUPSIDEGAP2CROWS lookback = max(BodyLong, BodyShort) + 2
__host__ __device__ inline int upside_gap_2_crows_lookback() {
  int bodyLong = candle_setting_period(CandleSettingType::BodyLong);
  int bodyShort = candle_setting_period(CandleSettingType::BodyShort);
  int maxPeriod = bodyLong > bodyShort ? bodyLong : bodyShort;
  return maxPeriod + 2;
}

__host__ __device__ inline float x_side_gap_3_methods(
    float o1, float h1, float l1, float c1, float o2, float h2, float l2,
    float c2, float o3, float h3, float l3, float c3) {
  int color1 = c1 > o1 ? 1 : (c1 < o1 ? -1 : 0);
  int color2 = c2 > o2 ? 1 : (c2 < o2 ? -1 : 0);
  int color3 = c3 > o3 ? 1 : (c3 < o3 ? -1 : 0);
  bool same12 = color1 != 0 && color1 == color2;
  bool opp3 = color3 == -color2;
  bool openIn2 = o3 < fmaxf(o2, c2) && o3 > fminf(o2, c2);
  bool closeIn1 = c3 < fmaxf(o1, c1) && c3 > fminf(o1, c1);
  bool gapCond =
      (color1 == 1 && fminf(o2, c2) > fmaxf(o1, c1)) ||
      (color1 == -1 && fmaxf(o2, c2) < fminf(o1, c1));
  if (same12 && opp3 && openIn2 && closeIn1 && gapCond)
    return static_cast<float>(color1 * 100);
  return 0.0f;
}

// TA-Lib: CDLXSIDEGAP3METHODS lookback = 2
__host__ __device__ inline int x_side_gap_3_methods_lookback() { return 2; }

__host__ __device__ inline float
tasuki_gap(float o1, float h1, float l1, float c1, float o2, float h2,
           float l2, float c2, float o3, float h3, float l3, float c3) {
  bool bull = c1 > o1 && c2 > o2 && o2 > c1 && c3 < o3 && o3 >= o2 &&
              o3 <= c2 && c3 > c1 && c3 < o2;
  bool bear = c1 < o1 && c2 < o2 && o2 < c1 && c3 > o3 && o3 <= o2 &&
              o3 >= c2 && c3 < c1 && c3 > o2;
  if (bull)
    return 100.0f;
  if (bear)
    return -100.0f;
  return 0.0f;
}

// TA-Lib: CDLTASUKIGAP lookback = Near + 2
__host__ __device__ inline int tasuki_gap_lookback() {
  return candle_setting_period(CandleSettingType::Near) + 2;
}

__host__ __device__ inline float
rise_fall_3_methods(float o1, float h1, float l1, float c1, float o2, float h2,
                    float l2, float c2, float o3, float h3, float l3, float c3,
                    float o4, float h4, float l4, float c4, float o5, float h5,
                    float l5, float c5) {
  bool rising = c1 > o1 && c2 < o2 && c3 < o3 && c4 < o4 && h2 <= h1 &&
                l2 >= l1 && h3 <= h1 && l3 >= l1 && h4 <= h1 && l4 >= l1 &&
                c5 > o5 && c5 > c1;
  bool falling = c1 < o1 && c2 > o2 && c3 > o3 && c4 > o4 && h2 <= h1 &&
                 l2 >= l1 && h3 <= h1 && l3 >= l1 && h4 <= h1 && l4 >= l1 &&
                 c5 < o5 && c5 < c1;
  if (rising)
    return 100.0f;
  if (falling)
    return -100.0f;
  return 0.0f;
}

// TA-Lib: CDLRISEFALL3METHODS lookback = BodyLong + 4
__host__ __device__ inline int rise_fall_3_methods_lookback() {
  return candle_setting_period(CandleSettingType::BodyLong) + 4;
}

// ============================================================================
// Literal TA-Lib ports — each cdl_*_talib below reproduces the decision
// block of the corresponding ta_CDL*.c verbatim. ALL arithmetic runs in
// DOUBLE (like TA-Lib) over the float32-quantized inputs, so borderline
// candles resolve identically and signals match bit-for-bit.
// Return value is the signed TA-Lib signal (-100/0/+-80/+100, HIKKAKE
// confirmations +-200).
// ============================================================================

// TA_CANDLEGAPUP/DOWN: gap between candle RANGES (shadows included).
__host__ __device__ inline bool candle_gap_up(const float* high, const float* low,
                                              int i2, int i1) {
  return low[i2] > high[i1];
}

__host__ __device__ inline bool candle_gap_down(const float* high, const float* low,
                                                int i2, int i1) {
  return high[i2] < low[i1];
}

__host__ __device__ inline double candle_range_value_d(CandleRangeType type,
                                                       double o, double h,
                                                       double l, double c) {
  switch (type) {
    case CandleRangeType::RealBody:
      return fabs(c - o);
    case CandleRangeType::HighLow:
      return h - l;
    case CandleRangeType::Shadows:
      return (h - fmax(o, c)) + (fmin(o, c) - l);
  }
  return 0.0;
}

// TA_CANDLEAVERAGE in double: rolling sum over the avgPeriod bars strictly
// BEFORE `idx`, divided by avgPeriod; the avgPeriod==0 settings use the
// candle at `idx` itself.
__host__ __device__ inline double candle_average_d(CandleSettingType type,
                                                   const float* open,
                                                   const float* high,
                                                   const float* low,
                                                   const float* close,
                                                   int idx) {
  CandleSettingDef def = get_candle_setting(type);
  if (idx < 0) idx = 0;
  const double factor = static_cast<double>(def.factor);
  const double divisor = (def.rangeType == CandleRangeType::Shadows) ? 2.0 : 1.0;

  if (def.avgPeriod <= 0) {
    return factor * candle_range_value_d(def.rangeType, open[idx], high[idx],
                                         low[idx], close[idx]) / divisor;
  }

  int start = idx - def.avgPeriod;
  if (start < 0) start = 0;
  int count = idx - start;
  if (count <= 0) {
    return factor * candle_range_value_d(def.rangeType, open[idx], high[idx],
                                         low[idx], close[idx]) / divisor;
  }

  double sum = 0.0;
  for (int i = start; i < idx; ++i) {
    sum += candle_range_value_d(def.rangeType, open[i], high[i], low[i], close[i]);
  }
  return factor * (sum / static_cast<double>(count)) / divisor;
}

#define TACUDA_CDL_ARGS const float* open, const float* high, const float* low, const float* close, int i
#define TACUDA_AVG(SET, IDX) candle_average_d(CandleSettingType::SET, open, high, low, close, (IDX))
#define TACUDA_RB(IDX) fabs(static_cast<double>(close[(IDX)]) - static_cast<double>(open[(IDX)]))
#define TACUDA_US(IDX) (static_cast<double>(high[(IDX)]) - fmax(static_cast<double>(open[(IDX)]), static_cast<double>(close[(IDX)])))
#define TACUDA_LS(IDX) (fmin(static_cast<double>(open[(IDX)]), static_cast<double>(close[(IDX)])) - static_cast<double>(low[(IDX)]))
#define TACUDA_COLOR(IDX) candle_color(open[(IDX)], close[(IDX)])
#define TACUDA_RBGAPUP(I2, I1) real_body_gap_up(open[(I2)], close[(I2)], open[(I1)], close[(I1)])
#define TACUDA_RBGAPDOWN(I2, I1) real_body_gap_down(open[(I2)], close[(I2)], open[(I1)], close[(I1)])

__host__ __device__ inline int cdl_belt_hold_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      ((TACUDA_COLOR(i) == 1 && TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i)) ||
       (TACUDA_COLOR(i) == -1 && TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i))))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_closing_marubozu_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      ((TACUDA_COLOR(i) == 1 && TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i)) ||
       (TACUDA_COLOR(i) == -1 && TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i))))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_marubozu_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_high_wave_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_US(i) > TACUDA_AVG(ShadowVeryLong, i) &&
      TACUDA_LS(i) > TACUDA_AVG(ShadowVeryLong, i))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_long_legged_doji_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i) &&
      (TACUDA_LS(i) > TACUDA_AVG(ShadowLong, i) ||
       TACUDA_US(i) > TACUDA_AVG(ShadowLong, i)))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_long_line_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowShort, i) &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowShort, i))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_short_line_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowShort, i) &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowShort, i))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_spinning_top_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_US(i) > TACUDA_RB(i) &&
      TACUDA_LS(i) > TACUDA_RB(i))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_rickshaw_man_talib(TACUDA_CDL_ARGS) {
  const double mid = static_cast<double>(low[i]) +
                     (static_cast<double>(high[i]) - static_cast<double>(low[i])) / 2.0;
  if (TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i) &&
      TACUDA_LS(i) > TACUDA_AVG(ShadowLong, i) &&
      TACUDA_US(i) > TACUDA_AVG(ShadowLong, i) &&
      static_cast<double>(fminf(open[i], close[i])) <= mid + TACUDA_AVG(Near, i) &&
      static_cast<double>(fmaxf(open[i], close[i])) >= mid - TACUDA_AVG(Near, i))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_takuri_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      TACUDA_LS(i) > TACUDA_AVG(ShadowVeryLong, i))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_homing_pigeon_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -1 && TACUDA_COLOR(i) == -1 &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_RB(i) <= TACUDA_AVG(BodyShort, i) &&
      open[i] < open[i - 1] && close[i] > close[i - 1])
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_matching_low_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -1 && TACUDA_COLOR(i) == -1 &&
      close[i] <= close[i - 1] + TACUDA_AVG(Equal, i - 1) &&
      close[i] >= close[i - 1] - TACUDA_AVG(Equal, i - 1))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_on_neck_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_COLOR(i) == 1 && open[i] < low[i - 1] &&
      close[i] <= low[i - 1] + TACUDA_AVG(Equal, i - 1) &&
      close[i] >= low[i - 1] - TACUDA_AVG(Equal, i - 1))
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_thrusting_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_COLOR(i) == 1 && open[i] < low[i - 1] &&
      close[i] > close[i - 1] + TACUDA_AVG(Equal, i - 1) &&
      static_cast<double>(close[i]) <=
          static_cast<double>(close[i - 1]) + TACUDA_RB(i - 1) * 0.5)
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_separating_lines_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -TACUDA_COLOR(i) &&
      open[i] <= open[i - 1] + TACUDA_AVG(Equal, i - 1) &&
      open[i] >= open[i - 1] - TACUDA_AVG(Equal, i - 1) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      ((TACUDA_COLOR(i) == 1 && TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i)) ||
       (TACUDA_COLOR(i) == -1 && TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i))))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_counterattack_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -TACUDA_COLOR(i) &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      close[i] <= close[i - 1] + TACUDA_AVG(Equal, i - 1) &&
      close[i] >= close[i - 1] - TACUDA_AVG(Equal, i - 1))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_gap_side_side_white_talib(TACUDA_CDL_ARGS) {
  if (((TACUDA_RBGAPUP(i - 1, i - 2) && TACUDA_RBGAPUP(i, i - 2)) ||
       (TACUDA_RBGAPDOWN(i - 1, i - 2) && TACUDA_RBGAPDOWN(i, i - 2))) &&
      TACUDA_COLOR(i - 1) == 1 && TACUDA_COLOR(i) == 1 &&
      TACUDA_RB(i) >= TACUDA_RB(i - 1) - TACUDA_AVG(Near, i - 1) &&
      TACUDA_RB(i) <= TACUDA_RB(i - 1) + TACUDA_AVG(Near, i - 1) &&
      open[i] >= open[i - 1] - TACUDA_AVG(Equal, i - 1) &&
      open[i] <= open[i - 1] + TACUDA_AVG(Equal, i - 1))
    return TACUDA_RBGAPUP(i - 1, i - 2) ? 100 : -100;
  return 0;
}

__host__ __device__ inline int cdl_upside_gap_2_crows_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == 1 &&
      TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyShort, i - 1) &&
      TACUDA_RBGAPUP(i - 1, i - 2) &&
      TACUDA_COLOR(i) == -1 &&
      open[i] > open[i - 1] && close[i] < close[i - 1] &&
      close[i] > close[i - 2])
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_three_inside_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyShort, i - 1) &&
      fmaxf(close[i - 1], open[i - 1]) < fmaxf(close[i - 2], open[i - 2]) &&
      fminf(close[i - 1], open[i - 1]) > fminf(close[i - 2], open[i - 2]) &&
      ((TACUDA_COLOR(i - 2) == 1 && TACUDA_COLOR(i) == -1 && close[i] < open[i - 2]) ||
       (TACUDA_COLOR(i - 2) == -1 && TACUDA_COLOR(i) == 1 && close[i] > open[i - 2])))
    return -TACUDA_COLOR(i - 2) * 100;
  return 0;
}

__host__ __device__ inline int cdl_three_black_crows_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 3) == 1 &&
      TACUDA_COLOR(i - 2) == -1 &&
      TACUDA_LS(i - 2) < TACUDA_AVG(ShadowVeryShort, i - 2) &&
      TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_LS(i - 1) < TACUDA_AVG(ShadowVeryShort, i - 1) &&
      TACUDA_COLOR(i) == -1 &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      open[i - 1] < open[i - 2] && open[i - 1] > close[i - 2] &&
      open[i] < open[i - 1] && open[i] > close[i - 1] &&
      high[i - 3] > close[i - 2] &&
      close[i - 2] > close[i - 1] && close[i - 1] > close[i])
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_three_stars_in_south_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == -1 && TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_COLOR(i) == -1 &&
      TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_LS(i - 2) > TACUDA_AVG(ShadowLong, i - 2) &&
      TACUDA_RB(i - 1) < TACUDA_RB(i - 2) &&
      open[i - 1] > close[i - 2] && open[i - 1] <= high[i - 2] &&
      low[i - 1] < close[i - 2] &&
      low[i - 1] >= low[i - 2] &&
      TACUDA_LS(i - 1) > TACUDA_AVG(ShadowVeryShort, i - 1) &&
      TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      low[i] > low[i - 1] && high[i] < high[i - 1])
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_tristar_talib(TACUDA_CDL_ARGS) {
  // NOTE (TA-Lib verbatim): all three doji tests use the average anchored
  // at i-2 — including the 2nd and 3rd candles.
  if (TACUDA_RB(i - 2) <= TACUDA_AVG(BodyDoji, i - 2) &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyDoji, i - 2) &&
      TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i - 2)) {
    int out = 0;
    if (TACUDA_RBGAPUP(i - 1, i - 2) &&
        fmaxf(open[i], close[i]) < fmaxf(open[i - 1], close[i - 1]))
      out = -100;
    if (TACUDA_RBGAPDOWN(i - 1, i - 2) &&
        fminf(open[i], close[i]) > fminf(open[i - 1], close[i - 1]))
      out = 100;
    return out;
  }
  return 0;
}

__host__ __device__ inline int cdl_conceal_baby_swallow_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 3) == -1 && TACUDA_COLOR(i - 2) == -1 &&
      TACUDA_COLOR(i - 1) == -1 && TACUDA_COLOR(i) == -1 &&
      TACUDA_LS(i - 3) < TACUDA_AVG(ShadowVeryShort, i - 3) &&
      TACUDA_US(i - 3) < TACUDA_AVG(ShadowVeryShort, i - 3) &&
      TACUDA_LS(i - 2) < TACUDA_AVG(ShadowVeryShort, i - 2) &&
      TACUDA_US(i - 2) < TACUDA_AVG(ShadowVeryShort, i - 2) &&
      TACUDA_RBGAPDOWN(i - 1, i - 2) &&
      TACUDA_US(i - 1) > TACUDA_AVG(ShadowVeryShort, i - 1) &&
      high[i - 1] > close[i - 2] &&
      high[i] > high[i - 1] && low[i] < low[i - 1])
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_abandoned_baby_talib(TACUDA_CDL_ARGS,
                                                        double penetration = 0.3) {
  if (TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyDoji, i - 1) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyShort, i) &&
      ((TACUDA_COLOR(i - 2) == 1 && TACUDA_COLOR(i) == -1 &&
        close[i] < close[i - 2] - TACUDA_RB(i - 2) * penetration &&
        candle_gap_up(high, low, i - 1, i - 2) &&
        candle_gap_down(high, low, i, i - 1)) ||
       (TACUDA_COLOR(i - 2) == -1 && TACUDA_COLOR(i) == 1 &&
        close[i] > close[i - 2] + TACUDA_RB(i - 2) * penetration &&
        candle_gap_down(high, low, i - 1, i - 2) &&
        candle_gap_up(high, low, i, i - 1))))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_advance_block_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == 1 && TACUDA_COLOR(i - 1) == 1 &&
      TACUDA_COLOR(i) == 1 &&
      close[i] > close[i - 1] && close[i - 1] > close[i - 2] &&
      open[i - 1] > open[i - 2] &&
      open[i - 1] <= close[i - 2] + TACUDA_AVG(Near, i - 2) &&
      open[i] > open[i - 1] &&
      open[i] <= close[i - 1] + TACUDA_AVG(Near, i - 1) &&
      TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_US(i - 2) < TACUDA_AVG(ShadowShort, i - 2) &&
      ((TACUDA_RB(i - 1) < TACUDA_RB(i - 2) - TACUDA_AVG(Far, i - 2) &&
        TACUDA_RB(i) < TACUDA_RB(i - 1) + TACUDA_AVG(Near, i - 1)) ||
       (TACUDA_RB(i) < TACUDA_RB(i - 1) - TACUDA_AVG(Far, i - 1)) ||
       (TACUDA_RB(i) < TACUDA_RB(i - 1) && TACUDA_RB(i - 1) < TACUDA_RB(i - 2) &&
        (TACUDA_US(i) > TACUDA_AVG(ShadowShort, i) ||
         TACUDA_US(i - 1) > TACUDA_AVG(ShadowShort, i - 1))) ||
       (TACUDA_RB(i) < TACUDA_RB(i - 1) &&
        TACUDA_US(i) > TACUDA_AVG(ShadowLong, i))))
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_stalled_pattern_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == 1 && TACUDA_COLOR(i - 1) == 1 &&
      TACUDA_COLOR(i) == 1 &&
      close[i] > close[i - 1] && close[i - 1] > close[i - 2] &&
      TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_US(i - 1) < TACUDA_AVG(ShadowVeryShort, i - 1) &&
      open[i - 1] > open[i - 2] &&
      open[i - 1] <= close[i - 2] + TACUDA_AVG(Near, i - 2) &&
      TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      open[i] >= close[i - 1] - TACUDA_RB(i) - TACUDA_AVG(Near, i - 1))
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_mat_hold_talib(TACUDA_CDL_ARGS,
                                                  double penetration = 0.5) {
  if (TACUDA_RB(i - 4) > TACUDA_AVG(BodyLong, i - 4) &&
      TACUDA_RB(i - 3) < TACUDA_AVG(BodyShort, i - 3) &&
      TACUDA_RB(i - 2) < TACUDA_AVG(BodyShort, i - 2) &&
      TACUDA_RB(i - 1) < TACUDA_AVG(BodyShort, i - 1) &&
      TACUDA_COLOR(i - 4) == 1 && TACUDA_COLOR(i - 3) == -1 &&
      TACUDA_COLOR(i) == 1 &&
      TACUDA_RBGAPUP(i - 3, i - 4) &&
      fminf(open[i - 2], close[i - 2]) < close[i - 4] &&
      fminf(open[i - 1], close[i - 1]) < close[i - 4] &&
      fminf(open[i - 2], close[i - 2]) > close[i - 4] - TACUDA_RB(i - 4) * penetration &&
      fminf(open[i - 1], close[i - 1]) > close[i - 4] - TACUDA_RB(i - 4) * penetration &&
      fmaxf(close[i - 2], open[i - 2]) < open[i - 3] &&
      fmaxf(close[i - 1], open[i - 1]) < fmaxf(close[i - 2], open[i - 2]) &&
      open[i] > close[i - 1] &&
      close[i] > fmaxf(fmaxf(high[i - 3], high[i - 2]), high[i - 1]))
    return 100;
  return 0;
}

// HIKKAKE family: TA-Lib is a small state machine (pattern index +
// confirmation within 3 bars, one confirmation per pattern, a new pattern
// overrides pending confirmations). Both are reproduced statelessly from a
// bounded window: the confirmation at bar i requires the most recent
// pattern bar j in [i-3, i-1] with no earlier confirmation in (j, i).

__host__ __device__ inline int cdl_hikkake_pattern_at(const float* high, const float* low,
                                                      int i) {
  if (high[i - 1] < high[i - 2] && low[i - 1] > low[i - 2] &&
      ((high[i] < high[i - 1] && low[i] < low[i - 1]) ||
       (high[i] > high[i - 1] && low[i] > low[i - 1])))
    return 100 * (high[i] < high[i - 1] ? 1 : -1);
  return 0;
}

__host__ __device__ inline int cdl_hikkake_talib(TACUDA_CDL_ARGS) {
  int pat = cdl_hikkake_pattern_at(high, low, i);
  if (pat != 0) return pat;
  // Find the most recent pattern in the previous 3 bars (needs i-2 >= 2).
  for (int j = i - 1; j >= i - 3 && j >= 2; --j) {
    int pj = cdl_hikkake_pattern_at(high, low, j);
    if (pj == 0) continue;
    // Confirmation for pattern at j fires at the FIRST bar in (j, j+3]
    // meeting the close condition; a later bar must see none in between.
    for (int m = j + 1; m <= i; ++m) {
      bool conf = (pj > 0 && close[m] > high[j - 1]) ||
                  (pj < 0 && close[m] < low[j - 1]);
      // A new pattern between j and i would have been found first by the
      // outer loop (j scans from the most recent), so only confirmation
      // ordering matters here.
      if (conf) {
        if (m == i) return pj + (pj > 0 ? 100 : -100);
        break;  // confirmation already consumed before bar i
      }
    }
    break;  // only the most recent pattern can confirm
  }
  return 0;
}

__host__ __device__ inline int cdl_hikkake_mod_pattern_at(TACUDA_CDL_ARGS) {
  if (high[i - 2] < high[i - 3] && low[i - 2] > low[i - 3] &&
      high[i - 1] < high[i - 2] && low[i - 1] > low[i - 2] &&
      ((high[i] < high[i - 1] && low[i] < low[i - 1] &&
        close[i - 2] <= low[i - 2] + TACUDA_AVG(Near, i - 2)) ||
       (high[i] > high[i - 1] && low[i] > low[i - 1] &&
        close[i - 2] >= high[i - 2] - TACUDA_AVG(Near, i - 2))))
    return 100 * (high[i] < high[i - 1] ? 1 : -1);
  return 0;
}

__host__ __device__ inline int cdl_hikkake_mod_talib(TACUDA_CDL_ARGS) {
  int pat = cdl_hikkake_mod_pattern_at(open, high, low, close, i);
  if (pat != 0) return pat;
  for (int j = i - 1; j >= i - 3 && j >= 3; --j) {
    int pj = cdl_hikkake_mod_pattern_at(open, high, low, close, j);
    if (pj == 0) continue;
    for (int m = j + 1; m <= i; ++m) {
      bool conf = (pj > 0 && close[m] > high[j - 1]) ||
                  (pj < 0 && close[m] < low[j - 1]);
      if (conf) {
        if (m == i) return pj + (pj > 0 ? 100 : -100);
        break;
      }
    }
    break;
  }
  return 0;
}

// ---- Remaining literal ports (0.7.0): all 61 TA-Lib patterns covered ----

__host__ __device__ inline int cdl_doji_talib(TACUDA_CDL_ARGS) {
  return (TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i)) ? 100 : 0;
}

__host__ __device__ inline int cdl_hammer_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_LS(i) > TACUDA_AVG(ShadowLong, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      static_cast<double>(fminf(close[i], open[i])) <=
          static_cast<double>(low[i - 1]) + TACUDA_AVG(Near, i - 1))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_inverted_hammer_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_US(i) > TACUDA_AVG(ShadowLong, i) &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      TACUDA_RBGAPDOWN(i, i - 1))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_hanging_man_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_LS(i) > TACUDA_AVG(ShadowLong, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      static_cast<double>(fminf(close[i], open[i])) >=
          static_cast<double>(high[i - 1]) - TACUDA_AVG(Near, i - 1))
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_shooting_star_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_US(i) > TACUDA_AVG(ShadowLong, i) &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      TACUDA_RBGAPUP(i, i - 1))
    return -100;
  return 0;
}

// TA-Lib ENGULFING: +-100 for a strict engulf, +-80 when one real-body end
// matches exactly.
__host__ __device__ inline int cdl_engulfing_talib(TACUDA_CDL_ARGS) {
  bool white = TACUDA_COLOR(i) == 1 && TACUDA_COLOR(i - 1) == -1 &&
               ((close[i] >= open[i - 1] && open[i] < close[i - 1]) ||
                (close[i] > open[i - 1] && open[i] <= close[i - 1]));
  bool black = TACUDA_COLOR(i) == -1 && TACUDA_COLOR(i - 1) == 1 &&
               ((open[i] >= close[i - 1] && close[i] < open[i - 1]) ||
                (open[i] > close[i - 1] && close[i] <= open[i - 1]));
  if (white || black) {
    if (open[i] != close[i - 1] && close[i] != open[i - 1])
      return TACUDA_COLOR(i) * 100;
    return TACUDA_COLOR(i) * 80;
  }
  return 0;
}

__host__ __device__ inline int cdl_three_white_soldiers_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == 1 &&
      TACUDA_US(i - 2) < TACUDA_AVG(ShadowVeryShort, i - 2) &&
      TACUDA_COLOR(i - 1) == 1 &&
      TACUDA_US(i - 1) < TACUDA_AVG(ShadowVeryShort, i - 1) &&
      TACUDA_COLOR(i) == 1 &&
      TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      close[i] > close[i - 1] && close[i - 1] > close[i - 2] &&
      open[i - 1] > open[i - 2] &&
      static_cast<double>(open[i - 1]) <=
          static_cast<double>(close[i - 2]) + TACUDA_AVG(Near, i - 2) &&
      open[i] > open[i - 1] &&
      static_cast<double>(open[i]) <=
          static_cast<double>(close[i - 1]) + TACUDA_AVG(Near, i - 1) &&
      TACUDA_RB(i - 1) > TACUDA_RB(i - 2) - TACUDA_AVG(Far, i - 2) &&
      TACUDA_RB(i) > TACUDA_RB(i - 1) - TACUDA_AVG(Far, i - 1) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyShort, i))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_dark_cloud_cover_talib(TACUDA_CDL_ARGS,
                                                          double penetration = 0.5) {
  if (TACUDA_COLOR(i - 1) == 1 &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_COLOR(i) == -1 &&
      open[i] > high[i - 1] &&
      close[i] > open[i - 1] &&
      static_cast<double>(close[i]) <
          static_cast<double>(close[i - 1]) - TACUDA_RB(i - 1) * penetration)
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_doji_star_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i) &&
      ((TACUDA_COLOR(i - 1) == 1 && TACUDA_RBGAPUP(i, i - 1)) ||
       (TACUDA_COLOR(i - 1) == -1 && TACUDA_RBGAPDOWN(i, i - 1))))
    return -TACUDA_COLOR(i - 1) * 100;
  return 0;
}

__host__ __device__ inline int cdl_dragonfly_doji_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i) &&
      TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      TACUDA_LS(i) > TACUDA_AVG(ShadowVeryShort, i))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_gravestone_doji_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i) &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      TACUDA_US(i) > TACUDA_AVG(ShadowVeryShort, i))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_evening_doji_star_talib(TACUDA_CDL_ARGS,
                                                           double penetration = 0.3) {
  if (TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_COLOR(i - 2) == 1 &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyDoji, i - 1) &&
      TACUDA_RBGAPUP(i - 1, i - 2) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyShort, i) &&
      TACUDA_COLOR(i) == -1 &&
      static_cast<double>(close[i]) <
          static_cast<double>(close[i - 2]) - TACUDA_RB(i - 2) * penetration)
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_evening_star_talib(TACUDA_CDL_ARGS,
                                                      double penetration = 0.3) {
  if (TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_COLOR(i - 2) == 1 &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyShort, i - 1) &&
      TACUDA_RBGAPUP(i - 1, i - 2) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyShort, i) &&
      TACUDA_COLOR(i) == -1 &&
      static_cast<double>(close[i]) <
          static_cast<double>(close[i - 2]) - TACUDA_RB(i - 2) * penetration)
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_morning_doji_star_talib(TACUDA_CDL_ARGS,
                                                           double penetration = 0.3) {
  if (TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_COLOR(i - 2) == -1 &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyDoji, i - 1) &&
      TACUDA_RBGAPDOWN(i - 1, i - 2) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyShort, i) &&
      TACUDA_COLOR(i) == 1 &&
      static_cast<double>(close[i]) >
          static_cast<double>(close[i - 2]) + TACUDA_RB(i - 2) * penetration)
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_morning_star_talib(TACUDA_CDL_ARGS,
                                                      double penetration = 0.3) {
  if (TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_COLOR(i - 2) == -1 &&
      TACUDA_RB(i - 1) <= TACUDA_AVG(BodyShort, i - 1) &&
      TACUDA_RBGAPDOWN(i - 1, i - 2) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyShort, i) &&
      TACUDA_COLOR(i) == 1 &&
      static_cast<double>(close[i]) >
          static_cast<double>(close[i - 2]) + TACUDA_RB(i - 2) * penetration)
    return 100;
  return 0;
}

// TA-Lib HARAMI: +-100 for strict engulfing of the 2nd body, +-80 when one
// end matches exactly.
__host__ __device__ inline int cdl_harami_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_RB(i) <= TACUDA_AVG(BodyShort, i)) {
    float hi2 = fmaxf(close[i], open[i]);
    float lo2 = fminf(close[i], open[i]);
    float hi1 = fmaxf(close[i - 1], open[i - 1]);
    float lo1 = fminf(close[i - 1], open[i - 1]);
    if (hi2 < hi1 && lo2 > lo1)
      return -TACUDA_COLOR(i - 1) * 100;
    if (hi2 <= hi1 && lo2 >= lo1)
      return -TACUDA_COLOR(i - 1) * 80;
  }
  return 0;
}

__host__ __device__ inline int cdl_harami_cross_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_RB(i) <= TACUDA_AVG(BodyDoji, i)) {
    float hi2 = fmaxf(close[i], open[i]);
    float lo2 = fminf(close[i], open[i]);
    float hi1 = fmaxf(close[i - 1], open[i - 1]);
    float lo1 = fminf(close[i - 1], open[i - 1]);
    if (hi2 < hi1 && lo2 > lo1)
      return -TACUDA_COLOR(i - 1) * 100;
    if (hi2 <= hi1 && lo2 >= lo1)
      return -TACUDA_COLOR(i - 1) * 80;
  }
  return 0;
}

__host__ __device__ inline bool cdl_kicking_common_talib(TACUDA_CDL_ARGS) {
  return TACUDA_COLOR(i - 1) == -TACUDA_COLOR(i) &&
         TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
         TACUDA_US(i - 1) < TACUDA_AVG(ShadowVeryShort, i - 1) &&
         TACUDA_LS(i - 1) < TACUDA_AVG(ShadowVeryShort, i - 1) &&
         TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
         TACUDA_US(i) < TACUDA_AVG(ShadowVeryShort, i) &&
         TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i) &&
         ((TACUDA_COLOR(i - 1) == -1 && candle_gap_up(high, low, i, i - 1)) ||
          (TACUDA_COLOR(i - 1) == 1 && candle_gap_down(high, low, i, i - 1)));
}

__host__ __device__ inline int cdl_kicking_talib(TACUDA_CDL_ARGS) {
  if (cdl_kicking_common_talib(open, high, low, close, i))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

// Signal color follows the LONGER marubozu.
__host__ __device__ inline int cdl_kicking_by_length_talib(TACUDA_CDL_ARGS) {
  if (cdl_kicking_common_talib(open, high, low, close, i))
    return TACUDA_COLOR(TACUDA_RB(i) > TACUDA_RB(i - 1) ? i : i - 1) * 100;
  return 0;
}

// Three black candles with very short lower shadows, declining closes,
// each opening "very close to" (within the Equal average of) the prior
// close. Literal port of ta_CDLIDENTICAL3CROWS.c.
__host__ __device__ inline int cdl_identical_three_crows_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == -1 &&
      TACUDA_LS(i - 2) < TACUDA_AVG(ShadowVeryShort, i - 2) &&
      TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_LS(i - 1) < TACUDA_AVG(ShadowVeryShort, i - 1) &&
      TACUDA_COLOR(i) == -1 &&
      TACUDA_LS(i) < TACUDA_AVG(ShadowVeryShort, i) &&
      close[i - 2] > close[i - 1] && close[i - 1] > close[i] &&
      static_cast<double>(open[i - 1]) <=
          static_cast<double>(close[i - 2]) + TACUDA_AVG(Equal, i - 2) &&
      static_cast<double>(open[i - 1]) >=
          static_cast<double>(close[i - 2]) - TACUDA_AVG(Equal, i - 2) &&
      static_cast<double>(open[i]) <=
          static_cast<double>(close[i - 1]) + TACUDA_AVG(Equal, i - 1) &&
      static_cast<double>(open[i]) >=
          static_cast<double>(close[i - 1]) - TACUDA_AVG(Equal, i - 1))
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_ladder_bottom_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 4) == -1 && TACUDA_COLOR(i - 3) == -1 &&
      TACUDA_COLOR(i - 2) == -1 &&
      open[i - 4] > open[i - 3] && open[i - 3] > open[i - 2] &&
      close[i - 4] > close[i - 3] && close[i - 3] > close[i - 2] &&
      TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_US(i - 1) > TACUDA_AVG(ShadowVeryShort, i - 1) &&
      TACUDA_COLOR(i) == 1 &&
      open[i] > open[i - 1] &&
      close[i] > high[i - 1])
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_in_neck_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_COLOR(i) == 1 && open[i] < low[i - 1] &&
      static_cast<double>(close[i]) <=
          static_cast<double>(close[i - 1]) + TACUDA_AVG(Equal, i - 1) &&
      close[i] >= close[i - 1])
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_piercing_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_RB(i - 1) > TACUDA_AVG(BodyLong, i - 1) &&
      TACUDA_COLOR(i) == 1 &&
      TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      open[i] < low[i - 1] &&
      close[i] < open[i - 1] &&
      static_cast<double>(close[i]) >
          static_cast<double>(close[i - 1]) + TACUDA_RB(i - 1) * 0.5)
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_stick_sandwich_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == -1 && TACUDA_COLOR(i - 1) == 1 &&
      TACUDA_COLOR(i) == -1 &&
      low[i - 1] > close[i - 2] &&
      static_cast<double>(close[i]) <=
          static_cast<double>(close[i - 2]) + TACUDA_AVG(Equal, i - 2) &&
      static_cast<double>(close[i]) >=
          static_cast<double>(close[i - 2]) - TACUDA_AVG(Equal, i - 2))
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_tasuki_gap_talib(TACUDA_CDL_ARGS) {
  if ((TACUDA_RBGAPUP(i - 1, i - 2) &&
       TACUDA_COLOR(i - 1) == 1 && TACUDA_COLOR(i) == -1 &&
       open[i] < close[i - 1] && open[i] > open[i - 1] &&
       close[i] < open[i - 1] &&
       close[i] > fmaxf(close[i - 2], open[i - 2]) &&
       fabs(TACUDA_RB(i - 1) - TACUDA_RB(i)) < TACUDA_AVG(Near, i - 1)) ||
      (TACUDA_RBGAPDOWN(i - 1, i - 2) &&
       TACUDA_COLOR(i - 1) == -1 && TACUDA_COLOR(i) == 1 &&
       open[i] < open[i - 1] && open[i] > close[i - 1] &&
       close[i] > open[i - 1] &&
       close[i] < fminf(close[i - 2], open[i - 2]) &&
       fabs(TACUDA_RB(i - 1) - TACUDA_RB(i)) < TACUDA_AVG(Near, i - 1)))
    return TACUDA_COLOR(i - 1) * 100;
  return 0;
}

__host__ __device__ inline int cdl_three_line_strike_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 3) == TACUDA_COLOR(i - 2) &&
      TACUDA_COLOR(i - 2) == TACUDA_COLOR(i - 1) &&
      TACUDA_COLOR(i) == -TACUDA_COLOR(i - 1) &&
      static_cast<double>(open[i - 2]) >=
          static_cast<double>(fminf(open[i - 3], close[i - 3])) - TACUDA_AVG(Near, i - 3) &&
      static_cast<double>(open[i - 2]) <=
          static_cast<double>(fmaxf(open[i - 3], close[i - 3])) + TACUDA_AVG(Near, i - 3) &&
      static_cast<double>(open[i - 1]) >=
          static_cast<double>(fminf(open[i - 2], close[i - 2])) - TACUDA_AVG(Near, i - 2) &&
      static_cast<double>(open[i - 1]) <=
          static_cast<double>(fmaxf(open[i - 2], close[i - 2])) + TACUDA_AVG(Near, i - 2) &&
      ((TACUDA_COLOR(i - 1) == 1 &&
        close[i - 1] > close[i - 2] && close[i - 2] > close[i - 3] &&
        open[i] > close[i - 1] && close[i] < open[i - 3]) ||
       (TACUDA_COLOR(i - 1) == -1 &&
        close[i - 1] < close[i - 2] && close[i - 2] < close[i - 3] &&
        open[i] < close[i - 1] && close[i] > open[i - 3])))
    return TACUDA_COLOR(i - 1) * 100;
  return 0;
}

__host__ __device__ inline int cdl_three_outside_talib(TACUDA_CDL_ARGS) {
  if ((TACUDA_COLOR(i - 1) == 1 && TACUDA_COLOR(i - 2) == -1 &&
       close[i - 1] > open[i - 2] && open[i - 1] < close[i - 2] &&
       close[i] > close[i - 1]) ||
      (TACUDA_COLOR(i - 1) == -1 && TACUDA_COLOR(i - 2) == 1 &&
       open[i - 1] > close[i - 2] && close[i - 1] < open[i - 2] &&
       close[i] < close[i - 1]))
    return TACUDA_COLOR(i - 1) * 100;
  return 0;
}

__host__ __device__ inline int cdl_breakaway_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i - 4) > TACUDA_AVG(BodyLong, i - 4) &&
      TACUDA_COLOR(i - 4) == TACUDA_COLOR(i - 3) &&
      TACUDA_COLOR(i - 3) == TACUDA_COLOR(i - 1) &&
      TACUDA_COLOR(i - 1) == -TACUDA_COLOR(i) &&
      ((TACUDA_COLOR(i - 4) == -1 &&
        TACUDA_RBGAPDOWN(i - 3, i - 4) &&
        high[i - 2] < high[i - 3] && low[i - 2] < low[i - 3] &&
        high[i - 1] < high[i - 2] && low[i - 1] < low[i - 2] &&
        close[i] > open[i - 3] && close[i] < close[i - 4]) ||
       (TACUDA_COLOR(i - 4) == 1 &&
        TACUDA_RBGAPUP(i - 3, i - 4) &&
        high[i - 2] > high[i - 3] && low[i - 2] > low[i - 3] &&
        high[i - 1] > high[i - 2] && low[i - 1] > low[i - 2] &&
        close[i] < open[i - 3] && close[i] > close[i - 4])))
    return TACUDA_COLOR(i) * 100;
  return 0;
}

__host__ __device__ inline int cdl_two_crows_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == 1 &&
      TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_COLOR(i - 1) == -1 &&
      TACUDA_RBGAPUP(i - 1, i - 2) &&
      TACUDA_COLOR(i) == -1 &&
      open[i] < open[i - 1] && open[i] > close[i - 1] &&
      close[i] > open[i - 2] && close[i] < close[i - 2])
    return -100;
  return 0;
}

__host__ __device__ inline int cdl_unique_3_river_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_RB(i - 2) > TACUDA_AVG(BodyLong, i - 2) &&
      TACUDA_COLOR(i - 2) == -1 &&
      TACUDA_COLOR(i - 1) == -1 &&
      close[i - 1] > close[i - 2] && open[i - 1] <= open[i - 2] &&
      low[i - 1] < low[i - 2] &&
      TACUDA_RB(i) < TACUDA_AVG(BodyShort, i) &&
      TACUDA_COLOR(i) == 1 &&
      open[i] > low[i - 1])
    return 100;
  return 0;
}

__host__ __device__ inline int cdl_xside_gap_3_methods_talib(TACUDA_CDL_ARGS) {
  if (TACUDA_COLOR(i - 2) == TACUDA_COLOR(i - 1) &&
      TACUDA_COLOR(i - 1) == -TACUDA_COLOR(i) &&
      open[i] < fmaxf(close[i - 1], open[i - 1]) &&
      open[i] > fminf(close[i - 1], open[i - 1]) &&
      close[i] < fmaxf(close[i - 2], open[i - 2]) &&
      close[i] > fminf(close[i - 2], open[i - 2]) &&
      ((TACUDA_COLOR(i - 2) == 1 && TACUDA_RBGAPUP(i - 1, i - 2)) ||
       (TACUDA_COLOR(i - 2) == -1 && TACUDA_RBGAPDOWN(i - 1, i - 2))))
    return TACUDA_COLOR(i - 2) * 100;
  return 0;
}

__host__ __device__ inline int cdl_rise_fall_3_methods_talib(TACUDA_CDL_ARGS) {
  const int c4 = TACUDA_COLOR(i - 4);
  if (TACUDA_RB(i - 4) > TACUDA_AVG(BodyLong, i - 4) &&
      TACUDA_RB(i - 3) < TACUDA_AVG(BodyShort, i - 3) &&
      TACUDA_RB(i - 2) < TACUDA_AVG(BodyShort, i - 2) &&
      TACUDA_RB(i - 1) < TACUDA_AVG(BodyShort, i - 1) &&
      TACUDA_RB(i) > TACUDA_AVG(BodyLong, i) &&
      c4 == -TACUDA_COLOR(i - 3) &&
      TACUDA_COLOR(i - 3) == TACUDA_COLOR(i - 2) &&
      TACUDA_COLOR(i - 2) == TACUDA_COLOR(i - 1) &&
      TACUDA_COLOR(i - 1) == -TACUDA_COLOR(i) &&
      fminf(open[i - 3], close[i - 3]) < high[i - 4] &&
      fmaxf(open[i - 3], close[i - 3]) > low[i - 4] &&
      fminf(open[i - 2], close[i - 2]) < high[i - 4] &&
      fmaxf(open[i - 2], close[i - 2]) > low[i - 4] &&
      fminf(open[i - 1], close[i - 1]) < high[i - 4] &&
      fmaxf(open[i - 1], close[i - 1]) > low[i - 4] &&
      close[i - 2] * c4 < close[i - 3] * c4 &&
      close[i - 1] * c4 < close[i - 2] * c4 &&
      open[i] * c4 > close[i - 1] * c4 &&
      close[i] * c4 > close[i - 4] * c4)
    return 100 * c4;
  return 0;
}

#endif
