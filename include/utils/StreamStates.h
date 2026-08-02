/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_STREAM_STATES_H
#define TACUDA_STREAM_STATES_H

/**
 * @file StreamStates.h
 * @brief Incremental (streaming) indicator state machines, usable from both
 *        host and device code — the single source of truth behind the
 *        single-series streams in api_stream.cpp and the batch streams in
 *        api_stream_batch.cu.
 *
 * Every state mirrors the exact recurrence of its full-series counterpart
 * (the referenced kernel), so appending bar-by-bar reproduces the tail of a
 * full recompute. Accumulators are double, like TA-Lib's own.
 *
 * States are POD and fixed-size: window families keep their ring buffer in a
 * caller-owned slab (`float*`) instead of owning memory, so an array of
 * states can be memcpy'd to the device and indexed by series.
 *
 * Contract of every family:
 *   void  init(params...)          reset to "no bars seen"
 *   int   ring_size() const        floats of scratch this state needs
 *   int   lookback() const         bars before the first non-NaN output
 *   int   outputs() const          values written by step()
 *   void  step(float* ring, const StreamBar&, float* out)
 */

#include <cmath>

#if defined(__CUDACC__)
#define TACUDA_HD __host__ __device__
#else
#define TACUDA_HD
#endif

namespace tacuda {
namespace streaming {

TACUDA_HD inline float stream_nan() {
#if defined(__CUDA_ARCH__)
  return nanf("");
#else
  return NAN;
#endif
}

TACUDA_HD inline bool stream_isnan(double x) {
#if defined(__CUDA_ARCH__)
  return isnan(x);
#else
  return std::isnan(x);
#endif
}

TACUDA_HD inline bool stream_isfinite(double x) {
#if defined(__CUDA_ARCH__)
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

TACUDA_HD inline double stream_sqrt(double x) {
#if defined(__CUDA_ARCH__)
  return sqrt(x);
#else
  return std::sqrt(x);
#endif
}

TACUDA_HD inline float stream_fabsf(float x) {
#if defined(__CUDA_ARCH__)
  return fabsf(x);
#else
  return std::fabs(x);
#endif
}

/** @brief One closed bar. Families read only the fields they need. */
struct StreamBar {
  float high;
  float low;
  float close;
  float volume;
};

/**
 * @brief Ring over the last `period` values held in a caller-owned buffer.
 * at(0) is the oldest sample once the ring is full.
 */
struct Ring {
  int period;
  int pos;
  int filled;

  TACUDA_HD void init(int p) {
    period = p;
    pos = 0;
    filled = 0;
  }

  TACUDA_HD void push(float *buf, float v) {
    buf[pos] = v;
    if (++pos == period) pos = 0;
    if (filled < period) ++filled;
  }

  TACUDA_HD float at(const float *buf, int k) const {
    int idx = pos + k;
    if (idx >= period) idx -= period;
    return buf[idx];
  }

  TACUDA_HD bool full() const { return filled == period; }
};

// ---------------------------------------------------------------------------
// Window families. The window is re-summed from the ring every bar
// (O(period), independent of history) because the full-series kernels also
// sum each window from scratch; a rolling add/subtract accumulator would
// drift away from them over long histories.
// ---------------------------------------------------------------------------

/** @brief Mirrors smaFinalizeFromPrefixKernel / smaKernelSmallWindow (SMA.cu). */
struct SmaState {
  Ring ring;

  TACUDA_HD void init(int period) { ring.init(period); }
  TACUDA_HD int ring_size() const { return ring.period; }
  TACUDA_HD int lookback() const { return ring.period - 1; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD void step(float *rbuf, const StreamBar &bar, float *out) {
    ring.push(rbuf, bar.close);
    if (!ring.full()) {
      out[0] = stream_nan();
      return;
    }
    double sum = 0.0;
    for (int k = 0; k < ring.period; ++k) sum += static_cast<double>(ring.at(rbuf, k));
    out[0] = static_cast<float>(sum / static_cast<double>(ring.period));
  }
};

/** @brief Mirrors wmaFinalizeFromPrefixKernel (WMA.cu): weights 1..P oldest->newest. */
struct WmaState {
  Ring ring;
  double denom;

  TACUDA_HD void init(int period) {
    ring.init(period);
    denom = 0.5 * static_cast<double>(period) * static_cast<double>(period + 1);
  }
  TACUDA_HD int ring_size() const { return ring.period; }
  TACUDA_HD int lookback() const { return ring.period - 1; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD void step(float *rbuf, const StreamBar &bar, float *out) {
    ring.push(rbuf, bar.close);
    if (!ring.full()) {
      out[0] = stream_nan();
      return;
    }
    double weighted = 0.0;
    for (int k = 0; k < ring.period; ++k) {
      weighted += static_cast<double>(ring.at(rbuf, k)) * static_cast<double>(k + 1);
    }
    out[0] = static_cast<float>(weighted / denom);
  }
};

/**
 * @brief Mirrors bbands_finalize_from_moments_kernel (BBANDS.cu): population
 * variance over the window, clamped at zero, float casts at the very end.
 */
struct BbandsState {
  Ring ring;
  float upperMul;
  float lowerMul;

  TACUDA_HD void init(int period, float up, float dn) {
    ring.init(period);
    upperMul = up;
    lowerMul = dn;
  }
  TACUDA_HD int ring_size() const { return ring.period; }
  TACUDA_HD int lookback() const { return ring.period - 1; }
  TACUDA_HD int outputs() const { return 3; }

  TACUDA_HD void step(float *rbuf, const StreamBar &bar, float *out) {
    ring.push(rbuf, bar.close);
    if (!ring.full()) {
      out[0] = out[1] = out[2] = stream_nan();
      return;
    }
    const int period = ring.period;
    double sum = 0.0;
    for (int k = 0; k < period; ++k) sum += static_cast<double>(ring.at(rbuf, k));
    const double mean = sum / static_cast<double>(period);

    double m2 = 0.0;
    for (int k = 0; k < period; ++k) {
      const double d = static_cast<double>(ring.at(rbuf, k)) - mean;
      m2 += d * d;
    }
    double var = m2 / static_cast<double>(period);
    if (var < 0.0) var = 0.0;
    const double sd = stream_sqrt(var);

    out[0] = static_cast<float>(mean + static_cast<double>(upperMul) * sd);
    out[1] = static_cast<float>(mean);
    out[2] = static_cast<float>(mean - static_cast<double>(lowerMul) * sd);
  }
};

// ---------------------------------------------------------------------------
// Recursive families — O(1) per bar, no scratch.
// ---------------------------------------------------------------------------

/**
 * @brief Mirrors EMA.cu: SMA seed over the first `period` bars written at
 * index period-1, then y = (1-alpha)*y + alpha*x in double.
 */
struct EmaState {
  int period;
  double alpha;
  double a;
  long long idx;
  double seedSum;
  double y;
  int seeded;

  TACUDA_HD void init(int p) {
    period = p;
    alpha = 2.0 / (static_cast<double>(p) + 1.0);
    a = 1.0 - alpha;
    idx = 0;
    seedSum = 0.0;
    y = 0.0;
    seeded = 0;
  }
  TACUDA_HD int ring_size() const { return 0; }
  TACUDA_HD int lookback() const { return period - 1; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD void step(float *, const StreamBar &bar, float *out) {
    const long long i = idx++;
    const double x = static_cast<double>(bar.close);
    if (!seeded) {
      seedSum += x;
      if (i < period - 1) {
        out[0] = stream_nan();
        return;
      }
      y = seedSum / static_cast<double>(period);
      seeded = 1;
      out[0] = static_cast<float>(y);
      return;
    }
    y = a * y + alpha * x;
    out[0] = static_cast<float>(y);
  }
};

/**
 * @brief Mirrors rsi_wilder_one_series (RSI.cu), including TA-Lib's
 * flat-window convention (both averages zero yields 0, not 50) and NaN
 * poisoning of the Wilder recursion.
 */
struct RsiState {
  int period;
  double invP;
  long long idx;
  double prev;
  double sumG;
  double sumL;
  double avgG;
  double avgL;

  TACUDA_HD void init(int p) {
    period = p;
    invP = 1.0 / static_cast<double>(p);
    idx = 0;
    prev = 0.0;
    sumG = sumL = 0.0;
    avgG = avgL = 0.0;
  }
  TACUDA_HD int ring_size() const { return 0; }
  TACUDA_HD int lookback() const { return period; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD float value() const {
    if (avgL == 0.0) return (avgG == 0.0) ? 0.0f : 100.0f;
    if (avgG == 0.0) return 0.0f;
    const double rs = avgG / avgL;
    return static_cast<float>(100.0 - 100.0 / (1.0 + rs));
  }

  TACUDA_HD void step(float *, const StreamBar &bar, float *out) {
    const long long i = idx++;
    const double x = static_cast<double>(bar.close);
    if (i == 0) {
      prev = x;
      out[0] = stream_nan();
      return;
    }
    const double d = x - prev;
    prev = x;

    double g, l;
    if (stream_isnan(d)) {
      g = d;
      l = d;
    } else {
      g = d > 0.0 ? d : 0.0;
      l = d < 0.0 ? -d : 0.0;
    }

    if (i <= period) {
      sumG += g;
      sumL += l;
      if (i < period) {
        out[0] = stream_nan();
        return;
      }
      avgG = sumG / static_cast<double>(period);
      avgL = sumL / static_cast<double>(period);
      out[0] = value();
      return;
    }
    avgG = (avgG * static_cast<double>(period - 1) + g) * invP;
    avgL = (avgL * static_cast<double>(period - 1) + l) * invP;
    out[0] = value();
  }
};

/**
 * @brief Wilder ATR core shared by ATR and NATR. Mirrors
 * trueRangeBatchedKernel + initialAtrKernel (ATR.cu): TR is evaluated in
 * float exactly like the kernel, the seed is the mean of TR[1..period] and
 * lands at index `period`.
 */
struct AtrCore {
  int period;
  long long idx;
  float prevClose;
  double trSum;
  double atr;
  int ready;

  TACUDA_HD void init(int p) {
    period = p;
    idx = 0;
    prevClose = 0.0f;
    trSum = 0.0;
    atr = 0.0;
    ready = 0;
  }

  TACUDA_HD bool step_core(const StreamBar &bar) {
    const long long i = idx++;
    if (i == 0) {
      prevClose = bar.close;
      ready = 0;
      return false;
    }
    const float hl = bar.high - bar.low;
    const float hc = stream_fabsf(bar.high - prevClose);
    const float lc = stream_fabsf(bar.low - prevClose);
    float tr = hl;
    if (hc > tr) tr = hc;
    if (lc > tr) tr = lc;
    prevClose = bar.close;

    if (i <= period) {
      trSum += static_cast<double>(tr);
      if (i < period) {
        ready = 0;
        return false;
      }
      atr = trSum / static_cast<double>(period);
      ready = 1;
      return true;
    }
    atr = (atr * static_cast<double>(period - 1) + static_cast<double>(tr)) /
          static_cast<double>(period);
    ready = 1;
    return true;
  }
};

struct AtrState {
  AtrCore core;

  TACUDA_HD void init(int p) { core.init(p); }
  TACUDA_HD int ring_size() const { return 0; }
  TACUDA_HD int lookback() const { return core.period; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD void step(float *, const StreamBar &bar, float *out) {
    out[0] = core.step_core(bar) ? static_cast<float>(core.atr) : stream_nan();
  }
};

/**
 * @brief Mirrors applyAffineNormalizedKernel (NATR.cu): TA-Lib's TA_IS_ZERO
 * guard on the close, normalization performed in float.
 */
struct NatrState {
  AtrCore core;

  TACUDA_HD void init(int p) { core.init(p); }
  TACUDA_HD int ring_size() const { return 0; }
  TACUDA_HD int lookback() const { return core.period; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD void step(float *, const StreamBar &bar, float *out) {
    if (!core.step_core(bar)) {
      out[0] = stream_nan();
      return;
    }
    const float atr = static_cast<float>(core.atr);
    const float c = bar.close;
    out[0] = (stream_fabsf(c) < 1e-14f) ? 0.0f : 100.0f * atr / c;
  }
};

/**
 * @brief Mirrors obvContribKernel + inclusive scan (OBV.cu): bar 0
 * contributes the raw volume, later bars the signed volume.
 */
struct ObvState {
  long long idx;
  float prevClose;
  double acc;

  TACUDA_HD void init() {
    idx = 0;
    prevClose = 0.0f;
    acc = 0.0;
  }
  TACUDA_HD int ring_size() const { return 0; }
  TACUDA_HD int lookback() const { return 0; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD void step(float *, const StreamBar &bar, float *out) {
    const long long i = idx++;
    if (i == 0) {
      acc += static_cast<double>(bar.volume);
    } else {
      const double p = static_cast<double>(bar.close);
      const double prev = static_cast<double>(prevClose);
      double s = 0.0;
      if (p > prev) s = 1.0;
      else if (p < prev) s = -1.0;
      acc += s * static_cast<double>(bar.volume);
    }
    prevClose = bar.close;
    out[0] = static_cast<float>(acc);
  }
};

/**
 * @brief Mirrors ADTerm + inclusive scan (AD.cu): non-finite bars and zero
 * ranges contribute nothing.
 */
struct AdState {
  double acc;

  TACUDA_HD void init() { acc = 0.0; }
  TACUDA_HD int ring_size() const { return 0; }
  TACUDA_HD int lookback() const { return 0; }
  TACUDA_HD int outputs() const { return 1; }

  TACUDA_HD void step(float *, const StreamBar &bar, float *out) {
    const double hi = static_cast<double>(bar.high);
    const double lo = static_cast<double>(bar.low);
    const double cl = static_cast<double>(bar.close);
    const double vol = static_cast<double>(bar.volume);

    double term = 0.0;
    if (stream_isfinite(hi) && stream_isfinite(lo) && stream_isfinite(cl) &&
        stream_isfinite(vol)) {
      const double range = hi - lo;
      if (range > 0.0) {
        term = (((cl - lo) - (hi - cl)) / range) * vol;
      }
    }
    acc += term;
    out[0] = static_cast<float>(acc);
  }
};

/**
 * @brief Mirrors the all-EMA path of MACDEXT::calculate (MACDEXT.cu), i.e.
 * ct_macd with maType = CT_MA_EMA. Two details are load-bearing: both EMA
 * seeds are taken at macdLookback = slowPeriod-1 (the fast seed is the
 * fast-length SMA *ending* there, not one at fastPeriod-1), and the signal
 * EMA consumes the float-rounded MACD line. All three outputs stay NaN
 * until the signal lookback, like TA-Lib's shared outBegIdx.
 */
struct MacdState {
  int fastPeriod;
  int slowPeriod;
  int signalPeriod;
  int macdLookback;
  int signalLookback;
  double fastAlpha;
  double slowAlpha;
  double signalAlpha;
  Ring fastRing;  // holds the last fastPeriod closes for the fast seed
  long long idx;
  double slowSeedSum;
  double fastEma;
  double slowEma;
  double signalD;
  double signalSeedSum;
  float macdF;
  float signalF;

  TACUDA_HD void init(int fast, int slow, int signal) {
    // TA-Lib normalizes reversed periods (slow = max) instead of rejecting.
    if (slow < fast) {
      const int t = fast;
      fast = slow;
      slow = t;
    }
    fastPeriod = fast;
    slowPeriod = slow;
    signalPeriod = signal;
    macdLookback = slow - 1;
    signalLookback = macdLookback + signal - 1;
    fastAlpha = 2.0 / (static_cast<double>(fast) + 1.0);
    slowAlpha = 2.0 / (static_cast<double>(slow) + 1.0);
    signalAlpha = 2.0 / (static_cast<double>(signal) + 1.0);
    fastRing.init(fast);
    idx = 0;
    slowSeedSum = 0.0;
    fastEma = slowEma = 0.0;
    signalD = 0.0;
    signalSeedSum = 0.0;
    macdF = stream_nan();
    signalF = stream_nan();
  }
  TACUDA_HD int ring_size() const { return fastRing.period; }
  TACUDA_HD int lookback() const { return signalLookback; }
  TACUDA_HD int outputs() const { return 3; }

  TACUDA_HD void step(float *rbuf, const StreamBar &bar, float *out) {
    const long long i = idx++;
    const double x = static_cast<double>(bar.close);

    fastRing.push(rbuf, bar.close);
    if (i < slowPeriod) slowSeedSum += x;

    if (i == macdLookback) {
      double fastSum = 0.0;
      for (int k = 0; k < fastPeriod; ++k) {
        fastSum += static_cast<double>(fastRing.at(rbuf, k));
      }
      fastEma = fastSum / static_cast<double>(fastPeriod);
      slowEma = slowSeedSum / static_cast<double>(slowPeriod);
      macdF = static_cast<float>(fastEma - slowEma);
    } else if (i > macdLookback) {
      fastEma = (1.0 - fastAlpha) * fastEma + fastAlpha * x;
      slowEma = (1.0 - slowAlpha) * slowEma + slowAlpha * x;
      macdF = static_cast<float>(fastEma - slowEma);
    }

    if (i >= macdLookback && i <= signalLookback) {
      signalSeedSum += static_cast<double>(macdF);
      if (i == signalLookback) {
        signalD = signalSeedSum / static_cast<double>(signalPeriod);
        signalF = static_cast<float>(signalD);
      }
    } else if (i > signalLookback) {
      signalD = (1.0 - signalAlpha) * signalD +
                signalAlpha * static_cast<double>(macdF);
      signalF = static_cast<float>(signalD);
    }

    if (i < signalLookback) {
      out[0] = out[1] = out[2] = stream_nan();
      return;
    }
    out[0] = macdF;
    out[1] = signalF;
    out[2] = (stream_isnan(macdF) || stream_isnan(signalF))
                 ? stream_nan()
                 : macdF - signalF;
  }
};

}  // namespace streaming
}  // namespace tacuda

#endif  // TACUDA_STREAM_STATES_H
