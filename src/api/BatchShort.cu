/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "BatchShort.h"
#include "BatchFused.h"
#include <utils/BatchRows.h>
#include <utils/LeadingNaN.cuh>
#include <utils/CudaUtils.h>
#include <algorithm>

namespace tacuda::detail {
namespace {
using F = ShortFamily;

struct Matrices {
    const float* in[5] = {};
    float* out[2] = {};
};

struct Ema {
    double sum = 0.0, value = 0.0;
    int count = 0;

    __device__ float step(float x, int period, double alpha) {
        if (period == 1) return x;
        if (count == 0 && isnan(x)) return nanf("");
        if (count < period) {
            sum += x;
            if (++count < period) return nanf("");
            value = static_cast<float>(sum / period);
        } else {
            value = (1.0 - alpha) * value + alpha * x;
        }
        return static_cast<float>(value);
    }
};

template <F family>
__device__ void ema_cascade(const float* in, float* out, int n, int p, float v) {
    constexpr int stages = family == F::Dema ? 2 : family == F::T3 ? 6 : 3;
    Ema ema[stages];
    const double alpha = 2.0 / (p + 1.0);
    const float c1 = -v * v * v;
    const float c2 = 3 * v * v + 3 * v * v * v;
    const float c3 = -3 * v - 6 * v * v - 3 * v * v * v;
    const float c4 = 1 + 3 * v + 3 * v * v + v * v * v;
    float previous = nanf("");
    for (int i = 0; i < n; ++i) {
        float values[stages];
        float x = in[i];
#pragma unroll
        for (int j = 0; j < stages; ++j) {
            x = ema[j].step(x, p, alpha);
            values[j] = x;
        }
        if constexpr (family == F::Dema) out[i] = 2.0f * values[0] - values[1];
        if constexpr (family == F::Tema) out[i] = 3.0f * values[0] - 3.0f * values[1] + values[2];
        if constexpr (family == F::T3)
            out[i] = c1 * values[5] + c2 * values[4] + c3 * values[3] + c4 * values[2];
        if constexpr (family == F::Trix) {
            out[i] = previous != 0.0f ? (x - previous) / previous * 100.0f : 0.0f;
            previous = x;
        }
    }
}

__device__ void kama(const float* x, float* out, int n, int p, int fast, int slow) {
    const int first = leading_nan_count(x, n);
    for (int i = 0; i < n; ++i) out[i] = nanf("");
    if (p >= n - first) return;
    const float fastAlpha = 2.0f / (fast + 1.0f), slowAlpha = 2.0f / (slow + 1.0f);
    double value = x[first + p - 1];
    double volatility = 0.0;
    for (int i = first + 1; i < first + p; ++i) volatility += fabs(double(x[i]) - x[i - 1]);
    for (int i = first + p; i < n; ++i) {
        volatility += fabs(double(x[i]) - x[i - 1]);
        const double change = double(x[i]) - x[i - p];
        const double ratio = volatility <= change || fabs(volatility) < 1e-8
                             ? 1.0 : fabs(change) / volatility;
        const double s = ratio * (double(fastAlpha) - slowAlpha) + slowAlpha;
        const double weight = static_cast<float>(s * s);
        value = (1.0 - weight) * value + weight * x[i];
        out[i] = static_cast<float>(value);
        volatility -= fabs(double(x[i - p + 1]) - x[i - p]);
    }
}

template <F family>
__device__ void oscillator(const float* x, float* out, int n, int p, int q) {
    if (p > q) { int temp = p; p = q; q = temp; }
    const double alpha = 2.0 / (p + 1.0), beta = 2.0 / (q + 1.0);
    if constexpr (family == F::Apo || family == F::Ppo) {
        Ema fast, slow;
        for (int i = 0; i < n; ++i) {
            const float f = fast.step(x[i], p, alpha), s = slow.step(x[i], q, beta);
            if constexpr (family == F::Apo) out[i] = f - s;
            else out[i] = isnan(f) || isnan(s) ? nanf("")
                          : fabsf(s) < 1e-14f ? 0.0f : 100.0f * (f - s) / s;
        }
        return;
    }
    const int first = family == F::MacdLine ? leading_nan_count(x, n) : 0;
    if (q > n - first) {
        for (int i = 0; i < n; ++i) out[i] = nanf("");
        return;
    }
    const int seed = first + q - 1;
    const int fastBegin = family == F::MacdLine ? seed - p + 1 : first;
    double f = 0.0, s = 0.0;
    for (int i = first; i <= seed; ++i) {
        s += x[i];
        if (i >= fastBegin) {
            if (i - fastBegin < p) {
                f += x[i];
                if (i - fastBegin == p - 1) f /= p;
            } else {
                f = (1.0 - alpha) * f + alpha * x[i];
            }
        }
        if (i < seed) out[i] = nanf("");
    }
    for (int i = 0; i < first; ++i) out[i] = nanf("");
    s /= q;
    for (int i = seed; i < n; ++i) {
        if (i > seed) {
            f = (1.0 - alpha) * f + alpha * x[i];
            s = (1.0 - beta) * s + beta * x[i];
        }
        if constexpr (family == F::Pvo)
            out[i] = fabs(s) < 1e-14 ? 0.0f : static_cast<float>(100.0 * (f - s) / s);
        else
            out[i] = static_cast<float>(f - s);
    }
}

__device__ double sum_window(const float* x, int p, bool weighted) {
    double sum = 0.0;
    for (int j = 0; j < p; ++j) sum += double(x[j]) * (weighted ? j + 1.0 : 1.0);
    return sum;
}

__device__ void moving_average(const float* x, float* out, int n, int p, int type) {
    if (type == 3) { ema_cascade<F::Dema>(x, out, n, p, 0); return; }
    if (type == 4) { ema_cascade<F::Tema>(x, out, n, p, 0); return; }
    if (type == 6) { kama(x, out, n, p, 2, 30); return; }
    if (type == 8) { ema_cascade<F::T3>(x, out, n, p, 0.7f); return; }
    if (type == 1) {
        Ema ema;
        const double alpha = 2.0 / (p + 1.0);
        for (int i = 0; i < n; ++i) out[i] = ema.step(x[i], p, alpha);
        return;
    }
    // The dispatcher handles SMA/WMA in parallel and excludes MAMA.
    // The remaining type is the two-stage triangular average.
    const int a = (p + 1) / 2, b = p - a + 1;
    for (int i = 0; i < n; ++i) {
        if (i < p - 1) { out[i] = nanf(""); continue; }
        const float* window = x + (i - p + 1);
        double sum = 0.0;
        for (int j = 0; j < b; ++j) sum += static_cast<float>(sum_window(window + j, a, false) / a);
        out[i] = static_cast<float>(sum / b);
    }
}

template <F family>
__device__ void directional(const float* high, const float* low, const float* close,
                             float* out, int n, int p) {
    constexpr bool dmOnly = family == F::PlusDm || family == F::MinusDm;
    constexpr bool diOnly = family == F::PlusDi || family == F::MinusDi;
    constexpr bool adx = family == F::Adx || family == F::Adxr;
    const double decay = (p - 1.0) / p;
    const float dmDecay = static_cast<float>(p - 1) / p;
    double tr = 0.0, positive = 0.0, negative = 0.0, average = 0.0;
    float dm = 0.0f, history[family == F::Adxr ? kShortWindow : 1];
    if constexpr (!dmOnly && !diOnly) tr = high[0] - low[0];
    for (int i = 0; i < n; ++i) {
        out[i] = nanf("");
        if (p > n) continue;
        float plus = 0.0f, minus = 0.0f, range = 0.0f;
        if (i > 0) {
            const float up = high[i] - high[i - 1], down = low[i - 1] - low[i];
            plus = up > 0.0f && up > down ? up : 0.0f;
            minus = down > 0.0f && down > up ? down : 0.0f;
            if constexpr (!dmOnly) {
                range = fmaxf(high[i] - low[i],
                          fmaxf(fabsf(high[i] - close[i - 1]), fabsf(low[i] - close[i - 1])));
            }
        }
        if constexpr (dmOnly) {
            const float movement = family == F::PlusDm ? plus : minus;
            dm = i < p ? dm + movement : dmDecay * dm + movement;
            if (i >= p - 1) out[i] = dm;
        } else {
            if (i < p) {
                tr += range; positive += plus; negative += minus;
                continue;
            }
            tr = decay * tr + range;
            positive = decay * positive + plus;
            negative = decay * negative + minus;
            const float t = static_cast<float>(tr);
            if constexpr (diOnly) {
                const float movement = static_cast<float>(family == F::PlusDi ? positive : negative);
                out[i] = t == 0.0f ? 0.0f : 100.0f * movement / t;
            } else {
                const float plusDi = t > 0.0f ? 100.0f * static_cast<float>(positive) / t : 0.0f;
                const float minusDi = t > 0.0f ? 100.0f * static_cast<float>(negative) / t : 0.0f;
                const float total = plusDi + minusDi;
                const float dx = total > 0.0f ? 100.0f * fabsf(plusDi - minusDi) / total : 0.0f;
                if constexpr (!adx) out[i] = dx;
                else {
                    if (i < p + p) {
                        average += dx;
                        if (i != p + p - 1) continue;
                        average /= p;
                    } else average = decay * average + double(dx) / p;
                    const float current = static_cast<float>(average);
                    if constexpr (family == F::Adx) out[i] = current;
                    else {
                        const int slot = i % p;
                        if (p == 1) out[i] = current;
                        else if (i >= 3 * p - 2) out[i] = 0.5f * (current + history[(i + 1) % p]);
                        history[slot] = current;
                    }
                }
            }
        }
    }
}

__device__ void cmo(const float* x, float* out, int n, int p) {
    const int first = leading_nan_count(x, n);
    for (int i = 0; i < n; ++i) out[i] = nanf("");
    if (p >= n - first) return;
    const float inverse = 1.0f / p;
    const float decay = (p - 1.0f) * inverse;
    const int seed = first + p;
    double gains = 0.0, losses = 0.0;
    for (int i = first + 1; i <= seed; ++i) {
        const float d = x[i] - x[i - 1];
        if (d > 0.0f) gains += d;
        if (d < 0.0f) losses -= d;
    }
    float gain = static_cast<float>(gains / p), loss = static_cast<float>(losses / p);
    for (int i = seed; i < n; ++i) {
        if (i > seed) {
            const float d = x[i] - x[i - 1];
            gain = decay * gain + (d > 0.0f ? d * inverse : 0.0f);
            loss = decay * loss + (d < 0.0f ? -d * inverse : 0.0f);
        }
        const float total = gain + loss;
        out[i] = total == 0.0f ? 0.0f : 100.0f * (gain - loss) / total;
    }
}

__device__ void stoch_rsi(const float* x, float* fastK, float* fastD, int n,
                          int p, int k, int d) {
    for (int i = 0; i < n; ++i) fastK[i] = fastD[i] = nanf("");
    const int first = leading_nan_count(x, n);
    if (p >= n - first || k + d - 2 >= n - p) return;
    const float inverse = 1.0f / p;
    const float decay = static_cast<float>((p - 1.0) / p);
    const int seed = first + p;
    double gains = 0.0, losses = 0.0;
    for (int i = first + 1; i <= seed; ++i) {
        const float diff = x[i] - x[i - 1];
        if (isnan(diff)) return;
        if (diff > 0.0f) gains += diff;
        else losses -= diff;
    }
    float gain = static_cast<float>(gains / p), loss = static_cast<float>(losses / p);
    float rsi[kShortWindow], rawK[kShortWindow];
    for (int i = seed; i < n; ++i) {
        if (i > seed) {
            const float diff = x[i] - x[i - 1];
            if (isnan(diff)) return;
            gain = decay * gain + (diff > 0.0f ? diff * inverse : 0.0f);
            loss = decay * loss + (diff < 0.0f ? -diff * inverse : 0.0f);
        }
        const float value = loss == 0.0f ? (gain == 0.0f ? 50.0f : 100.0f)
                          : gain == 0.0f ? 0.0f : 100.0f - 100.0f / (1.0f + gain / loss);
        if (isnan(value)) return;
        rsi[i % k] = value;
        if (i < seed + k - 1) continue;
        float hi = value, lo = value;
        for (int j = 1; j < k; ++j) {
            hi = fmaxf(hi, rsi[(i - j) % k]);
            lo = fminf(lo, rsi[(i - j) % k]);
        }
        const float kv = hi == lo ? 0.0f : (value - lo) / (hi - lo) * 100.0f;
        rawK[i % d] = kv;
        if (i >= p + k + d - 2) fastK[i] = kv;
        if (i < seed + k + d - 2) continue;
        float sum = 0.0f;
        for (int j = 0; j < d; ++j) sum += rawK[(i - j) % d];
        fastD[i] = sum / d;
    }
}

template <F family>
__device__ void volume_index(const float* close, const float* volume, float* out, int n) {
    int i = 0;
    while (i < n - 1 && (isnan(close[i]) || isnan(volume[i]))) out[i++] = nanf("");
    double state = 1000.0;
    out[i++] = static_cast<float>(state);
    for (; i < n; ++i) {
        const bool changed = family == F::Nvi ? volume[i] < volume[i - 1] : volume[i] > volume[i - 1];
        if (changed && close[i - 1] != 0.0f) state *= double(close[i]) / close[i - 1];
        out[i] = static_cast<float>(state);
    }
}

__device__ double relative_return(const float* x, int i) {
    const double previous = x[i - 1];
    return fabs(previous) < 1.0e-14 ? 0.0 : (double(x[i]) - previous) / previous;
}

__device__ void beta_rows(const float* x, const float* y, float* out, int n, int p) {
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    double ringX[kShortWindow], ringY[kShortWindow];
    out[0] = nanf("");
    for (int i = 1; i < n; ++i) {
        const int slot = (i - 1) % p;
        if (i > p) {
            const double a = ringX[slot], b = ringY[slot];
            sx -= a; sy -= b; sxx -= a * a; sxy -= a * b;
        }
        const double a = relative_return(x, i), b = relative_return(y, i);
        ringX[slot] = a; ringY[slot] = b;
        sx += a; sy += b; sxx += a * a; sxy += a * b;
        if (i < p) { out[i] = nanf(""); continue; }
        const double denominator = p * sxx - sx * sx;
        out[i] = fabs(denominator) < 1e-14 ? 0.0f
                                          : static_cast<float>((p * sxy - sx * sy) / denominator);
    }
}

__device__ void imi_rows(const float* open, const float* close, float* out, int n, int p) {
    bool poisoned = false;
    for (int i = 0; i < n; ++i) {
        poisoned |= !isfinite(double(close[i]) - open[i]);
        const int first = i - p + 1;
        if (first < 0 || poisoned) { out[i] = nanf(""); continue; }
        double up = 0.0, down = 0.0;
        for (int j = first; j <= i; ++j) {
            const double diff = double(close[j]) - open[j];
            if (diff > 0.0) up += diff;
            else down -= diff;
        }
        out[i] = up + down == 0.0 ? 50.0f : static_cast<float>(100.0 * (up / (up + down)));
    }
}

// Each block finds its row's first invalid bar. Its threads then evaluate
// independent, centered windows, so longish rows can use the whole block.
__global__ void correl_rows(Matrices m, int length, int series, int p) {
    __shared__ int invalid;
    for (size_t row = blockIdx.x; row < static_cast<size_t>(series); row += gridDim.x) {
        const size_t offset = row * length;
        const float* x = m.in[0] + offset;
        const float* y = m.in[1] + offset;
        if (threadIdx.x == 0) invalid = length;
        __syncthreads();
        for (int i = threadIdx.x; i < length; i += blockDim.x) {
            if (!isfinite(x[i]) || !isfinite(y[i])) { atomicMin(&invalid, i); break; }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < length; i += blockDim.x) {
            const int first = i - p + 1;
            if (first < 0) { m.out[0][offset + i] = nanf(""); continue; }
            if (i >= invalid) { m.out[0][offset + i] = 0.0f; continue; }
            double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
            for (int j = first; j <= i; ++j) {
                const double a = double(x[j]) - x[first];
                const double b = double(y[j]) - y[first];
                sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b;
            }
            const double vx = sxx - sx * sx / p, vy = syy - sy * sy / p;
            const double product = vx * vy;
            m.out[0][offset + i] = vx > 0.0 && vy > 0.0 && product > 1.0e-14
                   ? static_cast<float>((sxy - sx * sy / p) / sqrt(product)) : 0.0f;
        }
        __syncthreads();
    }
}

__global__ void direct_ma(Matrices m, int length, size_t total, int p, bool weighted) {
    const double divisor = weighted ? p * (p + 1.0) / 2.0 : p;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        m.out[0][i] = i % length < static_cast<size_t>(p - 1) ? nanf("")
            : static_cast<float>(sum_window(m.in[0] + (i - (p - 1)), p, weighted) / divisor);
    }
}

template <F family>
__device__ void club(const float* const* x, float* out, int n, ShortOptions params) {
    const double alpha = 2.0 / (params.p + 1.0);
    double state = 0.0;
    bool seeded = false;
    for (int i = 0; i < n; ++i) {
        if constexpr (family == F::Xaverage) {
            const float a = x[0][i];
            if (!isfinite(a)) { out[i] = nanf(""); continue; }
            state = seeded ? (1.0 - alpha) * state + alpha * a : a;
            seeded = true;
        } else if constexpr (family == F::Wobv) {
            const float range = x[1][i] - x[2][i];
            if (range > 0.0f) state += (double(x[3][i]) - x[0][i]) / range * x[4][i];
        } else {
            if (i == 0) { out[i] = nanf(""); continue; }
            if constexpr (family == F::Obtr) {
                const double prev = x[2][i - 1];
                const double range = fmax(double(x[0][i]) - x[1][i],
                                     fmax(fabs(x[0][i] - prev), fabs(x[1][i] - prev)));
                if (x[2][i] > prev) state += range;
                else if (x[2][i] < prev) state -= range;
            }
            if constexpr (family == F::Hpi) {
                const double mid = (double(x[0][i]) + x[1][i]) / 2.0;
                const double previous = (double(x[0][i - 1]) + x[1][i - 1]) / 2.0;
                const double raw = (mid - previous) * x[2][i];
                const double factor = params.factor;
                state = (1.0 - factor) * state + factor * raw;
            }
            if constexpr (family == F::Asi) {
                const double h = x[1][i], l = x[2][i], c = x[3][i], o = x[0][i];
                const double cy = x[3][i - 1], oy = x[0][i - 1];
                const double hc = fabs(h - cy), lc = fabs(l - cy);
                const double hl = h - l, coy = fabs(cy - oy);
                double r;
                if (hc >= lc && hc >= hl) r = hc - 0.5 * lc + 0.25 * coy;
                else if (lc >= hc && lc >= hl) r = lc - 0.5 * hc + 0.25 * coy;
                else r = hl + 0.25 * coy;
                if (r > 1e-12) {
                    const double k = fmax(hc, lc);
                    state += 50.0 * ((c - cy) + 0.5 * (c - o) + 0.25 * (cy - oy))
                             / r * (k / fmax(1e-12, hl > 0.0 ? hl : k));
                }
            }
        }
        out[i] = static_cast<float>(state);
    }
}

template <F family, int NI, int NO = 1>
__global__ void run_rows(Matrices matrices, int length, int series, ShortOptions opt) {
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < static_cast<size_t>(series); row += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t offset = row * length;
        const float* x[5] = {};
        float* y[2] = {};
#pragma unroll
        for (int j = 0; j < NI; ++j) x[j] = matrices.in[j] + offset;
#pragma unroll
        for (int j = 0; j < NO; ++j) y[j] = matrices.out[j] + offset;
        if constexpr (family == F::Dema || family == F::Tema || family == F::T3 || family == F::Trix)
            ema_cascade<family>(x[0], y[0], length, opt.p, opt.factor);
        else if constexpr (family == F::Kama)
            kama(x[0], y[0], length, opt.p, opt.q, opt.r);
        else if constexpr (family == F::Cmo)
            cmo(x[0], y[0], length, opt.p);
        else if constexpr (family == F::Apo || family == F::Ppo || family == F::Pvo || family == F::MacdLine)
            oscillator<family>(x[0], y[0], length, opt.p, opt.q);
        else if constexpr (family == F::Ma)
            moving_average(x[0], y[0], length, opt.p, opt.q);
        else if constexpr (family == F::StochRsi)
            stoch_rsi(x[0], y[0], y[1], length, opt.p, opt.q, opt.r);
        else if constexpr (family == F::Nvi || family == F::Pvi)
            volume_index<family>(x[0], x[1], y[0], length);
        else if constexpr (family == F::Beta)
            beta_rows(x[0], x[1], y[0], length, opt.p);
        else if constexpr (family == F::Imi)
            imi_rows(x[0], x[1], y[0], length, opt.p);
        else if constexpr (family == F::Adx || family == F::Adxr || family == F::Dx ||
                           family == F::PlusDm || family == F::MinusDm ||
                           family == F::PlusDi || family == F::MinusDi)
            directional<family>(x[0], x[1], x[2], y[0], length, opt.p);
        else club<family>(x, y[0], length, opt);
    }
}

}  // namespace

bool short_batch_shape(int length, int series) {
    return length <= kShortLength && (batch_mode() == BatchMode::Force ||
          (series >= 32 && length <= 4LL * series));
}

void launch_short_rows(ShortFamily family, const float* const* inputs, int nIn,
                       float* const* outputs, int nOut, int length, int series,
                       ShortOptions options, cudaStream_t stream) {
    Matrices matrices;
    for (int j = 0; j < nIn; ++j) matrices.in[j] = inputs[j];
    for (int j = 0; j < nOut; ++j) matrices.out[j] = outputs[j];
    if (family == F::Ma && (options.q == 0 || options.q == 2)) {
        const size_t total = static_cast<size_t>(length) * series;
        const int blocks = static_cast<int>(std::min<size_t>((total + 255) / 256, 65535));
        direct_ma<<<blocks, 256, 0, stream>>>(matrices, length, total, options.p, options.q == 2);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int blocks = std::min(1 + (series - 1) / kRowThreads, 65535);
#define LAUNCH(name, ...) case F::name: run_rows<F::name, __VA_ARGS__><<<blocks, kRowThreads, 0, stream>>>( \
        matrices, length, series, options); break
    switch (family) {
        LAUNCH(Dema, 1); LAUNCH(Tema, 1); LAUNCH(T3, 1); LAUNCH(Kama, 1);
        LAUNCH(Trix, 1); LAUNCH(Cmo, 1);
        LAUNCH(Adx, 3); LAUNCH(Adxr, 3); LAUNCH(Dx, 3);
        LAUNCH(PlusDi, 3); LAUNCH(MinusDi, 3); LAUNCH(PlusDm, 2); LAUNCH(MinusDm, 2);
        LAUNCH(Apo, 1); LAUNCH(Ppo, 1); LAUNCH(Pvo, 1);
        LAUNCH(Ma, 1); LAUNCH(MacdLine, 1); LAUNCH(StochRsi, 1, 2);
        LAUNCH(Imi, 2); LAUNCH(Nvi, 2); LAUNCH(Pvi, 2); LAUNCH(Beta, 2);
        case F::Correl:
            correl_rows<<<std::min(series, 65535), 128, 0, stream>>>(matrices, length, series, options.p);
            break;
        LAUNCH(Asi, 4); LAUNCH(Obtr, 3); LAUNCH(Wobv, 5); LAUNCH(Hpi, 3); LAUNCH(Xaverage, 1);
    }
#undef LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace tacuda::detail
