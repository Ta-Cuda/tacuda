/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// ClubPack.cu — STAD club indicators (see ClubPack.h). Reference semantics:
// the CPU implementations in tools/optimizer/src/strategy (EXPERIMENTS pack).
//
// Cumulative lines (ASI, OBTR, WOBV) are prefix sums of per-bar terms that
// depend only on bars i-1..i, and HPI / XAverage are first-order affine
// recurrences — all five run as parallel scans (double accumulation) like
// EMA/RSI. Only SAREL is a genuine state machine and keeps the single-thread
// kernel (the SAR precedent). Windowed indicators use one thread per element.

#include <indicators/ClubPack.h>
#include <utils/AffineScan.cuh>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cmath>
#include <stdexcept>

namespace {

inline void fill_nan(float* out, int size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(out, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
}

// ---------------------------------------------------------------------------
// Prefix-sum recursions: per-bar term kernels + thrust scan + float cast.
// ---------------------------------------------------------------------------

__global__ void asiTermKernel(const float* __restrict__ open,
                              const float* __restrict__ high,
                              const float* __restrict__ low,
                              const float* __restrict__ close,
                              double* __restrict__ term, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i == 0) {
            term[0] = 0.0;
            continue;
        }
        const double h = high[i], l = low[i], c = close[i], o = open[i];
        const double cy = close[i - 1], oy = open[i - 1];
        const double hc = fabs(h - cy), lc = fabs(l - cy);
        const double hl = h - l, coy = fabs(cy - oy);
        double r;
        if (hc >= lc && hc >= hl) r = hc - 0.5 * lc + 0.25 * coy;
        else if (lc >= hc && lc >= hl) r = lc - 0.5 * hc + 0.25 * coy;
        else r = hl + 0.25 * coy;
        double si = 0.0;
        if (r > 1e-12) {
            const double k = fmax(hc, lc);
            si = 50.0 * ((c - cy) + 0.5 * (c - o) + 0.25 * (cy - oy)) / r *
                 (k / fmax(1e-12, hl > 0.0 ? hl : k));
        }
        term[i] = si;
    }
}

__global__ void obtrTermKernel(const float* __restrict__ high,
                               const float* __restrict__ low,
                               const float* __restrict__ close,
                               double* __restrict__ term, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i == 0) {
            term[0] = 0.0;
            continue;
        }
        const double cy = close[i - 1];
        const double tr = fmax(static_cast<double>(high[i]) - low[i],
                               fmax(fabs(high[i] - cy), fabs(low[i] - cy)));
        double t = 0.0;
        if (close[i] > close[i - 1]) t = tr;
        else if (close[i] < close[i - 1]) t = -tr;
        term[i] = t;
    }
}

__global__ void wobvTermKernel(const float* __restrict__ open,
                               const float* __restrict__ high,
                               const float* __restrict__ low,
                               const float* __restrict__ close,
                               const float* __restrict__ volume,
                               double* __restrict__ term, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        const float rng = high[i] - low[i];
        term[i] = rng > 0.0f
                      ? (static_cast<double>(close[i]) - open[i]) / rng * volume[i]
                      : 0.0;
    }
}

// Writes the scanned prefix into the float output from firstValid onward.
__global__ void castPrefixKernel(const double* __restrict__ prefix,
                                 float* __restrict__ out, int size,
                                 int firstValid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i >= firstValid) out[i] = static_cast<float>(prefix[i]);
    }
}

void scan_double_inplace(double* term, int size, cudaStream_t stream) {
    thrust::device_ptr<double> p(term);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), p, p + size, p);
}

// ---------------------------------------------------------------------------
// Affine recursions: HPI (y = (1-f)*y + f*raw) via AffineScan; XAverage via a
// seed-aware affine monoid that treats NaN inputs as identities.
// ---------------------------------------------------------------------------

using AffineD = tacuda::detail::AffineTransform1<double>;
using AffineComposeD = tacuda::detail::AffineCompose1<double>;

__global__ void hpiTransformKernel(const float* __restrict__ high,
                                   const float* __restrict__ low,
                                   const float* __restrict__ volume,
                                   AffineD* __restrict__ trans, int size,
                                   float factor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i == 0) {
            trans[0] = AffineD{1.0, 0.0};
            continue;
        }
        const double mid = (static_cast<double>(high[i]) + low[i]) / 2.0;
        const double midPrev =
            (static_cast<double>(high[i - 1]) + low[i - 1]) / 2.0;
        const double raw = (mid - midPrev) * volume[i];
        const double f = static_cast<double>(factor);
        trans[i] = AffineD{1.0 - f, f * raw};
    }
}

// y0 = 0, so the recurrence value at i is just the composed addend b.
__global__ void hpiFinalizeKernel(const AffineD* __restrict__ prefix,
                                  float* __restrict__ out, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i >= 1) out[i] = static_cast<float>(prefix[i].b);
    }
}

// Seed-aware EMA element: applied to a seeded state v the segment yields
// A*v + B; applied to an unseeded state it yields C (only meaningful when
// hasFinite). NaN inputs are identities, finite inputs seed with themselves.
struct SeededEma {
    double A;
    double B;
    double C;
    int hasFinite;
};

struct SeededEmaCompose {
    __host__ __device__ SeededEma operator()(const SeededEma& x,
                                             const SeededEma& y) const {
        SeededEma r;
        r.A = y.A * x.A;
        r.B = y.A * x.B + y.B;
        r.C = x.hasFinite ? (y.A * x.C + y.B) : y.C;
        r.hasFinite = x.hasFinite | y.hasFinite;
        return r;
    }
};

__global__ void xaverageElementKernel(const float* __restrict__ in,
                                      SeededEma* __restrict__ elems, int size,
                                      double alpha) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        const float v = in[i];
        if (isfinite(v)) {
            elems[i] = SeededEma{1.0 - alpha, alpha * static_cast<double>(v),
                                 static_cast<double>(v), 1};
        } else {
            elems[i] = SeededEma{1.0, 0.0, 0.0, 0};
        }
    }
}

__global__ void xaverageFinalizeKernel(const float* __restrict__ in,
                                       const SeededEma* __restrict__ prefix,
                                       float* __restrict__ out, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        // The output is defined exactly at finite input bars (a finite input
        // implies the prefix is seeded); NaN bars keep the NaN fill.
        if (isfinite(in[i])) out[i] = static_cast<float>(prefix[i].C);
    }
}

// ---------------------------------------------------------------------------
// SAREL: a genuine state machine (trend flips, acceleration clamps) — the
// only club recursion that stays on a single GPU thread, like SAR.
// ---------------------------------------------------------------------------

__global__ void sarElKernel(const float* __restrict__ high,
                            const float* __restrict__ low,
                            const float* __restrict__ close,
                            float* __restrict__ out, int size,
                            float accFact, float maxAf) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (size < 2) return;
    bool isLong = close[1] >= close[0];
    float af = accFact;
    float ep = isLong ? high[1] : low[1];
    float sar = isLong ? low[0] : high[0];
    out[1] = sar;
    for (int i = 2; i < size; ++i) {
        sar = sar + af * (ep - sar);
        if (isLong) {
            sar = fminf(sar, fminf(low[i - 1], low[i - 2]));
            if (low[i] < sar) {
                isLong = false;
                sar = ep;
                ep = low[i];
                af = accFact;
            } else if (high[i] > ep) {
                ep = high[i];
                af = fminf(maxAf, af + accFact);
            }
        } else {
            sar = fmaxf(sar, fmaxf(high[i - 1], high[i - 2]));
            if (high[i] > sar) {
                isLong = true;
                sar = ep;
                ep = high[i];
                af = accFact;
            } else if (low[i] < ep) {
                ep = low[i];
                af = fminf(maxAf, af + accFact);
            }
        }
        out[i] = sar;
    }
}

// ---------------------------------------------------------------------------
// Windowed kernels (one thread per element, grid-stride)
// ---------------------------------------------------------------------------

__global__ void marketStrengthKernel(const float* __restrict__ close,
                                     float* __restrict__ out, int size,
                                     int len) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i - len < 0) continue;
        double total = 0.0, ups = 0.0, dns = 0.0;
        for (int k = 0; k < len; ++k) {
            const double d = static_cast<double>(close[i - k]) - close[i - k - 1];
            total += d;
            if (d > 0.0) ups += d;
            else dns += -d;
        }
        float ms;
        if (total >= 0.0) ms = ups > 0.0 ? static_cast<float>(100.0 * total / ups) : 0.0f;
        else ms = dns > 0.0 ? static_cast<float>(100.0 * total / dns) : 0.0f;
        out[i] = ms;
    }
}

__global__ void trendScoreKernel(const float* __restrict__ close,
                                 float* __restrict__ out, int size,
                                 int lookback) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i < lookback + 10) continue;
        int t = 0;
        for (int k = 1; k <= 10; ++k) {
            t += close[i] >= close[i - lookback - k] ? 1 : -1;
        }
        out[i] = static_cast<float>(t);
    }
}

__global__ void refDevKernel(const float* __restrict__ in,
                             float* __restrict__ out, int size, int smaLen,
                             int sumLen) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i < smaLen - 1 + sumLen - 1) continue;
        double ndv = 0.0, tdv = 0.0;
        for (int k = 0; k < sumLen; ++k) {
            const int j = i - k;
            double sma = 0.0;
            for (int m = 0; m < smaLen; ++m) sma += in[j - m];
            const double drd = in[j] - sma / smaLen;
            ndv += drd;
            tdv += fabs(drd);
        }
        if (tdv > 0.0) out[i] = static_cast<float>(100.0 * ndv / tdv);
    }
}

__global__ void momentsKernel(const float* __restrict__ in,
                              float* __restrict__ out, int size, int len,
                              bool wantKurtosis) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i < len - 1) continue;
        double mean = 0.0;
        for (int k = 0; k < len; ++k) mean += in[i - k];
        mean /= len;
        double m2 = 0.0, m3 = 0.0, m4 = 0.0;
        for (int k = 0; k < len; ++k) {
            const double d = static_cast<double>(in[i - k]) - mean;
            const double d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        m2 /= len;
        m3 /= len;
        m4 /= len;
        if (m2 <= 0.0) {
            out[i] = 0.0f;
        } else if (wantKurtosis) {
            out[i] = static_cast<float>(m4 / (m2 * m2) - 3.0);
        } else {
            out[i] = static_cast<float>(m3 / pow(m2, 1.5));
        }
    }
}

__global__ void pivotKernel(const float* __restrict__ in,
                            float* __restrict__ out, int size, int strength,
                            bool wantHigh) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        const int b = i - strength;
        if (b < strength) continue;
        bool isPivot = true;
        for (int k = 1; k <= strength && isPivot; ++k) {
            if (wantHigh) {
                isPivot = in[b] > in[b - k] && in[b] >= in[b + k];
            } else {
                isPivot = in[b] < in[b - k] && in[b] <= in[b + k];
            }
        }
        if (isPivot) out[i] = in[b];
    }
}

// mode: 0=mean, 1=max, 2=min, 3=stddev (population)
__global__ void rollingStrictKernel(const float* __restrict__ in,
                                    float* __restrict__ out, int size,
                                    int len, int mode) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (i < len - 1) continue;
        bool ok = true;
        double sum = 0.0, sq = 0.0;
        float mx = in[i], mn = in[i];
        for (int k = 0; k < len; ++k) {
            const float v = in[i - k];
            if (!isfinite(v)) { ok = false; break; }
            sum += v;
            sq += static_cast<double>(v) * v;
            mx = fmaxf(mx, v);
            mn = fminf(mn, v);
        }
        if (!ok) continue;
        if (mode == 0) {
            out[i] = static_cast<float>(sum / len);
        } else if (mode == 1) {
            out[i] = mx;
        } else if (mode == 2) {
            out[i] = mn;
        } else {
            const double mean = sum / len;
            const double var = fmax(0.0, sq / len - mean * mean);
            out[i] = static_cast<float>(sqrt(var));
        }
    }
}

inline void check_period(int period, int size, const char* what) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument(std::string(what) + ": invalid period");
    }
}

} // namespace

namespace tacuda {

// --- ASI -------------------------------------------------------------------

void ASI::calculate(const float* open, const float* high, const float* low,
                    const float* close, float* output, int size,
                    cudaStream_t stream) noexcept(false) {
    if (size <= 0) throw std::invalid_argument("ASI: invalid size");
    fill_nan(output, size, stream);
    auto term = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);
    asiTermKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        open, high, low, close, term.get(), size);
    CUDA_CHECK(cudaGetLastError());
    scan_double_inplace(term.get(), size, stream);
    castPrefixKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        term.get(), output, size, 1);
    CUDA_CHECK(cudaGetLastError());
}

void ASI::calculate(const float* input, float* output, int size,
                    cudaStream_t stream) noexcept(false) {
    calculate(input, input + size, input + 2 * size, input + 3 * size, output,
              size, stream);
}

// --- SAREL -----------------------------------------------------------------

SAREL::SAREL(float accFact, float maxAccel)
    : accFact(accFact), maxAccel(maxAccel) {
    if (!(accFact > 0.0f) || !(maxAccel >= accFact)) {
        throw std::invalid_argument("SAREL: invalid acceleration parameters");
    }
}

void SAREL::calculate(const float* high, const float* low, const float* close,
                      float* output, int size,
                      cudaStream_t stream) noexcept(false) {
    if (size <= 0) throw std::invalid_argument("SAREL: invalid size");
    fill_nan(output, size, stream);
    sarElKernel<<<1, 1, 0, stream>>>(high, low, close, output, size, accFact,
                                     maxAccel);
    CUDA_CHECK(cudaGetLastError());
}

void SAREL::calculate(const float* input, float* output, int size,
                      cudaStream_t stream) noexcept(false) {
    calculate(input, input + size, input + 2 * size, output, size, stream);
}

// --- OBTR ------------------------------------------------------------------

void OBTR::calculate(const float* high, const float* low, const float* close,
                     float* output, int size,
                     cudaStream_t stream) noexcept(false) {
    if (size <= 0) throw std::invalid_argument("OBTR: invalid size");
    fill_nan(output, size, stream);
    auto term = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);
    obtrTermKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        high, low, close, term.get(), size);
    CUDA_CHECK(cudaGetLastError());
    scan_double_inplace(term.get(), size, stream);
    castPrefixKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        term.get(), output, size, 1);
    CUDA_CHECK(cudaGetLastError());
}

void OBTR::calculate(const float* input, float* output, int size,
                     cudaStream_t stream) noexcept(false) {
    calculate(input, input + size, input + 2 * size, output, size, stream);
}

// --- WOBV ------------------------------------------------------------------

void WOBV::calculate(const float* open, const float* high, const float* low,
                     const float* close, const float* volume, float* output,
                     int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) throw std::invalid_argument("WOBV: invalid size");
    auto term = acquireDeviceBufferAsync<double>(static_cast<size_t>(size), stream);
    wobvTermKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        open, high, low, close, volume, term.get(), size);
    CUDA_CHECK(cudaGetLastError());
    scan_double_inplace(term.get(), size, stream);
    castPrefixKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        term.get(), output, size, 0);
    CUDA_CHECK(cudaGetLastError());
}

void WOBV::calculate(const float* input, float* output, int size,
                     cudaStream_t stream) noexcept(false) {
    calculate(input, input + size, input + 2 * size, input + 3 * size,
              input + 4 * size, output, size, stream);
}

// --- HPI -------------------------------------------------------------------

HPI::HPI(float factor) : factor(factor) {
    if (!(factor > 0.0f) || factor > 1.0f) {
        throw std::invalid_argument("HPI: factor must be in (0, 1]");
    }
}

void HPI::calculate(const float* high, const float* low, const float* volume,
                    float* output, int size,
                    cudaStream_t stream) noexcept(false) {
    if (size <= 0) throw std::invalid_argument("HPI: invalid size");
    fill_nan(output, size, stream);
    auto trans = acquireDeviceBufferAsync<AffineD>(static_cast<size_t>(size), stream);
    hpiTransformKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        high, low, volume, trans.get(), size, factor);
    CUDA_CHECK(cudaGetLastError());
    tacuda::detail::affine_inclusive_scan<AffineD, AffineComposeD>(
        trans.get(), trans.get(), size, stream);
    hpiFinalizeKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        trans.get(), output, size);
    CUDA_CHECK(cudaGetLastError());
}

void HPI::calculate(const float* input, float* output, int size,
                    cudaStream_t stream) noexcept(false) {
    calculate(input, input + size, input + 2 * size, output, size, stream);
}

// --- MarketStrength ----------------------------------------------------------

MarketStrength::MarketStrength(int period) : period(period) {}

void MarketStrength::calculate(const float* input, float* output, int size,
                               cudaStream_t stream) noexcept(false) {
    // First valid bar is `period` (the window reads back to i-period).
    if (period <= 0) {
        throw std::invalid_argument("MarketStrength: invalid period");
    }
    if (period >= size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    fill_nan(output, size, stream);
    marketStrengthKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, period);
    CUDA_CHECK(cudaGetLastError());
}

// --- TrendScore --------------------------------------------------------------

TrendScore::TrendScore(int lookback) : lookback(lookback) {
    if (lookback < 0) throw std::invalid_argument("TrendScore: invalid lookback");
}

void TrendScore::calculate(const float* input, float* output, int size,
                           cudaStream_t stream) noexcept(false) {
    if (size <= lookback + 10) {
        throw std::invalid_argument("TrendScore: series shorter than lookback+10");
    }
    fill_nan(output, size, stream);
    trendScoreKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, lookback);
    CUDA_CHECK(cudaGetLastError());
}

// --- RefDev ------------------------------------------------------------------

RefDev::RefDev(int smaPeriod, int sumPeriod)
    : smaPeriod(smaPeriod), sumPeriod(sumPeriod) {}

void RefDev::calculate(const float* input, float* output, int size,
                       cudaStream_t stream) noexcept(false) {
    check_period(smaPeriod, size, "RefDev");
    check_period(sumPeriod, size, "RefDev");
    if (smaPeriod - 1 + sumPeriod - 1 >= size) {
        throw std::invalid_argument("RefDev: window exceeds series");
    }
    fill_nan(output, size, stream);
    refDevKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, smaPeriod, sumPeriod);
    CUDA_CHECK(cudaGetLastError());
}

// --- RollingSkew / RollingKurtosis -------------------------------------------

RollingSkew::RollingSkew(int period) : period(period) {}

void RollingSkew::calculate(const float* input, float* output, int size,
                            cudaStream_t stream) noexcept(false) {
    check_period(period, size, "RollingSkew");
    fill_nan(output, size, stream);
    momentsKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, period, false);
    CUDA_CHECK(cudaGetLastError());
}

RollingKurtosis::RollingKurtosis(int period) : period(period) {}

void RollingKurtosis::calculate(const float* input, float* output, int size,
                                cudaStream_t stream) noexcept(false) {
    check_period(period, size, "RollingKurtosis");
    fill_nan(output, size, stream);
    momentsKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, period, true);
    CUDA_CHECK(cudaGetLastError());
}

// --- PivotHigh / PivotLow ----------------------------------------------------

PivotHigh::PivotHigh(int strength) : strength(strength) {}

void PivotHigh::calculate(const float* input, float* output, int size,
                          cudaStream_t stream) noexcept(false) {
    if (strength <= 0 || 2 * strength >= size) {
        throw std::invalid_argument("PivotHigh: invalid strength");
    }
    fill_nan(output, size, stream);
    pivotKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, strength, true);
    CUDA_CHECK(cudaGetLastError());
}

PivotLow::PivotLow(int strength) : strength(strength) {}

void PivotLow::calculate(const float* input, float* output, int size,
                         cudaStream_t stream) noexcept(false) {
    if (strength <= 0 || 2 * strength >= size) {
        throw std::invalid_argument("PivotLow: invalid strength");
    }
    fill_nan(output, size, stream);
    pivotKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, strength, false);
    CUDA_CHECK(cudaGetLastError());
}

// --- XAverage ----------------------------------------------------------------

XAverage::XAverage(int period) : period(period) {}

void XAverage::calculate(const float* input, float* output, int size,
                         cudaStream_t stream) noexcept(false) {
    check_period(period, size, "XAverage");
    fill_nan(output, size, stream);
    const double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    auto elems = acquireDeviceBufferAsync<SeededEma>(static_cast<size_t>(size), stream);
    xaverageElementKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, elems.get(), size, alpha);
    CUDA_CHECK(cudaGetLastError());
    {
        thrust::device_ptr<SeededEma> p(elems.get());
        thrust::inclusive_scan(thrust::cuda::par.on(stream), p, p + size, p,
                               SeededEmaCompose{});
    }
    xaverageFinalizeKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, elems.get(), output, size);
    CUDA_CHECK(cudaGetLastError());
}

// --- NaN-strict rolling stats -------------------------------------------------

RollingMean::RollingMean(int period) : period(period) {}

void RollingMean::calculate(const float* input, float* output, int size,
                            cudaStream_t stream) noexcept(false) {
    check_period(period, size, "RollingMean");
    fill_nan(output, size, stream);
    rollingStrictKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, period, 0);
    CUDA_CHECK(cudaGetLastError());
}

RollingMax::RollingMax(int period) : period(period) {}

void RollingMax::calculate(const float* input, float* output, int size,
                           cudaStream_t stream) noexcept(false) {
    check_period(period, size, "RollingMax");
    fill_nan(output, size, stream);
    rollingStrictKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, period, 1);
    CUDA_CHECK(cudaGetLastError());
}

RollingMin::RollingMin(int period) : period(period) {}

void RollingMin::calculate(const float* input, float* output, int size,
                           cudaStream_t stream) noexcept(false) {
    check_period(period, size, "RollingMin");
    fill_nan(output, size, stream);
    rollingStrictKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, period, 2);
    CUDA_CHECK(cudaGetLastError());
}

RollingStd::RollingStd(int period) : period(period) {}

void RollingStd::calculate(const float* input, float* output, int size,
                           cudaStream_t stream) noexcept(false) {
    check_period(period, size, "RollingStd");
    fill_nan(output, size, stream);
    rollingStrictKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
        input, output, size, period, 3);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda
