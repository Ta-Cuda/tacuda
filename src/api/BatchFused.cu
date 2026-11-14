/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "BatchFused.h"

#include <utils/ErrorState.h>
#include <utils/StreamStates.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace tacuda { namespace detail {
namespace {
using F = FusedFamily;
using namespace tacuda::streaming;
constexpr int kWindowLimit = 64;

struct Matrices {
    const float* in[4] = {};
    float* out[3] = {};
};

template <typename State, int NI, int NO>
__global__ void recurrent(Matrices m, int length, int series, State initial) {
    for (size_t s = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         s < static_cast<size_t>(series); s += static_cast<size_t>(gridDim.x) * blockDim.x) {
        State state = initial;
        const size_t row = s * length;
        for (int t = 0; t < length; ++t) {
            const size_t i = row + t;
            StreamBar b{};
            if constexpr (NI == 1) b.close = m.in[0][i];
            if constexpr (NI == 2) { b.close = m.in[0][i]; b.volume = m.in[1][i]; }
            if constexpr (NI >= 3) { b.high = m.in[0][i]; b.low = m.in[1][i]; b.close = m.in[2][i]; }
            if constexpr (NI == 4) b.volume = m.in[3][i];
            float out[NO];
            state.step(nullptr, b, out);
            for (int j = 0; j < NO; ++j) m.out[j][i] = out[j];
        }
    }
}

__global__ void mfi_rows(Matrices m, int length, int series, int period) {
    for (size_t s = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         s < static_cast<size_t>(series); s += static_cast<size_t>(gridDim.x) * blockDim.x) {
        precise::MfiState state;
        state.init(period);
        double ring[2 * kWindowLimit];
        const size_t row = s * length;
        for (int t = 0; t < length; ++t) {
            const size_t i = row + t;
            StreamBar b{m.in[0][i], m.in[1][i], m.in[2][i], m.in[3][i]};
            float value;
            state.step(ring, b, &value);
            m.out[0][i] = value;
        }
    }
}

template <F Family>
__global__ void window(Matrices m, int length, size_t total, FusedCall c) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int t = static_cast<int>(i % length);
        const long long lookback = Family == F::Stochastic
            ? static_cast<long long>(c.p) - 1 + 2LL * (c.q - 1) : c.p - 1;
        if (t < lookback) {
            m.out[0][i] = stream_nan();
            if constexpr (Family == F::Bbands) m.out[1][i] = m.out[2][i] = stream_nan();
            if constexpr (Family == F::Stochastic) m.out[1][i] = stream_nan();
            continue;
        }
        if constexpr (Family == F::Stochastic) {
            // Preserve the shipped float STOCH arithmetic and summation order.
            float fk[2 * kWindowLimit - 1];
            const int nf = 2 * c.q - 1;
            for (int k = 0; k < nf; ++k) {
                const size_t j = i - (nf - 1 - k);
                float hi = m.in[0][j], lo = m.in[1][j];
                for (int v = 1; v < c.p; ++v) {
                    hi = fmaxf(hi, m.in[0][j - v]);
                    lo = fminf(lo, m.in[1][j - v]);
                }
                const float range = hi - lo;
                fk[k] = range == 0 ? 0.0f : 100.0f * (m.in[2][j] - lo) / range;
            }
            float d = 0, k0 = 0;
            for (int j = 0; j < c.q; ++j) {
                float sum = 0;
                for (int k = 0; k < c.q; ++k) sum += fk[nf - 1 - j - k];
                const float sk = sum / c.q;
                if (j == 0) k0 = sk;
                d += sk;
            }
            m.out[0][i] = k0; m.out[1][i] = d / c.q;
        } else if constexpr (Family == F::Min || Family == F::Max) {
            const size_t start = i - (c.p - 1);
            float value = m.in[0][start];
            for (int k = 1; k < c.p; ++k) {
                if constexpr (Family == F::Max) value = fmaxf(value, m.in[0][start + k]);
                else value = fminf(value, m.in[0][start + k]);
            }
            m.out[0][i] = value;
        } else {
            double sum = 0, squares = 0;
            const size_t start = i - (c.p - 1);
            for (int k = 0; k < c.p; ++k) {
                const double x = m.in[0][start + k];
                if constexpr (Family == F::Wma) sum += x * (k + 1.0);
                else sum += x;
                if constexpr (Family == F::Stddev || Family == F::Bbands) squares += x * x;
            }
            if constexpr (Family == F::Sum) m.out[0][i] = static_cast<float>(sum);
            if constexpr (Family == F::Sma) m.out[0][i] = static_cast<float>(sum / c.p);
            if constexpr (Family == F::Wma) m.out[0][i] = static_cast<float>(sum / (0.5 * c.p * (c.p + 1.0)));
            if constexpr (Family == F::Stddev || Family == F::Bbands) {
                const double mean = sum / c.p;
                double var = squares / c.p - mean * mean;
                if (var < 0) var = 0;
                const double sd = sqrt(var);
                if constexpr (Family == F::Stddev) m.out[0][i] = static_cast<float>(sd);
                else {
                    m.out[0][i] = static_cast<float>(mean + c.up * sd);
                    m.out[1][i] = static_cast<float>(mean);
                    m.out[2][i] = static_cast<float>(mean - c.down * sd);
                }
            }
        }
    }
}

bool is_window(F f) {
    return f == F::Sma || f == F::Sum || f == F::Wma || f == F::Min || f == F::Max ||
           f == F::Stddev || f == F::Bbands || f == F::Stochastic;
}
}

bool FusedCall::enabled(int length, int series) const {
    const char* mode = std::getenv("TACUDA_BATCH_FUSED");
    if (family == F::None || (mode && std::strcmp(mode, "0") == 0)) return false;
    const bool force = mode && std::strcmp(mode, "1") == 0;
    // Invalid parameters go through the existing entry point's validation.
    if (is_window(family)) {
        if (p <= 0 || p > kWindowLimit) return false;
        if (family == F::Stochastic && (q <= 0 || q > kWindowLimit ||
            static_cast<long long>(2 * q - 1) * p + q * q > 1024)) return false;
        return true;
    }
    if (family == F::Mfi && (p < 2 || p > kWindowLimit)) return false;
    if (family == F::Adosc && (p < 2 || q < 2 || p > 100000 || q > 100000)) return false;
    if (family == F::Macd && (p < 2 || q < 2 || r < 1 || maType != 1 ||
                            p > 100000 || q > 100000 || r > 100000)) return false;
    if ((family == F::Atr || family == F::Natr || family == F::Ema || family == F::Rsi) && p <= 0) return false;
    return force || static_cast<long long>(length) <= 4LL * series;
}

ctStatus_t FusedCall::launch(const float* const* in, float* const* out,
                             int length, int series, cudaStream_t stream) const {
    Matrices m;
    const int ni = family == F::Ad || family == F::Adosc || family == F::Mfi ? 4
                 : family == F::Atr || family == F::Natr || family == F::Stochastic ? 3
                 : family == F::Obv ? 2 : 1;
    const int no = family == F::Bbands || family == F::Macd ? 3 : family == F::Stochastic ? 2 : 1;
    for (int i = 0; i < ni; ++i) m.in[i] = in[i];
    for (int i = 0; i < no; ++i) m.out[i] = out[i];
    const size_t total = static_cast<size_t>(length) * series;
    const int blocks = static_cast<int>(std::min<size_t>((total + 255) / 256, 65535));
    const int rows = static_cast<int>(std::min<size_t>((static_cast<size_t>(series) + 63) / 64, 65535));
    switch (family) {
#define WINDOW(name) case F::name: window<F::name><<<blocks, 256, 0, stream>>>(m, length, total, *this); break
        WINDOW(Sma); WINDOW(Wma); WINDOW(Sum); WINDOW(Min); WINDOW(Max);
        WINDOW(Stddev); WINDOW(Bbands); WINDOW(Stochastic);
#undef WINDOW
        case F::Mfi: mfi_rows<<<rows, 64, 0, stream>>>(m, length, series, p); break;
#define REC(name, type, ni, no, ...) case F::name: { type state; state.init(__VA_ARGS__); \
    recurrent<type, ni, no><<<rows, 64, 0, stream>>>(m, length, series, state); break; }
        REC(Ema, precise::EmaState, 1, 1, p);
        REC(Rsi, precise::RsiState, 1, 1, p);
        REC(Atr, precise::AtrState, 3, 1, p);
        REC(Natr, precise::NatrState, 3, 1, p);
        REC(Macd, precise::MacdState, 1, 3, p, q, r);
        REC(Obv, ObvState, 2, 1);
        REC(Ad, AdState, 4, 1);
        REC(Adosc, precise::AdoscState, 4, 1, p, q);
#undef REC
        default: return CT_STATUS_UNSUPPORTED;
    }
    const cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) return CT_STATUS_SUCCESS;
    set_last_error(cudaGetErrorString(err));
    return CT_STATUS_KERNEL_FAILED;
}

}}  // namespace tacuda::detail
