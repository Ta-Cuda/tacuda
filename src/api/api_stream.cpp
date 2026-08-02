/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Streaming (incremental update) API — see docs/DESIGN_STREAMING_API.md.
//
// Single-series streams carry their state on the host in double precision:
// a one-bar update is a handful of FLOPs, while a GPU roundtrip would pay
// two PCIe latencies plus a launch to do the same work. Each state machine
// below mirrors the exact recurrence of its full-series counterpart (kernel
// referenced per class), so appending bar-by-bar reproduces the tail of a
// full recompute.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "tacuda_stream.h"
#include <utils/ErrorState.h>
#include <utils/StreamStates.h>

namespace {

constexpr std::uint32_t kStreamMagic = 0x54435331u;  // 'TCS1'
constexpr std::uint32_t kStreamDead = 0xDEADBEEFu;

enum class Family {
  SMA, EMA, WMA, RSI, ATR, NATR, OBV, AD, MACD, BBANDS
};

using Bar = tacuda::streaming::StreamBar;

class StreamState {
public:
  virtual ~StreamState() = default;
  virtual Family family() const = 0;
  virtual int outputs() const = 0;
  virtual int lookback() const = 0;
  virtual void reset() = 0;
  // Consumes one closed bar and writes outputs() values.
  virtual void step(const Bar &bar, float *out) = 0;
};

// Adapter over a POD state from StreamStates.h. The POD holds the recurrence
// (shared verbatim with the batch/device path); the adapter owns the ring
// buffer the POD expects and remembers how to re-seed itself on reset.
template <typename Pod, Family kFamily>
class PodStream final : public StreamState {
public:
  explicit PodStream(std::function<void(Pod &)> seed) : seed_(std::move(seed)) {
    seed_(pod_);
    ring_.assign(static_cast<std::size_t>(pod_.ring_size()), 0.0f);
  }

  Family family() const override { return kFamily; }
  int outputs() const override { return pod_.outputs(); }
  int lookback() const override { return pod_.lookback(); }

  void reset() override {
    seed_(pod_);
    std::fill(ring_.begin(), ring_.end(), 0.0f);
  }

  void step(const Bar &bar, float *out) override {
    pod_.step(ring_.data(), bar, out);
  }

private:
  Pod pod_{};
  std::vector<float> ring_;
  std::function<void(Pod &)> seed_;
};

namespace st = tacuda::streaming;

using SmaStream = PodStream<st::SmaState, Family::SMA>;
using EmaStream = PodStream<st::EmaState, Family::EMA>;
using WmaStream = PodStream<st::WmaState, Family::WMA>;
using RsiStream = PodStream<st::RsiState, Family::RSI>;
using AtrStream = PodStream<st::AtrState, Family::ATR>;
using NatrStream = PodStream<st::NatrState, Family::NATR>;
using ObvStream = PodStream<st::ObvState, Family::OBV>;
using AdStream = PodStream<st::AdState, Family::AD>;
using MacdStream = PodStream<st::MacdState, Family::MACD>;
using BbandsStream = PodStream<st::BbandsState, Family::BBANDS>;
}  // namespace

// ---------------------------------------------------------------------------
// Handle plumbing
// ---------------------------------------------------------------------------

struct ctStream {
  std::uint32_t magic;
  std::unique_ptr<StreamState> impl;
  long long bars;
};

namespace {

// Same contract as api.cpp's api_guard: clear the thread-local error on
// entry, never let an exception cross the C boundary.
template <typename Fn>
ctStatus_t stream_guard(const char *what, Fn &&fn) noexcept {
  tacuda::detail::clear_last_error();
  try {
    return fn();
  } catch (const std::invalid_argument &e) {
    tacuda::detail::set_last_error(e.what());
    return CT_STATUS_INVALID_ARGUMENT;
  } catch (const std::bad_alloc &) {
    tacuda::detail::set_last_error(what ? what : "allocation failed");
    return CT_STATUS_ALLOC_FAILED;
  } catch (const std::exception &e) {
    tacuda::detail::set_last_error(e.what());
    return CT_STATUS_KERNEL_FAILED;
  } catch (...) {
    tacuda::detail::set_last_error(what ? what : "unknown error");
    return CT_STATUS_KERNEL_FAILED;
  }
}

bool valid_handle(const ctStream *s) {
  return s != nullptr && s->magic == kStreamMagic && s->impl != nullptr;
}

ctStatus_t reject(const std::string &msg) {
  tacuda::detail::set_last_error(msg);
  return CT_STATUS_INVALID_ARGUMENT;
}

// TA-Lib's own BAD_PARAM range per function: most families start at 2, the
// ones that degenerate gracefully (ATR/NATR reduce to TRANGE, a MACD signal
// of 1 tracks the line) start at 1.
ctStatus_t check_period(const char *what, int period, int minPeriod = 2) {
  if (period < minPeriod || period > 100000) {
    return reject(std::string(what) + ": period must be in [" +
                  std::to_string(minPeriod) + ", 100000] (TA-Lib BAD_PARAM)");
  }
  return CT_STATUS_SUCCESS;
}

ctStatus_t make_stream(std::unique_ptr<StreamState> impl, ctStream **outStream) {
  if (!outStream) return reject("stream create: null out parameter");
  *outStream = nullptr;
  *outStream = new ctStream{kStreamMagic, std::move(impl), 0};
  return CT_STATUS_SUCCESS;
}

// Runs the state machine over n bars assembled by `makeBar`, scattering each
// step's outputs into up to three caller arrays.
template <typename MakeBar>
ctStatus_t run_append(const char *what, ctStream *stream, Family family,
                      int n, float *o0, float *o1, float *o2,
                      MakeBar &&makeBar) {
  if (!valid_handle(stream)) {
    return reject(std::string(what) + ": invalid or destroyed stream handle");
  }
  if (stream->impl->family() != family) {
    return reject(std::string(what) + ": stream was created for a different indicator");
  }
  if (n <= 0) return reject(std::string(what) + ": n must be > 0");

  const int outCount = stream->impl->outputs();
  float *outs[3] = {o0, o1, o2};
  for (int k = 0; k < outCount; ++k) {
    if (!outs[k]) return reject(std::string(what) + ": null output pointer");
  }

  float scratch[3];
  for (int j = 0; j < n; ++j) {
    stream->impl->step(makeBar(j), scratch);
    for (int k = 0; k < outCount; ++k) outs[k][j] = scratch[k];
  }
  stream->bars += n;
  return CT_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Lifetime and introspection
// ---------------------------------------------------------------------------

ctStatus_t ct_stream_destroy(ctStream_t *stream) {
  return stream_guard("ct_stream_destroy", [&]() -> ctStatus_t {
    if (!stream) return CT_STATUS_SUCCESS;  // NULL-safe by contract
    if (stream->magic != kStreamMagic) {
      return reject("ct_stream_destroy: invalid or already destroyed handle");
    }
    stream->magic = kStreamDead;
    stream->impl.reset();
    delete stream;
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_reset(ctStream_t *stream) {
  return stream_guard("ct_stream_reset", [&]() -> ctStatus_t {
    if (!valid_handle(stream)) {
      return reject("ct_stream_reset: invalid or destroyed stream handle");
    }
    stream->impl->reset();
    stream->bars = 0;
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_lookback(const ctStream_t *stream, int *outLookback) {
  return stream_guard("ct_stream_lookback", [&]() -> ctStatus_t {
    if (!valid_handle(stream)) {
      return reject("ct_stream_lookback: invalid or destroyed stream handle");
    }
    if (!outLookback) return reject("ct_stream_lookback: null out parameter");
    *outLookback = stream->impl->lookback();
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_bars(const ctStream_t *stream, long long *outBars) {
  return stream_guard("ct_stream_bars", [&]() -> ctStatus_t {
    if (!valid_handle(stream)) {
      return reject("ct_stream_bars: invalid or destroyed stream handle");
    }
    if (!outBars) return reject("ct_stream_bars: null out parameter");
    *outBars = stream->bars;
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_outputs(const ctStream_t *stream, int *outCount) {
  return stream_guard("ct_stream_outputs", [&]() -> ctStatus_t {
    if (!valid_handle(stream)) {
      return reject("ct_stream_outputs: invalid or destroyed stream handle");
    }
    if (!outCount) return reject("ct_stream_outputs: null out parameter");
    *outCount = stream->impl->outputs();
    return CT_STATUS_SUCCESS;
  });
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

ctStatus_t ct_sma_stream_create(int period, ctStream_t **outStream) {
  return stream_guard("ct_sma_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_sma_stream_create", period);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(std::unique_ptr<StreamState>(new SmaStream([period](st::SmaState &s) { s.init(period); })), outStream);
  });
}

ctStatus_t ct_ema_stream_create(int period, ctStream_t **outStream) {
  return stream_guard("ct_ema_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_ema_stream_create", period);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(std::unique_ptr<StreamState>(new EmaStream([period](st::EmaState &s) { s.init(period); })), outStream);
  });
}

ctStatus_t ct_wma_stream_create(int period, ctStream_t **outStream) {
  return stream_guard("ct_wma_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_wma_stream_create", period);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(std::unique_ptr<StreamState>(new WmaStream([period](st::WmaState &s) { s.init(period); })), outStream);
  });
}

ctStatus_t ct_rsi_stream_create(int period, ctStream_t **outStream) {
  return stream_guard("ct_rsi_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_rsi_stream_create", period);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(std::unique_ptr<StreamState>(new RsiStream([period](st::RsiState &s) { s.init(period); })), outStream);
  });
}

ctStatus_t ct_atr_stream_create(int period, ctStream_t **outStream) {
  return stream_guard("ct_atr_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_atr_stream_create", period, 1);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(std::unique_ptr<StreamState>(new AtrStream([period](st::AtrState &s) { s.init(period); })), outStream);
  });
}

ctStatus_t ct_natr_stream_create(int period, ctStream_t **outStream) {
  return stream_guard("ct_natr_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_natr_stream_create", period, 1);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(std::unique_ptr<StreamState>(new NatrStream([period](st::NatrState &s) { s.init(period); })), outStream);
  });
}

ctStatus_t ct_obv_stream_create(ctStream_t **outStream) {
  return stream_guard("ct_obv_stream_create", [&]() -> ctStatus_t {
    return make_stream(std::unique_ptr<StreamState>(new ObvStream([](st::ObvState &s) { s.init(); })), outStream);
  });
}

ctStatus_t ct_ad_stream_create(ctStream_t **outStream) {
  return stream_guard("ct_ad_stream_create", [&]() -> ctStatus_t {
    return make_stream(std::unique_ptr<StreamState>(new AdStream([](st::AdState &s) { s.init(); })), outStream);
  });
}

ctStatus_t ct_macd_stream_create(int fastPeriod, int slowPeriod, int signalPeriod,
                                 ctStream_t **outStream) {
  return stream_guard("ct_macd_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_macd_stream_create (fastPeriod)", fastPeriod);
    if (st != CT_STATUS_SUCCESS) return st;
    st = check_period("ct_macd_stream_create (slowPeriod)", slowPeriod);
    if (st != CT_STATUS_SUCCESS) return st;
    st = check_period("ct_macd_stream_create (signalPeriod)", signalPeriod, 1);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(
        std::unique_ptr<StreamState>(new MacdStream([fastPeriod, slowPeriod, signalPeriod](st::MacdState &s) {
          s.init(fastPeriod, slowPeriod, signalPeriod);
        })),
        outStream);
  });
}

ctStatus_t ct_bbands_stream_create(int period, float upperMul, float lowerMul,
                                   ctStream_t **outStream) {
  return stream_guard("ct_bbands_stream_create", [&]() -> ctStatus_t {
    ctStatus_t st = check_period("ct_bbands_stream_create", period);
    if (st != CT_STATUS_SUCCESS) return st;
    return make_stream(
        std::unique_ptr<StreamState>(new BbandsStream([period, upperMul, lowerMul](st::BbandsState &s) {
          s.init(period, upperMul, lowerMul);
        })),
        outStream);
  });
}

// ---------------------------------------------------------------------------
// Append
// ---------------------------------------------------------------------------

namespace {

ctStatus_t append_close(const char *what, ctStream *stream, Family family,
                        const float *close, int n, float *o0, float *o1,
                        float *o2) {
  if (!close) return reject(std::string(what) + ": null input pointer");
  return run_append(what, stream, family, n, o0, o1, o2, [&](int j) {
    Bar b{};
    b.close = close[j];
    return b;
  });
}

}  // namespace

ctStatus_t ct_sma_stream_append(ctStream_t *stream, const float *close, int n,
                                float *out) {
  return stream_guard("ct_sma_stream_append", [&]() -> ctStatus_t {
    return append_close("ct_sma_stream_append", stream, Family::SMA, close, n,
                        out, nullptr, nullptr);
  });
}

ctStatus_t ct_ema_stream_append(ctStream_t *stream, const float *close, int n,
                                float *out) {
  return stream_guard("ct_ema_stream_append", [&]() -> ctStatus_t {
    return append_close("ct_ema_stream_append", stream, Family::EMA, close, n,
                        out, nullptr, nullptr);
  });
}

ctStatus_t ct_wma_stream_append(ctStream_t *stream, const float *close, int n,
                                float *out) {
  return stream_guard("ct_wma_stream_append", [&]() -> ctStatus_t {
    return append_close("ct_wma_stream_append", stream, Family::WMA, close, n,
                        out, nullptr, nullptr);
  });
}

ctStatus_t ct_rsi_stream_append(ctStream_t *stream, const float *close, int n,
                                float *out) {
  return stream_guard("ct_rsi_stream_append", [&]() -> ctStatus_t {
    return append_close("ct_rsi_stream_append", stream, Family::RSI, close, n,
                        out, nullptr, nullptr);
  });
}

ctStatus_t ct_macd_stream_append(ctStream_t *stream, const float *close, int n,
                                 float *outMacd, float *outSignal,
                                 float *outHist) {
  return stream_guard("ct_macd_stream_append", [&]() -> ctStatus_t {
    return append_close("ct_macd_stream_append", stream, Family::MACD, close, n,
                        outMacd, outSignal, outHist);
  });
}

ctStatus_t ct_bbands_stream_append(ctStream_t *stream, const float *close, int n,
                                   float *outUpper, float *outMiddle,
                                   float *outLower) {
  return stream_guard("ct_bbands_stream_append", [&]() -> ctStatus_t {
    return append_close("ct_bbands_stream_append", stream, Family::BBANDS, close,
                        n, outUpper, outMiddle, outLower);
  });
}

ctStatus_t ct_atr_stream_append(ctStream_t *stream, const float *high,
                                const float *low, const float *close, int n,
                                float *out) {
  return stream_guard("ct_atr_stream_append", [&]() -> ctStatus_t {
    if (!high || !low || !close) {
      return reject("ct_atr_stream_append: null input pointer");
    }
    return run_append("ct_atr_stream_append", stream, Family::ATR, n, out,
                      nullptr, nullptr, [&](int j) {
                        Bar b{};
                        b.high = high[j];
                        b.low = low[j];
                        b.close = close[j];
                        return b;
                      });
  });
}

ctStatus_t ct_natr_stream_append(ctStream_t *stream, const float *high,
                                 const float *low, const float *close, int n,
                                 float *out) {
  return stream_guard("ct_natr_stream_append", [&]() -> ctStatus_t {
    if (!high || !low || !close) {
      return reject("ct_natr_stream_append: null input pointer");
    }
    return run_append("ct_natr_stream_append", stream, Family::NATR, n, out,
                      nullptr, nullptr, [&](int j) {
                        Bar b{};
                        b.high = high[j];
                        b.low = low[j];
                        b.close = close[j];
                        return b;
                      });
  });
}

ctStatus_t ct_obv_stream_append(ctStream_t *stream, const float *close,
                                const float *volume, int n, float *out) {
  return stream_guard("ct_obv_stream_append", [&]() -> ctStatus_t {
    if (!close || !volume) {
      return reject("ct_obv_stream_append: null input pointer");
    }
    return run_append("ct_obv_stream_append", stream, Family::OBV, n, out,
                      nullptr, nullptr, [&](int j) {
                        Bar b{};
                        b.close = close[j];
                        b.volume = volume[j];
                        return b;
                      });
  });
}

ctStatus_t ct_ad_stream_append(ctStream_t *stream, const float *high,
                               const float *low, const float *close,
                               const float *volume, int n, float *out) {
  return stream_guard("ct_ad_stream_append", [&]() -> ctStatus_t {
    if (!high || !low || !close || !volume) {
      return reject("ct_ad_stream_append: null input pointer");
    }
    return run_append("ct_ad_stream_append", stream, Family::AD, n, out, nullptr,
                      nullptr, [&](int j) {
                        Bar b{};
                        b.high = high[j];
                        b.low = low[j];
                        b.close = close[j];
                        b.volume = volume[j];
                        return b;
                      });
  });
}

}  // extern "C"
