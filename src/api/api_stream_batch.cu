/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Batch (portfolio) streams: one call appends closed bars to every series of
// a portfolio. See docs/DESIGN_STREAMING_API.md (D2) and STREAMING_API.md.
//
// The state machines are the ones in utils/StreamStates.h — shared verbatim
// with the single-series path, so a batch of N series is numerically the same
// as N independent streams (asserted in tests/cpp/test_stream_batch.cpp).
//
// Two execution paths behind one API, because the crossover is a measurement,
// not an assumption: a per-bar state update is tens of nanoseconds, so small
// portfolios finish on one CPU core long before a kernel launch plus two
// transfers would return, while large ones amortize that fixed cost and win on
// the GPU. CT_STREAM_EXEC_AUTO switches at the measured crossover, documented
// in STREAMING_API.md.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tacuda_stream.h"
#include <utils/ErrorState.h>
#include <utils/StreamStates.h>
#include <utils/StreamHandleRegistry.h>

namespace st = tacuda::streaming;

namespace {

// Measured on an i3-12100F + RTX 3090 Ti (benchmarks/bench_stream_portfolio,
// 7-indicator pipeline, one bar per tick): CPU p50 is 71.9 us at 1024 series
// against 126.3 us on the GPU, and 317.8 us at 4096 against 161.6 us. The
// crossover sits between them.
constexpr int kGpuThreshold = 2048;

enum class Family { SMA, EMA, WMA, RSI, ATR, NATR, OBV, AD, MACD, BBANDS };

// Series-major input matrices; unused components stay null.
struct Inputs {
  const float *high;
  const float *low;
  const float *close;
  const float *volume;
};

inline st::StreamBar bar_at(const Inputs &in, std::size_t idx) {
  st::StreamBar b;
  b.high = in.high ? in.high[idx] : 0.0f;
  b.low = in.low ? in.low[idx] : 0.0f;
  b.close = in.close ? in.close[idx] : 0.0f;
  b.volume = in.volume ? in.volume[idx] : 0.0f;
  return b;
}

// ---------------------------------------------------------------------------
// Device kernel — one thread per series, state and ring resident on the GPU.
// ---------------------------------------------------------------------------

template <typename Pod>
__global__ void stream_batch_kernel(Pod *states, float *rings, int ringLen,
                                    const float *high, const float *low,
                                    const float *close, const float *volume,
                                    int numSeries, int n, int numOutputs,
                                    float *out0, float *out1, float *out2) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= numSeries) return;

  Pod state = states[s];  // work in registers/local, write back once
  float *ring = ringLen > 0 ? rings + static_cast<std::size_t>(s) * ringLen : nullptr;
  float *outs[3] = {out0, out1, out2};

  for (int j = 0; j < n; ++j) {
    const std::size_t idx = static_cast<std::size_t>(s) * n + j;
    st::StreamBar bar;
    bar.high = high ? high[idx] : 0.0f;
    bar.low = low ? low[idx] : 0.0f;
    bar.close = close ? close[idx] : 0.0f;
    bar.volume = volume ? volume[idx] : 0.0f;

    float scratch[3];
    state.step(ring, bar, scratch);
    for (int k = 0; k < numOutputs; ++k) outs[k][idx] = scratch[k];
  }
  states[s] = state;
}

// ---------------------------------------------------------------------------
// Type-erased engine over one POD family
// ---------------------------------------------------------------------------

class BatchEngine {
public:
  virtual ~BatchEngine() = default;
  virtual void resetAll() = 0;
  virtual void appendCpu(const Inputs &in, int n, float **outs, int from, int to) = 0;
  virtual ctStatus_t appendGpu(const Inputs &in, int n, float **outs) = 0;
  virtual void releaseDevice() = 0;
  virtual int numOutputs() const = 0;
  virtual int lookback() const = 0;
  virtual int ringLen() const = 0;
};

template <typename Pod>
class BatchEngineT final : public BatchEngine {
public:
  BatchEngineT(int numSeries, std::function<void(Pod &)> seed)
      : numSeries_(numSeries), seed_(std::move(seed)) {
    Pod probe{};
    seed_(probe);
    ringLen_ = probe.ring_size();
    outputs_ = probe.outputs();
    lookback_ = probe.lookback();

    states_.assign(static_cast<std::size_t>(numSeries_), probe);
    rings_.assign(static_cast<std::size_t>(numSeries_) *
                      static_cast<std::size_t>(ringLen_),
                  0.0f);
  }

  ~BatchEngineT() override { releaseDevice(); }

  void resetAll() override {
    Pod fresh{};
    seed_(fresh);
    std::fill(states_.begin(), states_.end(), fresh);
    std::fill(rings_.begin(), rings_.end(), 0.0f);
    deviceDirty_ = true;
  }

  void appendCpu(const Inputs &in, int n, float **outs, int from, int to) override {
    if (deviceLive_) syncFromDevice();
    for (int s = from; s < to; ++s) {
      Pod state = states_[static_cast<std::size_t>(s)];
      float *ring = ringLen_ > 0
                        ? rings_.data() + static_cast<std::size_t>(s) * ringLen_
                        : nullptr;
      for (int j = 0; j < n; ++j) {
        const std::size_t idx = static_cast<std::size_t>(s) * n + j;
        float scratch[3];
        state.step(ring, bar_at(in, idx), scratch);
        for (int k = 0; k < outputs_; ++k) outs[k][idx] = scratch[k];
      }
      states_[static_cast<std::size_t>(s)] = state;
    }
    deviceDirty_ = true;
  }

  ctStatus_t appendGpu(const Inputs &in, int n, float **outs) override {
    const std::size_t elems = static_cast<std::size_t>(numSeries_) * n;
    ctStatus_t status = ensureDevice();
    if (status != CT_STATUS_SUCCESS) return status;
    status = ensureIoBuffers(elems);
    if (status != CT_STATUS_SUCCESS) return status;

    // Only the components this family actually reads cross the bus.
    const float *src[4] = {in.high, in.low, in.close, in.volume};
    for (int c = 0; c < 4; ++c) {
      if (!src[c]) continue;
      if (cudaMemcpy(devIn_[c], src[c], elems * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        tacuda::detail::set_last_error("stream batch: device upload failed");
        return CT_STATUS_COPY_FAILED;
      }
    }

    const int block = 128;
    const int grid = (numSeries_ + block - 1) / block;
    stream_batch_kernel<Pod><<<grid, block>>>(
        devStates_, devRings_, ringLen_,
        in.high ? devIn_[0] : nullptr, in.low ? devIn_[1] : nullptr,
        in.close ? devIn_[2] : nullptr, in.volume ? devIn_[3] : nullptr,
        numSeries_, n, outputs_, devOut_[0], devOut_[1], devOut_[2]);
    if (cudaGetLastError() != cudaSuccess) {
      tacuda::detail::set_last_error("stream batch: kernel launch failed");
      return CT_STATUS_KERNEL_FAILED;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      tacuda::detail::set_last_error("stream batch: kernel execution failed");
      return CT_STATUS_KERNEL_FAILED;
    }
    for (int k = 0; k < outputs_; ++k) {
      if (cudaMemcpy(outs[k], devOut_[k], elems * sizeof(float),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        tacuda::detail::set_last_error("stream batch: device download failed");
        return CT_STATUS_COPY_FAILED;
      }
    }
    hostDirty_ = true;  // the device now holds the authoritative state
    return CT_STATUS_SUCCESS;
  }

  void releaseDevice() override {
    if (devStates_) { cudaFree(devStates_); devStates_ = nullptr; }
    if (devRings_) { cudaFree(devRings_); devRings_ = nullptr; }
    for (float *&p : devIn_) { if (p) { cudaFree(p); p = nullptr; } }
    for (float *&p : devOut_) { if (p) { cudaFree(p); p = nullptr; } }
    ioElems_ = 0;
    deviceLive_ = false;
  }

  int numOutputs() const override { return outputs_; }
  int lookback() const override { return lookback_; }
  int ringLen() const override { return ringLen_; }

private:
  // I/O buffers stay resident across appends: reallocating them per tick
  // would dominate the very latency this path exists to reduce.
  ctStatus_t ensureIoBuffers(std::size_t elems) {
    if (elems <= ioElems_) return CT_STATUS_SUCCESS;
    for (float *&p : devIn_) { if (p) { cudaFree(p); p = nullptr; } }
    for (float *&p : devOut_) { if (p) { cudaFree(p); p = nullptr; } }
    const std::size_t bytes = elems * sizeof(float);
    for (int c = 0; c < 4; ++c) {
      if (cudaMalloc(&devIn_[c], bytes) != cudaSuccess) {
        tacuda::detail::set_last_error("stream batch: input allocation failed");
        return CT_STATUS_ALLOC_FAILED;
      }
    }
    for (int k = 0; k < outputs_; ++k) {
      if (cudaMalloc(&devOut_[k], bytes) != cudaSuccess) {
        tacuda::detail::set_last_error("stream batch: output allocation failed");
        return CT_STATUS_ALLOC_FAILED;
      }
    }
    ioElems_ = elems;
    return CT_STATUS_SUCCESS;
  }

  ctStatus_t ensureDevice() {
    const std::size_t stateBytes = static_cast<std::size_t>(numSeries_) * sizeof(Pod);
    const std::size_t ringBytes = static_cast<std::size_t>(numSeries_) *
                                  static_cast<std::size_t>(ringLen_) * sizeof(float);
    if (!devStates_) {
      if (cudaMalloc(&devStates_, stateBytes) != cudaSuccess) return CT_STATUS_ALLOC_FAILED;
      deviceDirty_ = true;
    }
    if (!devRings_ && ringBytes > 0) {
      if (cudaMalloc(&devRings_, ringBytes) != cudaSuccess) return CT_STATUS_ALLOC_FAILED;
      deviceDirty_ = true;
    }
    if (deviceDirty_) {
      if (cudaMemcpy(devStates_, states_.data(), stateBytes,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
      }
      if (ringBytes > 0 &&
          cudaMemcpy(devRings_, rings_.data(), ringBytes,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
      }
      deviceDirty_ = false;
    }
    deviceLive_ = true;
    return CT_STATUS_SUCCESS;
  }

  void syncFromDevice() {
    if (!hostDirty_ || !devStates_) { hostDirty_ = false; return; }
    cudaMemcpy(states_.data(), devStates_,
               static_cast<std::size_t>(numSeries_) * sizeof(Pod),
               cudaMemcpyDeviceToHost);
    if (devRings_) {
      cudaMemcpy(rings_.data(), devRings_,
                 static_cast<std::size_t>(numSeries_) *
                     static_cast<std::size_t>(ringLen_) * sizeof(float),
                 cudaMemcpyDeviceToHost);
    }
    hostDirty_ = false;
  }

  int numSeries_;
  std::function<void(Pod &)> seed_;
  int ringLen_ = 0;
  int outputs_ = 1;
  int lookback_ = 0;
  std::vector<Pod> states_;
  std::vector<float> rings_;
  Pod *devStates_ = nullptr;
  float *devRings_ = nullptr;
  float *devIn_[4] = {nullptr, nullptr, nullptr, nullptr};
  float *devOut_[3] = {nullptr, nullptr, nullptr};
  std::size_t ioElems_ = 0;
  bool deviceLive_ = false;
  bool deviceDirty_ = true;   // host state not yet pushed to the device
  bool hostDirty_ = false;    // device state not yet pulled back to the host
};

}  // namespace

struct BatchEntry {
  Family family;
  int numSeries;
  long long bars;
  ctStreamExec_t exec;
  std::unique_ptr<BatchEngine> engine;
};

namespace {

tacuda::detail::StreamHandleRegistry<ctStreamBatch_t, BatchEntry, 2> batches;

template <typename Fn>
ctStatus_t batch_guard(const char *what, Fn &&fn) noexcept {
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

ctStatus_t reject(const std::string &msg) {
  tacuda::detail::set_last_error(msg);
  return CT_STATUS_INVALID_ARGUMENT;
}

ctStatus_t check_period(const char *what, int period, int minPeriod = 2) {
  if (period < minPeriod || period > 100000) {
    return reject(std::string(what) + ": period must be in [" +
                  std::to_string(minPeriod) + ", 100000] (TA-Lib BAD_PARAM)");
  }
  return CT_STATUS_SUCCESS;
}

ctStatus_t check_series(const char *what, int numSeries) {
  if (numSeries <= 0) return reject(std::string(what) + ": numSeries must be > 0");
  return CT_STATUS_SUCCESS;
}

template <typename Pod>
ctStatus_t make_batch(Family family, int numSeries, std::function<void(Pod &)> seed,
                      ctStreamBatch **out) {
  if (!out) return reject("stream batch create: null out parameter");
  *out = nullptr;
  auto engine = std::unique_ptr<BatchEngine>(
      new BatchEngineT<Pod>(numSeries, std::move(seed)));
  *out = batches.insert(std::make_unique<BatchEntry>(BatchEntry{family, numSeries, 0,
                           CT_STREAM_EXEC_AUTO, std::move(engine)}));
  return CT_STATUS_SUCCESS;
}

ctStatus_t run_batch_append(const char *what, ctStreamBatch *batch, Family family,
                            const Inputs &in, int n, float *o0, float *o1,
                            float *o2) {
  auto *state = batches.find(batch);
  if (!state) {
    return reject(std::string(what) + ": invalid or destroyed batch handle");
  }
  if (state->family != family) {
    return reject(std::string(what) + ": batch was created for a different indicator");
  }
  if (n <= 0) return reject(std::string(what) + ": n must be > 0");

  const int outCount = state->engine->numOutputs();
  float *outs[3] = {o0, o1, o2};
  for (int k = 0; k < outCount; ++k) {
    if (!outs[k]) return reject(std::string(what) + ": null output pointer");
  }

  ctStreamExec_t exec = state->exec;
  if (exec == CT_STREAM_EXEC_AUTO) {
    exec = state->numSeries >= kGpuThreshold ? CT_STREAM_EXEC_GPU : CT_STREAM_EXEC_CPU;
  }

  ctStatus_t status = CT_STATUS_SUCCESS;
  if (exec == CT_STREAM_EXEC_GPU) {
    status = state->engine->appendGpu(in, n, outs);
  } else {
    state->engine->appendCpu(in, n, outs, 0, state->numSeries);
  }
  if (status == CT_STATUS_SUCCESS) state->bars += n;
  return status;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Lifetime and introspection
// ---------------------------------------------------------------------------

ctStatus_t ct_stream_batch_destroy(ctStreamBatch_t *batch) {
  return batch_guard("ct_stream_batch_destroy", [&]() -> ctStatus_t {
    if (!batch) return CT_STATUS_SUCCESS;
    auto state = batches.remove(batch);
    if (!state) {
      return reject("ct_stream_batch_destroy: invalid or already destroyed handle");
    }
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_batch_reset(ctStreamBatch_t *batch) {
  return batch_guard("ct_stream_batch_reset", [&]() -> ctStatus_t {
    auto *state = batches.find(batch);
    if (!state) {
      return reject("ct_stream_batch_reset: invalid or destroyed batch handle");
    }
    state->engine->resetAll();
    state->bars = 0;
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_batch_lookback(const ctStreamBatch_t *batch, int *outLookback) {
  return batch_guard("ct_stream_batch_lookback", [&]() -> ctStatus_t {
    auto *state = batches.find(batch);
    if (!state) {
      return reject("ct_stream_batch_lookback: invalid or destroyed batch handle");
    }
    if (!outLookback) return reject("ct_stream_batch_lookback: null out parameter");
    *outLookback = state->engine->lookback();
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_batch_series(const ctStreamBatch_t *batch, int *outSeries) {
  return batch_guard("ct_stream_batch_series", [&]() -> ctStatus_t {
    auto *state = batches.find(batch);
    if (!state) {
      return reject("ct_stream_batch_series: invalid or destroyed batch handle");
    }
    if (!outSeries) return reject("ct_stream_batch_series: null out parameter");
    *outSeries = state->numSeries;
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_batch_outputs(const ctStreamBatch_t *batch, int *outCount) {
  return batch_guard("ct_stream_batch_outputs", [&]() -> ctStatus_t {
    auto *state = batches.find(batch);
    if (!state) {
      return reject("ct_stream_batch_outputs: invalid or destroyed batch handle");
    }
    if (!outCount) return reject("ct_stream_batch_outputs: null out parameter");
    *outCount = state->engine->numOutputs();
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_batch_bars(const ctStreamBatch_t *batch, long long *outBars) {
  return batch_guard("ct_stream_batch_bars", [&]() -> ctStatus_t {
    auto *state = batches.find(batch);
    if (!state) {
      return reject("ct_stream_batch_bars: invalid or destroyed batch handle");
    }
    if (!outBars) return reject("ct_stream_batch_bars: null out parameter");
    *outBars = state->bars;
    return CT_STATUS_SUCCESS;
  });
}

ctStatus_t ct_stream_batch_set_exec(ctStreamBatch_t *batch, ctStreamExec_t exec) {
  return batch_guard("ct_stream_batch_set_exec", [&]() -> ctStatus_t {
    auto *state = batches.find(batch);
    if (!state) {
      return reject("ct_stream_batch_set_exec: invalid or destroyed batch handle");
    }
    if (exec < CT_STREAM_EXEC_AUTO || exec > CT_STREAM_EXEC_GPU) {
      return reject("ct_stream_batch_set_exec: unknown execution policy");
    }
    state->exec = exec;
    return CT_STATUS_SUCCESS;
  });
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

#define TACUDA_BATCH_CREATE_PERIOD(fn, Pod, fam, minPeriod)                     \
  ctStatus_t fn(int period, int numSeries, ctStreamBatch_t **outBatch) {        \
    return batch_guard(#fn, [&]() -> ctStatus_t {                               \
      ctStatus_t status = check_period(#fn, period, minPeriod);                      \
      if (status != CT_STATUS_SUCCESS) return status;                                     \
      status = check_series(#fn, numSeries);                                         \
      if (status != CT_STATUS_SUCCESS) return status;                                     \
      return make_batch<Pod>(fam, numSeries,                                    \
                             [period](Pod &v) { v.init(period); }, outBatch); \
    });                                                                         \
  }

TACUDA_BATCH_CREATE_PERIOD(ct_sma_stream_batch_create, st::SmaState, Family::SMA, 2)
TACUDA_BATCH_CREATE_PERIOD(ct_ema_stream_batch_create, st::EmaState, Family::EMA, 2)
TACUDA_BATCH_CREATE_PERIOD(ct_wma_stream_batch_create, st::WmaState, Family::WMA, 2)
TACUDA_BATCH_CREATE_PERIOD(ct_rsi_stream_batch_create, st::RsiState, Family::RSI, 2)
TACUDA_BATCH_CREATE_PERIOD(ct_atr_stream_batch_create, st::AtrState, Family::ATR, 1)
TACUDA_BATCH_CREATE_PERIOD(ct_natr_stream_batch_create, st::NatrState, Family::NATR, 1)

#undef TACUDA_BATCH_CREATE_PERIOD

ctStatus_t ct_obv_stream_batch_create(int numSeries, ctStreamBatch_t **outBatch) {
  return batch_guard("ct_obv_stream_batch_create", [&]() -> ctStatus_t {
    ctStatus_t status = check_series("ct_obv_stream_batch_create", numSeries);
    if (status != CT_STATUS_SUCCESS) return status;
    return make_batch<st::ObvState>(Family::OBV, numSeries,
                                    [](st::ObvState &v) { v.init(); }, outBatch);
  });
}

ctStatus_t ct_ad_stream_batch_create(int numSeries, ctStreamBatch_t **outBatch) {
  return batch_guard("ct_ad_stream_batch_create", [&]() -> ctStatus_t {
    ctStatus_t status = check_series("ct_ad_stream_batch_create", numSeries);
    if (status != CT_STATUS_SUCCESS) return status;
    return make_batch<st::AdState>(Family::AD, numSeries,
                                   [](st::AdState &v) { v.init(); }, outBatch);
  });
}

ctStatus_t ct_macd_stream_batch_create(int fastPeriod, int slowPeriod,
                                       int signalPeriod, int numSeries,
                                       ctStreamBatch_t **outBatch) {
  return batch_guard("ct_macd_stream_batch_create", [&]() -> ctStatus_t {
    ctStatus_t status = check_period("ct_macd_stream_batch_create (fastPeriod)", fastPeriod);
    if (status != CT_STATUS_SUCCESS) return status;
    status = check_period("ct_macd_stream_batch_create (slowPeriod)", slowPeriod);
    if (status != CT_STATUS_SUCCESS) return status;
    status = check_period("ct_macd_stream_batch_create (signalPeriod)", signalPeriod, 1);
    if (status != CT_STATUS_SUCCESS) return status;
    status = check_series("ct_macd_stream_batch_create", numSeries);
    if (status != CT_STATUS_SUCCESS) return status;
    return make_batch<st::MacdState>(
        Family::MACD, numSeries,
        [fastPeriod, slowPeriod, signalPeriod](st::MacdState &v) {
          v.init(fastPeriod, slowPeriod, signalPeriod);
        },
        outBatch);
  });
}

ctStatus_t ct_bbands_stream_batch_create(int period, float upperMul, float lowerMul,
                                         int numSeries, ctStreamBatch_t **outBatch) {
  return batch_guard("ct_bbands_stream_batch_create", [&]() -> ctStatus_t {
    ctStatus_t status = check_period("ct_bbands_stream_batch_create", period);
    if (status != CT_STATUS_SUCCESS) return status;
    status = check_series("ct_bbands_stream_batch_create", numSeries);
    if (status != CT_STATUS_SUCCESS) return status;
    return make_batch<st::BbandsState>(
        Family::BBANDS, numSeries,
        [period, upperMul, lowerMul](st::BbandsState &v) {
          v.init(period, upperMul, lowerMul);
        },
        outBatch);
  });
}

// ---------------------------------------------------------------------------
// Append
// ---------------------------------------------------------------------------

#define TACUDA_BATCH_APPEND_CLOSE(fn, fam)                                      \
  ctStatus_t fn(ctStreamBatch_t *batch, const float *closes, int n, float *outs) { \
    return batch_guard(#fn, [&]() -> ctStatus_t {                               \
      if (!closes) return reject(#fn ": null input pointer");                   \
      Inputs in{nullptr, nullptr, closes, nullptr};                             \
      return run_batch_append(#fn, batch, fam, in, n, outs, nullptr, nullptr);  \
    });                                                                         \
  }

TACUDA_BATCH_APPEND_CLOSE(ct_sma_stream_batch_append, Family::SMA)
TACUDA_BATCH_APPEND_CLOSE(ct_ema_stream_batch_append, Family::EMA)
TACUDA_BATCH_APPEND_CLOSE(ct_wma_stream_batch_append, Family::WMA)
TACUDA_BATCH_APPEND_CLOSE(ct_rsi_stream_batch_append, Family::RSI)

#undef TACUDA_BATCH_APPEND_CLOSE

ctStatus_t ct_atr_stream_batch_append(ctStreamBatch_t *batch, const float *highs,
                                      const float *lows, const float *closes, int n,
                                      float *outs) {
  return batch_guard("ct_atr_stream_batch_append", [&]() -> ctStatus_t {
    if (!highs || !lows || !closes) {
      return reject("ct_atr_stream_batch_append: null input pointer");
    }
    Inputs in{highs, lows, closes, nullptr};
    return run_batch_append("ct_atr_stream_batch_append", batch, Family::ATR, in, n,
                            outs, nullptr, nullptr);
  });
}

ctStatus_t ct_natr_stream_batch_append(ctStreamBatch_t *batch, const float *highs,
                                       const float *lows, const float *closes, int n,
                                       float *outs) {
  return batch_guard("ct_natr_stream_batch_append", [&]() -> ctStatus_t {
    if (!highs || !lows || !closes) {
      return reject("ct_natr_stream_batch_append: null input pointer");
    }
    Inputs in{highs, lows, closes, nullptr};
    return run_batch_append("ct_natr_stream_batch_append", batch, Family::NATR, in, n,
                            outs, nullptr, nullptr);
  });
}

ctStatus_t ct_obv_stream_batch_append(ctStreamBatch_t *batch, const float *closes,
                                      const float *volumes, int n, float *outs) {
  return batch_guard("ct_obv_stream_batch_append", [&]() -> ctStatus_t {
    if (!closes || !volumes) {
      return reject("ct_obv_stream_batch_append: null input pointer");
    }
    Inputs in{nullptr, nullptr, closes, volumes};
    return run_batch_append("ct_obv_stream_batch_append", batch, Family::OBV, in, n,
                            outs, nullptr, nullptr);
  });
}

ctStatus_t ct_ad_stream_batch_append(ctStreamBatch_t *batch, const float *highs,
                                     const float *lows, const float *closes,
                                     const float *volumes, int n, float *outs) {
  return batch_guard("ct_ad_stream_batch_append", [&]() -> ctStatus_t {
    if (!highs || !lows || !closes || !volumes) {
      return reject("ct_ad_stream_batch_append: null input pointer");
    }
    Inputs in{highs, lows, closes, volumes};
    return run_batch_append("ct_ad_stream_batch_append", batch, Family::AD, in, n,
                            outs, nullptr, nullptr);
  });
}

ctStatus_t ct_macd_stream_batch_append(ctStreamBatch_t *batch, const float *closes,
                                       int n, float *outMacd, float *outSignal,
                                       float *outHist) {
  return batch_guard("ct_macd_stream_batch_append", [&]() -> ctStatus_t {
    if (!closes) return reject("ct_macd_stream_batch_append: null input pointer");
    Inputs in{nullptr, nullptr, closes, nullptr};
    return run_batch_append("ct_macd_stream_batch_append", batch, Family::MACD, in, n,
                            outMacd, outSignal, outHist);
  });
}

ctStatus_t ct_bbands_stream_batch_append(ctStreamBatch_t *batch, const float *closes,
                                         int n, float *outUpper, float *outMiddle,
                                         float *outLower) {
  return batch_guard("ct_bbands_stream_batch_append", [&]() -> ctStatus_t {
    if (!closes) return reject("ct_bbands_stream_batch_append: null input pointer");
    Inputs in{nullptr, nullptr, closes, nullptr};
    return run_batch_append("ct_bbands_stream_batch_append", batch, Family::BBANDS, in,
                            n, outUpper, outMiddle, outLower);
  });
}

}  // extern "C"
