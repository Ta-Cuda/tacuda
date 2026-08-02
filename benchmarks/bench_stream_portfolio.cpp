/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Per-tick latency of a whole-portfolio indicator pipeline over batch streams,
// and the crossover between the CPU and GPU execution policies.
//
// The pipeline is the M2 acceptance workload: 7 indicators (SMA, EMA, RSI,
// ATR, MACD, BBANDS, OBV) over N symbols, one closed bar appended per tick.
//
// Build (against an existing build tree):
//   g++ -O2 -std=c++17 -I include -I "$CUDA_HOME/include" \
//       benchmarks/bench_stream_portfolio.cpp -L build -ltacuda \
//       -L "$CUDA_HOME/lib64" -lcudart -o build/bench_stream_portfolio
//   LD_LIBRARY_PATH=build:"$CUDA_HOME/lib64" ./build/bench_stream_portfolio

#include <tacuda_stream.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double micros(Clock::duration d) {
  return std::chrono::duration<double, std::micro>(d).count();
}

// One indicator pipeline over the whole portfolio.
struct Pipeline {
  ctStreamBatch_t *sma = nullptr;
  ctStreamBatch_t *ema = nullptr;
  ctStreamBatch_t *rsi = nullptr;
  ctStreamBatch_t *atr = nullptr;
  ctStreamBatch_t *macd = nullptr;
  ctStreamBatch_t *bbands = nullptr;
  ctStreamBatch_t *obv = nullptr;

  bool create(int series, ctStreamExec_t exec) {
    const bool ok =
        ct_sma_stream_batch_create(20, series, &sma) == CT_STATUS_SUCCESS &&
        ct_ema_stream_batch_create(21, series, &ema) == CT_STATUS_SUCCESS &&
        ct_rsi_stream_batch_create(14, series, &rsi) == CT_STATUS_SUCCESS &&
        ct_atr_stream_batch_create(14, series, &atr) == CT_STATUS_SUCCESS &&
        ct_macd_stream_batch_create(12, 26, 9, series, &macd) == CT_STATUS_SUCCESS &&
        ct_bbands_stream_batch_create(20, 2.f, 2.f, series, &bbands) == CT_STATUS_SUCCESS &&
        ct_obv_stream_batch_create(series, &obv) == CT_STATUS_SUCCESS;
    if (!ok) return false;
    for (ctStreamBatch_t *h : {sma, ema, rsi, atr, macd, bbands, obv}) {
      if (ct_stream_batch_set_exec(h, exec) != CT_STATUS_SUCCESS) return false;
    }
    return true;
  }

  void destroy() {
    for (ctStreamBatch_t *h : {sma, ema, rsi, atr, macd, bbands, obv}) {
      ct_stream_batch_destroy(h);
    }
  }
};

struct Buffers {
  std::vector<float> high, low, close, volume;
  std::vector<float> o0, o1, o2;

  explicit Buffers(int series)
      : high(series), low(series), close(series), volume(series),
        o0(series), o1(series), o2(series) {}
};

bool tick(Pipeline &p, Buffers &b) {
  return ct_sma_stream_batch_append(p.sma, b.close.data(), 1, b.o0.data()) == CT_STATUS_SUCCESS &&
         ct_ema_stream_batch_append(p.ema, b.close.data(), 1, b.o0.data()) == CT_STATUS_SUCCESS &&
         ct_rsi_stream_batch_append(p.rsi, b.close.data(), 1, b.o0.data()) == CT_STATUS_SUCCESS &&
         ct_atr_stream_batch_append(p.atr, b.high.data(), b.low.data(), b.close.data(), 1,
                                    b.o0.data()) == CT_STATUS_SUCCESS &&
         ct_macd_stream_batch_append(p.macd, b.close.data(), 1, b.o0.data(), b.o1.data(),
                                     b.o2.data()) == CT_STATUS_SUCCESS &&
         ct_bbands_stream_batch_append(p.bbands, b.close.data(), 1, b.o0.data(),
                                       b.o1.data(), b.o2.data()) == CT_STATUS_SUCCESS &&
         ct_obv_stream_batch_append(p.obv, b.close.data(), b.volume.data(), 1,
                                    b.o0.data()) == CT_STATUS_SUCCESS;
}

void fill(Buffers &b, std::vector<float> &price, std::mt19937 &rng) {
  std::normal_distribution<float> step(0.0f, 0.5f);
  for (std::size_t i = 0; i < price.size(); ++i) {
    price[i] += step(rng);
    if (price[i] < 1.0f) price[i] = 1.0f;
    b.close[i] = price[i];
    b.high[i] = price[i] + 0.4f;
    b.low[i] = price[i] - 0.4f;
    b.volume[i] = 1500.0f;
  }
}

// Prices are generated outside the timed region: this measures the library,
// not the random number generator.
bool run(int series, ctStreamExec_t exec, const char *label, int ticks) {
  Pipeline p;
  if (!p.create(series, exec)) {
    std::printf("  %-10s create failed: %s\n", label, ct_get_last_error());
    p.destroy();
    return false;
  }
  Buffers b(series);
  std::vector<float> price(static_cast<std::size_t>(series), 100.0f);
  std::mt19937 rng(11u);

  for (int t = 0; t < 60; ++t) {  // past every lookback in the pipeline
    fill(b, price, rng);
    if (!tick(p, b)) {
      std::printf("  %-10s warm-up failed: %s\n", label, ct_get_last_error());
      p.destroy();
      return false;
    }
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(ticks));
  for (int t = 0; t < ticks; ++t) {
    fill(b, price, rng);
    const auto t0 = Clock::now();
    const bool ok = tick(p, b);
    samples.push_back(micros(Clock::now() - t0));
    if (!ok) {
      std::printf("  %-10s append failed: %s\n", label, ct_get_last_error());
      p.destroy();
      return false;
    }
  }
  std::sort(samples.begin(), samples.end());
  const double p50 = samples[samples.size() / 2];
  const double p99 = samples[static_cast<std::size_t>(samples.size() * 0.99)];
  std::printf("  %-10s p50 %9.1f us   p99 %9.1f us   %s\n", label, p50, p99,
              p99 < 1000.0 ? "" : "(over 1 ms)");
  p.destroy();
  return true;
}

}  // namespace

int main() {
  std::printf("portfolio tick: 7 indicators x N symbols, 1 bar appended per tick\n");
  std::printf("target: p99 < 1 ms at 256 symbols\n\n");

  for (int series : {64, 256, 1024, 2048, 4096, 16384, 65536}) {
    const int ticks = series >= 16384 ? 300 : 2000;
    std::printf("%d symbols (%d state updates per tick)\n", series, series * 7);
    run(series, CT_STREAM_EXEC_CPU, "cpu", ticks);
    run(series, CT_STREAM_EXEC_GPU, "gpu", ticks);
    run(series, CT_STREAM_EXEC_AUTO, "auto", ticks);
    std::printf("\n");
  }
  return 0;
}
