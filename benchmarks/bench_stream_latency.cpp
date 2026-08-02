/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Per-tick latency of the streaming API vs recomputing the whole series,
// which is what a live loop pays without incremental state.
//
// Build (against an existing build tree):
//   g++ -O2 -std=c++17 -I include benchmarks/bench_stream_latency.cpp \
//       -L build -ltacuda -L "$CUDA_HOME/lib64" -lcudart -o build/bench_stream
//   LD_LIBRARY_PATH=build:"$CUDA_HOME/lib64" ./build/bench_stream

#include <tacuda.h>
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

std::vector<float> random_walk(int n) {
  std::vector<float> v(n);
  std::mt19937 rng(20260802u);
  std::normal_distribution<float> step(0.0f, 0.5f);
  float p = 100.0f;
  for (int i = 0; i < n; ++i) {
    p += step(rng);
    if (p < 1.0f) p = 1.0f;
    v[i] = p;
  }
  return v;
}

// Median is the honest summary here: a per-tick cost distribution has a long
// right tail (scheduler, driver), and p50/p99 are what a live desk cares about.
void summarize(const char *label, std::vector<double> &samples) {
  std::sort(samples.begin(), samples.end());
  const double p50 = samples[samples.size() / 2];
  const double p99 = samples[static_cast<std::size_t>(samples.size() * 0.99)];
  std::printf("  %-38s p50 %10.2f us   p99 %10.2f us\n", label, p50, p99);
}

void run(int bars, int period, int ticks) {
  const std::vector<float> close = random_walk(bars + ticks);
  std::vector<float> out(bars + ticks);

  std::printf("history=%d bars, RSI(%d), %d simulated ticks\n", bars, period, ticks);

  // 1. Naive live loop: recompute the whole window on every closed bar. The
  //    window slides at a fixed length — growing it by one bar per tick would
  //    also make the device buffer pool cache a new size every tick, which
  //    inflates GPU memory instead of measuring latency.
  double lastFull = 0.0;
  {
    std::vector<double> samples;
    samples.reserve(ticks);
    for (int t = 0; t < ticks; ++t) {
      const auto t0 = Clock::now();
      // Window ends on the bar that just closed, so the final iteration lands
      // on the same absolute bar the stream finishes on (cross-checked below).
      const ctStatus_t rc = ct_rsi(close.data() + t + 1, out.data(), bars, period);
      samples.push_back(micros(Clock::now() - t0));
      if (rc != CT_STATUS_SUCCESS) {
        std::printf("  ct_rsi failed: %s\n", ct_get_last_error());
        return;
      }
      lastFull = out[bars - 1];
    }
    summarize("full recompute per tick (ct_rsi)", samples);
  }

  // 2. Streaming: replay the history once, then append one bar per tick.
  {
    ctStream_t *s = nullptr;
    if (ct_rsi_stream_create(period, &s) != CT_STATUS_SUCCESS) {
      std::printf("  stream create failed: %s\n", ct_get_last_error());
      return;
    }

    const auto warm0 = Clock::now();
    ct_rsi_stream_append(s, close.data(), bars, out.data());
    const double warmUs = micros(Clock::now() - warm0);
    std::printf("  %-38s %10.2f us  (%.1f M bars/s)\n", "history replay (one call)",
                warmUs, bars / warmUs);

    std::vector<double> samples;
    samples.reserve(ticks);
    for (int t = 0; t < ticks; ++t) {
      const auto t0 = Clock::now();
      ct_rsi_stream_append(s, close.data() + bars + t, 1, out.data() + bars + t);
      samples.push_back(micros(Clock::now() - t0));
    }
    summarize("stream append per tick", samples);
    // Both paths must have produced the same last value — otherwise one of
    // them silently did no work and the timings above are meaningless.
    const double lastStream = out[bars + ticks - 1];
    std::printf("  %-38s full=%.6f stream=%.6f\n", "last value cross-check",
                lastFull, lastStream);
    ct_stream_destroy(s);
  }
  std::printf("\n");
}

}  // namespace

int main() {
  std::printf("tacuda %s — streaming latency\n\n", ct_version());
  run(100000, 14, 2000);
  run(1000000, 14, 500);
  return 0;
}
