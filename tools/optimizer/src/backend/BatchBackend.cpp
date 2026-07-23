/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "backend/BatchBackend.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "backend/TacudaBackend.h"
#include "backend/TacudaBatchBackend.h"
#include "backend/TalibBackend.h"

namespace topt {

PerSymbolBatchBackend::PerSymbolBatchBackend(
    std::vector<std::unique_ptr<IIndicatorBackend>> backends, std::string name,
    int threads)
    : backends_(std::move(backends)), name_(std::move(name)) {
  if (backends_.empty()) {
    throw std::invalid_argument("per-symbol batch backend: no backends");
  }
  for (const auto& b : backends_) {
    if (!b) {
      throw std::invalid_argument("per-symbol batch backend: null backend");
    }
  }
  if (threads <= 0) {
    threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) threads = 1;
  }
  threads_ = std::min<int>(threads, static_cast<int>(backends_.size()));
}

void PerSymbolBatchBackend::compute(
    const IndicatorRequest& req, const std::vector<std::vector<float*>>& outs) {
  if (outs.size() != backends_.size()) {
    throw std::invalid_argument("per-symbol batch backend: wrong symbol count");
  }
  if (threads_ <= 1) {
    for (std::size_t s = 0; s < backends_.size(); ++s) {
      backends_[s]->compute(req, outs[s]);
    }
    return;
  }

  // Work-stealing over symbols; each worker only touches its own backends.
  std::atomic<std::size_t> next{0};
  std::exception_ptr firstError;
  std::mutex errorMutex;
  auto worker = [&]() {
    for (;;) {
      const std::size_t s = next.fetch_add(1);
      if (s >= backends_.size()) return;
      try {
        backends_[s]->compute(req, outs[s]);
      } catch (...) {
        std::lock_guard<std::mutex> lock(errorMutex);
        if (!firstError) firstError = std::current_exception();
        return;
      }
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads_));
  for (int t = 0; t < threads_; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  if (firstError) std::rethrow_exception(firstError);
}

std::unique_ptr<IBatchIndicatorBackend> create_batch_backend(
    BackendKind kind, const std::vector<Series>& series, std::string& log,
    int cpuThreads) {
  if (series.empty()) {
    throw std::invalid_argument("batch backend: no series");
  }

  auto make_talib = [&series, cpuThreads]() {
    std::vector<std::unique_ptr<IIndicatorBackend>> perSymbol;
    perSymbol.reserve(series.size());
    for (const Series& s : series) {
      perSymbol.push_back(std::make_unique<TalibBackend>(s));
    }
    return std::make_unique<PerSymbolBatchBackend>(
        std::move(perSymbol), "talib-per-symbol", cpuThreads);
  };

  switch (kind) {
  case BackendKind::Tacuda:
    log = "backend: tacuda batch API (GPU, forced)";
    return std::make_unique<TacudaBatchBackend>(series);
  case BackendKind::TacudaDevice: {
    // Benchmark baseline: the same GPU indicators, but one single-series
    // ct_*_d call (and its own column uploads) per symbol per request.
    std::vector<std::unique_ptr<IIndicatorBackend>> perSymbol;
    perSymbol.reserve(series.size());
    for (const Series& s : series) {
      perSymbol.push_back(std::make_unique<TacudaBackend>(s));
    }
    log = "backend: tacuda device API per symbol (GPU, forced)";
    return std::make_unique<PerSymbolBatchBackend>(std::move(perSymbol),
                                                   "tacuda-device-per-symbol");
  }
  case BackendKind::Talib: {
    auto backend = make_talib();
    log = "backend: talib per symbol (CPU, forced, " +
          std::to_string(backend->threads()) + " threads)";
    return backend;
  }
  case BackendKind::Auto:
    break;
  }

  if (TacudaBackend::available()) {
    log = "backend: tacuda batch API (GPU)";
    return std::make_unique<TacudaBatchBackend>(series);
  }
  std::string why;
  if (TalibBackend::available(why)) {
    auto backend = make_talib();
    log = "backend: talib per symbol (CPU fallback, no CUDA device, " +
          std::to_string(backend->threads()) + " threads)";
    return backend;
  }
  throw std::runtime_error(
      "no usable indicator backend: no CUDA device for tacuda and " + why);
}

} // namespace topt
