/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "optimize/GridSearch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

namespace topt {

VariantResult run_variant(const Series& bars, const IStrategy& strategy,
                          const ParamSet& ps, IndicatorCache& cache,
                          const ExecConfig& exec, const FitnessExpr* fitness,
                          SimResult* simOut, SimRange range, int minTrades) {
  VariantResult res;
  res.params = ps;
  if (!strategy.validate(ps)) {
    res.error = "invalid parameter combination";
    return res;
  }

  try {
    std::vector<IndicatorRequest> requests = strategy.indicators(ps);
    const int n = bars.size();
    const int rangeEnd = range.end < 0 ? n : range.end;

    std::vector<std::vector<const float*>> lines;
    lines.reserve(requests.size());
    int warmup = 0;
    for (const auto& req : requests) {
      const IndicatorCache::Entry& entry = cache.get(req);
      std::vector<const float*> ptrs;
      ptrs.reserve(entry.lines.size());
      for (const auto& line : entry.lines) {
        ptrs.push_back(line.data());
      }
      lines.push_back(std::move(ptrs));
      for (int fv : entry.firstValid) {
        warmup = std::max(warmup, fv);
      }
    }
    // A trade needs a decision bar >= warmup at or before rangeEnd-2.
    if (warmup >= rangeEnd - 1) {
      res.error = "insufficient data for indicator warm-up";
      return res;
    }

    IndicatorView view(std::move(lines));
    std::vector<Signal> sig(static_cast<std::size_t>(n), 0);
    strategy.signals(bars, ps, view, sig.data());
    // No trading until every requested indicator line has left its warm-up.
    std::fill(sig.begin(), sig.begin() + warmup, static_cast<Signal>(0));

    SimResult sim = simulate(bars, sig.data(), exec, range);
    res.metrics = compute_metrics(sim, exec.barsPerYear);
    if (minTrades > 0 && res.metrics.numTrades < minTrades) {
      res.error = "fewer than " + std::to_string(minTrades) + " trades (" +
                  std::to_string(res.metrics.numTrades) + ")";
      return res;
    }
    if (fitness) {
      res.fitness = fitness->eval(fitness_variables(res.metrics));
      if (!std::isfinite(res.fitness)) {
        throw std::runtime_error("fitness expression produced a non-finite value");
      }
    }
    res.valid = true;
    if (simOut) *simOut = std::move(sim);
  } catch (const std::exception& e) {
    res.valid = false;
    res.error = e.what();
  }
  return res;
}

std::vector<VariantResult> run_grid(const Series& bars, const IStrategy& strategy,
                                    const std::vector<ParamSet>& variants,
                                    IndicatorCache& cache, const GridConfig& cfg,
                                    const FitnessExpr& fitness) {
  const std::size_t total = variants.size();
  std::vector<VariantResult> results(total);

  int threads = cfg.threads > 0
                    ? cfg.threads
                    : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  auto startTime = std::chrono::steady_clock::now();

  std::size_t done = 0;
  std::size_t chunkIndex = 0;
  std::size_t i = 0;
  while (i < total) {
    // 1) Assemble a chunk whose unique indicator lines fit the cache budget.
    std::unordered_set<std::string> chunkKeys;
    std::size_t chunkBytes = 0;
    std::size_t chunkStart = i;
    while (i < total) {
      const ParamSet& ps = variants[i];
      std::size_t newBytes = 0;
      std::vector<std::string> newKeys;
      if (strategy.validate(ps)) {
        for (const auto& req : strategy.indicators(ps)) {
          std::string key = IndicatorCache::key_of(req);
          if (chunkKeys.count(key) == 0) {
            newBytes += cache.bytes_for(req);
            newKeys.push_back(std::move(key));
          }
        }
      }
      if (i > chunkStart && chunkBytes + newBytes > cfg.cacheBytes) {
        break;
      }
      for (auto& k : newKeys) chunkKeys.insert(std::move(k));
      chunkBytes += newBytes;
      ++i;
    }
    const std::size_t chunkEnd = i;
    ++chunkIndex;

    // 2) Compute unique indicator requests once (serial: one GPU stream).
    for (std::size_t k = chunkStart; k < chunkEnd; ++k) {
      const ParamSet& ps = variants[k];
      if (!strategy.validate(ps)) continue;
      for (const auto& req : strategy.indicators(ps)) {
        cache.get(req);
      }
    }

    // 3) Simulate the chunk's variants in parallel on CPU threads.
    std::atomic<std::size_t> next(chunkStart);
    auto worker = [&]() {
      for (;;) {
        std::size_t k = next.fetch_add(1);
        if (k >= chunkEnd) return;
        results[k] = run_variant(bars, strategy, variants[k], cache,
                                 cfg.exec, &fitness, nullptr, cfg.range,
                                 cfg.minTrades);
      }
    };
    std::vector<std::thread> pool;
    int poolSize = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(threads), chunkEnd - chunkStart));
    pool.reserve(static_cast<std::size_t>(poolSize));
    for (int t = 0; t < poolSize; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    done = chunkEnd;
    if (!cfg.quiet) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - startTime)
                         .count();
      std::fprintf(stderr,
                   "[grid] chunk %zu: variants %zu/%zu, %zu indicator computes, "
                   "cache %.1f MB, %.1f s elapsed\n",
                   chunkIndex, done, total, cache.compute_count(),
                   static_cast<double>(cache.bytes()) / (1024.0 * 1024.0),
                   static_cast<double>(elapsed) / 1000.0);
    }

    if (chunkEnd < total) {
      cache.clear(); // stay inside the memory budget for the next chunk
    }
  }
  return results;
}

void compute_robust_fitness(std::vector<VariantResult>& results,
                            const std::vector<std::size_t>& axisSizes) {
  std::size_t total = 1;
  for (std::size_t s : axisSizes) total *= s;
  if (total != results.size()) {
    throw std::invalid_argument(
        "robust fitness: " + std::to_string(results.size()) +
        " results do not form the full grid of " + std::to_string(total) +
        " variants");
  }
  const std::size_t dims = axisSizes.size();
  // Mixed-radix strides of enumerate_variants(): last axis varies fastest.
  std::vector<std::size_t> stride(dims, 1);
  for (std::size_t i = dims; i-- > 1;) stride[i - 1] = stride[i] * axisSizes[i];

  for (std::size_t k = 0; k < results.size(); ++k) {
    VariantResult& v = results[k];
    v.robustFitness = 0.0;
    v.neighborhood = 0;
    if (!v.valid) continue;
    double sum = v.fitness;
    int count = 1;
    for (std::size_t i = 0; i < dims; ++i) {
      if (axisSizes[i] < 2) continue;
      const std::size_t idx = (k / stride[i]) % axisSizes[i];
      if (idx > 0) {
        const VariantResult& nb = results[k - stride[i]];
        if (nb.valid) { sum += nb.fitness; ++count; }
      }
      if (idx + 1 < axisSizes[i]) {
        const VariantResult& nb = results[k + stride[i]];
        if (nb.valid) { sum += nb.fitness; ++count; }
      }
    }
    v.robustFitness = sum / count;
    v.neighborhood = count;
  }
}

} // namespace topt
