/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "portfolio/Portfolio.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace topt {

namespace {

// Timestamps usable for alignment: present, non-zero, strictly increasing.
bool usable_timestamps(const Series& s) {
  if (s.ts.size() != s.close.size() || s.ts.empty()) return false;
  if (s.ts.front() == 0) return false;
  for (std::size_t i = 1; i < s.ts.size(); ++i) {
    if (s.ts[i] <= s.ts[i - 1]) return false;
  }
  return true;
}

// Keeps only bars whose timestamp is in `keep` (both sides sorted ascending).
Series filter_by_ts(const Series& s, const std::vector<std::int64_t>& keep) {
  Series out;
  out.ts.reserve(keep.size());
  out.open.reserve(keep.size());
  out.high.reserve(keep.size());
  out.low.reserve(keep.size());
  out.close.reserve(keep.size());
  out.volume.reserve(keep.size());

  std::size_t k = 0;
  for (std::size_t i = 0; i < s.ts.size() && k < keep.size(); ++i) {
    // keep is a subset of s.ts, so every kept timestamp matches exactly.
    if (s.ts[i] != keep[k]) continue;
    out.ts.push_back(s.ts[i]);
    out.open.push_back(s.open[i]);
    out.high.push_back(s.high[i]);
    out.low.push_back(s.low[i]);
    out.close.push_back(s.close[i]);
    out.volume.push_back(s.volume[i]);
    ++k;
  }
  return out;
}

int first_finite(const float* line, int n) {
  for (int i = 0; i < n; ++i) {
    if (std::isfinite(line[i])) return i;
  }
  return n;
}

} // namespace

void align_portfolio(std::vector<Series>& series, std::string& log) {
  if (series.empty()) {
    throw std::invalid_argument("portfolio: no series to align");
  }
  if (series.size() == 1) {
    log = "portfolio: single series, no alignment needed";
    return;
  }

  bool allTs = true;
  for (const Series& s : series) {
    if (!usable_timestamps(s)) {
      allTs = false;
      break;
    }
  }

  if (!allTs) {
    const int n0 = series.front().size();
    for (const Series& s : series) {
      if (s.size() != n0) {
        throw std::invalid_argument(
            "portfolio: series lengths differ (" + std::to_string(n0) + " vs " +
            std::to_string(s.size()) +
            ") and timestamps are unavailable for alignment");
      }
    }
    log = "portfolio: equal-length series, aligned by bar index";
    return;
  }

  // Intersection of all timestamp sequences (each strictly increasing).
  std::vector<std::int64_t> common = series.front().ts;
  for (std::size_t i = 1; i < series.size() && !common.empty(); ++i) {
    std::vector<std::int64_t> next;
    next.reserve(common.size());
    std::set_intersection(common.begin(), common.end(), series[i].ts.begin(),
                          series[i].ts.end(), std::back_inserter(next));
    common.swap(next);
  }
  if (common.empty()) {
    throw std::invalid_argument("portfolio: series share no common timestamps");
  }

  bool alreadyAligned = true;
  for (const Series& s : series) {
    if (s.ts.size() != common.size()) {
      alreadyAligned = false;
      break;
    }
  }
  if (alreadyAligned) {
    log = "portfolio: timestamps already aligned (" +
          std::to_string(common.size()) + " bars)";
    return;
  }

  std::size_t maxLen = 0;
  for (const Series& s : series) {
    maxLen = std::max(maxLen, s.ts.size());
  }
  for (Series& s : series) {
    s = filter_by_ts(s, common);
  }
  log = "portfolio: aligned by timestamp intersection, kept " +
        std::to_string(common.size()) + " of " + std::to_string(maxLen) +
        " bars";
}

PortfolioResult run_portfolio(const std::vector<Series>& aligned,
                              const std::vector<std::string>& names,
                              const IStrategy& strategy, const ParamSet& ps,
                              IBatchIndicatorBackend& backend,
                              const PortfolioConfig& cfg,
                              const FitnessExpr* fitness) {
  const int numSymbols = static_cast<int>(aligned.size());
  if (numSymbols == 0) {
    throw std::invalid_argument("portfolio: no symbols");
  }
  if (names.size() != aligned.size()) {
    throw std::invalid_argument("portfolio: names/series count mismatch");
  }
  if (backend.num_series() != numSymbols) {
    throw std::invalid_argument("portfolio: backend symbol count mismatch");
  }
  const int n = aligned.front().size();
  for (const Series& s : aligned) {
    if (s.size() != n) {
      throw std::invalid_argument("portfolio: series are not aligned");
    }
  }
  if (!strategy.validate(ps)) {
    throw std::invalid_argument("invalid parameter combination: " +
                                ps.to_string());
  }

  ExecConfig symExec = cfg.exec;
  symExec.capital = cfg.exec.capital / numSymbols;

  using Clock = std::chrono::steady_clock;
  const auto indicatorStart = Clock::now();

  // One batch-backend call per indicator request covers every symbol.
  const std::vector<IndicatorRequest> requests = strategy.indicators(ps);
  // store[request][symbol][line] -> n floats
  std::vector<std::vector<std::vector<std::vector<float>>>> store;
  store.reserve(requests.size());
  for (const IndicatorRequest& req : requests) {
    const int nOut = family_outputs(req.family);
    std::vector<std::vector<std::vector<float>>> symbolLines(
        static_cast<std::size_t>(numSymbols));
    std::vector<std::vector<float*>> outs(static_cast<std::size_t>(numSymbols));
    for (int s = 0; s < numSymbols; ++s) {
      auto& lines = symbolLines[static_cast<std::size_t>(s)];
      lines.assign(static_cast<std::size_t>(nOut),
                   std::vector<float>(static_cast<std::size_t>(n)));
      for (auto& line : lines) {
        outs[static_cast<std::size_t>(s)].push_back(line.data());
      }
    }
    backend.compute(req, outs);
    store.push_back(std::move(symbolLines));
  }

  PortfolioResult res;
  res.bars = n;
  res.symbols.resize(static_cast<std::size_t>(numSymbols));
  res.indicatorSec =
      std::chrono::duration<double>(Clock::now() - indicatorStart).count();
  const auto signalStart = Clock::now();

  for (int s = 0; s < numSymbols; ++s) {
    SymbolResult& sr = res.symbols[static_cast<std::size_t>(s)];
    sr.name = names[static_cast<std::size_t>(s)];

    std::vector<std::vector<const float*>> lines;
    lines.reserve(requests.size());
    int warmup = 0;
    for (std::size_t r = 0; r < requests.size(); ++r) {
      std::vector<const float*> ptrs;
      for (const auto& line : store[r][static_cast<std::size_t>(s)]) {
        ptrs.push_back(line.data());
        warmup = std::max(warmup, first_finite(line.data(), n));
      }
      lines.push_back(std::move(ptrs));
    }
    if (warmup >= n - 1) {
      sr.error = "insufficient data for indicator warm-up";
      continue;
    }

    try {
      IndicatorView view(std::move(lines));
      std::vector<Signal> sig(static_cast<std::size_t>(n), 0);
      strategy.signals(aligned[static_cast<std::size_t>(s)], ps, view,
                       sig.data());
      std::fill(sig.begin(), sig.begin() + warmup, static_cast<Signal>(0));

      sr.sim = simulate(aligned[static_cast<std::size_t>(s)], sig.data(),
                        symExec);
      sr.metrics = compute_metrics(sr.sim, symExec.barsPerYear);
      sr.valid = true;
      ++res.validSymbols;
    } catch (const std::exception& e) {
      sr.valid = false;
      sr.error = e.what();
    }
  }
  res.signalSec =
      std::chrono::duration<double>(Clock::now() - signalStart).count();

  // Aggregate: equity is the sum over symbols (skipped symbols stay flat at
  // their capital share), trades are merged chronologically.
  res.equity.assign(static_cast<std::size_t>(n), 0.0);
  double exposureSum = 0.0;
  for (const SymbolResult& sr : res.symbols) {
    if (sr.valid) {
      for (int t = 0; t < n; ++t) {
        res.equity[static_cast<std::size_t>(t)] +=
            sr.sim.equity[static_cast<std::size_t>(t)];
      }
      res.trades.insert(res.trades.end(), sr.sim.trades.begin(),
                        sr.sim.trades.end());
      exposureSum += sr.metrics.exposure;
    } else {
      for (int t = 0; t < n; ++t) {
        res.equity[static_cast<std::size_t>(t)] += symExec.capital;
      }
    }
  }
  std::stable_sort(res.trades.begin(), res.trades.end(),
                   [](const Trade& a, const Trade& b) {
                     if (a.exitBar != b.exitBar) return a.exitBar < b.exitBar;
                     return a.entryBar < b.entryBar;
                   });

  SimResult agg;
  agg.trades = res.trades;
  agg.equity = res.equity;
  agg.initialCapital = cfg.exec.capital;
  agg.finalEquity = res.equity.back();
  res.metrics = compute_metrics(agg, cfg.exec.barsPerYear);
  res.metrics.exposure = exposureSum / numSymbols;
  if (fitness) {
    res.fitness = fitness->eval(fitness_variables(res.metrics));
  }
  return res;
}

} // namespace topt
