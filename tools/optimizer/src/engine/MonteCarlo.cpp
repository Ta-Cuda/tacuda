/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "engine/MonteCarlo.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace topt {

namespace {

// Max peak-to-valley decline of the equity path capital + cumsum(pnls).
double path_max_drawdown(const std::vector<double>& pnls, double capital) {
  double equity = capital;
  double peak = capital;
  double maxDd = 0.0;
  for (double pnl : pnls) {
    equity += pnl;
    peak = std::max(peak, equity);
    maxDd = std::max(maxDd, peak - equity);
  }
  return maxDd;
}

// Nearest-rank percentile of an ascending-sorted sample, q in (0, 1).
double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  double rank = std::ceil(q * static_cast<double>(sorted.size()));
  auto idx = static_cast<std::size_t>(std::max(1.0, rank)) - 1;
  return sorted[std::min(idx, sorted.size() - 1)];
}

McPercentiles summarize(std::vector<double>& sample, bool worstIsMax) {
  std::sort(sample.begin(), sample.end());
  McPercentiles p;
  p.p5 = percentile(sample, 0.05);
  p.p25 = percentile(sample, 0.25);
  p.p50 = percentile(sample, 0.50);
  p.p75 = percentile(sample, 0.75);
  p.p95 = percentile(sample, 0.95);
  p.worst = worstIsMax ? sample.back() : sample.front();
  return p;
}

} // namespace

const char* to_string(McMode m) {
  return m == McMode::Permute ? "permute" : "resample";
}

McMode parse_mc_mode(const std::string& text) {
  if (text == "permute") return McMode::Permute;
  if (text == "resample") return McMode::Resample;
  throw std::invalid_argument("unknown Monte-Carlo mode '" + text +
                              "' (use permute|resample)");
}

McResult run_monte_carlo(const std::vector<Trade>& trades, double capital,
                         const McConfig& cfg) {
  if (cfg.iterations <= 0) {
    throw std::invalid_argument("monte-carlo: iterations must be positive");
  }

  McResult res;
  res.iterations = cfg.iterations;
  res.mode = cfg.mode;
  res.numTrades = static_cast<int>(trades.size());
  if (trades.empty()) return res;

  std::vector<double> pnls;
  pnls.reserve(trades.size());
  for (const Trade& t : trades) pnls.push_back(t.pnl);

  double totalPnl = 0.0;
  for (double p : pnls) totalPnl += p;
  res.originalNetProfit = totalPnl;
  res.originalMaxDrawdown = path_max_drawdown(pnls, capital);

  std::mt19937_64 rng(cfg.seed);
  std::vector<double> nps, dds;
  nps.reserve(static_cast<std::size_t>(cfg.iterations));
  dds.reserve(static_cast<std::size_t>(cfg.iterations));

  std::vector<double> sample = pnls;
  int netLosses = 0;
  int ddAbove = 0;
  for (int it = 0; it < cfg.iterations; ++it) {
    double np;
    if (cfg.mode == McMode::Permute) {
      std::shuffle(sample.begin(), sample.end(), rng);
      np = totalPnl;
    } else {
      std::uniform_int_distribution<std::size_t> pick(0, pnls.size() - 1);
      np = 0.0;
      for (std::size_t k = 0; k < pnls.size(); ++k) {
        sample[k] = pnls[pick(rng)];
        np += sample[k];
      }
    }
    double dd = path_max_drawdown(sample, capital);
    nps.push_back(np);
    dds.push_back(dd);
    if (np < 0.0) ++netLosses;
    if (dd > res.originalMaxDrawdown) ++ddAbove;
  }

  res.netProfit = summarize(nps, /*worstIsMax=*/false);
  res.maxDrawdown = summarize(dds, /*worstIsMax=*/true);
  res.probNetLoss = static_cast<double>(netLosses) / cfg.iterations;
  res.probDdAboveOriginal = static_cast<double>(ddAbove) / cfg.iterations;
  return res;
}

} // namespace topt
