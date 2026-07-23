/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "optimize/WalkForward.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace topt {

namespace {

// Best valid variant of a finished IS optimization; -1 when none is valid.
// Ties keep the earlier variant (stable, deterministic).
std::ptrdiff_t best_valid(const std::vector<VariantResult>& results) {
  std::ptrdiff_t best = -1;
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (!results[i].valid) continue;
    if (best < 0 || results[i].fitness > results[static_cast<std::size_t>(best)].fitness) {
      best = static_cast<std::ptrdiff_t>(i);
    }
  }
  return best;
}

} // namespace

std::vector<WfoWindow> make_wfo_windows(int n, int windows, double isFraction) {
  if (windows < 1) throw std::invalid_argument("wfo: windows must be >= 1");
  if (!(isFraction > 0.0) || !(isFraction < 1.0)) {
    throw std::invalid_argument("wfo: in-sample fraction must be in (0, 1)");
  }
  if (n < 2) throw std::invalid_argument("wfo: series too short");

  const double ratio = isFraction / (1.0 - isFraction); // L_is / L_oos
  const int oosLen = static_cast<int>(std::floor(n / (windows + ratio)));
  if (oosLen < 1) {
    throw std::invalid_argument("wfo: series too short for " + std::to_string(windows) +
                                " windows at is=" + std::to_string(isFraction));
  }
  const int isLen = static_cast<int>(std::floor(oosLen * ratio));
  if (isLen < 1) {
    throw std::invalid_argument("wfo: in-sample segment would be empty; "
                                "reduce windows or increase --is");
  }

  std::vector<WfoWindow> out;
  out.reserve(static_cast<std::size_t>(windows));
  for (int w = 0; w < windows; ++w) {
    WfoWindow win;
    win.isBegin = w * oosLen;
    win.isEnd = win.isBegin + isLen;
    win.oosEnd = win.isEnd + oosLen;
    out.push_back(win);
  }
  return out;
}

double compute_wfe(double isNetProfit, double isBars, double oosNetProfit,
                   double oosBars) {
  if (!(isBars > 0.0) || !(oosBars > 0.0) || !(isNetProfit > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return (oosNetProfit / oosBars) / (isNetProfit / isBars);
}

WfoResult run_wfo(const Series& bars, const IStrategy& strategy,
                  const std::vector<GridAxis>& axes,
                  const std::vector<std::pair<std::string, double>>& sets,
                  IndicatorCache& cache, const WfoConfig& cfg,
                  const FitnessExpr& fitness) {
  const int n = bars.size();
  const std::vector<WfoWindow> windows = make_wfo_windows(n, cfg.windows, cfg.isFraction);

  // Grid variants are the same for every window; enumerate once.
  std::vector<ParamSet> variants;
  if (cfg.optimizer == WfoOptimizer::Grid) {
    variants = enumerate_variants(strategy.params(), axes, sets);
    if (variants.empty()) throw std::invalid_argument("wfo: empty variant grid");
  }

  WfoResult res;
  res.windows.reserve(windows.size());
  res.tailUnused = n - windows.back().oosEnd;

  const double capital = cfg.grid.exec.capital;
  double stitchedFinal = capital;
  std::vector<Trade> allTrades;
  int totalBarsInPosition = 0;

  double sumIsNp = 0.0, sumIsBars = 0.0, sumOosNp = 0.0, sumOosBars = 0.0;

  for (std::size_t w = 0; w < windows.size(); ++w) {
    const WfoWindow& win = windows[w];
    WfoWindowResult wr;
    wr.window = win;

    // --- Optimize on the in-sample segment.
    if (cfg.optimizer == WfoOptimizer::Grid) {
      GridConfig g = cfg.grid;
      g.range = SimRange{win.isBegin, win.isEnd};
      g.quiet = true; // window-level progress is printed below
      std::vector<VariantResult> results =
          run_grid(bars, strategy, variants, cache, g, fitness);
      res.isEvaluations += results.size();
      std::ptrdiff_t best = best_valid(results);
      if (best >= 0) {
        const VariantResult& v = results[static_cast<std::size_t>(best)];
        wr.optimized = true;
        wr.params = v.params;
        wr.isFitness = v.fitness;
        wr.isMetrics = v.metrics;
      }
    } else {
      GeneticConfig ga = cfg.genetic;
      ga.eval = cfg.grid;
      ga.eval.range = SimRange{win.isBegin, win.isEnd};
      ga.quiet = true;
      GeneticResult gres = run_genetic(bars, strategy, axes, sets, cache, ga, fitness);
      res.isEvaluations += gres.evaluations;
      if (gres.has_best()) {
        const VariantResult& v = gres.evaluated[gres.bestIndex];
        wr.optimized = true;
        wr.params = v.params;
        wr.isFitness = v.fitness;
        wr.isMetrics = v.metrics;
      }
    }

    // --- Evaluate the chosen parameters out-of-sample.
    if (wr.optimized) {
      SimResult oosSim;
      VariantResult oos = run_variant(bars, strategy, wr.params, cache, cfg.grid.exec,
                                      &fitness, &oosSim, SimRange{win.isEnd, win.oosEnd});
      if (oos.valid) {
        wr.oosFitness = oos.fitness;
        wr.oosMetrics = oos.metrics;
        wr.oosTrades = std::move(oosSim.trades);
        wr.oosEquity = std::move(oosSim.equity);
        totalBarsInPosition += oosSim.barsInPosition;

        sumIsNp += wr.isMetrics.netProfit;
        sumIsBars += static_cast<double>(win.is_bars());
        sumOosNp += wr.oosMetrics.netProfit;
        sumOosBars += static_cast<double>(win.oos_bars());
      } else {
        wr.optimized = false;
        wr.note = "OOS evaluation failed: " + oos.error;
      }
    } else if (wr.note.empty()) {
      wr.note = "no valid variant on the in-sample segment";
    }

    // --- Stitch this window's OOS equity (flat when skipped).
    if (wr.optimized) {
      for (double eq : wr.oosEquity) {
        res.stitchedEquity.push_back(stitchedFinal + (eq - capital));
      }
      stitchedFinal += wr.oosEquity.back() - capital;
      for (const Trade& t : wr.oosTrades) allTrades.push_back(t);
    } else {
      res.stitchedEquity.insert(res.stitchedEquity.end(),
                                static_cast<std::size_t>(win.oos_bars()), stitchedFinal);
    }

    if (!cfg.grid.quiet) {
      std::string line = wr.optimized
                             ? wr.params.to_string() + ", IS fitness " +
                                   std::to_string(wr.isFitness) + ", OOS fitness " +
                                   std::to_string(wr.oosFitness)
                             : wr.note;
      std::fprintf(stderr, "[wfo] window %zu/%zu: %s\n", w + 1, windows.size(),
                   line.c_str());
    }

    res.windows.push_back(std::move(wr));
    if (cache.bytes() > cfg.grid.cacheBytes) cache.clear();
  }

  // --- Overall OOS metrics over the stitched curve.
  SimResult stitched;
  stitched.trades = std::move(allTrades);
  stitched.equity = res.stitchedEquity;
  stitched.initialCapital = capital;
  stitched.finalEquity = res.stitchedEquity.empty() ? capital : res.stitchedEquity.back();
  stitched.barsInPosition = totalBarsInPosition;
  res.oosOverall = compute_metrics(stitched, cfg.grid.exec.barsPerYear);

  res.wfe = compute_wfe(sumIsNp, sumIsBars, sumOosNp, sumOosBars);
  return res;
}

} // namespace topt
