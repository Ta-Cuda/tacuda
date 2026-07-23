/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_REPORT_REPORT_H
#define TACUDA_OPT_REPORT_REPORT_H

#include <cstdio>
#include <string>
#include <vector>

#include "engine/MonteCarlo.h"
#include "optimize/GridSearch.h"
#include "optimize/WalkForward.h"
#include "portfolio/Portfolio.h"

namespace topt {

// Detailed console report for a single backtest.
void print_backtest_report(std::FILE* out, const std::string& strategyName,
                           const VariantResult& result, const SimResult& sim,
                           const ExecConfig& exec);

// Top-N console table for a set of evaluated variants, sorted by fitness
// (descending). `title` names the run kind ("Grid results", "Genetic
// results"). Returns indices of the printed rows into `results`.
std::vector<std::size_t> print_grid_report(std::FILE* out,
                                           const std::vector<VariantResult>& results,
                                           const std::string& fitnessText, int topN,
                                           const char* title = "Grid results");

// CSV with one row per variant (params, metrics, fitness, status).
void write_grid_csv(const std::string& path, const std::vector<VariantResult>& results);

// JSON summary: run config echo, best variant, top-N array.
void write_grid_json(const std::string& path, const std::string& strategyName,
                     const std::string& fitnessText,
                     const std::vector<VariantResult>& results, int topN);

// JSON report for one backtest, optionally with the trade list and a
// Monte-Carlo section (mc may be nullptr).
void write_backtest_json(const std::string& path, const std::string& strategyName,
                         const VariantResult& result, const SimResult& sim,
                         bool includeTrades, const McResult* mc = nullptr);

// Console block with Monte-Carlo percentiles (after a backtest/portfolio
// report).
void print_mc_report(std::FILE* out, const McResult& mc);

// Console report for a portfolio run: per-symbol table plus aggregated
// metrics (and the fitness line when fitnessText is non-empty).
void print_portfolio_report(std::FILE* out, const std::string& strategyName,
                            const ParamSet& params, const PortfolioResult& res,
                            const ExecConfig& exec,
                            const std::string& fitnessText);

// JSON report for a portfolio run: symbols array, aggregated metrics,
// optional per-symbol trade lists and Monte-Carlo section.
void write_portfolio_json(const std::string& path, const std::string& strategyName,
                          const ParamSet& params, const PortfolioResult& res,
                          bool includeTrades, const McResult* mc);

// Console report for a walk-forward run: per-window table (ranges, chosen
// params, IS/OOS fitness) plus stitched OOS summary and WFE.
void print_wfo_report(std::FILE* out, const std::string& strategyName,
                      const WfoResult& res, const WfoConfig& cfg,
                      const std::string& fitnessText);

// JSON report for a walk-forward run: windows array + overall OOS metrics.
void write_wfo_json(const std::string& path, const std::string& strategyName,
                    const std::string& fitnessText, const WfoResult& res,
                    const WfoConfig& cfg);

} // namespace topt

#endif // TACUDA_OPT_REPORT_REPORT_H
