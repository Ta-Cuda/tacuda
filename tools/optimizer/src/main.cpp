/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// tacuda-opt: console trading-strategy optimizer on top of TaCuda.
// Indicators run on the GPU through the TaCuda device API; when no CUDA
// device is available the tool falls back to a dynamically loaded TA-Lib.
// See docs/STRATEGY_OPTIMIZER.md for the user guide.

#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include "backend/Backend.h"
#include "backend/BatchBackend.h"
#include "cli/Args.h"
#include "core/Fitness.h"
#include "data/CsvLoader.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"
#include "engine/Metrics.h"
#include "engine/MonteCarlo.h"
#include "optimize/Genetic.h"
#include "optimize/GridSearch.h"
#include "optimize/WalkForward.h"
#include "portfolio/Portfolio.h"
#include "report/Report.h"
#include "strategy/IStrategy.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start)
             .count() /
         1000.0;
}

topt::Series load_data(const topt::CliOptions& opt) {
  if (!opt.dataPaths.empty()) {
    return topt::load_csv(opt.dataPaths.front());
  }
  return topt::generate_synthetic(opt.syntheticBars, opt.seed);
}

// Fixed parameters: strategy defaults overridden by --set.
topt::ParamSet fixed_params(const topt::IStrategy& strategy,
                            const topt::CliOptions& opt) {
  topt::ParamSet ps;
  for (const auto& def : strategy.params()) {
    ps.values.emplace_back(def.name, def.defval);
  }
  for (const auto& kv : opt.sets) {
    if (!ps.has(kv.first)) {
      throw std::invalid_argument("unknown parameter '" + kv.first +
                                  "' for strategy " + opt.strategy);
    }
    ps.set(kv.first, kv.second);
  }
  return ps;
}

topt::McConfig make_mc_config(const topt::CliOptions& opt) {
  topt::McConfig mc;
  mc.iterations = opt.mcIterations;
  mc.mode = topt::parse_mc_mode(opt.mcMode);
  mc.seed = opt.mcSeed;
  return mc;
}

// "data/ES_5min.csv" -> "ES_5min".
std::string symbol_name_from_path(const std::string& path) {
  std::size_t slash = path.find_last_of("/\\");
  std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
  std::size_t dot = base.find_last_of('.');
  if (dot != std::string::npos && dot > 0) base = base.substr(0, dot);
  return base.empty() ? path : base;
}

int cmd_list() {
  std::printf("built-in strategies:\n");
  for (const auto& name : topt::strategy_names()) {
    auto strategy = topt::make_strategy(name);
    std::printf("\n  %s\n", name.c_str());
    for (const auto& def : strategy->params()) {
      std::printf("    %-10s default=%-8g range %g..%g step %g%s\n",
                  def.name.c_str(), def.defval, def.minv, def.maxv, def.step,
                  def.integer ? " (int)" : "");
    }
  }
  return 0;
}

int cmd_backtest(const topt::CliOptions& opt) {
  auto start = Clock::now();
  topt::Series bars = load_data(opt);
  auto strategy = topt::make_strategy(opt.strategy);

  std::string backendLog;
  auto backend = topt::create_backend(opt.backend, bars, backendLog);
  if (!opt.quiet) {
    std::fprintf(stderr, "%s; %d bars loaded in %.2f s\n", backendLog.c_str(),
                 bars.size(), seconds_since(start));
  }

  topt::ParamSet ps = fixed_params(*strategy, opt);
  topt::FitnessExpr fitness =
      topt::FitnessExpr::parse(opt.fitness, topt::fitness_variable_names());

  topt::IndicatorCache cache(bars, *backend);
  topt::SimResult sim;
  topt::VariantResult result = topt::run_variant(bars, *strategy, ps, cache,
                                                 opt.exec, &fitness, &sim);

  topt::print_backtest_report(stdout, opt.strategy, result, sim, opt.exec);
  if (result.valid) {
    std::printf("fitness (%s) = %.6g\n", opt.fitness.c_str(), result.fitness);
  }

  topt::McResult mc;
  bool hasMc = false;
  if (opt.mcIterations > 0 && result.valid) {
    mc = topt::run_monte_carlo(sim.trades, opt.exec.capital, make_mc_config(opt));
    topt::print_mc_report(stdout, mc);
    hasMc = true;
  }

  if (!opt.outJson.empty()) {
    topt::write_backtest_json(opt.outJson, opt.strategy, result, sim,
                              opt.jsonTrades, hasMc ? &mc : nullptr);
    if (!opt.quiet) std::fprintf(stderr, "wrote %s\n", opt.outJson.c_str());
  }
  if (!opt.quiet) {
    std::fprintf(stderr, "total %.2f s\n", seconds_since(start));
  }
  return result.valid ? 0 : 2;
}

int cmd_portfolio(const topt::CliOptions& opt) {
  auto start = Clock::now();

  std::vector<topt::Series> series;
  std::vector<std::string> names;
  if (!opt.dataPaths.empty()) {
    for (const std::string& path : opt.dataPaths) {
      series.push_back(topt::load_csv(path));
      names.push_back(symbol_name_from_path(path));
    }
  } else {
    const int k = opt.symbols == -1 ? 4 : opt.symbols;
    for (int i = 0; i < k; ++i) {
      series.push_back(topt::generate_synthetic(
          opt.syntheticBars, opt.seed + static_cast<std::uint64_t>(i)));
      names.push_back("syn" + std::to_string(i + 1));
    }
  }

  std::string alignLog;
  topt::align_portfolio(series, alignLog);
  if (!opt.quiet) std::fprintf(stderr, "%s\n", alignLog.c_str());

  auto strategy = topt::make_strategy(opt.strategy);
  std::string backendLog;
  // --threads parallelizes the per-symbol CPU indicator path across symbols
  // (0 = all cores, matching the simulation phase); GPU backends ignore it.
  auto backend =
      topt::create_batch_backend(opt.backend, series, backendLog, opt.threads);
  if (!opt.quiet) {
    std::fprintf(stderr, "%s; %zu symbols x %d bars loaded in %.2f s\n",
                 backendLog.c_str(), series.size(), series.front().size(),
                 seconds_since(start));
  }

  topt::ParamSet ps = fixed_params(*strategy, opt);
  topt::FitnessExpr fitness =
      topt::FitnessExpr::parse(opt.fitness, topt::fitness_variable_names());

  topt::PortfolioConfig cfg;
  cfg.exec = opt.exec;
  topt::PortfolioResult res = topt::run_portfolio(series, names, *strategy, ps,
                                                  *backend, cfg, &fitness);

  topt::print_portfolio_report(stdout, opt.strategy, ps, res, opt.exec,
                               opt.fitness);

  topt::McResult mc;
  bool hasMc = false;
  if (opt.mcIterations > 0) {
    mc = topt::run_monte_carlo(res.trades, opt.exec.capital, make_mc_config(opt));
    topt::print_mc_report(stdout, mc);
    hasMc = true;
  }

  if (!opt.outJson.empty()) {
    topt::write_portfolio_json(opt.outJson, opt.strategy, ps, res,
                               opt.jsonTrades, hasMc ? &mc : nullptr);
    if (!opt.quiet) std::fprintf(stderr, "wrote %s\n", opt.outJson.c_str());
  }
  if (!opt.quiet) {
    std::fprintf(stderr, "portfolio total: %.2f s, %d/%zu symbols, %zu trades\n",
                 seconds_since(start), res.validSymbols, res.symbols.size(),
                 res.trades.size());
  }
  return res.has_valid() ? 0 : 2;
}

topt::GridConfig make_grid_config(const topt::CliOptions& opt) {
  topt::GridConfig cfg;
  cfg.exec = opt.exec;
  cfg.threads = opt.threads;
  cfg.cacheBytes = opt.cacheMb << 20;
  cfg.quiet = opt.quiet;
  return cfg;
}

topt::GeneticConfig make_genetic_config(const topt::CliOptions& opt) {
  topt::GeneticConfig ga;
  ga.eval = make_grid_config(opt);
  ga.population = opt.population;
  ga.maxGenerations = opt.generations;
  ga.elite = opt.elite;
  ga.tournament = opt.tournament;
  ga.crossoverProb = opt.crossoverProb;
  ga.mutationProb = opt.mutationProb;
  ga.mutationSigmaSteps = opt.mutationSigma;
  ga.stagnationLimit = opt.stagnation;
  ga.seed = opt.gaSeed;
  ga.quiet = opt.quiet;
  return ga;
}

int cmd_grid(const topt::CliOptions& opt) {
  auto start = Clock::now();
  topt::Series bars = load_data(opt);
  auto strategy = topt::make_strategy(opt.strategy);

  std::string backendLog;
  auto backend = topt::create_backend(opt.backend, bars, backendLog);
  if (!opt.quiet) {
    std::fprintf(stderr, "%s; %d bars loaded in %.2f s\n", backendLog.c_str(),
                 bars.size(), seconds_since(start));
  }

  std::vector<topt::ParamSet> variants =
      topt::enumerate_variants(strategy->params(), opt.gridAxes, opt.sets);
  topt::FitnessExpr fitness =
      topt::FitnessExpr::parse(opt.fitness, topt::fitness_variable_names());
  if (!opt.quiet) {
    std::fprintf(stderr, "grid: %zu variants of %s\n", variants.size(),
                 opt.strategy.c_str());
  }

  topt::GridConfig cfg = make_grid_config(opt);

  topt::IndicatorCache cache(bars, *backend);
  std::vector<topt::VariantResult> results =
      topt::run_grid(bars, *strategy, variants, cache, cfg, fitness);

  topt::print_grid_report(stdout, results, opt.fitness, opt.top);
  if (!opt.outCsv.empty()) {
    topt::write_grid_csv(opt.outCsv, results);
    if (!opt.quiet) std::fprintf(stderr, "wrote %s\n", opt.outCsv.c_str());
  }
  if (!opt.outJson.empty()) {
    topt::write_grid_json(opt.outJson, opt.strategy, opt.fitness, results, opt.top);
    if (!opt.quiet) std::fprintf(stderr, "wrote %s\n", opt.outJson.c_str());
  }
  if (!opt.quiet) {
    std::fprintf(stderr, "grid total: %.2f s for %zu variants on %d bars\n",
                 seconds_since(start), variants.size(), bars.size());
  }
  return 0;
}

int cmd_genetic(const topt::CliOptions& opt) {
  auto start = Clock::now();
  topt::Series bars = load_data(opt);
  auto strategy = topt::make_strategy(opt.strategy);

  std::string backendLog;
  auto backend = topt::create_backend(opt.backend, bars, backendLog);
  if (!opt.quiet) {
    std::fprintf(stderr, "%s; %d bars loaded in %.2f s\n", backendLog.c_str(),
                 bars.size(), seconds_since(start));
  }

  topt::FitnessExpr fitness =
      topt::FitnessExpr::parse(opt.fitness, topt::fitness_variable_names());
  topt::GeneticConfig ga = make_genetic_config(opt);

  topt::IndicatorCache cache(bars, *backend);
  topt::GeneticResult res =
      topt::run_genetic(bars, *strategy, opt.gridAxes, opt.sets, cache, ga, fitness);

  topt::print_grid_report(stdout, res.evaluated, opt.fitness, opt.top,
                          "Genetic results");
  std::printf("generations %d, unique variants evaluated %zu\n", res.generations,
              res.evaluations);
  if (!opt.outCsv.empty()) {
    topt::write_grid_csv(opt.outCsv, res.evaluated);
    if (!opt.quiet) std::fprintf(stderr, "wrote %s\n", opt.outCsv.c_str());
  }
  if (!opt.outJson.empty()) {
    topt::write_grid_json(opt.outJson, opt.strategy, opt.fitness, res.evaluated, opt.top);
    if (!opt.quiet) std::fprintf(stderr, "wrote %s\n", opt.outJson.c_str());
  }
  if (!opt.quiet) {
    std::fprintf(stderr, "genetic total: %.2f s, %zu evaluations on %d bars\n",
                 seconds_since(start), res.evaluations, bars.size());
  }
  return res.has_best() ? 0 : 2;
}

int cmd_wfo(const topt::CliOptions& opt) {
  auto start = Clock::now();
  topt::Series bars = load_data(opt);
  auto strategy = topt::make_strategy(opt.strategy);

  std::string backendLog;
  auto backend = topt::create_backend(opt.backend, bars, backendLog);
  if (!opt.quiet) {
    std::fprintf(stderr, "%s; %d bars loaded in %.2f s\n", backendLog.c_str(),
                 bars.size(), seconds_since(start));
  }

  topt::FitnessExpr fitness =
      topt::FitnessExpr::parse(opt.fitness, topt::fitness_variable_names());

  topt::WfoConfig cfg;
  cfg.grid = make_grid_config(opt);
  cfg.genetic = make_genetic_config(opt);
  cfg.optimizer = opt.wfoOptimizer == "genetic" ? topt::WfoOptimizer::Genetic
                                                : topt::WfoOptimizer::Grid;
  cfg.windows = opt.windows;
  cfg.isFraction = opt.isFraction;

  topt::IndicatorCache cache(bars, *backend);
  topt::WfoResult res =
      topt::run_wfo(bars, *strategy, opt.gridAxes, opt.sets, cache, cfg, fitness);

  topt::print_wfo_report(stdout, opt.strategy, res, cfg, opt.fitness);
  if (!opt.outJson.empty()) {
    topt::write_wfo_json(opt.outJson, opt.strategy, opt.fitness, res, cfg);
    if (!opt.quiet) std::fprintf(stderr, "wrote %s\n", opt.outJson.c_str());
  }
  if (!opt.quiet) {
    std::fprintf(stderr, "wfo total: %.2f s, %zu IS evaluations on %d bars\n",
                 seconds_since(start), res.isEvaluations, bars.size());
  }

  bool anyOptimized = false;
  for (const auto& w : res.windows) {
    if (w.optimized) anyOptimized = true;
  }
  return anyOptimized ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
  try {
    topt::CliOptions opt = topt::parse_cli(argc, argv);
    if (opt.command == "help") {
      std::printf("%s", topt::usage_text().c_str());
      return 0;
    }
    if (opt.command == "list") return cmd_list();
    if (opt.command == "backtest") return cmd_backtest(opt);
    if (opt.command == "genetic") return cmd_genetic(opt);
    if (opt.command == "wfo") return cmd_wfo(opt);
    if (opt.command == "portfolio") return cmd_portfolio(opt);
    return cmd_grid(opt);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tacuda-opt: %s\n", e.what());
    return 1;
  }
}
