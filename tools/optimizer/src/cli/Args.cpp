/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "cli/Args.h"

#include <cstdlib>
#include <stdexcept>

namespace topt {

namespace {

double to_double(const std::string& v, const std::string& flag) {
  char* end = nullptr;
  double d = std::strtod(v.c_str(), &end);
  if (!end || *end != '\0' || v.empty()) {
    throw std::invalid_argument("bad numeric value '" + v + "' for " + flag);
  }
  return d;
}

long long to_ll(const std::string& v, const std::string& flag) {
  char* end = nullptr;
  long long r = std::strtoll(v.c_str(), &end, 10);
  if (!end || *end != '\0' || v.empty()) {
    throw std::invalid_argument("bad integer value '" + v + "' for " + flag);
  }
  return r;
}

} // namespace

std::string usage_text() {
  return R"(tacuda-opt — GPU-accelerated trading strategy optimizer (TaCuda)

USAGE
  tacuda-opt <command> [options]

COMMANDS
  backtest   run one backtest with fixed parameters
  grid       exhaustive grid-search over strategy parameters
  genetic    genetic search over the same parameter space as grid
  wfo        walk-forward optimization (rolling IS/OOS windows)
  portfolio  multi-symbol backtest (shared params, batch API indicators)
  list       print built-in strategies and their parameters

DATA (exactly one source; portfolio accepts multiple --data files)
  --data PATH            OHLCV CSV: header (time,open,high,low,close[,volume])
                         or headerless [ts,]o,h,l,c[,v]; ',' or ';' separated
                         (portfolio: repeat --data once per symbol)
  --synthetic N          generate N synthetic bars instead of reading a file
  --seed S               synthetic data seed (default 42)
  --symbols K            portfolio + --synthetic: number of series (default 4)

STRATEGY & PARAMETERS
  --strategy NAME        ma_cross | rsi_reversion | bbands_breakout | exp01_*..
                         exp13_* (EXPERIMENTS pack; see `tacuda-opt list`)
  --set name=value       fix one parameter (repeatable)
  --grid name=a..b:s     grid axis, inclusive range with step (repeatable)
  --grid name=v1,v2,v3   grid axis, explicit value list

EXECUTION MODEL
  --commission-bps X     commission per order side, basis points (default 0)
  --commission-fixed X   commission per order side, currency (default 0)
  --slippage-ticks X     fill slippage per side, in ticks (default 0)
  --tick-size X          tick size (default 0.01)
  --qty X                position size in units (default 1)
  --capital X            starting capital (default 10000)
  --bars-per-year N      annualization for Sharpe/Sortino (default 252)

OPTIMIZATION
  --fitness EXPR         metric expression, e.g. "netprofit - 2*maxdd"
                         variables: netprofit grossprofit grossloss pf maxdd
                         maxddpct sharpe sortino winrate trades avgtrade exposure
  --top N                rows in the console table / JSON top (default 20)
  --threads N            simulation threads; in `portfolio` also parallelizes
                         the per-symbol CPU (TA-Lib) indicator phase across
                         symbols (default: all cores; GPU backends ignore it)
  --cache-mb N           indicator cache budget per chunk, MB (default 2048)

GENETIC (genetic command; wfo with --optimizer genetic)
  --population N         population size (default 128)
  --generations N        max generations (default 60)
  --elite N              individuals copied unchanged (default 8)
  --tournament N         tournament selection size (default 4)
  --crossover-prob X     uniform-crossover probability per child (default 0.9)
  --mutation-prob X      mutation probability per gene (default 0.15)
  --mutation-sigma X     gaussian mutation sigma, in axis steps (default 2)
  --stagnation N         early stop after N generations w/o improvement (default 12)
  --ga-seed S            genetic RNG seed (default 42)

WALK-FORWARD (wfo command)
  --windows K            number of rolling windows (default 6)
  --is FRACTION          in-sample share of each window, 0..1 (default 0.75)
  --optimizer KIND       per-window optimizer: grid | genetic (default grid)

MONTE-CARLO (backtest and portfolio commands)
  --mc N                 randomize the trade sequence N times and report
                         netProfit / maxDD percentiles (default 0 = off)
  --mc-mode MODE         permute (reorder realized trades) | resample
                         (bootstrap with replacement) (default permute)
  --mc-seed S            Monte-Carlo RNG seed (default 42)

BACKEND
  --backend KIND         auto | tacuda | tacuda-device | talib  (default auto:
                         GPU when available, otherwise TA-Lib CPU fallback).
                         tacuda-device forces the single-series device API even
                         in portfolio mode (one ct_*_d call per symbol) — the
                         benchmark baseline for the batch API

OUTPUT
  --out-csv PATH         grid: all variants as CSV
  --out-json PATH        grid: summary JSON; backtest: metrics JSON
  --json-trades          include the trade list in backtest JSON
  --quiet                suppress progress output on stderr

EXAMPLES
  tacuda-opt backtest --data ES_5min.csv --strategy ma_cross \
      --set fast=12 --set slow=48 --commission-bps 0.5
  tacuda-opt grid --synthetic 1000000 --strategy ma_cross \
      --grid fast=5..60:5 --grid slow=20..240:20 --fitness sharpe --top 20
  tacuda-opt genetic --synthetic 1000000 --strategy ma_cross \
      --grid fast=2..100:1 --grid slow=10..500:5 --population 128 --generations 40
  tacuda-opt wfo --data ES_5min.csv --strategy ma_cross \
      --grid fast=5..60:5 --grid slow=20..240:20 --windows 6 --is 0.75
  tacuda-opt portfolio --data ES.csv --data NQ.csv --data CL.csv \
      --strategy ma_cross --set fast=12 --set slow=48 --mc 1000
)";
}

CliOptions parse_cli(int argc, const char* const* argv) {
  CliOptions opt;
  if (argc < 2) {
    throw std::invalid_argument("missing command\n\n" + usage_text());
  }
  opt.command = argv[1];
  if (opt.command == "--help" || opt.command == "-h" || opt.command == "help") {
    opt.command = "help";
    return opt;
  }
  if (opt.command != "backtest" && opt.command != "grid" && opt.command != "genetic" &&
      opt.command != "wfo" && opt.command != "portfolio" && opt.command != "list") {
    throw std::invalid_argument(
        "unknown command '" + opt.command +
        "' (expected backtest, grid, genetic, wfo, portfolio or list)");
  }

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    std::string value;
    auto eq = arg.find('=');
    bool inlineValue = false;
    if (arg.rfind("--", 0) == 0 && eq != std::string::npos) {
      value = arg.substr(eq + 1);
      arg = arg.substr(0, eq);
      inlineValue = true;
    }
    auto need_value = [&]() -> std::string {
      if (inlineValue) return value;
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + arg);
      }
      return argv[++i];
    };

    if (arg == "--data") {
      opt.dataPaths.push_back(need_value());
    } else if (arg == "--synthetic") {
      opt.syntheticBars = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--seed") {
      opt.seed = static_cast<std::uint64_t>(to_ll(need_value(), arg));
    } else if (arg == "--symbols") {
      opt.symbols = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--strategy") {
      opt.strategy = need_value();
    } else if (arg == "--grid") {
      opt.gridAxes.push_back(parse_grid_spec(need_value()));
    } else if (arg == "--set") {
      opt.sets.push_back(parse_set_spec(need_value()));
    } else if (arg == "--commission-bps") {
      opt.exec.commissionBps = to_double(need_value(), arg);
    } else if (arg == "--commission-fixed") {
      opt.exec.commissionFixed = to_double(need_value(), arg);
    } else if (arg == "--slippage-ticks") {
      opt.exec.slippageTicks = to_double(need_value(), arg);
    } else if (arg == "--tick-size") {
      opt.exec.tickSize = to_double(need_value(), arg);
    } else if (arg == "--qty") {
      opt.exec.qty = to_double(need_value(), arg);
    } else if (arg == "--capital") {
      opt.exec.capital = to_double(need_value(), arg);
    } else if (arg == "--bars-per-year") {
      opt.exec.barsPerYear = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--fitness") {
      opt.fitness = need_value();
    } else if (arg == "--top") {
      opt.top = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--threads") {
      opt.threads = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--cache-mb") {
      opt.cacheMb = static_cast<std::size_t>(to_ll(need_value(), arg));
    } else if (arg == "--backend") {
      opt.backend = parse_backend_kind(need_value());
    } else if (arg == "--population") {
      opt.population = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--generations") {
      opt.generations = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--elite") {
      opt.elite = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--tournament") {
      opt.tournament = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--crossover-prob") {
      opt.crossoverProb = to_double(need_value(), arg);
    } else if (arg == "--mutation-prob") {
      opt.mutationProb = to_double(need_value(), arg);
    } else if (arg == "--mutation-sigma") {
      opt.mutationSigma = to_double(need_value(), arg);
    } else if (arg == "--stagnation") {
      opt.stagnation = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--ga-seed") {
      opt.gaSeed = static_cast<std::uint64_t>(to_ll(need_value(), arg));
    } else if (arg == "--windows") {
      opt.windows = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--is") {
      opt.isFraction = to_double(need_value(), arg);
    } else if (arg == "--optimizer") {
      opt.wfoOptimizer = need_value();
    } else if (arg == "--mc") {
      opt.mcIterations = static_cast<int>(to_ll(need_value(), arg));
    } else if (arg == "--mc-mode") {
      opt.mcMode = need_value();
    } else if (arg == "--mc-seed") {
      opt.mcSeed = static_cast<std::uint64_t>(to_ll(need_value(), arg));
    } else if (arg == "--out-csv") {
      opt.outCsv = need_value();
    } else if (arg == "--out-json") {
      opt.outJson = need_value();
    } else if (arg == "--json-trades") {
      opt.jsonTrades = true;
    } else if (arg == "--quiet") {
      opt.quiet = true;
    } else if (arg == "--help" || arg == "-h") {
      opt.command = "help";
      return opt;
    } else {
      throw std::invalid_argument("unknown option '" + arg + "'");
    }
  }

  if (opt.command == "list") return opt;

  if (opt.strategy.empty()) {
    throw std::invalid_argument("--strategy is required for " + opt.command);
  }
  bool hasCsv = !opt.dataPaths.empty();
  bool hasSynthetic = opt.syntheticBars > 0;
  if (hasCsv == hasSynthetic) {
    throw std::invalid_argument(
        "specify exactly one data source: --data PATH or --synthetic N");
  }
  if (opt.command != "portfolio" && opt.dataPaths.size() > 1) {
    throw std::invalid_argument(
        "multiple --data files are only valid for the portfolio command");
  }
  if (opt.symbols != -1 &&
      !(opt.command == "portfolio" && hasSynthetic)) {
    throw std::invalid_argument(
        "--symbols is only valid for portfolio with --synthetic");
  }
  if (opt.symbols == 0 || opt.symbols < -1) {
    throw std::invalid_argument("--symbols must be >= 1");
  }
  if ((opt.command == "grid" || opt.command == "genetic" || opt.command == "wfo") &&
      opt.gridAxes.empty()) {
    throw std::invalid_argument(opt.command + " command needs at least one --grid axis");
  }
  if (opt.command == "wfo") {
    if (opt.windows < 1) {
      throw std::invalid_argument("--windows must be >= 1");
    }
    if (!(opt.isFraction > 0.0) || !(opt.isFraction < 1.0)) {
      throw std::invalid_argument("--is must be in (0, 1)");
    }
    if (opt.wfoOptimizer != "grid" && opt.wfoOptimizer != "genetic") {
      throw std::invalid_argument("--optimizer must be grid or genetic");
    }
  }
  if (opt.command == "portfolio" && !opt.gridAxes.empty()) {
    throw std::invalid_argument(
        "portfolio uses fixed parameters (--set), not --grid axes");
  }
  if (opt.mcIterations < 0) {
    throw std::invalid_argument("--mc must be >= 0");
  }
  if (opt.mcIterations > 0 &&
      opt.command != "backtest" && opt.command != "portfolio") {
    throw std::invalid_argument(
        "--mc is only valid for backtest and portfolio commands");
  }
  if (opt.mcMode != "permute" && opt.mcMode != "resample") {
    throw std::invalid_argument("--mc-mode must be permute or resample");
  }
  return opt;
}

} // namespace topt
