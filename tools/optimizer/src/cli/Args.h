/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_CLI_ARGS_H
#define TACUDA_OPT_CLI_ARGS_H

#include <cstdint>
#include <string>
#include <vector>

#include "backend/Backend.h"
#include "core/ParamSpace.h"
#include "engine/Simulator.h"

namespace topt {

// Parsed command line for tacuda-opt.
// Commands: backtest, grid, genetic, wfo, portfolio, list.
struct CliOptions {
  std::string command;

  // Data source: CSV file(s) or a synthetic series. Multiple --data files
  // are allowed for the portfolio command only.
  std::vector<std::string> dataPaths;
  int syntheticBars = 0;
  std::uint64_t seed = 42;
  int symbols = -1; // portfolio + --synthetic: series count (-1 = default 4)

  std::string strategy;
  std::vector<GridAxis> gridAxes;                       // --grid
  std::vector<std::pair<std::string, double>> sets;     // --set

  ExecConfig exec;
  std::string fitness = "netprofit";
  BackendKind backend = BackendKind::Auto;

  int top = 20;
  int threads = 0;
  std::size_t cacheMb = 2048;
  bool quiet = false;
  bool jsonTrades = false;

  // Genetic optimizer (genetic command; wfo with --optimizer genetic).
  int population = 128;
  int generations = 60;
  int elite = 8;
  int tournament = 4;
  double crossoverProb = 0.9;
  double mutationProb = 0.15;
  double mutationSigma = 2.0;
  int stagnation = 12;
  std::uint64_t gaSeed = 42;

  // Walk-forward (wfo command).
  int windows = 6;
  double isFraction = 0.75;
  std::string wfoOptimizer = "grid"; // grid | genetic

  // Monte-Carlo robustness (backtest and portfolio commands).
  int mcIterations = 0;              // 0 = off
  std::string mcMode = "permute";    // permute | resample
  std::uint64_t mcSeed = 42;

  std::string outCsv;
  std::string outJson;
};

// Throws std::invalid_argument with a usage-style message on errors.
CliOptions parse_cli(int argc, const char* const* argv);

// Full usage text (also printed on --help).
std::string usage_text();

} // namespace topt

#endif // TACUDA_OPT_CLI_ARGS_H
