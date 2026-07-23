/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "report/Report.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace topt {

namespace {

std::string fmt(double v, int prec = 2) {
  if (!std::isfinite(v)) return "nan";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
  return buf;
}

// Ranks valid variants by fitness descending; ties keep enumeration order.
std::vector<std::size_t> rank_by_fitness(const std::vector<VariantResult>& results) {
  std::vector<std::size_t> idx;
  idx.reserve(results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i].valid) idx.push_back(i);
  }
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    return results[a].fitness > results[b].fitness;
  });
  return idx;
}

void json_escape(std::string& out, const std::string& s) {
  out += '"';
  for (char c : s) {
    switch (c) {
    case '"': out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n"; break;
    case '\r': out += "\\r"; break;
    case '\t': out += "\\t"; break;
    default: out += c; break;
    }
  }
  out += '"';
}

std::string json_number(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  return buf;
}

void append_params_json(std::string& out, const ParamSet& ps) {
  out += '{';
  bool first = true;
  for (const auto& kv : ps.values) {
    if (!first) out += ',';
    first = false;
    json_escape(out, kv.first);
    out += ':';
    out += json_number(kv.second);
  }
  out += '}';
}

void append_metrics_json(std::string& out, const MetricsResult& m) {
  out += '{';
  out += "\"netProfit\":" + json_number(m.netProfit);
  out += ",\"grossProfit\":" + json_number(m.grossProfit);
  out += ",\"grossLoss\":" + json_number(m.grossLoss);
  out += ",\"profitFactor\":" + json_number(m.profitFactor);
  out += ",\"maxDrawdown\":" + json_number(m.maxDrawdown);
  out += ",\"maxDrawdownPct\":" + json_number(m.maxDrawdownPct);
  out += ",\"sharpe\":" + json_number(m.sharpe);
  out += ",\"sortino\":" + json_number(m.sortino);
  out += ",\"winRate\":" + json_number(m.winRate);
  out += ",\"avgTrade\":" + json_number(m.avgTrade);
  out += ",\"exposure\":" + json_number(m.exposure);
  out += ",\"numTrades\":" + std::to_string(m.numTrades);
  out += '}';
}

std::ofstream open_file(const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write file: " + path);
  return f;
}

} // namespace

void print_backtest_report(std::FILE* out, const std::string& strategyName,
                           const VariantResult& result, const SimResult& sim,
                           const ExecConfig& exec) {
  const MetricsResult& m = result.metrics;
  std::fprintf(out, "\n=== Backtest: %s [%s] ===\n", strategyName.c_str(),
               result.params.to_string().c_str());
  if (!result.valid) {
    std::fprintf(out, "FAILED: %s\n", result.error.c_str());
    return;
  }
  std::fprintf(out, "%-22s %14s\n", "metric", "value");
  std::fprintf(out, "%-22s %14s\n", "----------------------", "--------------");
  std::fprintf(out, "%-22s %14s\n", "Net profit", fmt(m.netProfit).c_str());
  std::fprintf(out, "%-22s %14s\n", "Gross profit", fmt(m.grossProfit).c_str());
  std::fprintf(out, "%-22s %14s\n", "Gross loss", fmt(m.grossLoss).c_str());
  std::fprintf(out, "%-22s %14s\n", "Profit factor", fmt(m.profitFactor, 3).c_str());
  std::fprintf(out, "%-22s %14s\n", "Max drawdown", fmt(m.maxDrawdown).c_str());
  std::fprintf(out, "%-22s %13s%%\n", "Max drawdown %", fmt(m.maxDrawdownPct * 100.0).c_str());
  std::fprintf(out, "%-22s %14s\n", "Sharpe (annual)", fmt(m.sharpe, 3).c_str());
  std::fprintf(out, "%-22s %14s\n", "Sortino (annual)", fmt(m.sortino, 3).c_str());
  std::fprintf(out, "%-22s %14d\n", "Trades", m.numTrades);
  std::fprintf(out, "%-22s %13s%%\n", "Win rate", fmt(m.winRate * 100.0, 1).c_str());
  std::fprintf(out, "%-22s %14s\n", "Avg trade", fmt(m.avgTrade).c_str());
  std::fprintf(out, "%-22s %13s%%\n", "Exposure", fmt(m.exposure * 100.0, 1).c_str());
  std::fprintf(out, "%-22s %14s\n", "Final equity", fmt(sim.finalEquity).c_str());
  std::fprintf(out, "%-22s %14s\n", "Initial capital", fmt(sim.initialCapital).c_str());
  std::fprintf(out, "(fills: next bar open, commission %.2f bps + %.2f fixed, "
                    "slippage %.1f tick(s) of %.4f)\n",
               exec.commissionBps, exec.commissionFixed, exec.slippageTicks, exec.tickSize);
}

std::vector<std::size_t> print_grid_report(std::FILE* out,
                                           const std::vector<VariantResult>& results,
                                           const std::string& fitnessText, int topN,
                                           const char* title) {
  std::vector<std::size_t> ranked = rank_by_fitness(results);
  std::size_t invalid = results.size() - ranked.size();

  std::fprintf(out, "\n=== %s: %zu variants (%zu skipped), fitness = %s ===\n",
               title, results.size(), invalid, fitnessText.c_str());
  std::fprintf(out, "%4s  %-32s %12s %12s %9s %8s %8s %7s %7s\n", "#", "params",
               "fitness", "netProfit", "maxDD%", "PF", "Sharpe", "trades", "win%");
  std::size_t shown = std::min<std::size_t>(ranked.size(),
                                            topN > 0 ? static_cast<std::size_t>(topN)
                                                     : ranked.size());
  for (std::size_t r = 0; r < shown; ++r) {
    const VariantResult& v = results[ranked[r]];
    const MetricsResult& m = v.metrics;
    std::fprintf(out, "%4zu  %-32s %12s %12s %9s %8s %8s %7d %7s\n", r + 1,
                 v.params.to_string().c_str(), fmt(v.fitness, 4).c_str(),
                 fmt(m.netProfit).c_str(), fmt(m.maxDrawdownPct * 100.0, 1).c_str(),
                 fmt(m.profitFactor, 2).c_str(), fmt(m.sharpe, 2).c_str(),
                 m.numTrades, fmt(m.winRate * 100.0, 1).c_str());
  }
  ranked.resize(shown);
  return ranked;
}

void write_grid_csv(const std::string& path, const std::vector<VariantResult>& results) {
  std::ofstream f = open_file(path);
  f << "rank";
  if (!results.empty()) {
    for (const auto& kv : results.front().params.values) {
      f << ',' << kv.first;
    }
  }
  f << ",fitness,netProfit,grossProfit,grossLoss,profitFactor,maxDrawdown,"
       "maxDrawdownPct,sharpe,sortino,winRate,avgTrade,exposure,numTrades,status\n";

  std::vector<std::size_t> ranked = rank_by_fitness(results);
  std::vector<std::size_t> order = ranked;
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (!results[i].valid) order.push_back(i);
  }
  std::size_t rank = 0;
  for (std::size_t idx : order) {
    const VariantResult& v = results[idx];
    ++rank;
    f << (v.valid ? std::to_string(rank) : "");
    for (const auto& kv : v.params.values) {
      f << ',' << json_number(kv.second);
    }
    const MetricsResult& m = v.metrics;
    f << ',' << json_number(v.fitness) << ',' << json_number(m.netProfit) << ','
      << json_number(m.grossProfit) << ',' << json_number(m.grossLoss) << ','
      << json_number(m.profitFactor) << ',' << json_number(m.maxDrawdown) << ','
      << json_number(m.maxDrawdownPct) << ',' << json_number(m.sharpe) << ','
      << json_number(m.sortino) << ',' << json_number(m.winRate) << ','
      << json_number(m.avgTrade) << ',' << json_number(m.exposure) << ','
      << m.numTrades << ',' << (v.valid ? "ok" : v.error) << '\n';
  }
}

void write_grid_json(const std::string& path, const std::string& strategyName,
                     const std::string& fitnessText,
                     const std::vector<VariantResult>& results, int topN) {
  std::vector<std::size_t> ranked = rank_by_fitness(results);
  std::size_t shown = std::min<std::size_t>(ranked.size(),
                                            topN > 0 ? static_cast<std::size_t>(topN)
                                                     : ranked.size());
  std::string out;
  out += "{\"strategy\":";
  json_escape(out, strategyName);
  out += ",\"fitness\":";
  json_escape(out, fitnessText);
  out += ",\"variants\":" + std::to_string(results.size());
  out += ",\"evaluated\":" + std::to_string(ranked.size());
  if (!ranked.empty()) {
    const VariantResult& best = results[ranked.front()];
    out += ",\"best\":{\"params\":";
    append_params_json(out, best.params);
    out += ",\"fitness\":" + json_number(best.fitness);
    out += ",\"metrics\":";
    append_metrics_json(out, best.metrics);
    out += '}';
  }
  out += ",\"top\":[";
  for (std::size_t r = 0; r < shown; ++r) {
    if (r) out += ',';
    const VariantResult& v = results[ranked[r]];
    out += "{\"rank\":" + std::to_string(r + 1) + ",\"params\":";
    append_params_json(out, v.params);
    out += ",\"fitness\":" + json_number(v.fitness);
    out += ",\"metrics\":";
    append_metrics_json(out, v.metrics);
    out += '}';
  }
  out += "]}";

  std::ofstream f = open_file(path);
  f << out << '\n';
}

namespace {

void append_trades_json(std::string& out, const std::vector<Trade>& trades) {
  out += '[';
  for (std::size_t i = 0; i < trades.size(); ++i) {
    const Trade& t = trades[i];
    if (i) out += ',';
    out += "{\"entryBar\":" + std::to_string(t.entryBar) +
           ",\"exitBar\":" + std::to_string(t.exitBar) +
           ",\"dir\":" + std::to_string(t.dir) +
           ",\"entryPrice\":" + json_number(t.entryPrice) +
           ",\"exitPrice\":" + json_number(t.exitPrice) +
           ",\"pnl\":" + json_number(t.pnl) + "}";
  }
  out += ']';
}

void append_mc_percentiles_json(std::string& out, const McPercentiles& p) {
  out += "{\"p5\":" + json_number(p.p5);
  out += ",\"p25\":" + json_number(p.p25);
  out += ",\"p50\":" + json_number(p.p50);
  out += ",\"p75\":" + json_number(p.p75);
  out += ",\"p95\":" + json_number(p.p95);
  out += ",\"worst\":" + json_number(p.worst);
  out += '}';
}

void append_mc_json(std::string& out, const McResult& mc) {
  out += "{\"mode\":";
  json_escape(out, to_string(mc.mode));
  out += ",\"iterations\":" + std::to_string(mc.iterations);
  out += ",\"numTrades\":" + std::to_string(mc.numTrades);
  out += ",\"originalNetProfit\":" + json_number(mc.originalNetProfit);
  out += ",\"originalMaxDrawdown\":" + json_number(mc.originalMaxDrawdown);
  out += ",\"netProfit\":";
  append_mc_percentiles_json(out, mc.netProfit);
  out += ",\"maxDrawdown\":";
  append_mc_percentiles_json(out, mc.maxDrawdown);
  out += ",\"probNetLoss\":" + json_number(mc.probNetLoss);
  out += ",\"probDdAboveOriginal\":" + json_number(mc.probDdAboveOriginal);
  out += '}';
}

} // namespace

void write_backtest_json(const std::string& path, const std::string& strategyName,
                         const VariantResult& result, const SimResult& sim,
                         bool includeTrades, const McResult* mc) {
  std::string out;
  out += "{\"strategy\":";
  json_escape(out, strategyName);
  out += ",\"params\":";
  append_params_json(out, result.params);
  out += ",\"valid\":";
  out += result.valid ? "true" : "false";
  if (!result.valid) {
    out += ",\"error\":";
    json_escape(out, result.error);
  }
  out += ",\"metrics\":";
  append_metrics_json(out, result.metrics);
  out += ",\"initialCapital\":" + json_number(sim.initialCapital);
  out += ",\"finalEquity\":" + json_number(sim.finalEquity);
  if (mc) {
    out += ",\"monteCarlo\":";
    append_mc_json(out, *mc);
  }
  if (includeTrades) {
    out += ",\"trades\":";
    append_trades_json(out, sim.trades);
  }
  out += '}';

  std::ofstream f = open_file(path);
  f << out << '\n';
}

void print_mc_report(std::FILE* out, const McResult& mc) {
  std::fprintf(out, "\n=== Monte-Carlo: %s, %d iterations, %d trades ===\n",
               to_string(mc.mode), mc.iterations, mc.numTrades);
  if (mc.numTrades == 0) {
    std::fprintf(out, "no trades — nothing to randomize\n");
    return;
  }
  std::fprintf(out, "%-12s %10s %10s %10s %10s %10s %10s\n", "", "P5", "P25",
               "P50", "P75", "P95", "worst");
  const McPercentiles& np = mc.netProfit;
  const McPercentiles& dd = mc.maxDrawdown;
  std::fprintf(out, "%-12s %10s %10s %10s %10s %10s %10s\n", "netProfit",
               fmt(np.p5).c_str(), fmt(np.p25).c_str(), fmt(np.p50).c_str(),
               fmt(np.p75).c_str(), fmt(np.p95).c_str(), fmt(np.worst).c_str());
  std::fprintf(out, "%-12s %10s %10s %10s %10s %10s %10s\n", "maxDD",
               fmt(dd.p5).c_str(), fmt(dd.p25).c_str(), fmt(dd.p50).c_str(),
               fmt(dd.p75).c_str(), fmt(dd.p95).c_str(), fmt(dd.worst).c_str());
  if (mc.mode == McMode::Permute) {
    std::fprintf(out, "(net profit is order-invariant under permutation)\n");
  }
  std::fprintf(out,
               "original order: netProfit %s, maxDD %s (trade-level equity)\n",
               fmt(mc.originalNetProfit).c_str(),
               fmt(mc.originalMaxDrawdown).c_str());
  std::fprintf(out, "P(net loss) %s%%   P(maxDD > original) %s%%\n",
               fmt(mc.probNetLoss * 100.0, 1).c_str(),
               fmt(mc.probDdAboveOriginal * 100.0, 1).c_str());
}

void print_portfolio_report(std::FILE* out, const std::string& strategyName,
                            const ParamSet& params, const PortfolioResult& res,
                            const ExecConfig& exec,
                            const std::string& fitnessText) {
  std::fprintf(out, "\n=== Portfolio backtest: %s [%s], %zu symbols, %d bars ===\n",
               strategyName.c_str(), params.to_string().c_str(),
               res.symbols.size(), res.bars);
  std::fprintf(out, "%-16s %-8s %12s %9s %8s %8s %7s %7s\n", "symbol", "status",
               "netProfit", "maxDD%", "PF", "Sharpe", "trades", "win%");
  for (const SymbolResult& sr : res.symbols) {
    if (sr.valid) {
      const MetricsResult& m = sr.metrics;
      std::fprintf(out, "%-16s %-8s %12s %9s %8s %8s %7d %7s\n", sr.name.c_str(),
                   "ok", fmt(m.netProfit).c_str(),
                   fmt(m.maxDrawdownPct * 100.0, 1).c_str(),
                   fmt(m.profitFactor, 2).c_str(), fmt(m.sharpe, 2).c_str(),
                   m.numTrades, fmt(m.winRate * 100.0, 1).c_str());
    } else {
      std::fprintf(out, "%-16s %-8s %s\n", sr.name.c_str(), "SKIPPED",
                   sr.error.c_str());
    }
  }

  const MetricsResult& m = res.metrics;
  std::fprintf(out, "\n--- portfolio (aggregated, %d/%zu symbols) ---\n",
               res.validSymbols, res.symbols.size());
  std::fprintf(out, "%-22s %14s\n", "Net profit", fmt(m.netProfit).c_str());
  std::fprintf(out, "%-22s %14s\n", "Profit factor", fmt(m.profitFactor, 3).c_str());
  std::fprintf(out, "%-22s %14s\n", "Max drawdown", fmt(m.maxDrawdown).c_str());
  std::fprintf(out, "%-22s %13s%%\n", "Max drawdown %",
               fmt(m.maxDrawdownPct * 100.0).c_str());
  std::fprintf(out, "%-22s %14s\n", "Sharpe (annual)", fmt(m.sharpe, 3).c_str());
  std::fprintf(out, "%-22s %14s\n", "Sortino (annual)", fmt(m.sortino, 3).c_str());
  std::fprintf(out, "%-22s %14d\n", "Trades", m.numTrades);
  std::fprintf(out, "%-22s %13s%%\n", "Win rate", fmt(m.winRate * 100.0, 1).c_str());
  std::fprintf(out, "%-22s %13s%%\n", "Avg exposure",
               fmt(m.exposure * 100.0, 1).c_str());
  std::fprintf(out, "%-22s %14s\n", "Final equity",
               res.equity.empty() ? "nan" : fmt(res.equity.back()).c_str());
  std::fprintf(out, "%-22s %13.3fs\n", "Indicator phase", res.indicatorSec);
  std::fprintf(out, "%-22s %13.3fs\n", "Signals+simulation", res.signalSec);
  std::fprintf(out, "(capital %s split equally: %s per symbol)\n",
               fmt(exec.capital).c_str(),
               fmt(exec.capital / static_cast<double>(res.symbols.size())).c_str());
  if (!fitnessText.empty()) {
    std::fprintf(out, "fitness (%s) = %.6g\n", fitnessText.c_str(), res.fitness);
  }
}

void write_portfolio_json(const std::string& path, const std::string& strategyName,
                          const ParamSet& params, const PortfolioResult& res,
                          bool includeTrades, const McResult* mc) {
  std::string out;
  out += "{\"strategy\":";
  json_escape(out, strategyName);
  out += ",\"params\":";
  append_params_json(out, params);
  out += ",\"symbols\":" + std::to_string(res.symbols.size());
  out += ",\"validSymbols\":" + std::to_string(res.validSymbols);
  out += ",\"bars\":" + std::to_string(res.bars);
  out += ",\"metrics\":";
  append_metrics_json(out, res.metrics);
  out += ",\"fitness\":" + json_number(res.fitness);
  out += ",\"timing\":{\"indicatorSec\":" + json_number(res.indicatorSec) +
         ",\"signalSec\":" + json_number(res.signalSec) + "}";
  if (mc) {
    out += ",\"monteCarlo\":";
    append_mc_json(out, *mc);
  }
  out += ",\"perSymbol\":[";
  for (std::size_t s = 0; s < res.symbols.size(); ++s) {
    const SymbolResult& sr = res.symbols[s];
    if (s) out += ',';
    out += "{\"name\":";
    json_escape(out, sr.name);
    out += ",\"valid\":";
    out += sr.valid ? "true" : "false";
    if (!sr.valid) {
      out += ",\"error\":";
      json_escape(out, sr.error);
    } else {
      out += ",\"metrics\":";
      append_metrics_json(out, sr.metrics);
      if (includeTrades) {
        out += ",\"trades\":";
        append_trades_json(out, sr.sim.trades);
      }
    }
    out += '}';
  }
  out += "]}";

  std::ofstream f = open_file(path);
  f << out << '\n';
}

void print_wfo_report(std::FILE* out, const std::string& strategyName,
                      const WfoResult& res, const WfoConfig& cfg,
                      const std::string& fitnessText) {
  std::fprintf(out,
               "\n=== Walk-forward: %s, %zu windows, IS %.0f%%, optimizer %s, "
               "fitness = %s ===\n",
               strategyName.c_str(), res.windows.size(), cfg.isFraction * 100.0,
               cfg.optimizer == WfoOptimizer::Grid ? "grid" : "genetic",
               fitnessText.c_str());
  std::fprintf(out, "%4s  %-19s %-19s %-28s %12s %12s %12s %7s\n", "#", "IS bars",
               "OOS bars", "params", "IS fit", "OOS fit", "OOS NP", "trades");
  for (std::size_t w = 0; w < res.windows.size(); ++w) {
    const WfoWindowResult& wr = res.windows[w];
    char isRange[32], oosRange[32];
    std::snprintf(isRange, sizeof(isRange), "%d..%d", wr.window.isBegin, wr.window.isEnd);
    std::snprintf(oosRange, sizeof(oosRange), "%d..%d", wr.window.isEnd, wr.window.oosEnd);
    if (wr.optimized) {
      std::fprintf(out, "%4zu  %-19s %-19s %-28s %12s %12s %12s %7d\n", w + 1, isRange,
                   oosRange, wr.params.to_string().c_str(), fmt(wr.isFitness, 4).c_str(),
                   fmt(wr.oosFitness, 4).c_str(), fmt(wr.oosMetrics.netProfit).c_str(),
                   wr.oosMetrics.numTrades);
    } else {
      std::fprintf(out, "%4zu  %-19s %-19s SKIPPED: %s\n", w + 1, isRange, oosRange,
                   wr.note.c_str());
    }
  }

  const MetricsResult& m = res.oosOverall;
  std::fprintf(out, "\n=== OOS summary (stitched equity, %zu bars) ===\n",
               res.stitchedEquity.size());
  std::fprintf(out, "%-22s %14s\n", "Net profit", fmt(m.netProfit).c_str());
  std::fprintf(out, "%-22s %14s\n", "Profit factor", fmt(m.profitFactor, 3).c_str());
  std::fprintf(out, "%-22s %14s\n", "Max drawdown", fmt(m.maxDrawdown).c_str());
  std::fprintf(out, "%-22s %13s%%\n", "Max drawdown %", fmt(m.maxDrawdownPct * 100.0).c_str());
  std::fprintf(out, "%-22s %14s\n", "Sharpe (annual)", fmt(m.sharpe, 3).c_str());
  std::fprintf(out, "%-22s %14d\n", "Trades", m.numTrades);
  std::fprintf(out, "%-22s %13s%%\n", "Win rate", fmt(m.winRate * 100.0, 1).c_str());
  std::fprintf(out, "%-22s %14s\n", "WFE", fmt(res.wfe, 3).c_str());
  std::fprintf(out, "(IS evaluations %zu, unused tail bars %d)\n", res.isEvaluations,
               res.tailUnused);
}

void write_wfo_json(const std::string& path, const std::string& strategyName,
                    const std::string& fitnessText, const WfoResult& res,
                    const WfoConfig& cfg) {
  std::string out;
  out += "{\"strategy\":";
  json_escape(out, strategyName);
  out += ",\"fitness\":";
  json_escape(out, fitnessText);
  out += ",\"optimizer\":";
  json_escape(out, cfg.optimizer == WfoOptimizer::Grid ? "grid" : "genetic");
  out += ",\"windows\":" + std::to_string(res.windows.size());
  out += ",\"isFraction\":" + json_number(cfg.isFraction);
  out += ",\"wfe\":" + json_number(res.wfe);
  out += ",\"isEvaluations\":" + std::to_string(res.isEvaluations);
  out += ",\"tailUnused\":" + std::to_string(res.tailUnused);
  out += ",\"oosOverall\":";
  append_metrics_json(out, res.oosOverall);
  out += ",\"windowResults\":[";
  for (std::size_t w = 0; w < res.windows.size(); ++w) {
    const WfoWindowResult& wr = res.windows[w];
    if (w) out += ',';
    out += "{\"window\":" + std::to_string(w + 1);
    out += ",\"isBegin\":" + std::to_string(wr.window.isBegin);
    out += ",\"isEnd\":" + std::to_string(wr.window.isEnd);
    out += ",\"oosEnd\":" + std::to_string(wr.window.oosEnd);
    out += ",\"optimized\":";
    out += wr.optimized ? "true" : "false";
    if (wr.optimized) {
      out += ",\"params\":";
      append_params_json(out, wr.params);
      out += ",\"isFitness\":" + json_number(wr.isFitness);
      out += ",\"isMetrics\":";
      append_metrics_json(out, wr.isMetrics);
      out += ",\"oosFitness\":" + json_number(wr.oosFitness);
      out += ",\"oosMetrics\":";
      append_metrics_json(out, wr.oosMetrics);
    } else {
      out += ",\"note\":";
      json_escape(out, wr.note);
    }
    out += '}';
  }
  out += "]}";

  std::ofstream f = open_file(path);
  f << out << '\n';
}

} // namespace topt
