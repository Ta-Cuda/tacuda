/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "engine/Metrics.h"

#include <algorithm>
#include <cmath>

namespace topt {

MetricsResult compute_metrics(const SimResult& sim, int barsPerYear) {
  MetricsResult m;
  const auto& eq = sim.equity;
  const int n = static_cast<int>(eq.size());
  if (n == 0) return m;

  m.netProfit = sim.finalEquity - sim.initialCapital;
  m.numTrades = static_cast<int>(sim.trades.size());

  int wins = 0;
  double tradePnlSum = 0.0;
  for (const Trade& t : sim.trades) {
    tradePnlSum += t.pnl;
    if (t.pnl > 0.0) {
      m.grossProfit += t.pnl;
      ++wins;
    } else {
      m.grossLoss += -t.pnl;
    }
  }
  m.profitFactor = m.grossLoss > 0.0 ? m.grossProfit / m.grossLoss
                                     : (m.grossProfit > 0.0 ? 1e9 : 0.0);
  m.winRate = m.numTrades > 0 ? static_cast<double>(wins) / m.numTrades : 0.0;
  m.avgTrade = m.numTrades > 0 ? tradePnlSum / m.numTrades : 0.0;
  m.exposure = n > 0 ? static_cast<double>(sim.barsInPosition) / n : 0.0;

  // Drawdowns over the close-marked equity curve.
  double peak = eq[0];
  for (int i = 0; i < n; ++i) {
    peak = std::max(peak, eq[static_cast<std::size_t>(i)]);
    double dd = peak - eq[static_cast<std::size_t>(i)];
    m.maxDrawdown = std::max(m.maxDrawdown, dd);
    if (peak > 0.0) {
      m.maxDrawdownPct = std::max(m.maxDrawdownPct, dd / peak);
    }
  }

  // Sharpe / Sortino from per-bar equity differences (population std).
  if (n >= 2) {
    const int k = n - 1;
    double mean = 0.0;
    for (int i = 1; i < n; ++i) {
      mean += eq[static_cast<std::size_t>(i)] - eq[static_cast<std::size_t>(i - 1)];
    }
    mean /= k;
    double var = 0.0, downVar = 0.0;
    for (int i = 1; i < n; ++i) {
      double d = eq[static_cast<std::size_t>(i)] - eq[static_cast<std::size_t>(i - 1)];
      var += (d - mean) * (d - mean);
      double down = std::min(d, 0.0);
      downVar += down * down;
    }
    var /= k;
    downVar /= k;
    double ann = std::sqrt(static_cast<double>(barsPerYear));
    if (var > 0.0) m.sharpe = mean / std::sqrt(var) * ann;
    if (downVar > 0.0) m.sortino = mean / std::sqrt(downVar) * ann;
  }
  return m;
}

const std::vector<std::string>& fitness_variable_names() {
  static const std::vector<std::string> names = {
      "netprofit", "grossprofit", "grossloss", "pf", "profitfactor",
      "maxdd", "maxddpct", "sharpe", "sortino", "winrate",
      "trades", "numtrades", "avgtrade", "exposure"};
  return names;
}

std::map<std::string, double> fitness_variables(const MetricsResult& m) {
  return {
      {"netprofit", m.netProfit},
      {"grossprofit", m.grossProfit},
      {"grossloss", m.grossLoss},
      {"pf", m.profitFactor},
      {"profitfactor", m.profitFactor},
      {"maxdd", m.maxDrawdown},
      {"maxddpct", m.maxDrawdownPct},
      {"sharpe", m.sharpe},
      {"sortino", m.sortino},
      {"winrate", m.winRate},
      {"trades", static_cast<double>(m.numTrades)},
      {"numtrades", static_cast<double>(m.numTrades)},
      {"avgtrade", m.avgTrade},
      {"exposure", m.exposure},
  };
}

} // namespace topt
