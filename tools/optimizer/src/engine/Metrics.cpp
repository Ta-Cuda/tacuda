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

  // Drawdowns over the close-marked equity curve. The peak starts at the
  // funded capital: a ranged segment can already trade on its first bar, in
  // which case eq[0] contains a result rather than the starting point.
  double peak = sim.initialCapital;
  for (int i = 0; i < n; ++i) {
    peak = std::max(peak, eq[static_cast<std::size_t>(i)]);
    double dd = peak - eq[static_cast<std::size_t>(i)];
    m.maxDrawdown = std::max(m.maxDrawdown, dd);
    if (peak > 0.0) {
      m.maxDrawdownPct = std::max(m.maxDrawdownPct, dd / peak);
    }
  }

  // Sharpe / Sortino from per-bar equity differences (population std). The
  // first bar is an observation only when it moved equity away from the funded
  // capital, so a run that starts flat keeps its n-1 differences.
  const double firstMove = eq[0] - sim.initialCapital;
  const double moveEps = 1e-9 * std::max(1.0, std::fabs(sim.initialCapital));
  const int first = std::fabs(firstMove) > moveEps ? 0 : 1;
  const int k = n - first;
  if (k >= 1) {
    const auto delta = [&](int i) {
      return i == 0 ? firstMove
                    : eq[static_cast<std::size_t>(i)] - eq[static_cast<std::size_t>(i - 1)];
    };
    double mean = 0.0;
    for (int i = first; i < n; ++i) mean += delta(i);
    mean /= k;
    double var = 0.0, downVar = 0.0;
    for (int i = first; i < n; ++i) {
      double d = delta(i);
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
