/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "engine/Simulator.h"

#include <cmath>
#include <stdexcept>

namespace topt {

namespace {

struct OpenPosition {
  int dir = 0;
  int entryBar = 0;
  double entryPrice = 0.0;
  double entryCost = 0.0;
};

struct ExecState {
  double cash = 0.0;
  double posUnits = 0.0; // signed units currently held
  OpenPosition open;
  std::vector<Trade>* trades = nullptr;
  const ExecConfig* cfg = nullptr;
};

int sign_of(double v) { return v > 0.0 ? 1 : (v < 0.0 ? -1 : 0); }

// Executes a transition to `targetUnits` at base price `basePx` on bar `bar`.
void execute(ExecState& st, int bar, double basePx, double targetUnits) {
  const ExecConfig& cfg = *st.cfg;
  double delta = targetUnits - st.posUnits;
  if (delta == 0.0) return;

  double slip = cfg.slippageTicks * cfg.tickSize;
  double px = delta > 0.0 ? basePx + slip : basePx - slip;
  double notional = std::fabs(delta) * px;
  double commission = notional * cfg.commissionBps / 1e4 + cfg.commissionFixed;
  double perUnitCost = commission / std::fabs(delta);

  st.cash -= delta * px;
  st.cash -= commission;

  int oldSign = sign_of(st.posUnits);
  int newSign = sign_of(targetUnits);

  if (oldSign != 0 && newSign != oldSign) {
    // Closing the current episode (possibly the closing half of a reversal).
    double closingUnits = std::fabs(st.posUnits);
    double closeCost = perUnitCost * closingUnits;
    Trade t;
    t.entryBar = st.open.entryBar;
    t.exitBar = bar;
    t.dir = st.open.dir;
    t.entryPrice = st.open.entryPrice;
    t.exitPrice = px;
    t.pnl = (px - st.open.entryPrice) * st.open.dir * closingUnits -
            st.open.entryCost - closeCost;
    st.trades->push_back(t);
    st.open = OpenPosition{};
  }
  if (newSign != 0 && newSign != oldSign) {
    // Opening a new episode (possibly the opening half of a reversal).
    double openingUnits = std::fabs(targetUnits);
    st.open.dir = newSign;
    st.open.entryBar = bar;
    st.open.entryPrice = px;
    st.open.entryCost = perUnitCost * openingUnits;
  }
  st.posUnits = targetUnits;
}

} // namespace

SimResult simulate(const Series& bars, const Signal* desired, const ExecConfig& cfg,
                   SimRange range) {
  const int n = bars.size();
  if (n <= 0) throw std::invalid_argument("simulate: empty series");
  if (!desired) throw std::invalid_argument("simulate: null signals");
  if (cfg.qty <= 0.0) throw std::invalid_argument("simulate: qty must be positive");

  const int begin = range.begin;
  const int end = range.end < 0 ? n : range.end;
  if (begin < 0 || begin >= end || end > n) {
    throw std::invalid_argument("simulate: bad range [" + std::to_string(begin) + ", " +
                                std::to_string(end) + ") for " + std::to_string(n) +
                                " bars");
  }

  SimResult res;
  res.initialCapital = cfg.capital;
  res.equity.resize(static_cast<std::size_t>(end - begin));

  ExecState st;
  st.cash = cfg.capital;
  st.trades = &res.trades;
  st.cfg = &cfg;

  for (int i = begin; i < end; ++i) {
    if (i > 0) {
      // Fill the decision made at the close of bar i-1 at this bar's open
      // (for i == begin > 0 that is the boundary decision from before the
      // segment — formed on data the segment may legitimately see).
      int want = desired[i - 1];
      double target = static_cast<double>(want) * cfg.qty;
      execute(st, i, static_cast<double>(bars.open[static_cast<std::size_t>(i)]), target);
    }
    if (st.posUnits != 0.0) ++res.barsInPosition;
    res.equity[static_cast<std::size_t>(i - begin)] =
        st.cash + st.posUnits * static_cast<double>(bars.close[static_cast<std::size_t>(i)]);
  }

  if (st.posUnits != 0.0) {
    // Force-close at the segment's last close so the final trade completes.
    execute(st, end - 1, static_cast<double>(bars.close[static_cast<std::size_t>(end - 1)]), 0.0);
    res.equity.back() = st.cash;
  }

  res.finalEquity = res.equity.back();
  return res;
}

} // namespace topt
