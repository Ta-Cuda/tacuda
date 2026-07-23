/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "data/Synthetic.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace topt {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Series generate_synthetic(int bars, std::uint64_t seed) {
  if (bars <= 0) {
    throw std::invalid_argument("synthetic: bar count must be positive");
  }
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  Series s;
  s.ts.reserve(bars);
  s.open.reserve(bars);
  s.high.reserve(bars);
  s.low.reserve(bars);
  s.close.reserve(bars);
  s.volume.reserve(bars);

  const double vol = 0.002;          // per-bar log volatility
  const double cycleAmp = 0.0006;    // regime cycle strength
  const double cyclePeriod = 900.0;  // bars per trend cycle
  double logPrice = std::log(100.0);
  std::int64_t ts = 1546300800; // 2019-01-01 00:00:00 UTC (Tuesday)
  double prevClose = std::exp(logPrice);
  // Daily bars, weekends skipped, so weekday-aware strategies see a
  // realistic Mon..Fri calendar. day 0 of the unix epoch was a Thursday.
  auto next_trading_day = [](std::int64_t t) {
    t += 86400;
    int dow = static_cast<int>(((t / 86400) % 7 + 7 + 4) % 7); // 0=Sun..6=Sat
    if (dow == 6) t += 2 * 86400;      // Saturday -> Monday
    else if (dow == 0) t += 86400;     // Sunday -> Monday
    return t;
  };

  for (int i = 0; i < bars; ++i) {
    double drift = cycleAmp * std::sin(2.0 * kPi * static_cast<double>(i) / cyclePeriod);
    logPrice += drift + vol * noise(rng);
    double close = std::exp(logPrice);
    double open = prevClose * (1.0 + 0.0003 * noise(rng));
    double body_hi = std::max(open, close);
    double body_lo = std::min(open, close);
    double high = body_hi * (1.0 + 0.0008 * uni(rng));
    double low = body_lo * (1.0 - 0.0008 * uni(rng));
    double volume = 1000.0 + 500.0 * uni(rng);

    s.ts.push_back(ts);
    s.open.push_back(static_cast<float>(open));
    s.high.push_back(static_cast<float>(high));
    s.low.push_back(static_cast<float>(low));
    s.close.push_back(static_cast<float>(close));
    s.volume.push_back(static_cast<float>(volume));

    prevClose = close;
    ts = next_trading_day(ts);
  }
  return s;
}

} // namespace topt
