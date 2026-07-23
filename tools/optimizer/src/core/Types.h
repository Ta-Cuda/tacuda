/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_CORE_TYPES_H
#define TACUDA_OPT_CORE_TYPES_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace topt {

// OHLCV column a request reads its input from.
enum class Column : std::uint8_t { Open = 0, High, Low, Close, Volume };

// Indicator families the optimizer knows how to compute on any backend.
// Families that read fixed OHLCV columns (ATR, ADX, CCI, STOCH, TRANGE,
// PLUS_DI, MINUS_DI use High/Low/Close; OBV uses Close+Volume) ignore
// IndicatorRequest::input.
enum class Family : std::uint8_t {
  SMA = 0,
  EMA,
  RSI,
  BBANDS,          // params: period, devUp, devDn -> upper, middle, lower
  ATR,             // params: period (H/L/C)
  WMA,             // params: period
  TRIMA,           // params: period (triangular MA, EasyLanguage TriAverage)
  MACD,            // params: fast, slow, signal -> macd, signal, hist
  STOCH,           // params: kPeriod, dPeriod -> slowK, slowD (H/L/C)
  CCI,             // params: period (H/L/C)
  ADX,             // params: period (H/L/C)
  PLUS_DI,         // params: period (H/L/C)
  MINUS_DI,        // params: period (H/L/C)
  OBV,             // params: none (Close+Volume)
  MOM,             // params: period (price - price[period])
  ROC,             // params: period (percent rate of change)
  STDDEV,          // params: period (population stddev, nbdev=1)
  LINEARREG,       // params: period (regression value at window end)
  LINEARREG_SLOPE, // params: period
  MAX,             // params: period (rolling highest)
  MIN,             // params: period (rolling lowest)
  TRANGE,          // params: none (H/L/C true range)
  // --- STAD club pack (native ct_* club kernels since 0.4.0) ---
  ASI,             // params: none (O/H/L/C accumulation swing index)
  SAR_EL,          // params: accFact, maxAccel (H/L/C EasyLanguage parabolic)
  OBTR,            // params: none (H/L/C on-balance true range)
  WOBV,            // params: none (O/H/L/C/V weighted OBV)
  HPI,             // params: factor (H/L/V Herrick-Payoff proxy)
  MARKET_STRENGTH, // params: period
  TRENDSCORE,      // params: lookback (Chande TrendScore, 10 comparisons)
  REFDEV,          // params: smaPeriod, sumPeriod (reference deviation RDV)
  SKEW,            // params: period (rolling population skewness)
  KURTOSIS,        // params: period (rolling population excess kurtosis)
  PIVOT_HIGH,      // params: strength (confirmed swing high level, else NaN)
  PIVOT_LOW,       // params: strength (confirmed swing low level, else NaN)
  XAVERAGE,        // params: period (EL XAverage; NaN-prefix aware)
  ROLLING_MEAN,    // params: period (NaN-strict window mean)
  ROLLING_MAX,     // params: period (NaN-strict window max)
  ROLLING_MIN,     // params: period (NaN-strict window min)
  ROLLING_STD,     // params: period (NaN-strict window population stddev)
};

const char* to_string(Column c);
const char* to_string(Family f);

// Number of output lines an indicator family produces (BBANDS -> 3).
int family_outputs(Family f);

// True for families that read one input line (req.input column or a chained
// upstream line) as opposed to fixed OHLCV column sets.
bool family_single_input(Family f);

// True for families that may consume a chained (derived) input line: the
// club-pack single-input families, whose kernels handle NaN warm-up
// prefixes of derived series explicitly.
bool family_chainable(Family f);

// In-memory OHLCV series. Timestamps are unix seconds (0 when absent).
struct Series {
  std::vector<std::int64_t> ts;
  std::vector<float> open, high, low, close, volume;

  int size() const { return static_cast<int>(close.size()); }
  const std::vector<float>& column(Column c) const;
};

// A concrete parameter assignment, ordered as the strategy declares them.
struct ParamSet {
  std::vector<std::pair<std::string, double>> values;

  double get(const std::string& name) const;
  int get_int(const std::string& name) const;
  bool has(const std::string& name) const;
  void set(const std::string& name, double v);
  std::string to_string() const; // "fast=12 slow=48"
};

// Parameter declared by a strategy: default value plus grid bounds.
struct ParamDef {
  std::string name;
  double defval = 0.0;
  double minv = 0.0;
  double maxv = 0.0;
  double step = 1.0;
  bool integer = true;
};

// Indicator computation requested by a strategy for a given ParamSet.
// A request may chain on another request's output line ("EMA of OBTR"):
// set `chain` to the upstream request and `chainLine` to its output line.
// Chaining is limited to family_chainable() downstream families over
// single-output upstream requests; backends compute the whole chain,
// GPU backends without host round-trips.
struct IndicatorRequest {
  Family family = Family::EMA;
  Column input = Column::Close;
  std::vector<double> params; // family-specific scalar parameters
  std::shared_ptr<const IndicatorRequest> chain; // derived-series input
  int chainLine = 0;
};

// Convenience factory for chained requests.
inline IndicatorRequest derived(Family family, std::vector<double> params,
                                IndicatorRequest upstream, int line = 0) {
  IndicatorRequest req;
  req.family = family;
  req.params = std::move(params);
  req.chain = std::make_shared<const IndicatorRequest>(std::move(upstream));
  req.chainLine = line;
  return req;
}

// Throws std::invalid_argument when a chained request violates the chaining
// contract (non-chainable downstream family, multi-output upstream, bad
// line index).
void validate_chain(const IndicatorRequest& req);

// Desired position per bar: -1 short, 0 flat, +1 long.
using Signal = std::int8_t;

} // namespace topt

#endif // TACUDA_OPT_CORE_TYPES_H
