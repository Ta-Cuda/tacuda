/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_STRATEGY_ISTRATEGY_H
#define TACUDA_OPT_STRATEGY_ISTRATEGY_H

#include <memory>
#include <string>
#include <vector>

#include "core/Types.h"

namespace topt {

// Read-only view of computed indicator lines for one variant:
// line(reqIdx, outIdx) is the output array of the strategy's
// indicators()[reqIdx], output number outIdx (BBANDS: 0=upper 1=middle 2=lower).
class IndicatorView {
public:
  explicit IndicatorView(std::vector<std::vector<const float*>> lines)
      : lines_(std::move(lines)) {}

  const float* line(std::size_t reqIdx, std::size_t outIdx = 0) const {
    return lines_.at(reqIdx).at(outIdx);
  }

private:
  std::vector<std::vector<const float*>> lines_;
};

// A trading strategy: declares tunable parameters, the indicators it needs
// for a given parameter assignment, and turns bars + indicators into
// per-bar target positions. Implementations must be stateless between
// signals() calls (one strategy object serves many variants in parallel).
class IStrategy {
public:
  virtual ~IStrategy() = default;

  virtual std::string name() const = 0;
  virtual std::vector<ParamDef> params() const = 0;

  // False for parameter combinations that make no sense (fast >= slow...).
  // Invalid variants are skipped by the optimizer and reported as such.
  virtual bool validate(const ParamSet& ps) const { (void)ps; return true; }

  virtual std::vector<IndicatorRequest> indicators(const ParamSet& ps) const = 0;

  // Fills out[0..n-1] with target positions decided at each bar's close.
  // The engine forces out[i] = 0 while any requested indicator is still in
  // its NaN warm-up, and executes out[i] at the open of bar i+1.
  virtual void signals(const Series& bars, const ParamSet& ps,
                       const IndicatorView& ind, Signal* out) const = 0;
};

// Built-in strategy registry.
std::vector<std::string> strategy_names();
std::unique_ptr<IStrategy> make_strategy(const std::string& name);

} // namespace topt

#endif // TACUDA_OPT_STRATEGY_ISTRATEGY_H
