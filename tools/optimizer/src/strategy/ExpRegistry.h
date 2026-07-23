/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_STRATEGY_EXPREGISTRY_H
#define TACUDA_OPT_STRATEGY_EXPREGISTRY_H

// Registration glue for the EXPERIMENTS strategy pack. Each ExpVolNN.cpp
// contributes the systems of one STAD volume; Strategies.cpp aggregates
// them behind strategy_names()/make_strategy().

#include <memory>
#include <vector>

#include "strategy/IStrategy.h"

namespace topt::exp {

struct Factory {
  const char* name;
  std::unique_ptr<IStrategy> (*make)();
};

void register_vol01(std::vector<Factory>& out);
void register_vol02(std::vector<Factory>& out);
void register_vol03(std::vector<Factory>& out);
void register_vol04(std::vector<Factory>& out);
void register_vol05(std::vector<Factory>& out);
void register_vol06(std::vector<Factory>& out);
void register_vol07(std::vector<Factory>& out);
void register_vol08(std::vector<Factory>& out);
void register_vol09(std::vector<Factory>& out);
void register_vol10(std::vector<Factory>& out);
void register_vol11(std::vector<Factory>& out);
void register_vol12(std::vector<Factory>& out);
void register_vol13(std::vector<Factory>& out);

} // namespace topt::exp

#endif // TACUDA_OPT_STRATEGY_EXPREGISTRY_H
