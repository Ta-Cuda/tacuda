/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "engine/IndicatorCache.h"

#include <cmath>
#include <cstdio>

namespace topt {

IndicatorCache::IndicatorCache(const Series& bars, IIndicatorBackend& backend)
    : bars_(bars), backend_(backend) {}

std::string IndicatorCache::key_of(const IndicatorRequest& req) {
  std::string key;
  key.reserve(32);
  key += to_string(req.family);
  key += '|';
  key += to_string(req.input);
  for (double p : req.params) {
    char buf[32];
    // Quantize to 1e-6 so 2.0 and 2.0000000001 share an entry.
    std::snprintf(buf, sizeof(buf), "|%lld",
                  static_cast<long long>(std::llround(p * 1e6)));
    key += buf;
  }
  if (req.chain) {
    // Derived-series input: fold the whole upstream chain into the key.
    key += "|<(";
    key += key_of(*req.chain);
    key += ")#";
    key += std::to_string(req.chainLine);
  }
  return key;
}

std::size_t IndicatorCache::bytes_for(const IndicatorRequest& req) const {
  return static_cast<std::size_t>(family_outputs(req.family)) *
         static_cast<std::size_t>(bars_.size()) * sizeof(float);
}

const IndicatorCache::Entry& IndicatorCache::get(const IndicatorRequest& req) {
  std::string key = key_of(req);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(key);
  if (it != entries_.end()) {
    return *it->second;
  }

  const int n = bars_.size();
  const int nOut = family_outputs(req.family);
  auto entry = std::make_unique<Entry>();
  entry->lines.resize(static_cast<std::size_t>(nOut));
  std::vector<float*> ptrs;
  ptrs.reserve(static_cast<std::size_t>(nOut));
  for (auto& line : entry->lines) {
    line.resize(static_cast<std::size_t>(n));
    ptrs.push_back(line.data());
  }
  backend_.compute(req, ptrs);
  ++computeCount_;

  entry->firstValid.resize(static_cast<std::size_t>(nOut), n);
  for (int k = 0; k < nOut; ++k) {
    const auto& line = entry->lines[static_cast<std::size_t>(k)];
    for (int i = 0; i < n; ++i) {
      if (std::isfinite(line[static_cast<std::size_t>(i)])) {
        entry->firstValid[static_cast<std::size_t>(k)] = i;
        break;
      }
    }
  }

  bytes_ += bytes_for(req);
  auto [pos, inserted] = entries_.emplace(std::move(key), std::move(entry));
  (void)inserted;
  return *pos->second;
}

void IndicatorCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  bytes_ = 0;
}

} // namespace topt
