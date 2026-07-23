/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_ENGINE_INDICATORCACHE_H
#define TACUDA_OPT_ENGINE_INDICATORCACHE_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/Backend.h"
#include "core/Types.h"

namespace topt {

// Host-side cache of computed indicator lines keyed by
// (family, input column, quantized parameters). Grid variants share
// overlapping requests — e.g. EMA(20) is needed by both (5,20) and (10,20)
// MA-cross variants — so each unique request hits the backend exactly once.
// get() is thread-safe; returned entry references stay valid until clear().
class IndicatorCache {
public:
  struct Entry {
    std::vector<std::vector<float>> lines; // one per family output
    std::vector<int> firstValid;           // per line: first finite index (n if none)
  };

  IndicatorCache(const Series& bars, IIndicatorBackend& backend);

  const Entry& get(const IndicatorRequest& req);

  // Exact textual key: no hash-collision aliasing between requests.
  static std::string key_of(const IndicatorRequest& req);
  // Bytes this request would add to the cache if it is not present yet.
  std::size_t bytes_for(const IndicatorRequest& req) const;

  std::size_t bytes() const { return bytes_; }
  std::size_t compute_count() const { return computeCount_; }
  void clear();

private:
  const Series& bars_;
  IIndicatorBackend& backend_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<Entry>> entries_;
  std::size_t bytes_ = 0;
  std::size_t computeCount_ = 0;
};

} // namespace topt

#endif // TACUDA_OPT_ENGINE_INDICATORCACHE_H
