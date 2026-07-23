/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "TestBackend.h"
#include "backend/Backend.h"
#include "backend/BatchBackend.h"
#include "backend/TacudaBackend.h"
#include "backend/TacudaBatchBackend.h"
#include "backend/TalibBackend.h"
#include "data/Synthetic.h"
#include "engine/IndicatorCache.h"

using namespace topt;
using topt::testing::TestBackend;

namespace {

// |a-b| <= max(absTol, relTol*|b|), NaN must match NaN.
void expect_lines_close(const std::vector<float>& got, const std::vector<float>& ref,
                        const char* what, double absTol = 0.02, double relTol = 2e-3) {
  ASSERT_EQ(got.size(), ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    bool gNan = !std::isfinite(got[i]);
    bool rNan = !std::isfinite(ref[i]);
    ASSERT_EQ(gNan, rNan) << what << ": NaN mismatch at " << i;
    if (gNan) continue;
    double tol = std::max(absTol, relTol * std::fabs(static_cast<double>(ref[i])));
    ASSERT_NEAR(got[i], ref[i], tol) << what << " at index " << i;
  }
}

// One request per Family member, so parity tests cover everything the
// strategies (core + EXPERIMENTS pack) can ask a backend for. Recursive
// Wilder-smoothing families and the cumulative OBV accumulate more float
// error over 20k bars than windowed ones, hence per-request tolerances.
struct SampleRequest {
  IndicatorRequest req;
  double absTol = 0.02;
  double relTol = 2e-3;
};

std::vector<SampleRequest> sample_requests() {
  return {
      {{Family::SMA, Column::Close, {20.0}}},
      {{Family::EMA, Column::Close, {12.0}}},
      {{Family::RSI, Column::Close, {14.0}}, 0.05, 2e-3},
      {{Family::BBANDS, Column::Close, {20.0, 2.0, 2.0}}},
      {{Family::ATR, Column::Close, {14.0}}, 0.05, 2e-3},
      {{Family::WMA, Column::Close, {10.0}}},
      {{Family::TRIMA, Column::Close, {18.0}}},
      {{Family::MACD, Column::Close, {12.0, 26.0, 9.0}}},
      {{Family::STOCH, Column::Close, {14.0, 3.0}}, 0.05, 2e-3},
      {{Family::CCI, Column::Close, {20.0}}, 0.05, 2e-3},
      {{Family::ADX, Column::Close, {14.0}}, 0.15, 1e-2},
      {{Family::PLUS_DI, Column::Close, {14.0}}, 0.15, 1e-2},
      {{Family::MINUS_DI, Column::Close, {14.0}}, 0.15, 1e-2},
      {{Family::OBV, Column::Close, {}}, 50.0, 2e-3},
      {{Family::MOM, Column::Close, {10.0}}},
      {{Family::ROC, Column::Close, {12.0}}},
      {{Family::STDDEV, Column::Close, {20.0}}},
      {{Family::LINEARREG, Column::Close, {14.0}}},
      {{Family::LINEARREG_SLOPE, Column::Close, {14.0}}},
      {{Family::MAX, Column::High, {20.0}}},
      {{Family::MIN, Column::Low, {20.0}}},
      {{Family::TRANGE, Column::Close, {}}},
      // --- club pack (0.4.0) ---
      {{Family::ASI, Column::Close, {}}, 0.5, 2e-3},
      {{Family::SAR_EL, Column::Close, {0.02, 0.2}}, 0.05, 1e-3},
      {{Family::OBTR, Column::Close, {}}, 0.5, 2e-3},
      {{Family::WOBV, Column::Close, {}}, 500.0, 2e-3},
      {{Family::HPI, Column::Close, {0.1}}, 5.0, 2e-3},
      {{Family::MARKET_STRENGTH, Column::Close, {5.0}}, 0.05, 2e-3},
      {{Family::TRENDSCORE, Column::Close, {10.0}}, 0.0, 0.0},
      {{Family::REFDEV, Column::Close, {15.0, 15.0}}, 0.1, 2e-3},
      {{Family::SKEW, Column::Close, {15.0}}, 0.01, 2e-3},
      {{Family::KURTOSIS, Column::Close, {40.0}}, 0.01, 2e-3},
      {{Family::PIVOT_HIGH, Column::High, {2.0}}, 0.0, 0.0},
      {{Family::PIVOT_LOW, Column::Low, {2.0}}, 0.0, 0.0},
      {{Family::XAVERAGE, Column::Close, {9.0}}},
      {{Family::ROLLING_MEAN, Column::Close, {10.0}}},
      {{Family::ROLLING_MAX, Column::High, {10.0}}, 0.0, 0.0},
      {{Family::ROLLING_MIN, Column::Low, {10.0}}, 0.0, 0.0},
      {{Family::ROLLING_STD, Column::Close, {20.0}}},
      // --- derived-series chains ---
      {derived(Family::XAVERAGE, {9.0},
               IndicatorRequest{Family::OBTR, Column::Close, {}}),
       0.5, 2e-3},
      {derived(Family::ROLLING_MEAN, {25.0},
               IndicatorRequest{Family::WOBV, Column::Close, {}}),
       500.0, 2e-3},
      {derived(Family::ROLLING_MAX, {10.0},
               IndicatorRequest{Family::HPI, Column::Close, {0.1}}),
       5.0, 2e-3},
      {derived(Family::ROLLING_MEAN, {18.0},
               IndicatorRequest{Family::TRENDSCORE, Column::Close, {10.0}}),
       0.01, 1e-3},
  };
}

void compare_backend_to_reference(IIndicatorBackend& backend, const Series& bars) {
  TestBackend reference(bars);
  const int n = bars.size();
  for (const auto& sample : sample_requests()) {
    const IndicatorRequest& req = sample.req;
    const int nOut = family_outputs(req.family);
    std::vector<std::vector<float>> got(static_cast<std::size_t>(nOut)),
        ref(static_cast<std::size_t>(nOut));
    std::vector<float*> gotPtrs, refPtrs;
    for (int k = 0; k < nOut; ++k) {
      got[static_cast<std::size_t>(k)].resize(static_cast<std::size_t>(n));
      ref[static_cast<std::size_t>(k)].resize(static_cast<std::size_t>(n));
      gotPtrs.push_back(got[static_cast<std::size_t>(k)].data());
      refPtrs.push_back(ref[static_cast<std::size_t>(k)].data());
    }
    backend.compute(req, gotPtrs);
    reference.compute(req, refPtrs);
    for (int k = 0; k < nOut; ++k) {
      std::string what = std::string(backend.name()) + "/" + to_string(req.family) +
                         "[" + std::to_string(k) + "]";
      expect_lines_close(got[static_cast<std::size_t>(k)],
                         ref[static_cast<std::size_t>(k)], what.c_str(),
                         sample.absTol, sample.relTol);
    }
  }
}

} // namespace

TEST(Backends, TacudaMatchesReference) {
  if (!TacudaBackend::available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  Series bars = generate_synthetic(20000, 33);
  TacudaBackend backend(bars);
  compare_backend_to_reference(backend, bars);
}

TEST(Backends, TalibMatchesReference) {
  std::string why;
  if (!TalibBackend::available(why)) {
    GTEST_SKIP() << why;
  }
  Series bars = generate_synthetic(20000, 33);
  TalibBackend backend(bars);
  compare_backend_to_reference(backend, bars);
}

TEST(Backends, KindParsing) {
  EXPECT_EQ(parse_backend_kind("auto"), BackendKind::Auto);
  EXPECT_EQ(parse_backend_kind("tacuda"), BackendKind::Tacuda);
  EXPECT_EQ(parse_backend_kind("tacuda-device"), BackendKind::TacudaDevice);
  EXPECT_EQ(parse_backend_kind("talib"), BackendKind::Talib);
  EXPECT_EQ(parse_backend_kind("cpu"), BackendKind::Talib);
  EXPECT_THROW(parse_backend_kind("magic"), std::invalid_argument);
}

TEST(Backends, AutoFactorySelectsSomethingWhenPossible) {
  Series bars = generate_synthetic(256, 1);
  std::string why;
  bool anyAvailable = TacudaBackend::available() || TalibBackend::available(why);
  std::string log;
  if (anyAvailable) {
    auto backend = create_backend(BackendKind::Auto, bars, log);
    ASSERT_NE(backend, nullptr);
    EXPECT_FALSE(log.empty());
  } else {
    EXPECT_THROW(create_backend(BackendKind::Auto, bars, log), std::runtime_error);
  }
}

TEST(Backends, TacudaBatchMatchesPerSymbolReference) {
  if (!TacudaBackend::available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  const int n = 8192;
  const int numSymbols = 5;
  std::vector<Series> series;
  for (int s = 0; s < numSymbols; ++s) {
    series.push_back(generate_synthetic(n, 100 + static_cast<std::uint64_t>(s)));
  }
  TacudaBatchBackend batch(series);
  ASSERT_EQ(batch.num_series(), numSymbols);

  for (const auto& sample : sample_requests()) {
    const IndicatorRequest& req = sample.req;
    const int nOut = family_outputs(req.family);
    std::vector<std::vector<std::vector<float>>> got(
        static_cast<std::size_t>(numSymbols),
        std::vector<std::vector<float>>(static_cast<std::size_t>(nOut),
                                        std::vector<float>(static_cast<std::size_t>(n))));
    std::vector<std::vector<float*>> outs(static_cast<std::size_t>(numSymbols));
    for (int s = 0; s < numSymbols; ++s) {
      for (int k = 0; k < nOut; ++k) {
        outs[static_cast<std::size_t>(s)].push_back(
            got[static_cast<std::size_t>(s)][static_cast<std::size_t>(k)].data());
      }
    }
    batch.compute(req, outs);

    for (int s = 0; s < numSymbols; ++s) {
      TestBackend reference(series[static_cast<std::size_t>(s)]);
      std::vector<std::vector<float>> ref(
          static_cast<std::size_t>(nOut),
          std::vector<float>(static_cast<std::size_t>(n)));
      std::vector<float*> refPtrs;
      for (int k = 0; k < nOut; ++k) {
        refPtrs.push_back(ref[static_cast<std::size_t>(k)].data());
      }
      reference.compute(req, refPtrs);
      for (int k = 0; k < nOut; ++k) {
        std::string what = "batch/" + std::string(to_string(req.family)) +
                           " sym" + std::to_string(s) + "[" + std::to_string(k) +
                           "]";
        expect_lines_close(got[static_cast<std::size_t>(s)][static_cast<std::size_t>(k)],
                           ref[static_cast<std::size_t>(k)], what.c_str(),
                           sample.absTol, sample.relTol);
      }
    }
  }
}

TEST(Backends, BatchFactoryPerSymbolFallbackWorks) {
  std::string why;
  if (!TacudaBackend::available() && !TalibBackend::available(why)) {
    GTEST_SKIP() << "no batch backend available: " << why;
  }
  std::vector<Series> series;
  for (int s = 0; s < 3; ++s) {
    series.push_back(generate_synthetic(512, 5 + static_cast<std::uint64_t>(s)));
  }
  std::string log;
  auto backend = create_batch_backend(BackendKind::Auto, series, log);
  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->num_series(), 3);
  EXPECT_FALSE(log.empty());
}

TEST(Backends, DerivedChainMatchesManualTwoStep) {
  Series bars = generate_synthetic(4000, 7);
  const int n = bars.size();
  TestBackend backend(bars);

  IndicatorRequest upstream{Family::WOBV, Column::Close, {}};
  IndicatorRequest chained = derived(Family::ROLLING_MEAN, {25.0}, upstream);

  std::vector<float> viaChain(static_cast<std::size_t>(n));
  std::vector<float*> outs{viaChain.data()};
  backend.compute(chained, outs);

  std::vector<float> wobv(static_cast<std::size_t>(n));
  std::vector<float*> upOuts{wobv.data()};
  backend.compute(upstream, upOuts);
  std::vector<float> manual(static_cast<std::size_t>(n));
  club::rolling_strict(wobv.data(), n, 25, 0, manual.data());

  expect_lines_close(viaChain, manual, "chained ROLLING_MEAN(WOBV)", 1e-6,
                     1e-9);
}

TEST(Backends, ChainKeysAreDistinct) {
  IndicatorRequest plain{Family::ROLLING_MEAN, Column::Close, {25.0}};
  IndicatorRequest overWobv = derived(
      Family::ROLLING_MEAN, {25.0},
      IndicatorRequest{Family::WOBV, Column::Close, {}});
  IndicatorRequest overObtr = derived(
      Family::ROLLING_MEAN, {25.0},
      IndicatorRequest{Family::OBTR, Column::Close, {}});
  const std::string k0 = IndicatorCache::key_of(plain);
  const std::string k1 = IndicatorCache::key_of(overWobv);
  const std::string k2 = IndicatorCache::key_of(overObtr);
  EXPECT_NE(k0, k1);
  EXPECT_NE(k1, k2);
  // Identical chains share a key (cache dedup across variants).
  EXPECT_EQ(k1, IndicatorCache::key_of(derived(
                    Family::ROLLING_MEAN, {25.0},
                    IndicatorRequest{Family::WOBV, Column::Close, {}})));
}

TEST(Backends, ChainValidationRejectsBadRequests) {
  // Non-chainable downstream family.
  IndicatorRequest bad =
      derived(Family::SMA, {20.0},
              IndicatorRequest{Family::OBTR, Column::Close, {}});
  EXPECT_THROW(validate_chain(bad), std::invalid_argument);

  // Multi-output upstream.
  IndicatorRequest badUp =
      derived(Family::ROLLING_MEAN, {10.0},
              IndicatorRequest{Family::BBANDS, Column::Close, {20.0, 2.0, 2.0}});
  EXPECT_THROW(validate_chain(badUp), std::invalid_argument);

  // Valid chain passes.
  IndicatorRequest ok =
      derived(Family::XAVERAGE, {9.0},
              IndicatorRequest{Family::OBTR, Column::Close, {}});
  EXPECT_NO_THROW(validate_chain(ok));
}
