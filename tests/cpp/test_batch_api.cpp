/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include <tacuda_device.h>
#include "test_utils.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <vector>

namespace {

std::vector<float> make_series(int seriesLen, int seed) {
  std::vector<float> v(seriesLen);
  for (int i = 0; i < seriesLen; ++i) {
    v[i] = 100.0f + 10.0f * std::sin(0.05f * i + seed) +
           0.3f * std::cos(0.7f * i * (seed + 1));
  }
  return v;
}

// numSeries independent series packed row-by-row (series-major matrix)
std::vector<float> make_matrix(int seriesLen, int numSeries, int seedBase) {
  std::vector<float> m(static_cast<size_t>(seriesLen) * numSeries);
  for (int s = 0; s < numSeries; ++s) {
    auto series = make_series(seriesLen, seedBase + s);
    std::copy(series.begin(), series.end(),
              m.begin() + static_cast<size_t>(s) * seriesLen);
  }
  return m;
}

std::vector<float> row(const std::vector<float>& m, int seriesLen, int s) {
  auto begin = m.begin() + static_cast<size_t>(s) * seriesLen;
  return {begin, begin + seriesLen};
}

}  // namespace

TEST(BatchAPI, SmaMatchesSingleSeries) {
  const int seriesLen = 1024, numSeries = 6, period = 20;
  auto inputs = make_matrix(seriesLen, numSeries, 1);
  std::vector<float> outputs(inputs.size(), 0.0f);

  ASSERT_EQ(ct_sma_batch(inputs.data(), outputs.data(), seriesLen, numSeries,
                         period, /*maxConcurrency=*/3),
            CT_STATUS_SUCCESS);

  for (int s = 0; s < numSeries; ++s) {
    auto in = row(inputs, seriesLen, s);
    std::vector<float> ref(seriesLen, 0.0f);
    ASSERT_EQ(ct_sma(in.data(), ref.data(), seriesLen, period),
              CT_STATUS_SUCCESS);
    expect_approx_equal(row(outputs, seriesLen, s), ref);
  }
}

TEST(BatchAPI, RsiMatchesSingleSeries) {
  const int seriesLen = 2048, numSeries = 5, period = 14;
  auto inputs = make_matrix(seriesLen, numSeries, 7);
  std::vector<float> outputs(inputs.size(), 0.0f);

  ASSERT_EQ(ct_rsi_batch(inputs.data(), outputs.data(), seriesLen, numSeries,
                         period, 0),
            CT_STATUS_SUCCESS);

  for (int s = 0; s < numSeries; ++s) {
    auto in = row(inputs, seriesLen, s);
    std::vector<float> ref(seriesLen, 0.0f);
    ASSERT_EQ(ct_rsi(in.data(), ref.data(), seriesLen, period),
              CT_STATUS_SUCCESS);
    expect_approx_equal(row(outputs, seriesLen, s), ref);
  }
}

TEST(BatchAPI, AtrMatchesSingleSeries) {
  const int seriesLen = 1500, numSeries = 4, period = 14;
  auto closes = make_matrix(seriesLen, numSeries, 3);
  auto highs = closes;
  auto lows = closes;
  for (size_t i = 0; i < closes.size(); ++i) {
    highs[i] += 1.5f;
    lows[i] -= 1.5f;
  }
  std::vector<float> outputs(closes.size(), 0.0f);

  ASSERT_EQ(ct_atr_batch(highs.data(), lows.data(), closes.data(),
                         outputs.data(), seriesLen, numSeries, period, 0.0f,
                         2),
            CT_STATUS_SUCCESS);

  for (int s = 0; s < numSeries; ++s) {
    auto h = row(highs, seriesLen, s);
    auto l = row(lows, seriesLen, s);
    auto c = row(closes, seriesLen, s);
    std::vector<float> ref(seriesLen, 0.0f);
    ASSERT_EQ(ct_atr(h.data(), l.data(), c.data(), ref.data(), seriesLen,
                     period, 0.0f),
              CT_STATUS_SUCCESS);
    expect_approx_equal(row(outputs, seriesLen, s), ref);
  }
}

TEST(BatchAPI, MacdMatchesSingleSeries) {
  const int seriesLen = 1024, numSeries = 3;
  auto inputs = make_matrix(seriesLen, numSeries, 11);
  std::vector<float> macds(inputs.size()), signals(inputs.size()),
      hists(inputs.size());

  ASSERT_EQ(ct_macd_batch(inputs.data(), macds.data(), signals.data(),
                          hists.data(), seriesLen, numSeries, 12, 26, 9,
                          CT_MA_EMA, 0),
            CT_STATUS_SUCCESS);

  for (int s = 0; s < numSeries; ++s) {
    auto in = row(inputs, seriesLen, s);
    std::vector<float> m(seriesLen), sg(seriesLen), h(seriesLen);
    ASSERT_EQ(ct_macd(in.data(), m.data(), sg.data(), h.data(), seriesLen, 12,
                      26, 9, CT_MA_EMA),
              CT_STATUS_SUCCESS);
    expect_approx_equal(row(macds, seriesLen, s), m);
    expect_approx_equal(row(signals, seriesLen, s), sg);
    expect_approx_equal(row(hists, seriesLen, s), h);
  }
}

TEST(BatchAPI, BbandsAndStochasticMatchSingleSeries) {
  const int seriesLen = 900, numSeries = 4, period = 20;
  auto inputs = make_matrix(seriesLen, numSeries, 5);
  std::vector<float> up(inputs.size()), mid(inputs.size()), lo(inputs.size());

  ASSERT_EQ(ct_bbands_batch(inputs.data(), up.data(), mid.data(), lo.data(),
                            seriesLen, numSeries, period, 2.0f, 2.0f, 0),
            CT_STATUS_SUCCESS);

  for (int s = 0; s < numSeries; ++s) {
    auto in = row(inputs, seriesLen, s);
    std::vector<float> ru(seriesLen), rm(seriesLen), rl(seriesLen);
    ASSERT_EQ(ct_bbands(in.data(), ru.data(), rm.data(), rl.data(), seriesLen,
                        period, 2.0f, 2.0f),
              CT_STATUS_SUCCESS);
    expect_approx_equal(row(up, seriesLen, s), ru);
    expect_approx_equal(row(mid, seriesLen, s), rm);
    expect_approx_equal(row(lo, seriesLen, s), rl);
  }

  auto closes = inputs;
  auto highs = closes;
  auto lows = closes;
  for (size_t i = 0; i < closes.size(); ++i) {
    highs[i] += 2.0f;
    lows[i] -= 2.0f;
  }
  std::vector<float> ks(closes.size()), ds(closes.size());
  ASSERT_EQ(ct_stochastic_batch(highs.data(), lows.data(), closes.data(),
                                ks.data(), ds.data(), seriesLen, numSeries, 14,
                                3, 0),
            CT_STATUS_SUCCESS);
  for (int s = 0; s < numSeries; ++s) {
    auto h = row(highs, seriesLen, s);
    auto l = row(lows, seriesLen, s);
    auto c = row(closes, seriesLen, s);
    std::vector<float> rk(seriesLen), rd(seriesLen);
    ASSERT_EQ(ct_stochastic(h.data(), l.data(), c.data(), rk.data(), rd.data(),
                            seriesLen, 14, 3),
              CT_STATUS_SUCCESS);
    expect_approx_equal(row(ks, seriesLen, s), rk);
    expect_approx_equal(row(ds, seriesLen, s), rd);
  }
}

TEST(BatchAPI, DeviceBatchMatchesHostBatch) {
  const int seriesLen = 1024, numSeries = 4, period = 10;
  auto inputs = make_matrix(seriesLen, numSeries, 9);
  const int total = seriesLen * numSeries;

  std::vector<float> hostOut(inputs.size(), 0.0f);
  ASSERT_EQ(ct_ema_batch(inputs.data(), hostOut.data(), seriesLen, numSeries,
                         period, 0),
            CT_STATUS_SUCCESS);

  float* dIn = ct_device_alloc(total);
  float* dOut = ct_device_alloc(total);
  ASSERT_NE(dIn, nullptr);
  ASSERT_NE(dOut, nullptr);
  ASSERT_EQ(ct_device_upload(dIn, inputs.data(), total, 0), CT_STATUS_SUCCESS);

  ASSERT_EQ(ct_ema_batch_d(dIn, dOut, seriesLen, numSeries, period, 3),
            CT_STATUS_SUCCESS);

  std::vector<float> deviceOut(inputs.size(), 0.0f);
  ASSERT_EQ(ct_device_download(deviceOut.data(), dOut, total, 0),
            CT_STATUS_SUCCESS);

  ct_device_free(dIn);
  ct_device_free(dOut);

  expect_approx_equal(deviceOut, hostOut);
}

TEST(BatchAPI, InvalidArgumentsRejected) {
  std::vector<float> buf(64, 1.0f);
  EXPECT_EQ(ct_sma_batch(nullptr, buf.data(), 32, 2, 5, 0),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_batch(buf.data(), buf.data(), 0, 2, 5, 0),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_batch(buf.data(), buf.data(), 32, 0, 5, 0),
            CT_STATUS_INVALID_ARGUMENT);
}

TEST(BatchAPI, VersionStringAvailable) {
  const char* v = ct_version();
  ASSERT_NE(v, nullptr);
  EXPECT_STREQ(v, CT_VERSION_STRING);
}

// ---------------------------------------------------------------------------
// Generated batch families (scripts/generate_batch_api.py): each host batch
// call must equal a per-series loop over the underlying ct_*_d function.
// ---------------------------------------------------------------------------

namespace {

struct BatchCase {
  const char* name;
  int numInputs;   // 1..4 taken from {closeM, highM, lowM, volumeM/openM}
  int numOutputs;  // 1..3
  // batch(hostIns, hostOuts, seriesLen, numSeries)
  std::function<ctStatus_t(const float* const*, float* const*, int, int)> batch;
  // singleD(deviceIns, deviceOuts, seriesLen) - per-series device call
  std::function<ctStatus_t(const float* const*, float* const*, int)> singleD;
};

void run_generated_case(const BatchCase& tc, int seriesLen, int numSeries) {
  SCOPED_TRACE(tc.name);
  auto base = make_matrix(seriesLen, numSeries, 21);
  auto high = base, low = base, open = base, volume = base;
  for (size_t i = 0; i < base.size(); ++i) {
    high[i] += 1.5f;
    low[i] -= 1.5f;
    open[i] += 0.25f;
    volume[i] = 1000.0f + 5.0f * (i % 97);
  }
  // Input slot order per family shape (see generate_batch_api.py):
  //   1 input : close
  //   2 inputs: high, low (aroon/medprice/...) or close, volume (obv)
  //   3 inputs: high, low, close
  //   4 inputs: high, low, close, volume (mfi/ad/adosc) or
  //             open, high, low, close (avgprice - handled by caller order)
  std::vector<const std::vector<float>*> slots;
  if (tc.numInputs == 1) {
    slots = {&base};
  } else if (tc.numInputs == 2) {
    slots = {&high, &low};
  } else if (tc.numInputs == 3) {
    slots = {&high, &low, &base};
  } else {
    slots = {&high, &low, &base, &volume};
  }

  const float* hostIns[4] = {};
  for (int j = 0; j < tc.numInputs; ++j) hostIns[j] = slots[j]->data();

  std::vector<std::vector<float>> outs(
      tc.numOutputs, std::vector<float>(base.size(), 0.0f));
  float* hostOuts[3] = {};
  for (int j = 0; j < tc.numOutputs; ++j) hostOuts[j] = outs[j].data();

  ASSERT_EQ(tc.batch(hostIns, hostOuts, seriesLen, numSeries),
            CT_STATUS_SUCCESS);

  // Reference: per-series device round-trip through the same ct_*_d call.
  std::vector<float*> dIn(tc.numInputs), dOut(tc.numOutputs);
  for (auto& p : dIn) ASSERT_NE(p = ct_device_alloc(seriesLen), nullptr);
  for (auto& p : dOut) ASSERT_NE(p = ct_device_alloc(seriesLen), nullptr);

  for (int s = 0; s < numSeries; ++s) {
    const float* ins[4] = {};
    float* douts[3] = {};
    for (int j = 0; j < tc.numInputs; ++j) {
      auto r = row(*slots[j], seriesLen, s);
      ASSERT_EQ(ct_device_upload(dIn[j], r.data(), seriesLen, 0),
                CT_STATUS_SUCCESS);
      ins[j] = dIn[j];
    }
    for (int j = 0; j < tc.numOutputs; ++j) douts[j] = dOut[j];
    ASSERT_EQ(tc.singleD(ins, douts, seriesLen), CT_STATUS_SUCCESS);
    for (int j = 0; j < tc.numOutputs; ++j) {
      std::vector<float> ref(seriesLen, 0.0f);
      ASSERT_EQ(ct_device_download(ref.data(), dOut[j], seriesLen, 0),
                CT_STATUS_SUCCESS);
      SCOPED_TRACE("series " + std::to_string(s) + " output " +
                   std::to_string(j));
      expect_approx_equal(row(outs[j], seriesLen, s), ref);
    }
  }
  for (auto p : dIn) ct_device_free(p);
  for (auto p : dOut) ct_device_free(p);
}

#define BATCH1(NAME, ...)                                                     \
  BatchCase {                                                                 \
    #NAME, 1, 1,                                                              \
        [](const float* const* in, float* const* out, int len, int n) {      \
          return ct_##NAME##_batch(in[0], out[0], len, n, ##__VA_ARGS__, 0); \
        },                                                                    \
        [](const float* const* in, float* const* out, int len) {             \
          return ct_##NAME##_d(in[0], out[0], len, ##__VA_ARGS__, 0);        \
        }                                                                     \
  }

#define BATCH_HLC(NAME, ...)                                                  \
  BatchCase {                                                                 \
    #NAME, 3, 1,                                                              \
        [](const float* const* in, float* const* out, int len, int n) {      \
          return ct_##NAME##_batch(in[0], in[1], in[2], out[0], len, n,      \
                                   ##__VA_ARGS__, 0);                        \
        },                                                                    \
        [](const float* const* in, float* const* out, int len) {             \
          return ct_##NAME##_d(in[0], in[1], in[2], out[0], len,             \
                               ##__VA_ARGS__, 0);                            \
        }                                                                     \
  }

#define BATCH_HLCV(NAME, ...)                                                 \
  BatchCase {                                                                 \
    #NAME, 4, 1,                                                              \
        [](const float* const* in, float* const* out, int len, int n) {      \
          return ct_##NAME##_batch(in[0], in[1], in[2], in[3], out[0], len,  \
                                   n, ##__VA_ARGS__, 0);                     \
        },                                                                    \
        [](const float* const* in, float* const* out, int len) {             \
          return ct_##NAME##_d(in[0], in[1], in[2], in[3], out[0], len,      \
                               ##__VA_ARGS__, 0);                            \
        }                                                                     \
  }

}  // namespace

TEST(BatchAPIGenerated, SingleInputFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  const std::vector<BatchCase> cases = {
      BATCH1(wma, 10), BATCH1(dema, 10), BATCH1(tema, 10), BATCH1(trima, 10),
      BATCH1(momentum, 10), BATCH1(roc, 10), BATCH1(rocp, 10),
      BATCH1(rocr, 10), BATCH1(rocr100, 10), BATCH1(cmo, 14),
      BATCH1(stddev, 20), BATCH1(var, 20), BATCH1(sum, 10), BATCH1(max, 10),
      BATCH1(min, 10), BATCH1(linearreg, 14), BATCH1(linearreg_slope, 14),
      BATCH1(linearreg_intercept, 14), BATCH1(linearreg_angle, 14),
      BATCH1(tsf, 14), BATCH1(kama, 10, 2, 30), BATCH1(t3, 5, 0.7f),
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, HlcFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  const std::vector<BatchCase> cases = {
      BATCH_HLC(natr, 14), BATCH_HLC(adx, 14), BATCH_HLC(adxr, 14),
      BATCH_HLC(dx, 14), BATCH_HLC(plus_di, 14), BATCH_HLC(minus_di, 14),
      BATCH_HLC(willr, 14), BATCH_HLC(cci, 14), BATCH_HLC(trange),
      BATCH_HLC(typprice), BATCH_HLC(wclprice), BATCH_HLC(ultosc, 7, 14, 28),
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, VolumeAndPriceFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  const std::vector<BatchCase> cases = {
      BATCH_HLCV(mfi, 14), BATCH_HLCV(ad), BATCH_HLCV(adosc, 3, 10),
      // obv: 2 inputs = close, volume; reuse HL slots via explicit lambdas
      BatchCase{"obv", 2, 1,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_obv_batch(in[0], in[1], out[0], len, n, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_obv_d(in[0], in[1], out[0], len, 0);
                }},
      BatchCase{"medprice", 2, 1,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_medprice_batch(in[0], in[1], out[0], len, n, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_medprice_d(in[0], in[1], out[0], len, 0);
                }},
      BatchCase{"aroonosc", 2, 1,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_aroonosc_batch(in[0], in[1], out[0], len, n, 14,
                                           0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_aroonosc_d(in[0], in[1], out[0], len, 14, 0);
                }},
      BatchCase{"avgprice", 4, 1,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_avgprice_batch(in[0], in[1], in[2], in[3], out[0],
                                           len, n, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_avgprice_d(in[0], in[1], in[2], in[3], out[0],
                                       len, 0);
                }},
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, MultiOutputFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  const std::vector<BatchCase> cases = {
      BatchCase{"aroon", 2, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_aroon_batch(in[0], in[1], out[0], out[1], len, n,
                                        14, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_aroon_d(in[0], in[1], out[0], out[1], len, 14, 0);
                }},
      BatchCase{"stochf", 3, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_stochf_batch(in[0], in[1], in[2], out[0], out[1],
                                         len, n, 14, 3, 0, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_stochastic_fast_d(in[0], in[1], in[2], out[0],
                                              out[1], len, 14, 3, 0, 0);
                }},
      BatchCase{"macdext", 1, 3,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_macdext_batch(in[0], out[0], out[1], out[2], len,
                                          n, 12, 1, 26, 1, 9, 1, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_macdext_d(in[0], out[0], out[1], out[2], len, 12,
                                      1, 26, 1, 9, 1, 0);
                }},
      BatchCase{"accbands", 3, 3,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_accbands_batch(in[0], in[1], in[2], out[0], out[1],
                                           out[2], len, n, 20, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_accbands_d(in[0], in[1], in[2], out[0], out[1],
                                       out[2], len, 20, 0);
                }},
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

// --- families whose ct_*_d wrappers are generated in 0.4.0 -----------------

TEST(BatchAPIGenerated, NewSingleInputFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  const std::vector<BatchCase> cases = {
      BATCH1(trix, 10), BATCH1(change, 10), BATCH1(midpoint, 14),
      BATCH1(avgdev, 14), BATCH1(maxindex, 14), BATCH1(minindex, 14),
      BATCH1(apo, 12, 26), BATCH1(ppo, 12, 26), BATCH1(pvo, 12, 26),
      BATCH1(ma, 10, 1), BATCH1(macd_line, 12, 26),
      BATCH1(ht_dcperiod), BATCH1(ht_dcphase), BATCH1(ht_trendline),
      BATCH1(ht_trendmode),
      // club pack
      BATCH1(market_strength, 5), BATCH1(trendscore, 10),
      BATCH1(refdev, 15, 15), BATCH1(skew, 15), BATCH1(kurtosis, 40),
      BATCH1(pivot_high, 2), BATCH1(pivot_low, 2), BATCH1(xaverage, 9),
      BATCH1(rolling_mean, 10), BATCH1(rolling_max, 10),
      BATCH1(rolling_min, 10), BATCH1(rolling_std, 10),
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, NewTwoInputFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  auto two = [](const char* name, auto batchFn, auto singleFn) {
    return BatchCase{name, 2, 1, batchFn, singleFn};
  };
  const std::vector<BatchCase> cases = {
      two("midprice",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_midprice_batch(in[0], in[1], out[0], len, n, 14, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_midprice_d(in[0], in[1], out[0], len, 14, 0);
          }),
      two("plus_dm",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_plus_dm_batch(in[0], in[1], out[0], len, n, 14, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_plus_dm_d(in[0], in[1], out[0], len, 14, 0);
          }),
      two("minus_dm",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_minus_dm_batch(in[0], in[1], out[0], len, n, 14, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_minus_dm_d(in[0], in[1], out[0], len, 14, 0);
          }),
      two("sar",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_sar_batch(in[0], in[1], out[0], len, n, 0.02f, 0.2f, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_sar_d(in[0], in[1], out[0], len, 0.02f, 0.2f, 0);
          }),
      two("sarext",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_sarext_batch(in[0], in[1], out[0], len, n, 0.f, 0.f,
                                   0.02f, 0.02f, 0.2f, 0.02f, 0.02f, 0.2f, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_sarext_d(in[0], in[1], out[0], len, 0.f, 0.f, 0.02f,
                               0.02f, 0.2f, 0.02f, 0.02f, 0.2f, 0);
          }),
      two("imi",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_imi_batch(in[0], in[1], out[0], len, n, 14, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_imi_d(in[0], in[1], out[0], len, 14, 0);
          }),
      two("nvi",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_nvi_batch(in[0], in[1], out[0], len, n, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_nvi_d(in[0], in[1], out[0], len, 0);
          }),
      two("pvi",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_pvi_batch(in[0], in[1], out[0], len, n, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_pvi_d(in[0], in[1], out[0], len, 0);
          }),
      two("correl",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_correl_batch(in[0], in[1], out[0], len, n, 14, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_correl_d(in[0], in[1], out[0], len, 14, 0);
          }),
      two("beta",
          [](const float* const* in, float* const* out, int len, int n) {
            return ct_beta_batch(in[0], in[1], out[0], len, n, 14, 0);
          },
          [](const float* const* in, float* const* out, int len) {
            return ct_beta_d(in[0], in[1], out[0], len, 14, 0);
          }),
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, MavpBatchMatchesSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  auto values = make_matrix(seriesLen, numSeries, 55);
  std::vector<float> periods(values.size());
  for (size_t i = 0; i < periods.size(); ++i) {
    periods[i] = 5.0f + static_cast<float>(i % 11);
  }
  std::vector<float> out(values.size(), 0.0f);
  ASSERT_EQ(ct_mavp_batch(values.data(), periods.data(), out.data(),
                          seriesLen, numSeries, 2, 30, 0, 0),
            CT_STATUS_SUCCESS);

  float* dV = ct_device_alloc(seriesLen);
  float* dP = ct_device_alloc(seriesLen);
  float* dO = ct_device_alloc(seriesLen);
  ASSERT_TRUE(dV && dP && dO);
  for (int s = 0; s < numSeries; ++s) {
    auto vRow = row(values, seriesLen, s);
    auto pRow = row(periods, seriesLen, s);
    ASSERT_EQ(ct_device_upload(dV, vRow.data(), seriesLen, 0),
              CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_device_upload(dP, pRow.data(), seriesLen, 0),
              CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_mavp_d(dV, dP, dO, seriesLen, 2, 30, 0, 0),
              CT_STATUS_SUCCESS);
    std::vector<float> ref(seriesLen, 0.0f);
    ASSERT_EQ(ct_device_download(ref.data(), dO, seriesLen, 0),
              CT_STATUS_SUCCESS);
    expect_approx_equal(row(out, seriesLen, s), ref);
  }
  ct_device_free(dV);
  ct_device_free(dP);
  ct_device_free(dO);
}

#define BATCH_OHLC(NAME)                                                      \
  BatchCase {                                                                 \
    #NAME, 4, 1,                                                              \
        [](const float* const* in, float* const* out, int len, int n) {      \
          return ct_##NAME##_batch(in[0], in[1], in[2], in[3], out[0], len,  \
                                   n, 0);                                    \
        },                                                                    \
        [](const float* const* in, float* const* out, int len) {             \
          return ct_##NAME##_d(in[0], in[1], in[2], in[3], out[0], len, 0);  \
        }                                                                     \
  }

TEST(BatchAPIGenerated, OhlcAndCandlestickFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 2;
  const std::vector<BatchCase> cases = {
      BATCH_OHLC(bop), BATCH_OHLC(asi),
      BATCH_OHLC(cdl_doji), BATCH_OHLC(cdl_hammer),
      BATCH_OHLC(cdl_inverted_hammer), BATCH_OHLC(cdl_bullish_engulfing),
      BATCH_OHLC(cdl_bearish_engulfing), BATCH_OHLC(cdl_three_white_soldiers),
      BATCH_OHLC(cdl_abandoned_baby), BATCH_OHLC(cdl_advance_block),
      BATCH_OHLC(cdl_belt_hold), BATCH_OHLC(cdl_breakaway),
      BATCH_OHLC(cdl_two_crows), BATCH_OHLC(cdl_three_black_crows),
      BATCH_OHLC(cdl_three_inside), BATCH_OHLC(cdl_three_outside),
      BATCH_OHLC(cdl_three_line_strike), BATCH_OHLC(cdl_three_stars_in_south),
      BATCH_OHLC(cdl_closing_marubozu), BATCH_OHLC(cdl_conceal_baby_swallow),
      BATCH_OHLC(cdl_counterattack), BATCH_OHLC(cdl_dark_cloud_cover),
      BATCH_OHLC(cdl_doji_star), BATCH_OHLC(cdl_dragonfly_doji),
      BATCH_OHLC(cdl_engulfing), BATCH_OHLC(cdl_evening_doji_star),
      BATCH_OHLC(cdl_evening_star), BATCH_OHLC(cdl_gap_side_side_white),
      BATCH_OHLC(cdl_gravestone_doji), BATCH_OHLC(cdl_hanging_man),
      BATCH_OHLC(cdl_harami), BATCH_OHLC(cdl_harami_cross),
      BATCH_OHLC(cdl_high_wave), BATCH_OHLC(cdl_hikkake),
      BATCH_OHLC(cdl_hikkake_mod), BATCH_OHLC(cdl_homing_pigeon),
      BATCH_OHLC(cdl_identical_three_crows), BATCH_OHLC(cdl_in_neck),
      BATCH_OHLC(cdl_kicking), BATCH_OHLC(cdl_kicking_by_length),
      BATCH_OHLC(cdl_ladder_bottom), BATCH_OHLC(cdl_long_legged_doji),
      BATCH_OHLC(cdl_long_line), BATCH_OHLC(cdl_marubozu),
      BATCH_OHLC(cdl_matching_low), BATCH_OHLC(cdl_mat_hold),
      BATCH_OHLC(cdl_morning_doji_star), BATCH_OHLC(cdl_morning_star),
      BATCH_OHLC(cdl_on_neck), BATCH_OHLC(cdl_piercing),
      BATCH_OHLC(cdl_rickshaw_man), BATCH_OHLC(cdl_rise_fall3_methods),
      BATCH_OHLC(cdl_separating_lines), BATCH_OHLC(cdl_shooting_star),
      BATCH_OHLC(cdl_short_line), BATCH_OHLC(cdl_spinning_top),
      BATCH_OHLC(cdl_stalled_pattern), BATCH_OHLC(cdl_stick_sandwich),
      BATCH_OHLC(cdl_takuri), BATCH_OHLC(cdl_tasuki_gap),
      BATCH_OHLC(cdl_thrusting), BATCH_OHLC(cdl_tristar),
      BATCH_OHLC(cdl_unique_3_river), BATCH_OHLC(cdl_upside_gap_2_crows),
      BATCH_OHLC(cdl_xside_gap_3_methods),
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, NewMultiOutputFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  const std::vector<BatchCase> cases = {
      BatchCase{"stochrsi", 1, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_stochrsi_batch(in[0], out[0], out[1], len, n, 14,
                                           5, 3, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_stochrsi_d(in[0], out[0], out[1], len, 14, 5, 3,
                                       0);
                }},
      BatchCase{"macdfix", 1, 3,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_macdfix_batch(in[0], out[0], out[1], out[2], len,
                                          n, 9, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_macdfix_d(in[0], out[0], out[1], out[2], len, 9,
                                      0);
                }},
      BatchCase{"mama", 1, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_mama_batch(in[0], out[0], out[1], len, n, 0.5f,
                                       0.05f, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_mama_d(in[0], out[0], out[1], len, 0.5f, 0.05f,
                                   0);
                }},
      BatchCase{"minmax", 1, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_minmax_batch(in[0], out[0], out[1], len, n, 14, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_minmax_d(in[0], out[0], out[1], len, 14, 0);
                }},
      BatchCase{"minmaxindex", 1, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_minmaxindex_batch(in[0], out[0], out[1], len, n,
                                              14, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_minmaxindex_d(in[0], out[0], out[1], len, 14, 0);
                }},
      BatchCase{"ht_phasor", 1, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_ht_phasor_batch(in[0], out[0], out[1], len, n, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_ht_phasor_d(in[0], out[0], out[1], len, 0);
                }},
      BatchCase{"ht_sine", 1, 2,
                [](const float* const* in, float* const* out, int len, int n) {
                  return ct_ht_sine_batch(in[0], out[0], out[1], len, n, 0);
                },
                [](const float* const* in, float* const* out, int len) {
                  return ct_ht_sine_d(in[0], out[0], out[1], len, 0);
                }},
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, ClubThreeInputFamiliesMatchSingleSeries) {
  const int seriesLen = 512, numSeries = 3;
  const std::vector<BatchCase> cases = {
      BATCH_HLC(sar_el, 0.02f, 0.2f),
      BATCH_HLC(obtr),
      BATCH_HLC(hpi, 0.1f),  // slots pass close as "volume": parity only
  };
  for (const auto& tc : cases) run_generated_case(tc, seriesLen, numSeries);
}

TEST(BatchAPIGenerated, DeviceVariantMatchesHostVariant) {
  const int seriesLen = 768, numSeries = 4, period = 14;
  auto base = make_matrix(seriesLen, numSeries, 33);
  auto high = base, low = base;
  for (size_t i = 0; i < base.size(); ++i) {
    high[i] += 1.0f;
    low[i] -= 1.0f;
  }
  const int total = seriesLen * numSeries;

  std::vector<float> hostOut(base.size(), 0.0f);
  ASSERT_EQ(ct_adx_batch(high.data(), low.data(), base.data(), hostOut.data(),
                         seriesLen, numSeries, period, 0),
            CT_STATUS_SUCCESS);

  float* dH = ct_device_alloc(total);
  float* dL = ct_device_alloc(total);
  float* dC = ct_device_alloc(total);
  float* dO = ct_device_alloc(total);
  ASSERT_TRUE(dH && dL && dC && dO);
  ASSERT_EQ(ct_device_upload(dH, high.data(), total, 0), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_device_upload(dL, low.data(), total, 0), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_device_upload(dC, base.data(), total, 0), CT_STATUS_SUCCESS);

  ASSERT_EQ(ct_adx_batch_d(dH, dL, dC, dO, seriesLen, numSeries, period, 2),
            CT_STATUS_SUCCESS);

  std::vector<float> deviceOut(base.size(), 0.0f);
  ASSERT_EQ(ct_device_download(deviceOut.data(), dO, total, 0),
            CT_STATUS_SUCCESS);
  ct_device_free(dH);
  ct_device_free(dL);
  ct_device_free(dC);
  ct_device_free(dO);

  expect_approx_equal(deviceOut, hostOut);
}
