/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Batch (portfolio) stream tests.
//
// The contract: a batch of N series is numerically identical to N independent
// single-series streams — same state machine, only the call shape and the
// execution policy differ. That must hold on both execution paths, so the
// parity cases run over CPU and GPU.

#include <tacuda_stream.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

constexpr int kSeries = 37;   // deliberately not a multiple of any block size
constexpr int kBars = 512;

struct Matrix {
  // series-major: series s occupies [s * bars, s * bars + bars)
  std::vector<float> high, low, close, volume;
};

Matrix make_matrix(int series, int bars) {
  Matrix m;
  const std::size_t n = static_cast<std::size_t>(series) * bars;
  m.high.resize(n);
  m.low.resize(n);
  m.close.resize(n);
  m.volume.resize(n);

  std::mt19937 rng(4242u);
  std::normal_distribution<float> step(0.0f, 0.6f);
  std::uniform_real_distribution<float> vol(500.0f, 5000.0f);

  for (int s = 0; s < series; ++s) {
    float price = 50.0f + static_cast<float>(s);
    for (int b = 0; b < bars; ++b) {
      // A flat stretch per series exercises the zero-range/flat-window paths.
      if (b < 60 || b >= 90) price += step(rng);
      if (price < 1.0f) price = 1.0f;
      const std::size_t i = static_cast<std::size_t>(s) * bars + b;
      m.close[i] = price;
      m.high[i] = price + 0.35f;
      m.low[i] = price - 0.35f;
      m.volume[i] = vol(rng);
    }
  }
  return m;
}

const Matrix &matrix() {
  static const Matrix m = make_matrix(kSeries, kBars);
  return m;
}

class BatchHandle {
public:
  ~BatchHandle() { ct_stream_batch_destroy(h_); }
  BatchHandle() = default;
  BatchHandle(const BatchHandle &) = delete;
  BatchHandle &operator=(const BatchHandle &) = delete;
  ctStreamBatch_t **out() { return &h_; }
  ctStreamBatch_t *get() const { return h_; }

private:
  ctStreamBatch_t *h_ = nullptr;
};

class SingleHandle {
public:
  ~SingleHandle() { ct_stream_destroy(h_); }
  SingleHandle() = default;
  SingleHandle(const SingleHandle &) = delete;
  SingleHandle &operator=(const SingleHandle &) = delete;
  ctStream_t **out() { return &h_; }
  ctStream_t *get() const { return h_; }

private:
  ctStream_t *h_ = nullptr;
};

// A batch must equal N single streams bit for bit: identical arithmetic, only
// a different loop around it.
void expect_identical(const std::string &what, const std::vector<float> &single,
                      const std::vector<float> &batch) {
  ASSERT_EQ(single.size(), batch.size()) << what;
  for (std::size_t i = 0; i < single.size(); ++i) {
    const bool aNan = std::isnan(single[i]);
    const bool bNan = std::isnan(batch[i]);
    ASSERT_EQ(aNan, bNan) << what << ": NaN mask differs at " << i;
    if (!aNan) {
      ASSERT_FLOAT_EQ(single[i], batch[i]) << what << ": value differs at " << i;
    }
  }
}

const ctStreamExec_t kExecs[] = {CT_STREAM_EXEC_CPU, CT_STREAM_EXEC_GPU};

const char *exec_name(ctStreamExec_t e) {
  switch (e) {
    case CT_STREAM_EXEC_CPU: return "cpu";
    case CT_STREAM_EXEC_GPU: return "gpu";
    default: return "auto";
  }
}

// Reference: run each series through its own single-series stream.
std::vector<float> reference_rsi(int period) {
  const auto &m = matrix();
  std::vector<float> out(static_cast<std::size_t>(kSeries) * kBars);
  for (int s = 0; s < kSeries; ++s) {
    SingleHandle h;
    EXPECT_EQ(ct_rsi_stream_create(period, h.out()), CT_STATUS_SUCCESS);
    EXPECT_EQ(ct_rsi_stream_append(h.get(), m.close.data() + s * kBars, kBars,
                                   out.data() + s * kBars),
              CT_STATUS_SUCCESS);
  }
  return out;
}

}  // namespace

TEST(TacudaStreamBatch, RsiMatchesSingleStreamsOnEveryExecPath) {
  const auto &m = matrix();
  const int period = 14;
  const std::vector<float> ref = reference_rsi(period);

  for (ctStreamExec_t exec : kExecs) {
    for (int chunk : {1, 7, kBars}) {
      BatchHandle b;
      ASSERT_EQ(ct_rsi_stream_batch_create(period, kSeries, b.out()),
                CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(b.get(), exec), CT_STATUS_SUCCESS);

      // Chunked appends need the bars of that chunk contiguous per series, so
      // the caller's matrix is re-packed the way a live feed would deliver it.
      std::vector<float> got(static_cast<std::size_t>(kSeries) * kBars);
      std::vector<float> inChunk(static_cast<std::size_t>(kSeries) * chunk);
      std::vector<float> outChunk(static_cast<std::size_t>(kSeries) * chunk);

      for (int off = 0; off < kBars; off += chunk) {
        const int n = std::min(chunk, kBars - off);
        for (int s = 0; s < kSeries; ++s) {
          for (int j = 0; j < n; ++j) {
            inChunk[static_cast<std::size_t>(s) * n + j] =
                m.close[static_cast<std::size_t>(s) * kBars + off + j];
          }
        }
        ASSERT_EQ(ct_rsi_stream_batch_append(b.get(), inChunk.data(), n,
                                             outChunk.data()),
                  CT_STATUS_SUCCESS)
            << ct_get_last_error();
        for (int s = 0; s < kSeries; ++s) {
          for (int j = 0; j < n; ++j) {
            got[static_cast<std::size_t>(s) * kBars + off + j] =
                outChunk[static_cast<std::size_t>(s) * n + j];
          }
        }
      }
      expect_identical(std::string("RSI exec=") + exec_name(exec) +
                           " chunk=" + std::to_string(chunk),
                       ref, got);
    }
  }
}

// One bulk append per family, compared against per-series single streams.
TEST(TacudaStreamBatch, EveryFamilyMatchesSingleStreams) {
  const auto &m = matrix();
  const std::size_t total = static_cast<std::size_t>(kSeries) * kBars;

  for (ctStreamExec_t exec : kExecs) {
    const std::string tag = std::string(" exec=") + exec_name(exec);

    {  // SMA
      std::vector<float> ref(total), got(total);
      for (int s = 0; s < kSeries; ++s) {
        SingleHandle h;
        ASSERT_EQ(ct_sma_stream_create(20, h.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_sma_stream_append(h.get(), m.close.data() + s * kBars, kBars,
                                       ref.data() + s * kBars),
                  CT_STATUS_SUCCESS);
      }
      BatchHandle b;
      ASSERT_EQ(ct_sma_stream_batch_create(20, kSeries, b.out()), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(b.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_sma_stream_batch_append(b.get(), m.close.data(), kBars, got.data()),
                CT_STATUS_SUCCESS);
      expect_identical("SMA" + tag, ref, got);
    }

    {  // EMA
      std::vector<float> ref(total), got(total);
      for (int s = 0; s < kSeries; ++s) {
        SingleHandle h;
        ASSERT_EQ(ct_ema_stream_create(21, h.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_ema_stream_append(h.get(), m.close.data() + s * kBars, kBars,
                                       ref.data() + s * kBars),
                  CT_STATUS_SUCCESS);
      }
      BatchHandle b;
      ASSERT_EQ(ct_ema_stream_batch_create(21, kSeries, b.out()), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(b.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ema_stream_batch_append(b.get(), m.close.data(), kBars, got.data()),
                CT_STATUS_SUCCESS);
      expect_identical("EMA" + tag, ref, got);
    }

    {  // WMA
      std::vector<float> ref(total), got(total);
      for (int s = 0; s < kSeries; ++s) {
        SingleHandle h;
        ASSERT_EQ(ct_wma_stream_create(18, h.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_wma_stream_append(h.get(), m.close.data() + s * kBars, kBars,
                                       ref.data() + s * kBars),
                  CT_STATUS_SUCCESS);
      }
      BatchHandle b;
      ASSERT_EQ(ct_wma_stream_batch_create(18, kSeries, b.out()), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(b.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_wma_stream_batch_append(b.get(), m.close.data(), kBars, got.data()),
                CT_STATUS_SUCCESS);
      expect_identical("WMA" + tag, ref, got);
    }

    {  // ATR and NATR
      std::vector<float> refA(total), gotA(total), refN(total), gotN(total);
      for (int s = 0; s < kSeries; ++s) {
        SingleHandle a, nt;
        ASSERT_EQ(ct_atr_stream_create(14, a.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_atr_stream_append(a.get(), m.high.data() + s * kBars,
                                       m.low.data() + s * kBars,
                                       m.close.data() + s * kBars, kBars,
                                       refA.data() + s * kBars),
                  CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_natr_stream_create(14, nt.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_natr_stream_append(nt.get(), m.high.data() + s * kBars,
                                        m.low.data() + s * kBars,
                                        m.close.data() + s * kBars, kBars,
                                        refN.data() + s * kBars),
                  CT_STATUS_SUCCESS);
      }
      BatchHandle ba, bn;
      ASSERT_EQ(ct_atr_stream_batch_create(14, kSeries, ba.out()), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(ba.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_atr_stream_batch_append(ba.get(), m.high.data(), m.low.data(),
                                           m.close.data(), kBars, gotA.data()),
                CT_STATUS_SUCCESS);
      expect_identical("ATR" + tag, refA, gotA);

      ASSERT_EQ(ct_natr_stream_batch_create(14, kSeries, bn.out()), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(bn.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_natr_stream_batch_append(bn.get(), m.high.data(), m.low.data(),
                                            m.close.data(), kBars, gotN.data()),
                CT_STATUS_SUCCESS);
      expect_identical("NATR" + tag, refN, gotN);
    }

    {  // OBV and AD
      std::vector<float> refO(total), gotO(total), refD(total), gotD(total);
      for (int s = 0; s < kSeries; ++s) {
        SingleHandle o, d;
        ASSERT_EQ(ct_obv_stream_create(o.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_obv_stream_append(o.get(), m.close.data() + s * kBars,
                                       m.volume.data() + s * kBars, kBars,
                                       refO.data() + s * kBars),
                  CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_ad_stream_create(d.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_ad_stream_append(d.get(), m.high.data() + s * kBars,
                                      m.low.data() + s * kBars,
                                      m.close.data() + s * kBars,
                                      m.volume.data() + s * kBars, kBars,
                                      refD.data() + s * kBars),
                  CT_STATUS_SUCCESS);
      }
      BatchHandle bo, bd;
      ASSERT_EQ(ct_obv_stream_batch_create(kSeries, bo.out()), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(bo.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_obv_stream_batch_append(bo.get(), m.close.data(), m.volume.data(),
                                           kBars, gotO.data()),
                CT_STATUS_SUCCESS);
      expect_identical("OBV" + tag, refO, gotO);

      ASSERT_EQ(ct_ad_stream_batch_create(kSeries, bd.out()), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(bd.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ad_stream_batch_append(bd.get(), m.high.data(), m.low.data(),
                                          m.close.data(), m.volume.data(), kBars,
                                          gotD.data()),
                CT_STATUS_SUCCESS);
      expect_identical("AD" + tag, refD, gotD);
    }

    {  // MACD — three output lines
      std::vector<float> refM(total), refS(total), refH(total);
      std::vector<float> gotM(total), gotS(total), gotH(total);
      for (int s = 0; s < kSeries; ++s) {
        SingleHandle h;
        ASSERT_EQ(ct_macd_stream_create(12, 26, 9, h.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_macd_stream_append(h.get(), m.close.data() + s * kBars, kBars,
                                        refM.data() + s * kBars,
                                        refS.data() + s * kBars,
                                        refH.data() + s * kBars),
                  CT_STATUS_SUCCESS);
      }
      BatchHandle b;
      ASSERT_EQ(ct_macd_stream_batch_create(12, 26, 9, kSeries, b.out()),
                CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(b.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_macd_stream_batch_append(b.get(), m.close.data(), kBars,
                                            gotM.data(), gotS.data(), gotH.data()),
                CT_STATUS_SUCCESS);
      expect_identical("MACD.line" + tag, refM, gotM);
      expect_identical("MACD.signal" + tag, refS, gotS);
      expect_identical("MACD.hist" + tag, refH, gotH);
    }

    {  // BBANDS — three output lines
      std::vector<float> refU(total), refMid(total), refL(total);
      std::vector<float> gotU(total), gotMid(total), gotL(total);
      for (int s = 0; s < kSeries; ++s) {
        SingleHandle h;
        ASSERT_EQ(ct_bbands_stream_create(20, 2.0f, 2.0f, h.out()), CT_STATUS_SUCCESS);
        ASSERT_EQ(ct_bbands_stream_append(h.get(), m.close.data() + s * kBars, kBars,
                                          refU.data() + s * kBars,
                                          refMid.data() + s * kBars,
                                          refL.data() + s * kBars),
                  CT_STATUS_SUCCESS);
      }
      BatchHandle b;
      ASSERT_EQ(ct_bbands_stream_batch_create(20, 2.0f, 2.0f, kSeries, b.out()),
                CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_stream_batch_set_exec(b.get(), exec), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_bbands_stream_batch_append(b.get(), m.close.data(), kBars,
                                              gotU.data(), gotMid.data(), gotL.data()),
                CT_STATUS_SUCCESS);
      expect_identical("BBANDS.upper" + tag, refU, gotU);
      expect_identical("BBANDS.middle" + tag, refMid, gotMid);
      expect_identical("BBANDS.lower" + tag, refL, gotL);
    }
  }
}

// Switching policy mid-stream must migrate state, not restart it: the values
// after the switch have to continue the same recursion.
TEST(TacudaStreamBatch, ExecPolicyCanChangeMidStream) {
  const auto &m = matrix();
  const std::size_t total = static_cast<std::size_t>(kSeries) * kBars;
  const std::vector<float> ref = reference_rsi(14);

  BatchHandle b;
  ASSERT_EQ(ct_rsi_stream_batch_create(14, kSeries, b.out()), CT_STATUS_SUCCESS);

  std::vector<float> got(total);
  const int third = kBars / 3;
  const ctStreamExec_t order[] = {CT_STREAM_EXEC_CPU, CT_STREAM_EXEC_GPU,
                                  CT_STREAM_EXEC_CPU};
  std::vector<float> inChunk, outChunk;

  int off = 0;
  for (int part = 0; part < 3; ++part) {
    const int n = (part == 2) ? (kBars - off) : third;
    ASSERT_EQ(ct_stream_batch_set_exec(b.get(), order[part]), CT_STATUS_SUCCESS);
    inChunk.assign(static_cast<std::size_t>(kSeries) * n, 0.0f);
    outChunk.assign(static_cast<std::size_t>(kSeries) * n, 0.0f);
    for (int s = 0; s < kSeries; ++s) {
      for (int j = 0; j < n; ++j) {
        inChunk[static_cast<std::size_t>(s) * n + j] =
            m.close[static_cast<std::size_t>(s) * kBars + off + j];
      }
    }
    ASSERT_EQ(ct_rsi_stream_batch_append(b.get(), inChunk.data(), n, outChunk.data()),
              CT_STATUS_SUCCESS)
        << ct_get_last_error();
    for (int s = 0; s < kSeries; ++s) {
      for (int j = 0; j < n; ++j) {
        got[static_cast<std::size_t>(s) * kBars + off + j] =
            outChunk[static_cast<std::size_t>(s) * n + j];
      }
    }
    off += n;
  }
  expect_identical("RSI across cpu->gpu->cpu", ref, got);
}

TEST(TacudaStreamBatch, Introspection) {
  BatchHandle sma, macd;
  ASSERT_EQ(ct_sma_stream_batch_create(30, 8, sma.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_macd_stream_batch_create(12, 26, 9, 8, macd.out()), CT_STATUS_SUCCESS);

  int lb = -1, series = -1, outs = -1;
  long long bars = -1;
  ASSERT_EQ(ct_stream_batch_lookback(sma.get(), &lb), CT_STATUS_SUCCESS);
  EXPECT_EQ(lb, 29);
  ASSERT_EQ(ct_stream_batch_lookback(macd.get(), &lb), CT_STATUS_SUCCESS);
  EXPECT_EQ(lb, 33);
  ASSERT_EQ(ct_stream_batch_series(sma.get(), &series), CT_STATUS_SUCCESS);
  EXPECT_EQ(series, 8);
  ASSERT_EQ(ct_stream_batch_outputs(macd.get(), &outs), CT_STATUS_SUCCESS);
  EXPECT_EQ(outs, 3);
  ASSERT_EQ(ct_stream_batch_bars(sma.get(), &bars), CT_STATUS_SUCCESS);
  EXPECT_EQ(bars, 0);

  std::vector<float> in(8 * 5, 1.0f), out(8 * 5, 0.0f);
  ASSERT_EQ(ct_sma_stream_batch_append(sma.get(), in.data(), 5, out.data()),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stream_batch_bars(sma.get(), &bars), CT_STATUS_SUCCESS);
  EXPECT_EQ(bars, 5);
}

TEST(TacudaStreamBatch, ResetEqualsFreshBatch) {
  const auto &m = matrix();
  const std::size_t total = static_cast<std::size_t>(kSeries) * kBars;

  BatchHandle b;
  ASSERT_EQ(ct_rsi_stream_batch_create(14, kSeries, b.out()), CT_STATUS_SUCCESS);
  std::vector<float> first(total), second(total);
  ASSERT_EQ(ct_rsi_stream_batch_append(b.get(), m.close.data(), kBars, first.data()),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stream_batch_reset(b.get()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_rsi_stream_batch_append(b.get(), m.close.data(), kBars, second.data()),
            CT_STATUS_SUCCESS);
  expect_identical("RSI after reset", first, second);

  long long bars = -1;
  ASSERT_EQ(ct_stream_batch_bars(b.get(), &bars), CT_STATUS_SUCCESS);
  EXPECT_EQ(bars, kBars);
}

TEST(TacudaStreamBatch, ResetAfterGpuAppendClearsDeviceState) {
  // Reset has to invalidate the resident device state too, not just the host
  // copy, or the next GPU append would resume from stale values.
  const auto &m = matrix();
  const std::size_t total = static_cast<std::size_t>(kSeries) * kBars;

  BatchHandle b;
  ASSERT_EQ(ct_rsi_stream_batch_create(14, kSeries, b.out()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stream_batch_set_exec(b.get(), CT_STREAM_EXEC_GPU), CT_STATUS_SUCCESS);

  std::vector<float> first(total), second(total);
  ASSERT_EQ(ct_rsi_stream_batch_append(b.get(), m.close.data(), kBars, first.data()),
            CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stream_batch_reset(b.get()), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_rsi_stream_batch_append(b.get(), m.close.data(), kBars, second.data()),
            CT_STATUS_SUCCESS);
  expect_identical("RSI after GPU reset", first, second);
}

TEST(TacudaStreamBatch, RejectsInvalidArguments) {
  ctStreamBatch_t *raw = nullptr;
  EXPECT_EQ(ct_sma_stream_batch_create(10, 0, &raw), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_batch_create(10, -3, &raw), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_batch_create(1, 4, &raw), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_batch_create(10, 4, nullptr), CT_STATUS_INVALID_ARGUMENT);

  BatchHandle b;
  ASSERT_EQ(ct_sma_stream_batch_create(10, 4, b.out()), CT_STATUS_SUCCESS);
  std::vector<float> in(4, 1.0f), out(4, 0.0f);
  EXPECT_EQ(ct_sma_stream_batch_append(b.get(), nullptr, 1, out.data()),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_batch_append(b.get(), in.data(), 1, nullptr),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_sma_stream_batch_append(b.get(), in.data(), 0, out.data()),
            CT_STATUS_INVALID_ARGUMENT);

  // Wrong family through the wrong entry point.
  EXPECT_EQ(ct_rsi_stream_batch_append(b.get(), in.data(), 1, out.data()),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(std::string(ct_get_last_error()).find("different indicator"),
            std::string::npos);

  EXPECT_EQ(ct_stream_batch_set_exec(b.get(), static_cast<ctStreamExec_t>(99)),
            CT_STATUS_INVALID_ARGUMENT);
}

TEST(TacudaStreamBatch, DestroyIsNullSafeAndRejectsReuse) {
  EXPECT_EQ(ct_stream_batch_destroy(nullptr), CT_STATUS_SUCCESS);

  ctStreamBatch_t *raw = nullptr;
  ASSERT_EQ(ct_ema_stream_batch_create(10, 4, &raw), CT_STATUS_SUCCESS);
  ASSERT_EQ(ct_stream_batch_destroy(raw), CT_STATUS_SUCCESS);

  std::vector<float> in(4, 1.0f), out(4, 0.0f);
  int lb = 0;
  EXPECT_EQ(ct_ema_stream_batch_append(raw, in.data(), 1, out.data()),
            CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_batch_reset(raw), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_batch_lookback(raw, &lb), CT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(ct_stream_batch_destroy(raw), CT_STATUS_INVALID_ARGUMENT);
}
