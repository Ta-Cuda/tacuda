/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// The *_ex API promises one set of semantics for every storage/compute pair.
//
// Three references keep each other honest: the plain float path (the one the
// TA-Lib corpus validates), a double-precision restatement of the same rules
// here, and the pairs against each other. Random-walk data matters for MACD: on
// a linear ramp a misplaced EMA seed produces exactly the same numbers.

#include <cuda_fp16.h>
#include <tacuda.h>
#include <tacuda_device.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Pair {
  ctDataType_t storage;
  ctDataType_t compute;
  const char *name;
};

const Pair kGenericPairs[] = {
    {CT_DTYPE_F16, CT_DTYPE_F32, "f16/f32"},
    {CT_DTYPE_F32, CT_DTYPE_F64, "f32/f64"},
    {CT_DTYPE_F64, CT_DTYPE_F32, "f64/f32"},
    {CT_DTYPE_F64, CT_DTYPE_F64, "f64/f64"},
};

std::size_t width(ctDataType_t type) {
  return type == CT_DTYPE_F16 ? sizeof(__half) : type == CT_DTYPE_F32 ? sizeof(float) : sizeof(double);
}

// A series held in the element type the native call will read.
class Stored {
public:
  Stored(ctDataType_t type, std::size_t count)
      : type_(type), count_(count), bytes_(count * width(type)) {}

  static Stored of(ctDataType_t type, const std::vector<double> &values) {
    Stored out(type, values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out.set(i, values[i]);
    return out;
  }

  void *data() { return bytes_.data(); }
  const void *data() const { return bytes_.data(); }

  std::vector<double> values() const {
    std::vector<double> out(count_);
    for (std::size_t i = 0; i < count_; ++i) out[i] = get(i);
    return out;
  }

private:
  void set(std::size_t i, double v) {
    if (type_ == CT_DTYPE_F16) {
      const __half h = __float2half(static_cast<float>(v));
      std::memcpy(bytes_.data() + i * sizeof(h), &h, sizeof(h));
    } else if (type_ == CT_DTYPE_F32) {
      const float f = static_cast<float>(v);
      std::memcpy(bytes_.data() + i * sizeof(f), &f, sizeof(f));
    } else {
      std::memcpy(bytes_.data() + i * sizeof(v), &v, sizeof(v));
    }
  }

  double get(std::size_t i) const {
    if (type_ == CT_DTYPE_F16) {
      __half h;
      std::memcpy(&h, bytes_.data() + i * sizeof(h), sizeof(h));
      return __half2float(h);
    }
    if (type_ == CT_DTYPE_F32) {
      float f;
      std::memcpy(&f, bytes_.data() + i * sizeof(f), sizeof(f));
      return f;
    }
    double d;
    std::memcpy(&d, bytes_.data() + i * sizeof(d), sizeof(d));
    return d;
  }

  ctDataType_t type_;
  std::size_t count_;
  std::vector<unsigned char> bytes_;
};

std::vector<double> random_walk(int n, unsigned seed = 7) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> step(0.0, 1.0);
  std::vector<double> out(static_cast<std::size_t>(n));
  double price = 100.0;
  for (double &v : out) {
    price += step(rng);
    v = price;
  }
  return out;
}

std::vector<double> flat(int n, double value = 100.0) {
  return std::vector<double>(static_cast<std::size_t>(n), value);
}

std::vector<double> rising(int n) {
  std::vector<double> out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = i + 1.0;
  return out;
}

std::vector<float> to_float(const std::vector<double> &values) {
  return std::vector<float>(values.begin(), values.end());
}

std::vector<double> to_double(const std::vector<float> &values) {
  return std::vector<double>(values.begin(), values.end());
}

void expect_close(const std::string &label, const std::vector<double> &got,
                  const std::vector<double> &want, double tolerance) {
  ASSERT_EQ(got.size(), want.size()) << label;
  for (std::size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(std::isnan(got[i]), std::isnan(want[i]))
        << label << ": NaN mask differs at " << i << " (got " << got[i] << ", want " << want[i] << ")";
    if (!std::isnan(got[i])) {
      ASSERT_NEAR(got[i], want[i], tolerance) << label << " at " << i;
    }
  }
}

double tolerance_for(const Pair &pair) {
  if (pair.storage == CT_DTYPE_F16) return 0.08;  // binary16 resolution near 100
  if (pair.compute == CT_DTYPE_F64 && pair.storage == CT_DTYPE_F64) return 1e-9;
  return 2e-2;
}

// --- double-precision restatements of the TA-Lib 0.7 rules ------------------

double rsi_from(double gain, double loss) {
  if (std::isnan(gain) || std::isnan(loss)) return kNaN;
  if (loss == 0.0) return gain == 0.0 ? 0.0 : 100.0;
  if (gain == 0.0) return 0.0;
  return 100.0 - 100.0 / (1.0 + gain / loss);
}

std::vector<double> reference_rsi(const std::vector<double> &x, int period) {
  std::vector<double> out(x.size(), kNaN);
  const int n = static_cast<int>(x.size());
  if (n <= period) return out;
  const auto split = [&](int i, double &g, double &l) {
    const double d = x[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i - 1)];
    g = std::isnan(d) ? d : (d > 0.0 ? d : 0.0);
    l = std::isnan(d) ? d : (d < 0.0 ? -d : 0.0);
  };
  double avgG = 0.0, avgL = 0.0, g, l;
  for (int i = 1; i <= period; ++i) {
    split(i, g, l);
    avgG += g;
    avgL += l;
  }
  avgG /= period;
  avgL /= period;
  out[static_cast<std::size_t>(period)] = rsi_from(avgG, avgL);
  for (int i = period + 1; i < n; ++i) {
    split(i, g, l);
    avgG = (avgG * (period - 1) + g) / period;
    avgL = (avgL * (period - 1) + l) / period;
    out[static_cast<std::size_t>(i)] = rsi_from(avgG, avgL);
  }
  return out;
}

struct MacdLines {
  std::vector<double> macd, signal, hist;
};

MacdLines reference_macd(const std::vector<double> &x, int fast, int slow, int sig) {
  const int n = static_cast<int>(x.size());
  MacdLines out{std::vector<double>(x.size(), kNaN), std::vector<double>(x.size(), kNaN),
                std::vector<double>(x.size(), kNaN)};
  const int macdLookback = slow - 1;
  const int signalLookback = macdLookback + sig - 1;
  if (signalLookback >= n) return out;
  double fastEma = 0.0, slowEma = 0.0;
  for (int i = macdLookback - fast + 1; i <= macdLookback; ++i) fastEma += x[static_cast<std::size_t>(i)];
  fastEma /= fast;
  for (int i = 0; i <= macdLookback; ++i) slowEma += x[static_cast<std::size_t>(i)];
  slowEma /= slow;
  const double fa = 2.0 / (fast + 1.0), sa = 2.0 / (slow + 1.0), ga = 2.0 / (sig + 1.0);
  double sum = 0.0, s = 0.0;
  for (int i = macdLookback; i < n; ++i) {
    if (i > macdLookback) {
      fastEma += fa * (x[static_cast<std::size_t>(i)] - fastEma);
      slowEma += sa * (x[static_cast<std::size_t>(i)] - slowEma);
    }
    const double line = fastEma - slowEma;
    if (i < signalLookback) {
      sum += line;
      continue;
    }
    s = i == signalLookback ? (sum + line) / sig : s + ga * (line - s);
    const std::size_t at = static_cast<std::size_t>(i);
    out.macd[at] = line;
    out.signal[at] = s;
    out.hist[at] = line - s;
  }
  return out;
}

// --- call helpers -----------------------------------------------------------

std::vector<double> rsi_ex(const Pair &pair, const std::vector<double> &values, int period) {
  Stored in = Stored::of(pair.storage, values);
  Stored out(pair.storage, values.size());
  const ctStatus_t status = ct_rsi_ex(in.data(), out.data(), static_cast<int>(values.size()), period,
                                      pair.storage, pair.compute, nullptr);
  EXPECT_EQ(status, CT_STATUS_SUCCESS) << pair.name;
  return out.values();
}

struct MacdResult {
  ctStatus_t status;
  MacdLines lines;
};

MacdResult macd_ex(const Pair &pair, const std::vector<double> &values, int fast, int slow,
                   int sig, ctMaType_t type = CT_MA_EMA) {
  Stored in = Stored::of(pair.storage, values);
  Stored m(pair.storage, values.size()), s(pair.storage, values.size()), h(pair.storage, values.size());
  MacdResult result;
  result.status = ct_macd_ex(in.data(), m.data(), s.data(), h.data(), static_cast<int>(values.size()),
                             fast, slow, sig, type, pair.storage, pair.compute, nullptr);
  result.lines = {m.values(), s.values(), h.values()};
  return result;
}

std::vector<double> plain_rsi(const std::vector<double> &values, int period) {
  std::vector<float> in = to_float(values), out(values.size());
  EXPECT_EQ(ct_rsi(in.data(), out.data(), static_cast<int>(in.size()), period, nullptr), CT_STATUS_SUCCESS);
  return to_double(out);
}

MacdLines plain_macd(const std::vector<double> &values, int fast, int slow, int sig,
                     ctMaType_t type = CT_MA_EMA) {
  std::vector<float> in = to_float(values), m(values.size()), s(values.size()), h(values.size());
  EXPECT_EQ(ct_macd(in.data(), m.data(), s.data(), h.data(), static_cast<int>(in.size()), fast, slow,
                    sig, type, nullptr),
            CT_STATUS_SUCCESS);
  return {to_double(m), to_double(s), to_double(h)};
}

// The values the native side actually sees after storage rounding.
std::vector<double> as_stored(const Pair &pair, const std::vector<double> &values) {
  return Stored::of(pair.storage, values).values();
}

}  // namespace

TEST(PrecisionEx, RsiAgreesWithPlainPathAndReference) {
  const std::vector<std::vector<double>> datasets = {random_walk(400), rising(200), flat(120)};
  for (const Pair &pair : kGenericPairs) {
    for (const auto &raw : datasets) {
      for (int period : {2, 14, 50}) {
        const std::vector<double> x = as_stored(pair, raw);
        const std::string label = std::string(pair.name) + " RSI(" + std::to_string(period) + ")";
        const std::vector<double> got = rsi_ex(pair, raw, period);
        expect_close(label + " vs plain", got, plain_rsi(x, period), 0.08);
        expect_close(label + " vs reference", got, reference_rsi(x, period), tolerance_for(pair));
      }
    }
  }
}

TEST(PrecisionEx, RsiOfFlatSeriesIsZeroNotOneHundred) {
  for (const Pair &pair : kGenericPairs) {
    const std::vector<double> got = rsi_ex(pair, flat(64), 14);
    for (std::size_t i = 14; i < got.size(); ++i) {
      ASSERT_EQ(got[i], 0.0) << pair.name << " at " << i;
    }
  }
}

TEST(PrecisionEx, RsiPropagatesInteriorNan) {
  std::vector<double> x = random_walk(200);
  x[100] = kNaN;
  for (const Pair &pair : kGenericPairs) {
    const std::vector<double> got = rsi_ex(pair, x, 14);
    EXPECT_FALSE(std::isnan(got[99])) << pair.name;
    for (std::size_t i = 100; i < got.size(); ++i) {
      ASSERT_TRUE(std::isnan(got[i])) << pair.name << " at " << i;
    }
  }
}

TEST(PrecisionEx, RsiShortSeriesIsAllNan) {
  for (const Pair &pair : kGenericPairs) {
    for (int size : {1, 2, 14}) {
      for (double v : rsi_ex(pair, rising(size), 14)) EXPECT_TRUE(std::isnan(v)) << pair.name;
    }
    const std::vector<double> justEnough = rsi_ex(pair, rising(15), 14);
    EXPECT_FALSE(std::isnan(justEnough[14])) << pair.name;
  }
}

TEST(PrecisionEx, MacdAgreesWithPlainPathAndReference) {
  const std::vector<std::vector<double>> datasets = {random_walk(400), rising(200), flat(120)};
  for (const Pair &pair : kGenericPairs) {
    for (const auto &raw : datasets) {
      const std::vector<double> x = as_stored(pair, raw);
      MacdResult got = macd_ex(pair, raw, 12, 26, 9);
      ASSERT_EQ(got.status, CT_STATUS_SUCCESS) << pair.name;
      MacdLines plain = plain_macd(x, 12, 26, 9);
      expect_close(std::string(pair.name) + " line vs plain", got.lines.macd, plain.macd, 0.08);
      expect_close(std::string(pair.name) + " signal vs plain", got.lines.signal, plain.signal, 0.08);
      expect_close(std::string(pair.name) + " hist vs plain", got.lines.hist, plain.hist, 0.08);
      MacdLines reference = reference_macd(x, 12, 26, 9);
      const double tolerance = tolerance_for(pair);
      expect_close(std::string(pair.name) + " line vs ref", got.lines.macd, reference.macd, tolerance);
      expect_close(std::string(pair.name) + " signal vs ref", got.lines.signal, reference.signal, tolerance);
    }
  }
}

TEST(PrecisionEx, MacdHasFiniteSignalAfterTheCommonLookback) {
  for (const Pair &pair : kGenericPairs) {
    MacdResult got = macd_ex(pair, rising(100), 12, 26, 9);
    ASSERT_EQ(got.status, CT_STATUS_SUCCESS) << pair.name;
    int finite = 0;
    for (std::size_t i = 0; i < got.lines.signal.size(); ++i) {
      ASSERT_EQ(std::isnan(got.lines.signal[i]), i < 33u) << pair.name << " at " << i;
      ASSERT_EQ(std::isnan(got.lines.macd[i]), i < 33u) << pair.name << " at " << i;
      ASSERT_EQ(std::isnan(got.lines.hist[i]), i < 33u) << pair.name << " at " << i;
      finite += std::isnan(got.lines.signal[i]) ? 0 : 1;
    }
    EXPECT_EQ(finite, 67) << pair.name;
  }
}

TEST(PrecisionEx, MacdReversedPeriodsMatchNormalOrder) {
  const std::vector<double> x = random_walk(200);
  for (const Pair &pair : kGenericPairs) {
    MacdResult normal = macd_ex(pair, x, 12, 26, 9);
    MacdResult reversed = macd_ex(pair, x, 26, 12, 9);
    ASSERT_EQ(reversed.status, CT_STATUS_SUCCESS) << pair.name;
    expect_close(pair.name, reversed.lines.macd, normal.lines.macd, 0.0);
    expect_close(pair.name, reversed.lines.signal, normal.lines.signal, 0.0);
  }
}

TEST(PrecisionEx, MacdFastLongerThanSeriesStaysInsideBuffers) {
  // Used to read and write fastPeriod elements of a 16-element buffer.
  for (const Pair &pair : kGenericPairs) {
    MacdResult got = macd_ex(pair, rising(16), 32, 4, 3);
    ASSERT_EQ(got.status, CT_STATUS_SUCCESS) << pair.name;
    for (double v : got.lines.signal) EXPECT_TRUE(std::isnan(v)) << pair.name;
  }
}

TEST(PrecisionEx, MacdEqualPeriodsHaveAZeroLine) {
  for (const Pair &pair : kGenericPairs) {
    MacdResult got = macd_ex(pair, random_walk(120), 12, 12, 3);
    ASSERT_EQ(got.status, CT_STATUS_SUCCESS) << pair.name;
    for (std::size_t i = 13; i < got.lines.macd.size(); ++i) {
      ASSERT_NEAR(got.lines.macd[i], 0.0, 1e-6) << pair.name << " at " << i;
    }
  }
}

TEST(PrecisionEx, MacdMaTypeIsNeverSilentlyIgnored) {
  const std::vector<double> x = random_walk(200);
  // float compute runs the same MACDEXT as the plain path, so it honours the type.
  const Pair viaFloat{CT_DTYPE_F64, CT_DTYPE_F32, "f64/f32"};
  MacdResult sma = macd_ex(viaFloat, x, 12, 26, 9, CT_MA_SMA);
  ASSERT_EQ(sma.status, CT_STATUS_SUCCESS);
  expect_close("SMA through f64/f32", sma.lines.macd, plain_macd(x, 12, 26, 9, CT_MA_SMA).macd, 0.02);
  // The double kernel is EMA-only and has to say so.
  const Pair viaDouble{CT_DTYPE_F64, CT_DTYPE_F64, "f64/f64"};
  EXPECT_EQ(macd_ex(viaDouble, x, 12, 26, 9, CT_MA_SMA).status, CT_STATUS_UNSUPPORTED);
}

TEST(PrecisionEx, OtherFamiliesAgreeWithPlainPath) {
  const std::vector<double> close = random_walk(400);
  std::vector<double> high = close, low = close;
  for (std::size_t i = 0; i < close.size(); ++i) {
    high[i] += 0.5 + static_cast<double>(i % 3) * 0.25;
    low[i] -= 0.5 + static_cast<double>(i % 5) * 0.125;
  }
  const int n = static_cast<int>(close.size());
  for (const Pair &pair : kGenericPairs) {
    const Stored in = Stored::of(pair.storage, close);
    const std::vector<float> fin = to_float(in.values());
    const std::string name = pair.name;
    {
      Stored out(pair.storage, close.size());
      std::vector<float> plain(close.size());
      ASSERT_EQ(ct_sma_ex(in.data(), out.data(), n, 20, pair.storage, pair.compute, nullptr), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_sma(fin.data(), plain.data(), n, 20, nullptr), CT_STATUS_SUCCESS);
      expect_close(name + " SMA", out.values(), to_double(plain), 0.08);
    }
    {
      Stored out(pair.storage, close.size());
      std::vector<float> plain(close.size());
      ASSERT_EQ(ct_ema_ex(in.data(), out.data(), n, 20, pair.storage, pair.compute, nullptr), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_ema(fin.data(), plain.data(), n, 20, nullptr), CT_STATUS_SUCCESS);
      expect_close(name + " EMA", out.values(), to_double(plain), 0.08);
    }
    {
      Stored up(pair.storage, close.size()), mid(pair.storage, close.size()), lo(pair.storage, close.size());
      std::vector<float> pu(close.size()), pm(close.size()), pl(close.size());
      ASSERT_EQ(ct_bbands_ex(in.data(), up.data(), mid.data(), lo.data(), n, 20, 2.0f, 2.0f, pair.storage,
                             pair.compute, nullptr), CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_bbands(fin.data(), pu.data(), pm.data(), pl.data(), n, 20, 2.0f, 2.0f, nullptr), CT_STATUS_SUCCESS);
      expect_close(name + " BBANDS upper", up.values(), to_double(pu), 0.1);
      expect_close(name + " BBANDS lower", lo.values(), to_double(pl), 0.1);
    }
    {
      const Stored h = Stored::of(pair.storage, high), l = Stored::of(pair.storage, low);
      const std::vector<float> fh = to_float(h.values()), fl = to_float(l.values());
      Stored out(pair.storage, close.size());
      std::vector<float> plain(close.size());
      ASSERT_EQ(ct_atr_ex(h.data(), l.data(), in.data(), out.data(), n, 14, 0.0f, pair.storage, pair.compute, nullptr),
                CT_STATUS_SUCCESS);
      ASSERT_EQ(ct_atr(fh.data(), fl.data(), fin.data(), plain.data(), n, 14, 0.0f, nullptr), CT_STATUS_SUCCESS);
      expect_close(name + " ATR", out.values(), to_double(plain), 0.08);
    }
  }
}

TEST(PrecisionEx, NonPositivePeriodIsAnErrorForEveryFamily) {
  const std::vector<double> x = random_walk(64);
  const int n = static_cast<int>(x.size());
  for (const Pair &pair : kGenericPairs) {
    const Stored in = Stored::of(pair.storage, x);
    Stored a(pair.storage, x.size()), b(pair.storage, x.size()), c(pair.storage, x.size());
    for (int bad : {0, -3}) {
      const std::string name = std::string(pair.name) + " period " + std::to_string(bad);
      EXPECT_EQ(ct_sma_ex(in.data(), a.data(), n, bad, pair.storage, pair.compute, nullptr), CT_STATUS_INVALID_ARGUMENT) << name;
      EXPECT_EQ(ct_ema_ex(in.data(), a.data(), n, bad, pair.storage, pair.compute, nullptr), CT_STATUS_INVALID_ARGUMENT) << name;
      EXPECT_EQ(ct_rsi_ex(in.data(), a.data(), n, bad, pair.storage, pair.compute, nullptr), CT_STATUS_INVALID_ARGUMENT) << name;
      EXPECT_EQ(ct_bbands_ex(in.data(), a.data(), b.data(), c.data(), n, bad, 2.0f, 2.0f, pair.storage, pair.compute, nullptr), CT_STATUS_INVALID_ARGUMENT) << name;
      EXPECT_EQ(ct_atr_ex(in.data(), in.data(), in.data(), a.data(), n, bad, 0.0f, pair.storage, pair.compute, nullptr), CT_STATUS_INVALID_ARGUMENT) << name;
      EXPECT_EQ(ct_macd_ex(in.data(), a.data(), b.data(), c.data(), n, bad, 26, 9, CT_MA_EMA, pair.storage, pair.compute, nullptr), CT_STATUS_INVALID_ARGUMENT) << name;
      EXPECT_EQ(ct_macd_ex(in.data(), a.data(), b.data(), c.data(), n, 12, 26, bad, CT_MA_EMA, pair.storage, pair.compute, nullptr), CT_STATUS_INVALID_ARGUMENT) << name;
    }
  }
}

TEST(PrecisionEx, UnsupportedPairIsRejected) {
  const std::vector<double> x = random_walk(32);
  const Stored in = Stored::of(CT_DTYPE_F16, x);
  Stored out(CT_DTYPE_F16, x.size());
  EXPECT_EQ(ct_rsi_ex(in.data(), out.data(), 32, 14, CT_DTYPE_F16, CT_DTYPE_F64, nullptr), CT_STATUS_UNSUPPORTED);
}

TEST(PrecisionEx, DeviceEntryPointMatchesHostEntryPoint) {
  const std::vector<double> x = random_walk(300);
  const int n = static_cast<int>(x.size());
  for (const Pair &pair : kGenericPairs) {
    const Stored in = Stored::of(pair.storage, x);
    void *dIn = ct_device_alloc_ex(n, pair.storage);
    void *dOut = ct_device_alloc_ex(n, pair.storage);
    ASSERT_TRUE(dIn && dOut);
    ASSERT_EQ(ct_device_upload_ex(dIn, in.data(), n, pair.storage, nullptr), CT_STATUS_SUCCESS);
    ASSERT_EQ(ct_rsi_ex_d(dIn, dOut, n, 14, pair.storage, pair.compute, nullptr), CT_STATUS_SUCCESS);
    ct_device_sync();
    Stored out(pair.storage, x.size());
    ASSERT_EQ(ct_device_download_ex(out.data(), dOut, n, pair.storage, nullptr), CT_STATUS_SUCCESS);
    ct_device_sync();
    ct_device_free_ex(dIn);
    ct_device_free_ex(dOut);
    expect_close(pair.name, out.values(), rsi_ex(pair, x, 14), 0.0);
  }
}
