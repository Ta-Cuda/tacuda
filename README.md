# 🚀 TACUDA

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![.NET](https://img.shields.io/badge/.NET-8.0%2B-purple.svg)](https://dotnet.microsoft.com/)
[![Functions](https://img.shields.io/badge/TA--Lib_API-161%2F161_functions-orange.svg)](#-comprehensive-indicator-suite-161-functions-from-ta-lib-071)
[![Website](https://img.shields.io/badge/Website-tacuda.com-blue.svg)](https://tacuda.com)

> **High-Performance CUDA-Accelerated Technical Analysis Library**
>
> **⚡ Up to 15.4x faster pipelines than TA-Lib** | **📊 161 functions from TA-Lib 0.7.1** | **🔧 Host / Device / Batch APIs** | 🌐 [tacuda.com](https://tacuda.com)
>
> Numerical compatibility with TA-Lib is verified per function by a strict
> differential corpus — identical NaN/Inf masks plus tolerance-bounded
> values over random walks, trend, flat, zero-crossing and short inputs
> (`benchmarks/differential_corpus.py`, **165/165 specs green as of 0.7.1**,
> covering the 161 functions in native TA-Lib 0.7.1 (Python wrapper 0.7.0) with exact-name coverage
> enforcement; all 61 candlestick patterns match TA-Lib signal-for-signal
> with no mask and no flip budget, and the report records how many signals
> each pattern actually produced). The few remaining corner-case notes live
> in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).

**Release status:** **0.8.0** is the current release on PyPI, npm and NuGet;
it adds the streaming API. Streaming requires a native library and bindings
from the same 0.8.0 release. The 161-function compatibility baseline is
pinned in `tests/requirements-reference.txt` (Python TA-Lib 0.7.0, whose
bundled native library reports 0.7.1 on Windows and 0.6.4 on Linux; both
pass the same strict corpus). TA-Lib 0.8.1 has 201 functions;
its 40 missing corpus specs and IMI/WILLR edge cases remain migration work,
not part of the full-coverage claim ([validation](docs/OPERATIONS.md#validation-snapshot--2026-10-07)).

TACUDA delivers lightning-fast technical analysis indicators powered by NVIDIA CUDA GPUs. Built for quantitative traders, researchers, and financial analysts who demand maximum performance from their computational workflows.

### 🎯 Quick Benchmark Results

| Metric | Value | Notes |
|--------|-------|-------|
| **vs TA-Lib** | **15.4x faster** | Device API, 1M bars, 7-indicator pipeline |
| **Peak indicator speedup** | **84.4x** | AVGDEV(100), 500K bars, host API |
| **Vector math** | **GPU-resident execution** | Separate NumPy and TA-Lib comparisons below |
| **Throughput** | **2M bars in 3.4ms** | Multi-indicator pipeline |

Implementation semantics (EMA seeding, warmup NaNs, indexing conventions) are documented in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).

---

## ✨ Key Features

### 🔥 **Performance First**
- **GPU-accelerated kernels** for massive parallel computation
- **Device API** (`ct_*_d`, 200 functions) for zero-copy multi-indicator pipelines
- **Host API** (`ct_*`) for simple single-indicator calls
- **Batch API** (`ct_*_batch[_d]`, 166 families) for multi-symbol portfolios
- **Pipelined workloads** with optional `cudaStream_t` support and an internal device buffer pool

### 📊 **Comprehensive Indicator Suite (161 functions from TA-Lib 0.7.1)**
- **Moving Averages**: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MAMA
- **Momentum**: RSI, MACD, MACDEXT (independent MA types), Stochastic, CCI, CMO, MOM, ROC, ROCP, ROCR, WILLR
- **Volatility**: ATR, NATR, TRANGE, Bollinger Bands, ACCBANDS
- **Trend**: ADX, ADXR, DX, Plus/Minus DI, Aroon, SAR, SAREXT
- **Volume**: AD, ADOSC, OBV, MFI, NVI, PVI
- **Statistical**: STDDEV, VAR, CORREL, LINEARREG, TSF, BETA
- **Price Transforms**: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE
- **Oscillators**: ULTOSC, APO, PPO, PVO, BOP, STOCHRSI
- **Vector Math** (element-wise, GPU-resident inputs): SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH, EXP, LN, LOG10, SQRT, CEIL, FLOOR, ADD, SUB, MULT, DIV

### 🕯️ **Candlestick Pattern Recognition (63 patterns)**
TA-Lib compatible pattern detection — all 61 TA-Lib patterns match signal-for-signal:
- **Single candle**: Doji, Hammer, InvertedHammer, HangingMan, ShootingStar, Marubozu, SpinningTop
- **Double candle**: Engulfing, Harami, HaramiCross, Piercing, DarkCloudCover, Kicking
- **Triple candle**: MorningStar, EveningStar, ThreeWhiteSoldiers, ThreeBlackCrows, ThreeInside, ThreeOutside
- **Complex patterns**: AbandonedBaby, Tristar, RiseFall3Methods, and many more

### 🔌 **Multi-Language Support**
- **C/C++**: Stable ABI with `extern "C"` interface (`ct_version()`, append-only within a MAJOR — see the [ABI policy](docs/CONTRIBUTING.md#abi-stability-policy))
- **Python**: Pythonic API with NumPy integration, `tacuda.OHLCV` structured inputs, `*_batch` portfolio calls
- **Node.js**: `koffi`-based bindings with TypedArray support, `Ohlcv` containers, `*Batch` + `device.*Batch` wrappers
- **C#**: Native .NET bindings — public `NativeMethods`, `OhlcvSeries` extension methods, `TacudaBatch` helpers

All bindings cover the full native surface (811 entry points: 559 host + 200 device + 52 streaming,
including the batch API and `_ex` precision variants) — verified by
`bindings/audit_bindings.py`, see [docs/API.md](docs/API.md#bindings).

### 📦 **Batched (multi-symbol) execution**
`ct_*_batch[_d]` for 166 indicator families (incl. all candlestick patterns): one call
processes a whole portfolio (series-major matrix), overlapping transfers and kernels
across an internal CUDA stream pool with per-call `maxConcurrency` control —
[docs/BATCH_API.md](docs/BATCH_API.md).

### Incremental updates (0.8.0)

SMA, EMA, WMA, RSI, ATR, NATR, OBV, AD, MACD and BBANDS support append-only
streams, with wrappers for Python, Node.js and C#. Single-series updates run
on the CPU; portfolio streams can use CPU or GPU. See
[STREAMING_API.md](docs/STREAMING_API.md) for lifetime, warm-up and execution policy.

```python
with tacuda.stream.Rsi(14) as rsi:
    rsi.append(history)
    latest = rsi.append([new_close])[-1]
```

### Validation and resource management
- **Warm-up markers** via leading `NaN` values for incomplete windows
- **Comprehensive test suite** with GoogleTest
- **Memory-safe** RAII patterns
- **Thread-safe** indicator registry

---

## 🛠️ Installation

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| **NVIDIA GPU** | Compute ≥ 7.5 for prebuilt binaries | Turing or newer; older GPUs (CC 6.0–7.0) work when building from source with `-DCMAKE_CUDA_ARCHITECTURES=<arch>` |
| **CUDA Toolkit** | 11.x / 12.x | 12.x recommended |
| **CMake** | ≥ 3.22 | |
| **Compiler** | C++17 | GCC, Clang, or MSVC |
| **Python** | ≥ 3.8 | *Optional for bindings* |
| **Node.js** | ≥ 18 | *Optional for JS binding* |
| **NET** | ≥ 8.0 | *Optional for C# binding* |

### Quick Build

```bash
# Clone repository
git clone https://github.com/Ta-Cuda/tacuda.git
cd tacuda

# Build core library
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run example
./build/example_app
```

On Windows use `scripts\build_win_dll.bat` (MSVC + Ninja + CUDA, see the
script header).

### Python Installation

```bash
pip install tacuda           # platform wheels bundle the native library

# or from a source checkout: the package is pure ctypes over your own build
pip install -e bindings/python
export TACUDA_LIBRARY=$PWD/build/libtacuda.so   # build-win\tacuda.dll on Windows

python -c "
import numpy as np, tacuda
data = np.arange(1, 11, dtype=np.float32)
print('SMA:', tacuda.sma(data, 5))
"
```

### Node.js Installation

```bash
cd bindings/nodejs
npm install

# Point to the built native library and run the test pipeline
TACUDA_LIBRARY=$PWD/../../build/libtacuda.so node test_pipeline.js
```

### C# Installation

```bash
# Build C# bindings
dotnet build bindings/csharp/ConsoleExample -c Release

# Ensure library is discoverable (Linux example)
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
dotnet run --project bindings/csharp/ConsoleExample
```

### Binding generation workflow

Python and C# bindings are generated from the public headers (`tacuda.h`,
`tacuda_device.h`); Node.js bindings are maintained in
`bindings/nodejs/index.js` and audited against the same headers. After editing
the headers, regenerate the artefacts and commit the results:

```bash
python bindings/generate_bindings.py
```

CTest contains a guard that fails if the checked-in bindings are stale.

---

## 🚀 Quick Start

### Python API

```python
import numpy as np
import tacuda

# 1M random-walk prices
prices = np.random.randn(1_000_000).cumsum().astype(np.float32) + 1000

sma = tacuda.sma(prices, 20)
rsi = tacuda.rsi(prices, 14)
macd = tacuda.macd(prices)                     # dict: macd / signal / hist

# Structured OHLCV inputs and multi-symbol batches
# ohlcv = tacuda.OHLCV.from_columns(o, h, l, c, v)
# atr = tacuda.atr(ohlcv, period=14)
# rows = tacuda.rsi_batch([series1, series2, series3], 14)

print("SMA:", sma[100:105])
print("RSI:", rsi[100:105])
```

### C++ API

```cpp
#include <tacuda.h>          // public C API — stable ABI
#include <tacuda/Context.h>  // optional C++ RAII layer (Context/DeviceBuffer)
#include <vector>

int main() {
    std::vector<float> prices(1'000'000, 100.0f);
    std::vector<float> sma_out(prices.size());
    std::vector<float> rsi_out(prices.size());

    // Host API: one call = upload + kernel + download
    ct_sma(prices.data(), sma_out.data(), (int)prices.size(), 20, nullptr);
    ct_rsi(prices.data(), rsi_out.data(), (int)prices.size(), 14, nullptr);

    // C++ zero-copy pipeline: keep data on the GPU between indicators
    tacuda::Context ctx;
    auto d_prices = ctx.upload(prices.data(), prices.size());
    auto d_sma = ctx.sma(d_prices, 20);
    auto d_rsi = ctx.rsi(d_prices, 14);
    ctx.download(d_rsi, rsi_out.data());
    return 0;
}
```

See [`src/main.cpp`](src/main.cpp) (built as `example_app`) and
[`examples/pipeline_example.cpp`](examples/pipeline_example.cpp) for complete
programs.

Warm-up depends on the indicator: SMA/EMA use `period-1` bars, RSI/ATR use
`period`, and MACD uses its combined slow/signal lookback. Most warm-up
outputs are leading `NaN`; integer signal functions use zero. See
[alignment conventions](docs/API.md#output-alignment-warmup-nan).

### C API (for FFI/P-Invoke)

```c
#include "tacuda.h"

int main() {
    float input[1000000];
    float output[1000000];
    
    // Host API: simple, transfers data each call
    ct_sma(input, output, 1000000, 20, NULL);
    ct_rsi(input, output, 1000000, 14, NULL);
    
    // Device API: zero-copy pipeline (3.7x faster for multiple indicators)
    // See "Device API - Zero-Copy Pipelines" section below
    
    return 0;
}
```

### 📦 OHLCV Container

#### C++
```cpp
#include <tacuda/OHLCVSeries.h>
#include <tacuda.h>

std::vector<float> open{1.0f, 2.0f, 1.5f};
std::vector<float> high{1.2f, 2.3f, 1.6f};
std::vector<float> low{0.9f, 1.9f, 1.3f};
std::vector<float> close{1.1f, 2.1f, 1.4f};

// Volume defaults to zero when omitted
tacuda::OHLCVSeries candles(open, high, low, close);
std::vector<float> imi(candles.size());
ct_imi(candles.open_data(), candles.close_data(), imi.data(), static_cast<int>(candles.size()), 3);
```

#### Python
```python
from tacuda import OHLCV, imi

ohlcv = OHLCV.from_columns(open, high, low, close, volume)
result = imi(ohlcv.open, ohlcv.close, period=3)
packed = ohlcv.column_major()  # [O1..On, H1..Hn, ...]
```

#### C#
```csharp
using Tacuda.Bindings;

var candles = new OhlcvSeries(open, high, low, close);
var output = new float[candles.Length];
NativeMethods.ct_imi(candles.Open, candles.Close, output, candles.Length, period: 3);
var columnMajor = candles.ToColumnMajor();
```


### C# API

```csharp
// dotnet add package Tacuda   (nupkg bundles win-x64 + linux-x64 natives)
using Tacuda.Bindings;

var prices = new float[1_000_000];
var sma = new float[prices.Length];
var rsi = new float[prices.Length];

// Fill prices with data...

// Full generated P/Invoke surface: NativeMethods.ct_*
NativeMethods.ct_sma(prices, sma, prices.Length, 20, IntPtr.Zero);
NativeMethods.ct_rsi(prices, rsi, prices.Length, 14, IntPtr.Zero);

// OHLCV extension methods and batch helpers
// var candles = new OhlcvSeries(open, high, low, close);
// var atr = candles.Atr(14);
// var rows = TacudaBatch.Rsi(new[] { s1, s2, s3 }, 14);

Console.WriteLine($"SMA[100]: {sma[100]}, RSI[100]: {rsi[100]}");
```

See [`bindings/csharp/ConsoleExample`](bindings/csharp/ConsoleExample) for a
complete test suite and pipeline (`dotnet run -- --test` / `-- --pipeline`).

---

## 📈 Performance Benchmarks

TACUDA delivers significant performance improvements over TA-Lib and CPU implementations.

### Multi-Indicator Pipeline Benchmark

Computing 7 indicators (SMA, EMA×2, RSI, MOM, ROC, STDDEV) on the same dataset:

| Data Size | TA-Lib (CPU) | Host API | Device API | vs TA-Lib |
|-----------|--------------|----------|------------|-----------|
| 100K bars | 1.52ms | 1.50ms | **0.41ms** | **3.7x** |
| 250K bars | 5.68ms | 2.30ms | **0.63ms** | **9.0x** |
| 500K bars | 11.20ms | 3.29ms | **0.91ms** | **12.3x** |
| 1M bars | 22.58ms | 5.62ms | **1.47ms** | **15.4x** |
| 2M bars | 46.51ms | 11.95ms | **3.39ms** | **13.7x** |

### API Comparison

| API | Use Case | Transfer Overhead |
|-----|----------|-------------------|
| **Host API** (`ct_*`) | Single indicator, simple scripts | H→D→H per call |
| **Device API** (`ct_*_d`) | Multi-indicator pipelines | Upload once, compute all, download once |
| **Batch API** (`ct_*_batch[_d]`) | Many symbols, same indicator | One matrix upload, stream-overlapped kernels |

**Device API Advantage**: Eliminates 7× host↔device transfers = **3.7x faster** than Host API

### Vector Math Benchmark (10M elements, RTX 3090 Ti)

Element-wise TA-Lib math functions on the Device API (data already on GPU):

| Function | tacuda device | NumPy f32 | TA-Lib f64 | Device vs NumPy | Device vs TA-Lib |
|----------|--------------:|----------:|-----------:|----------------:|-----------------:|
| SIN | **0.110ms** | 12.70ms | 62.21ms | 115.5x | 565.5x |
| ASIN | **0.111ms** | 83.49ms | 133.73ms | 752.2x | 1204.8x |
| SINH | **0.110ms** | 128.66ms | 141.42ms | 1169.6x | 1285.6x |
| EXP | **0.111ms** | 10.34ms | 35.00ms | 93.2x | 315.3x |
| SQRT | **0.110ms** | 6.99ms | 21.18ms | 63.5x | 192.5x |
| ADD | **0.163ms** | 7.08ms | 14.59ms | 43.4x | 89.5x |
| DIV | **0.160ms** | 7.18ms | 14.52ms | 44.9x | 90.8x |

These are historical timings; ratios are recomputed from the displayed
rounded values. Device timings exclude transfers. The two CPU baselines
have different precision and are reported separately.

Reproduce the full 19-function table with `python benchmarks/benchmark_vector_math.py`.

### Sequential indicators (SAR family): hybrid CPU path

Parabolic SAR is inherently recurrent, so the host API (`ct_sar`, `ct_sarext`) computes it on the CPU in double precision — no PCIe round-trip, on par with TA-Lib even at 1M bars. The device API (`ct_sar_d`, `ct_sarext_d`) still runs on the GPU so zero-copy pipelines are never interrupted.

### Memory Bandwidth Savings

```
Host API:   7 indicators × (upload + download) = 14 transfers = 56MB PCIe traffic
Device API: 1 upload + 1 download = 2 transfers = 8MB PCIe traffic
Savings:    7x less PCIe bandwidth!
```

### 🚀 Device API - Zero-Copy Pipelines

For maximum performance when computing multiple indicators on the same data.
Memory management is part of the C ABI; native callers need CUDA headers
for `cudaStream_t`:

```c
#include "tacuda.h"
#include "tacuda_device.h"

// Upload once
float* d_data = ct_device_alloc(size);
float* d_sma  = ct_device_alloc(size);
float* d_ema  = ct_device_alloc(size);
float* d_rsi  = ct_device_alloc(size);
ct_device_upload(d_data, h_data, size, NULL);

// Kernels only — nothing crosses PCIe
ct_sma_d(d_data, d_sma, size, 20, NULL);
ct_ema_d(d_data, d_ema, size, 12, NULL);
ct_rsi_d(d_data, d_rsi, size, 14, NULL);

// Download once
ct_device_download(h_rsi, d_rsi, size, NULL);
ct_device_free(d_data); ct_device_free(d_sma);
ct_device_free(d_ema);  ct_device_free(d_rsi);
```

**Surface**: all 200 single-series indicators have a `ct_*_d` variant
(declared in `include/tacuda_device.h`), plus `_ex_d` precision variants
(float16/32/64) and `ct_*_batch_d` for device-resident portfolios — see
[docs/DEVICE_API.md](docs/DEVICE_API.md).

---

### ⚙️ Exponential Moving Average Acceleration

The EMA family of indicators now share a single CUDA implementation based on a
prefix-scan formulation of the linear recurrence:

```
EMA[i] = α · x[i] + (1 − α) · EMA[i − 1]
```

Instead of iterating sequentially, we interpret each update as an affine
transformation and compute the cumulative product of these transforms with
`thrust::inclusive_scan`. This yields all intermediate EMA values in parallel,
which are then re-used across EMA, DEMA, TEMA, T3, TRIX and MACD calculations.
The shared helper keeps warm-up regions initialised to `NaN` while guaranteeing
identical numerical outputs to the previous per-kernel loops.

To reproduce the performance improvement for large smoothing periods, run the
dedicated benchmark:

```bash
python benchmarks/benchmark_ema_vs_talib.py
```

---

## 🏗️ Architecture

```
┌───────────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  Language bindings    │    │   Flat C ABI         │    │  CUDA kernels    │
│  Python / Node / C#   │───▶│   tacuda.h (559)     │───▶│  one indicator = │
│  (thin, generated)    │    │   tacuda_device.h    │    │  one .cu class   │
└───────────────────────┘    │   (200 ct_*_d)       │    └──────────────────┘
                             └──────────────────────┘             │
                             ┌──────────────────────┐    ┌──────────────────┐
                             │  C++ RAII layer      │    │  DeviceBufferPool│
                             │  Context /           │    │  (cached, stream-│
                             │  DeviceBuffer        │    │  ordered frees)  │
                             └──────────────────────┘    └──────────────────┘
```

### Core Components

- **Indicator classes** (`include/indicators/*.h` + `src/indicators/*.cu`): one CUDA implementation per indicator, shared helpers for EMA-scans, Wilder smoothing, candle settings and the Hilbert state machine
- **C ABI** (`src/api/*.cu`): thin `extern "C"` wrappers with a uniform error contract (`ctStatus_t` + `ct_get_last_error()`); the batch/device surface is generated from one family table (`scripts/generate_batch_api.py`)
- **Context / DeviceBuffer** (`include/tacuda/Context.h`): RAII C++ pipeline API; buffers may safely outlive the Context, cross-Context use is rejected
- **DeviceBufferPool**: allocation cache with stream-ordered release (capped via `ct_device_buffer_pool_set_max_cache_bytes`)
- **Bindings** (`bindings/`): Python (ctypes) and C# (P/Invoke) generated from the headers; Node.js (koffi) hand-written plus a generated batch/device layer — all audited for 811/811 coverage

---

## 🧪 Testing & Quality

```bash
# C++ unit tests (391 tests incl. optimizer + lifetime contracts)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j && ctest --test-dir build --output-on-failure

# Strict differential corpus vs installed TA-Lib (GPU required)
export TACUDA_LIBRARY=$PWD/build/libtacuda.so PYTHONPATH=$PWD/bindings/python
python benchmarks/differential_corpus.py

# Binding pipelines
python bindings/python/test_pipeline.py
node bindings/nodejs/test_pipeline.js
dotnet run --project bindings/csharp/ConsoleExample -c Release -- --test

# Performance benchmarks
python benchmarks/run_all_benchmarks.py
```

**Test Coverage:**
- ✅ Numerical parity vs TA-Lib: 165/165 corpus specs over 8 datasets (random walks, trend, flat, zeros, short inputs, candle fixtures)
- ✅ Edge cases (short inputs, NaN warm-up, zero denominators, parameter bounds)
- ✅ Memory safety: RAII buffers, stream-ordered pool, buffer/Context lifetime tests
- ✅ Binding coverage audit (`bindings/audit_bindings.py`, CI-gated)
- ✅ Cross-platform: Linux + Windows builds and GPU test runs in CI

---

## 🔧 Extending TACUDA

Adding a new indicator touches four places (see
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full workflow):

1. **Indicator class** — `include/indicators/MyIndicator.h` +
   `src/indicators/MyIndicator.cu` with a
   `calculate(const float* in, float* out, int size, cudaStream_t)` method;
   fill the warm-up region with `NaN` like the existing kernels.
2. **C ABI entry points** — either a hand-written `ct_my_indicator` wrapper in
   `src/api/`, or (preferred for single-series + batch + device coverage) a
   one-line family entry in `scripts/generate_batch_api.py`, then re-run it.
3. **Bindings** — `python bindings/generate_bindings.py` regenerates the
   Python/C# layers from the headers; add a Node wrapper in
   `bindings/nodejs/index.js` if it is not covered by the generated layer.
   `bindings/audit_bindings.py` must report full coverage.
4. **Tests** — a GoogleTest case in `tests/cpp/`, and, if TA-Lib has the same
   function, a spec in `benchmarks/talib_compare_helpers.py` so the
   differential corpus guards parity forever.

---

## ✅ Status & TODO

### ✅ Implemented as of 0.8.0

- [x] **All 161 functions from TA-Lib 0.7.1 implemented** (incl. MACDEXT with independent MA types and 19 element-wise vector math functions); numerical parity is verified per function by the strict differential corpus — 165/165 specs green, remaining corner-case notes in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)
- [x] **All 61 candlestick patterns** match TA-Lib signal-for-signal (double-precision literal ports, exact-equality oracle with hand-crafted activation fixtures for the rare patterns)
- [x] **Device API** (`ct_*_d`, 200 functions) for zero-copy multi-indicator pipelines, with `_ex_d` float16/32/64 precision variants
- [x] **Batch API**: 166 indicator families over a stream pool (`maxConcurrency` control)
- [x] Python (ctypes), Node.js (koffi) and .NET 8/9 bindings covering the full 811-symbol C ABI (audited in CI)
- [x] Packaging workflow for PyPI (platform wheels with natives), npm (tgz) and NuGet (nupkg with win-x64 + linux-x64 natives), GPU-gated publishing
- [x] **SAREXT** bit-for-bit TA-Lib semantics; SAR/SAREXT host API uses a fast CPU path
- [x] Memory-safety contracts: stream-ordered buffer pool, buffers may outlive their Context, cross-Context rejection — all covered by lifetime tests
- [x] `tacuda-opt` strategy optimizer (grid / genetic / walk-forward / portfolio / Monte-Carlo) with the 94-system EXPERIMENTS pack

### 🗺️ What's next

Current priorities live in **[docs/ROADMAP.md](docs/ROADMAP.md)**:

- finish the streaming line: site section, live-trading guide, the ADX group and STOCH/STOCHF as stream families;
- TA-Lib 0.8.1 coverage (40 missing corpus specs, IMI/WILLR edge semantics);
- persistent batch plans (deferred on measurement — see the roadmap) and multi-GPU dispatch;
- Conda/Docker/Jetson distribution;
- **v1.0.0** — ABI freeze, signed releases, GPU test matrix in CI, LTS branch.

---

## 📚 Documentation

- **[tacuda.com](https://tacuda.com)** - Project website: live benchmark tables and a page for each of the 94 [EXPERIMENTS](https://tacuda.com/experiments.html) systems
- **[API Reference](docs/API.md)** - Conventions (statuses, NaN warm-up, dtypes, streams), function catalog, binding surface, OHLCV containers
- **[Streaming API](docs/STREAMING_API.md)** - Incremental single-series and portfolio updates
- **[Device API](docs/DEVICE_API.md)** - Zero-copy GPU pipelines (`ct_*_d`, `_ex_d`, buffer pool)
- **[Batch API](docs/BATCH_API.md)** - Multi-symbol/portfolio execution (166 families)
- **[Implementation Notes](docs/IMPLEMENTATION_NOTES.md)** - TA-Lib parity semantics and corner cases
- **[Performance Guide](docs/PERFORMANCE.md)** - Which API is fast where, tuning rules
- **[Operations Guide](docs/OPERATIONS.md)** - Deployment, monitoring, upgrades, production adoption checklist
- **[Strategy Optimizer](docs/STRATEGY_OPTIMIZER.md)** - `tacuda-opt`: GPU grid search, genetic, walk-forward, portfolio, Monte-Carlo
- **[EXPERIMENTS pack](docs/EXPERIMENTS.md)** - 94 classic STAD Club systems ported to the optimizer
- **[Roadmap](docs/ROADMAP.md)** - Current state and planned milestones
- **[Changelog](docs/CHANGELOG.md)** - Canonical release history
- **[Contributing](docs/CONTRIBUTING.md)** - Development workflow, ABI stability policy, release criteria

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/Ta-Cuda/tacuda.git
cd tacuda

# Build with tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Contribution Guidelines

- 🎯 **Focus on performance**: Include benchmarks for performance-affecting changes
- 🧪 **Test coverage**: New features must include comprehensive tests
- 📝 **Documentation**: Update docs for API changes
- 🎨 **Code style**: Use `clang-format` and maintain consistency
- 💬 **Discussion**: Open an issue before major architectural changes

### Code Quality Standards

- ✅ All tests must pass
- ✅ No compiler warnings
- ✅ Memory leak-free
- ✅ Thread-safe implementations
- ✅ Proper error handling

---

## 💼 Commercial Support & Integration

TaCuda is completely free and open source (Apache 2.0).

However, integrating GPU acceleration into existing HFT/Algo-trading infrastructure often requires architectural expertise. I offer professional consulting services for:
*   **Custom Integration**: Embedding TaCuda into proprietary trading engines (C++, C#, Python).
*   **Performance Optimization**: Profiling and reducing latency in your specific pipeline.
*   **Strategy Implementation**: Porting complex strategies to CUDA/C++ for maximum speed.
*   **Infrastructure Design**: Building end-to-end zero-copy trading systems.

**[Contact me on LinkedIn](https://www.linkedin.com/in/deecalov)** to discuss your high-performance computing needs.

---

## 📄 License

`SPDX-License-Identifier: Apache-2.0`

Copyright (c) 2024-2026 [Paul Deecalov](mailto:paul@deecalov.com) — licensed under the
Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text; every source
file carries the SPDX tag in its signature header.

---

## 🙏 Acknowledgments

- **NVIDIA** for CUDA toolkit and documentation
- **TA-Lib** for algorithmic reference implementations  
- **NumPy** community for API design inspiration
- **GoogleTest** for testing framework

---

## ⚠️ Disclaimer

This project is independent and not affiliated with TA-Lib or any other trademark. "TA-style API" refers to interface similarity for user convenience only.

---

<div align="center">

**[⭐ Star this repo](https://github.com/Ta-Cuda/tacuda)** if you find TACUDA useful!

🌐 **[tacuda.com](https://tacuda.com)** | **Made with ❤️ by [Paul Deecalov](https://www.linkedin.com/in/deecalov)**

</div>
