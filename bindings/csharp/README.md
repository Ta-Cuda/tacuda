# TaCuda — .NET/C# Bindings

Managed bindings for the native `tacuda` CUDA library.
TA-Lib 0.7.1 baseline coverage (161 functions, 811 native entrypoints incl. the batch API): generated P/Invoke
signatures for host API, device API, MACDEXT, 19 vector math kernels, and typed
`_ex` entrypoints. The `DllImport` name is platform-neutral (`tacuda`), so the same
assembly loads `tacuda.dll`, `libtacuda.so`, or `libtacuda.dylib` automatically.

## Prerequisites

- .NET 8.0+ or .NET 9.0+
- NVIDIA GPU (Compute Capability >= 6.0)
- CUDA Toolkit 11.x or 12.x
- Built `tacuda` shared library (`tacuda.dll` / `libtacuda.so` / `libtacuda.dylib`)

## Package Scope

- `NativeMethods.ct_*` methods generated from `include/tacuda.h` / `include/tacuda_device.h`
- Precision-aware core entrypoints (`*_ex`) with runtime dtype pair selection
- OHLCV helper model (`OhlcvSeries`) for convenient column-major inputs

## Installation

### NuGet

```bash
dotnet add package Tacuda
```

### From Source

```bash
# Build the native library first
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Reference the binding project
dotnet add reference bindings/csharp/Tacuda.Bindings.csproj
```

> **Note:** The `tacuda` native library must be discoverable at runtime.
> Place it next to the published binary or add its location to `PATH` / `LD_LIBRARY_PATH`.

## Quick Start

```csharp
using Tacuda.Bindings;

// Prepare data
var prices = new float[100000];
var sma = new float[100000];
var rsi = new float[100000];

// Fill prices with your data...

// Calculate indicators (GPU-accelerated)
NativeMethods.ct_sma(prices, sma, prices.Length, 20);
NativeMethods.ct_rsi(prices, rsi, prices.Length, 14);

// MACD (multiple outputs)
var macd = new float[100000];
var signal = new float[100000];
var hist = new float[100000];
NativeMethods.ct_macd(prices, macd, signal, hist, prices.Length, 12, 26, 9, 0);

// OHLCV container
var candles = new OhlcvSeries(open, high, low, close, volume);
var output = new float[candles.Length];
NativeMethods.ct_atr(candles.High, candles.Low, candles.Close, output, candles.Length, 14);

// MACDEXT with independent MA types (1=EMA, 0=SMA, 2=WMA)
NativeMethods.ct_macdext(prices, macd, signal, hist, prices.Length,
                         12, 1, 26, 0, 9, 2);

// Vector math (element-wise GPU kernels)
var logs = new float[100000];
NativeMethods.ct_ln(prices, logs, prices.Length);

// Zero-copy device pipeline
IntPtr dIn = NativeMethods.ct_device_alloc(prices.Length);
IntPtr dOut = NativeMethods.ct_device_alloc(prices.Length);
NativeMethods.ct_device_upload(dIn, prices, prices.Length);
NativeMethods.ct_sma_d(dIn, dOut, prices.Length, 20);   // compute on GPU
NativeMethods.ct_sin_d(dOut, dOut, prices.Length);      // chain kernels
NativeMethods.ct_device_sync();
var result = new float[prices.Length];
NativeMethods.ct_device_download(result, dOut, prices.Length);
NativeMethods.ct_device_free(dIn);
NativeMethods.ct_device_free(dOut);
```

## Available Indicators (TA-Lib 0.7.1 baseline coverage)

All indicators are available as static `NativeMethods.ct_*` methods:

| Category | Methods |
|----------|---------|
| **Moving Averages** | `ct_sma`, `ct_ema`, `ct_wma`, `ct_dema`, `ct_tema`, `ct_trima`, `ct_kama`, `ct_t3`, `ct_mama`, `ct_mavp`, `ct_ma` |
| **Momentum** | `ct_rsi`, `ct_macd`, `ct_macdext`, `ct_macdfix`, `ct_stochastic`, `ct_stochf`, `ct_stochrsi`, `ct_cci`, `ct_cmo`, `ct_roc`, `ct_momentum`, `ct_willr`, `ct_apo`, `ct_ppo`, `ct_pvo`, `ct_trix`, `ct_ultosc`, `ct_bop` |
| **Volatility** | `ct_atr`, `ct_natr`, `ct_trange`, `ct_bbands`, `ct_accbands`, `ct_stddev`, `ct_var` |
| **Trend** | `ct_adx`, `ct_adxr`, `ct_dx`, `ct_plus_di`, `ct_minus_di`, `ct_plus_dm`, `ct_minus_dm`, `ct_sar`, `ct_sarext`, `ct_aroon`, `ct_aroon_osc` |
| **Volume** | `ct_ad`, `ct_adosc`, `ct_obv`, `ct_mfi`, `ct_imi`, `ct_nvi`, `ct_pvi` |
| **Statistical** | `ct_linearreg`, `ct_linearreg_slope`, `ct_linearreg_intercept`, `ct_linearreg_angle`, `ct_tsf`, `ct_correl`, `ct_beta`, `ct_avgdev` |
| **Hilbert Transform** | `ct_ht_dcperiod`, `ct_ht_dcphase`, `ct_ht_phasor`, `ct_ht_sine`, `ct_ht_trendline`, `ct_ht_trendmode` |
| **Vector Math (19)** | `ct_sin`, `ct_cos`, `ct_tan`, `ct_asin`, `ct_acos`, `ct_atan`, `ct_sinh`, `ct_cosh`, `ct_tanh`, `ct_exp`, `ct_ln`, `ct_log10`, `ct_sqrt`, `ct_ceil`, `ct_floor`, `ct_add`, `ct_sub`, `ct_mult`, `ct_div` |
| **Candlestick** | 63 `ct_cdl_*` patterns |

Device API variants (`ct_*_d` and `ct_*_device`) plus `ct_device_alloc/upload/download/sync/free`
and buffer-pool controls are available for zero-copy GPU pipelines.

## NuGet Packaging

To build the NuGet package with native libraries:

```bash
# 1. Build native libraries for target platforms
# 2. Place them in runtimes/ folder:
#    runtimes/win-x64/native/tacuda.dll
#    runtimes/linux-x64/native/libtacuda.so

# 3. Pack
cd bindings/csharp
dotnet pack -c Release
```

To publish:

```bash
dotnet nuget push bin/Release/*.nupkg --source https://api.nuget.org/v3/index.json --api-key <NUGET_API_KEY>
```

## Regenerating Bindings

After editing `include/tacuda.h`:

```bash
python bindings/generate_bindings.py
```

CTest verifies that the generated files are up to date.

## Running Tests

```bash
dotnet run --project bindings/csharp/ConsoleExample -- --test
# Runs 100 tests covering all indicator categories, vector math, MACDEXT and device API
```

## Changelog

Canonical release history is maintained in [`CHANGELOG.md`](../../docs/CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](../../LICENSE).

## Streaming and version pairing

The 0.8.0 bindings add incremental single-series and portfolio updates for
10 indicator families. Use a matching 0.8.0 native library; an older binary
does not export the streaming functions. See the
[Streaming API guide](https://github.com/Ta-Cuda/tacuda/blob/main/docs/STREAMING_API.md).
The 161-function compatibility baseline is native TA-Lib 0.7.1, not TA-Lib 0.8.1.
