# TaCuda — Python Bindings

Python package for the native `tacuda` CUDA library.
TA-Lib 0.7.1 baseline coverage (161 functions): indicators, MACDEXT, 19 vector math kernels,
candlestick patterns, plus typed core APIs with runtime dtype selection and the
zero-copy device API.

## Prerequisites

- NVIDIA GPU (Compute Capability >= 6.0)
- CUDA Toolkit 11.x or 12.x
- Built `tacuda` shared library (`tacuda.dll` / `libtacuda.so` / `libtacuda.dylib`)
- Python >= 3.8, NumPy >= 1.20

## Package Scope

- High-level NumPy wrappers for indicator functions
- Precision-aware core indicators (`sma`, `ema`, `rsi`, `macd`, `atr`, `bbands`) via `dtype="float16|float32|float64"`
- `OHLCV` convenience container and input validation helpers

## Installation

```bash
pip install tacuda
```

> **Note:** The `tacuda` native library must be built separately and placed where
> the Python bindings can find it. See "Library Location" below.

## Quick Start

```python
import numpy as np
from tacuda import sma, ema, rsi, macd, bbands, OHLCV

# Generate sample data
prices = np.random.randn(100_000).cumsum().astype(np.float32) + 100

# Moving averages
sma20 = sma(prices, 20)
ema12 = ema(prices, 12)

# RSI
rsi14 = rsi(prices, 14)

# MACD (returns macd_line, signal, histogram)
macd_line, signal, hist = macd(prices, 12, 26, 9)

# Bollinger Bands (returns upper, middle, lower)
upper, middle, lower = bbands(prices, 20, 2.0, 2.0)

# OHLCV container
ohlcv = OHLCV.from_columns(open_arr, high_arr, low_arr, close_arr, volume_arr)
```

### MACDEXT and vector math

```python
import tacuda

# Independent MA types per line (0=SMA 1=EMA 2=WMA ... 8=T3)
ext_macd, ext_signal, ext_hist = tacuda.macdext(
    prices, fastPeriod=12, fastType=1, slowPeriod=26, slowType=0,
    signalPeriod=9, signalType=2)

# Element-wise GPU kernels
log_prices = tacuda.ln(prices)
spread = tacuda.sub(high_arr, low_arr)
```

### Zero-copy device pipeline

```python
n = len(prices)
d_in = tacuda.device_alloc(n)
d_out = tacuda.device_alloc(n)
try:
    tacuda.device_upload(prices, d_in)      # 1 PCIe transfer in
    tacuda.sma_d(d_in, d_out, n, 20)        # compute on GPU
    tacuda.sin_d(d_out, d_out, n)           # chain kernels, zero copies
    tacuda.device_sync()
    result = tacuda.device_download(d_out, n)  # 1 PCIe transfer out
finally:
    tacuda.device_free(d_in)
    tacuda.device_free(d_out)
```

`device_upload` takes its element count from the array. An explicit `count` may
transfer a prefix but never more than the array holds: the native call receives
only a pointer, so this is checked before it. The raw `tacuda._lib` entry points
do not have this protection.

Single-series functions require a 1D array and raise `ValueError` otherwise,
because a matrix would be read as one concatenated series crossing symbol
boundaries. Use the `*_batch` functions for `(numSeries, seriesLen)` matrices.

## Available Indicators (TA-Lib 0.7.1 baseline coverage)

| Category | Indicators |
|----------|-----------|
| **Moving Averages** | SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MAMA, MAVP, MA |
| **Momentum** | RSI, MACD, MACDEXT, MACDFIX, Stochastic, StochF, StochRSI, CCI, CMO, ROC, ROCP, ROCR, ROCR100, MOM, WILLR, APO, PPO, PVO, TRIX, ULTOSC, BOP |
| **Volatility** | ATR, NATR, TRANGE, BBANDS, ACCBANDS, StdDev, VAR |
| **Trend** | ADX, ADXR, DX, +DI, -DI, +DM, -DM, SAR, SAREXT, Aroon, AroonOsc |
| **Volume** | AD, ADOSC, OBV, MFI, IMI, NVI, PVI |
| **Statistical** | LINEARREG, SLOPE, INTERCEPT, ANGLE, TSF, CORREL, BETA, AVGDEV, SUM, MAX, MIN |
| **Price Transform** | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, MIDPRICE |
| **Hilbert Transform** | HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE |
| **Vector Math (19)** | SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH, EXP, LN, LOG10, SQRT, CEIL, FLOOR, ADD, SUB, MULT, DIV |
| **Candlestick** | 63 patterns (Doji, Hammer, Engulfing, MorningStar, etc.) |
| **Device API** | `ct_*_d` variants of the above + `device_alloc/upload/download/sync/free` and buffer-pool controls |

## Library Location

The bindings search for the native library in this order:

1. `TACUDA_LIBRARY` environment variable (full path to shared library)
2. `TACUDA_LIBRARY_PATH` / `TACUDA_LIBRARY_DIR` + library name
3. System library path (`ctypes.util.find_library`)
4. Common build directories relative to the package

## Regenerating Metadata

When the C header changes, regenerate from the repository root:

```bash
python bindings/generate_bindings.py
```

## Build & Publish (PyPI)

Releases are built and published only by the `release-packages` workflow
(see [CONTRIBUTING.md](../../docs/CONTRIBUTING.md#release-criteria)): one
platform wheel per native plus a source-only sdist. For a local build, make
the sdist before any native is in `tacuda/` (setuptools packs package data
into it), then put exactly one native there and build only the wheel. Wipe
the setuptools cache before each build: `build/lib` is never pruned, so a
second wheel would otherwise also carry the previous build's native:

```bash
cd bindings/python
rm -rf build dist ./*.egg-info
python -m build --sdist          # tacuda/ must not contain a native yet
rm -rf build ./*.egg-info
cp /path/to/libtacuda.so tacuda/ # or tacuda.dll for a Windows wheel
python -m build --wheel
python -m twine check dist/*
```

## Changelog

Canonical release history is maintained in [`CHANGELOG.md`](../../docs/CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](../../LICENSE).

## Streaming and version pairing

Streaming was added in 0.8.0 for incremental single-series and portfolio
updates across 10 indicator families. Pair bindings with the native library
from the same release; 0.9.0 also includes precision, input-validation and stream
correctness fixes. A native older than 0.8.0 does not export the streaming
functions. See the
[Streaming API guide](https://github.com/Ta-Cuda/tacuda/blob/main/docs/STREAMING_API.md).
The 161-function compatibility baseline is native TA-Lib 0.7.1, not TA-Lib 0.8.1.

## Fused portfolio batches (0.9.0)

Seventeen indicator families can process a portfolio in one CUDA kernel
per call. Supported device batches are CUDA Graph compatible; user-stream
calls stay asynchronous. `TACUDA_BATCH_FUSED=auto|0|1` controls automatic,
per-series or forced fused dispatch. Period/shape limits apply, and the
double internal arithmetic can change the last float32 bits from 0.8.
See [batch dispatch and migration](https://github.com/Ta-Cuda/tacuda/blob/v0.9.0/docs/BATCH_API.md#fused-execution).
