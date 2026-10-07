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
    tacuda.device_upload(prices, d_in, n)   # 1 PCIe transfer in
    tacuda.sma_d(d_in, d_out, n, 20)        # compute on GPU
    tacuda.sin_d(d_out, d_out, n)           # chain kernels, zero copies
    tacuda.device_sync()
    result = tacuda.device_download(d_out, n)  # 1 PCIe transfer out
finally:
    tacuda.device_free(d_in)
    tacuda.device_free(d_out)
```

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

```bash
cd bindings/python
python -m build
python -m twine check dist/*
python -m twine upload dist/*
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
