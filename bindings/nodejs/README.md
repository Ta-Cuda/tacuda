# TaCuda — Node.js Bindings

Node.js package for the native `tacuda` CUDA library.
Full TA-Lib coverage (161 functions): indicators, candlestick patterns, MACDEXT,
19 vector math kernels, precision (`_ex`) variants, and the zero-copy device API.

## Prerequisites

- Node.js >= 18.0.0
- NVIDIA GPU (Compute Capability >= 6.0)
- CUDA Toolkit 11.x or 12.x
- Built `tacuda` shared library (`tacuda.dll` / `libtacuda.so` / `libtacuda.dylib`)

## Package Scope

- JavaScript wrappers for the complete 395-function native surface (host + device + batch API)
- Structured OHLCV inputs (`Ohlcv` class or plain `{open, high, low, close, volume}` objects)
- Batched multi-symbol execution: `rsiBatch`, `smaBatch`, `macdBatch`, ..., `device.*Batch`
- Full candlestick registry via `cdlPattern()` and `CDL_NAMES`
- Works with `Float32Array` inputs (and `Float64Array` for `_ex` precision variants)
- `tacuda.device.*` namespace for zero-copy GPU pipelines, `tacuda.bufferPool` controls

## Installation

```bash
npm install tacuda
```

> **Note:** The `tacuda` native library must be built separately and placed where
> the Node.js bindings can find it. See "Library Location" below.

## Quick Start

```javascript
const tacuda = require('tacuda');

const close = Float32Array.from(
    { length: 100000 }, (_, i) => 100 + Math.sin(i * 0.01) * 10
);

// Moving averages
const sma20 = tacuda.sma(close, 20);
const ema12 = tacuda.ema(close, 12);

// RSI
const rsi14 = tacuda.rsi(close, 14);

// MACD (returns { macd, signal, hist })
const macdResult = tacuda.macd(close, 12, 26, 9);

// Bollinger Bands (returns { upper, middle, lower })
const bbands = tacuda.bbands(close, 20, 2.0, 2.0);

// MACDEXT with independent MA types (returns { macd, signal, hist })
const ext = tacuda.macdext(close, 12, tacuda.MA_TYPE.EMA,
                           26, tacuda.MA_TYPE.SMA, 9, tacuda.MA_TYPE.WMA);

// Vector math (element-wise GPU kernels)
const logClose = tacuda.ln(close);
const spread = tacuda.sub(high, low);

// Candlestick patterns (63 available)
const doji = tacuda.cdlDoji(open, high, low, close);
const hammer = tacuda.cdlPattern('hammer', open, high, low, close);

console.log(`RSI[100]: ${rsi14[100]}`);
console.log(`MACD signal: ${macdResult.signal[100]}`);
```

### Zero-copy device pipeline

```javascript
const n = close.length;
const dIn = tacuda.device.alloc(n);      // GPU buffer (opaque pointer)
const dOut = tacuda.device.alloc(n);

tacuda.device.upload(dIn, close);        // 1 PCIe transfer in
tacuda.device.sma(dIn, dOut, n, 20);     // compute on GPU
tacuda.device.sin(dOut, dOut, n);        // chain more kernels, zero copies
tacuda.device.sync();
const result = tacuda.device.download(dOut, n);  // 1 PCIe transfer out

tacuda.device.free(dIn);
tacuda.device.free(dOut);
```

### Double precision (`_ex` variants)

```javascript
const close64 = Float64Array.from(close);
const sma64 = tacuda.smaEx(close64, 20);   // float64 storage + compute
const { macd } = tacuda.macdEx(close64, 12, 26, 9, tacuda.MA_TYPE.EMA);
```

## Available Indicators (full TA-Lib coverage)

### Moving Averages
`sma`, `ema`, `wma`, `dema`, `tema`, `trima`, `kama`, `t3`, `mama`, `mavp`, `ma`

### Momentum & Oscillators
`rsi`, `macd`, `macdext`, `macdfix`, `macdLine`, `stochastic`, `stochf`, `stochRsi`, `cci`, `cmo`, `roc`, `rocp`, `rocr`, `rocr100`, `momentum`, `change`, `willr`, `apo`, `ppo`, `pvo`, `trix`, `ultosc`, `bop`

### Volatility
`atr`, `natr`, `trange`, `bbands`, `accbands`, `stddev`, `variance`

### Trend & Directional
`adx`, `adxr`, `dx`, `plusDi`, `minusDi`, `plusDm`, `minusDm`, `sar`, `sarext`, `aroon`, `aroonOsc`

### Volume
`ad`, `adosc`, `obv`, `mfi`, `imi`, `nvi`, `pvi`

### Statistical
`linearReg`, `linearRegSlope`, `linearRegIntercept`, `linearRegAngle`, `tsf`, `correl`, `beta`, `avgdev`, `sum`, `max`, `min`, `maxIndex`, `minIndex`, `minMax`, `minMaxIndex`, `midpoint`

### Price Transform
`avgPrice`, `medPrice`, `typPrice`, `wclPrice`, `midPrice`

### Hilbert Transform
`htDcPeriod`, `htDcPhase`, `htPhasor`, `htSine`, `htTrendline`, `htTrendmode`

### Vector Math (19, element-wise)
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log10`, `sqrt`, `ceil`, `floor`, `add`, `sub`, `mult`, `div`

### Precision variants (`Float64Array` in/out)
`smaEx`, `emaEx`, `rsiEx`, `macdEx`, `atrEx`, `bbandsEx`

### Device API (`tacuda.device.*`, raw GPU pointers)
Memory: `alloc`, `allocEx`, `free`, `freeEx`, `upload`, `uploadEx`, `download`, `downloadEx`, `sync`.
Indicators: `sma`, `ema`, `wma`, `dema`, `tema`, `trima`, `kama`, `t3`, `rsi`, `momentum`, `roc`, `rocp`, `rocr`, `rocr100`, `cmo`, `stddev`, `var`, `sum`, `max`, `min`, `linearreg` family, `tsf`, `atr`, `natr`, `trange`, `adx`, `adxr`, `dx`, `plusDi`, `minusDi`, `willr`, `cci`, `macd`, `macdext`, `stochastic`, `stochasticFast`, `ultosc`, `bbands`, `accbands`, `ad`, `adosc`, `obv`, `mfi`, price transforms, `aroon`, `aroonosc`, all 19 vector math kernels, `_ex` precision variants, plus `change`, `trix`, `maxindex`, `minindex`, `ma`, `macdLine`, `apo`, `ppo`, `pvo`.
Pool: `tacuda.bufferPool.cleanup()`, `allocationCount()`, `setMaxCacheBytes(bytes)`.

### Candlestick Patterns (63)
`cdlDoji`, `cdlHammer`, `cdlEngulfing`, `cdlMorningStar`, `cdlEveningStar`, `cdlThreeWhiteSoldiers`, `cdlThreeBlackCrows`, and 56 more via `cdlPattern(name, open, high, low, close)`.

Full list available via `tacuda.CDL_NAMES`.

## Library Location

The bindings search for the native library in this order:

1. `TACUDA_LIBRARY` environment variable
2. `../../build/Release/tacuda.dll` (relative to this directory)
3. `../../build/Debug/tacuda.dll`
4. Common system locations

## Running Tests

```bash
npm test
# Runs 173 tests covering all indicators, vector math, MACDEXT, _ex and device API
```

## Build & Publish (npm)

```bash
cd bindings/nodejs
npm pack --dry-run
npm publish --access public
```

## Changelog

Canonical release history is maintained in [`CHANGELOG.md`](../../docs/CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](../../LICENSE).
