# TaCuda — Node.js Bindings

Version **0.9.3** covers the **237 TA-Lib 0.8.3 functions** with matching
native libraries and bindings. GPU RSI uses double-precision Wilder state.
The Python and Node.js loaders validate explicit native-library overrides.
See the [compatibility guide](https://github.com/Ta-Cuda/tacuda/blob/v0.9.3/docs/TA_LIB_COMPATIBILITY.md).
New streaming families are a separate API surface. Third-party formula
licenses ship in `THIRD_PARTY_NOTICES.txt`.

Node.js package for the native `tacuda` CUDA library.
TA-Lib 0.8.3 coverage (237 functions): indicators, candlestick patterns, MACDEXT,
19 vector math kernels, precision (`_ex`) variants, and the zero-copy device API.

## Prerequisites

- Node.js >= 18.0.0
- NVIDIA GPU (Compute Capability >= 6.0)
- For official binaries: a driver compatible with CUDA 12.9; source builds use the toolkit selected by CMake
- A native library matching the npm version

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

The npm package contains the JavaScript bindings. On Windows x64 you can use
the DLL from a matching Python wheel. With the CUDA 12 toolkit installed,
run these commands from your npm project in PowerShell:

```powershell
python -m pip install tacuda==0.9.3
npm install --save-exact tacuda@0.9.3
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
$env:TACUDA_LIBRARY = python -c "from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec('tacuda').origin).with_name('tacuda.dll'))"
node -e "const t=require('tacuda'); console.log(t.sma(Array.from({length:64},(_,i)=>i+1),20).at(-1))"  # 54.5
```

A native release download or a matching source build also works. Set
`TACUDA_LIBRARY` to its full path and start a new Node process.

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
4. On Windows, `../../Release/tacuda.dll`, then a DLL beside the bindings, then the working directory

## Running Tests

```bash
npm test
# Runs the binding pipeline, including indicators, batch, streaming and device API
```

## Build & Publish (npm)

```bash
cd bindings/nodejs
npm pack --dry-run
npm publish --access public
```

## Changelog

Canonical release history is maintained in [`CHANGELOG.md`](../../CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](../../LICENSE).

## Fused portfolio batches (0.9.0)

Seventeen indicator families can process a portfolio in one CUDA kernel
per call. Supported device batches are CUDA Graph compatible; user-stream
calls stay asynchronous. `TACUDA_BATCH_FUSED=auto|0|1` controls automatic,
per-series or forced fused dispatch. Period/shape limits apply, and the
double internal arithmetic can change the last float32 bits from 0.8.
See [batch dispatch and migration](https://github.com/Ta-Cuda/tacuda/blob/v0.9.3/docs/BATCH_API.md#fused-execution).
