# Batched Execution API (Multi-Symbol / Portfolio)

Added in v0.2.0, extended to 52 indicator families in v0.3.0 and to the
full indicator surface (166 families, including the 63 candlestick
patterns and the STAD club pack) in v0.4.0. The batch API computes the
same indicator over many independent series (symbols) in one call,
dispatching work across an internal pool of CUDA streams so transfers
and kernels overlap.

## Covered families (v0.4.0)

Every family below has a host `ct_<name>_batch` and a device
`ct_<name>_batch_d` entry point. As of 0.4.0 every indicator family with
a device (`ct_*_d`) function is batched — and every family that
previously lacked a device function got one (generated wrappers over the
CUDA indicator classes):

| Group | Families |
|-------|----------|
| Moving averages | SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, **MA, MAVP** |
| Momentum | RSI, MOMENTUM, ROC, ROCP, ROCR, ROCR100, CMO, MACD, MACDEXT, **MACDFIX, MACD_LINE, CHANGE, TRIX, APO, PPO, PVO, STOCHRSI**, STOCHASTIC, STOCHF, WILLR, CCI, ULTOSC |
| Trend / directional | ADX, ADXR, DX, +DI, -DI, **+DM, -DM**, AROON, AROONOSC, **SAR, SAREXT, MAMA** |
| Volatility / bands | ATR, NATR, TRANGE, BBANDS, ACCBANDS, STDDEV, VAR, **AVGDEV** |
| Statistics / regression | SUM, MAX, MIN, **MAXINDEX, MININDEX, MINMAX, MINMAXINDEX, MIDPOINT, MIDPRICE, CORREL, BETA**, LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE, TSF |
| Volume | OBV, MFI, AD, ADOSC, **NVI, PVI, IMI, BOP** |
| Price transforms | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE |
| Hilbert transform | **HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE** |
| Candlestick patterns | **all 63 `cdl_*` patterns** (`cdl_doji` … `cdl_xside_gap_3_methods`) |
| STAD club pack | **ASI, SAR_EL, OBTR, WOBV, HPI, MARKET_STRENGTH, TRENDSCORE, REFDEV, SKEW, KURTOSIS, PIVOT_HIGH, PIVOT_LOW, XAVERAGE, ROLLING_MEAN/MAX/MIN/STD** |

The STAD club pack (bold last row) are indicators from the Omega
Research System Trading & Development Club volumes with no TA-Lib
counterpart; they also get plain single-series host entry points
(`ct_asi`, `ct_obtr`, …). The `ROLLING_*`/`XAVERAGE` families are
NaN-strict / NaN-aware variants of SMA/MAX/MIN/STDDEV/EMA designed for
derived (chained) input lines with NaN warm-up prefixes.

Since 0.4.1 the cumulative club lines are parallel scans: OBTR/WOBV/ASI
are prefix sums of per-bar terms (thrust inclusive scan in double), HPI
is an `AffineScan` recurrence like EMA, and XAverage scans a seed-aware
affine monoid whose NaN inputs are identities — one series saturates the
GPU on its own. Genuinely sequential state machines (SAR, SAREXT,
SAR_EL, MAMA, HT_*) still run one GPU thread per series; their
throughput comes from batching many series, not from parallelizing one.

Vector-math functions (SIN, ADD, ...) are intentionally not batched:
element-wise kernels are shape-agnostic, so calling e.g. `ct_sin` on the
flattened `numSeries * seriesLen` buffer is already the optimal batch —
one kernel launch instead of `numSeries`.

New families are produced by `scripts/generate_batch_api.py` from a
single table; regenerate with:

```bash
python scripts/generate_batch_api.py   # header + CUDA + Node/C# wrappers
python bindings/generate_bindings.py   # ctypes / P-Invoke layers
```

## Data layout

All batch entry points take dense **series-major** matrices:
`numSeries` rows of `seriesLen` float32 elements, row `i` = series `i`,
rows contiguous in memory. Every series is processed independently with
identical parameters — exactly equivalent to calling the single-series
function per row, but with far less per-call overhead.

## Native C API

Host-matrix variants (upload → dispatch → download, synchronous):

```c
ctStatus_t ct_sma_batch (const float* inputs, float* outputs,
                         int seriesLen, int numSeries, int period,
                         int maxConcurrency, cudaStream_t stream);
ctStatus_t ct_ema_batch (...);          // same shape
ctStatus_t ct_rsi_batch (...);
ctStatus_t ct_atr_batch (const float* highs, const float* lows,
                         const float* closes, float* outputs, ..., float initial, ...);
ctStatus_t ct_macd_batch(const float* inputs, float* macds, float* signals,
                         float* hists, ..., int fastPeriod, int slowPeriod,
                         int signalPeriod, int maType, ...);
ctStatus_t ct_bbands_batch(...);        // upper/middle/lower outputs
ctStatus_t ct_stochastic_batch(...);    // K/D outputs
```

Device-matrix variants (`*_batch_d`) take GPU pointers of
`numSeries * seriesLen` floats and perform no host transfers.

### Scheduling semantics

- `maxConcurrency = 0` — auto: `min(numSeries, 4)` internal non-blocking
  streams (process-wide pool, reused across calls).
- `maxConcurrency = N` — cap the pool at N streams.
- `stream == NULL` is the legacy default stream, as everywhere in CUDA:
  `*_batch_d` makes the pool wait for work already queued there, then
  blocks until the batch is done.
- `stream != NULL` — everything runs ordered on the caller's stream;
  `maxConcurrency` is ignored and `*_batch_d` variants stay asynchronous
  (host variants still synchronize before returning, matching host-API
  semantics).
- Host matrices ≥ 1 MB are temporarily page-locked (`cudaHostRegister`)
  to enable truly async copies; registration failure silently degrades
  to pageable copies.
- Fan-out/fan-in is event-based: transfers on the base stream, kernels
  round-robin across the pool, downloads after all kernels complete.

### Error handling

Standard `ctStatus_t` codes; details via `ct_get_last_error()`.
`CT_STATUS_INVALID_ARGUMENT` for null pointers, `seriesLen <= 0`,
`numSeries <= 0`.

## Python

```python
import numpy as np, tacuda

matrix = np.asarray(closes_2d, dtype=np.float32)   # (numSeries, seriesLen)

rsi  = tacuda.rsi_batch(matrix, period=14, max_concurrency=4)
sma  = tacuda.sma_batch(matrix, period=20)
ema  = tacuda.ema_batch(matrix, period=12)
atr  = tacuda.atr_batch(highs, lows, closes, period=14)
m, s, h = tacuda.macd_batch(matrix)                # (macd, signal, hist)
u, mid, lo = tacuda.bbands_batch(matrix, period=20)
k, d = tacuda.stochastic_batch(highs, lows, closes, kPeriod=14, dPeriod=3)
```

Inputs are 2D `(numSeries, seriesLen)` arrays; outputs mirror the input
shape. Every other family follows the same convention — input matrices
first, then the family scalars (positional or by keyword), then optional
`max_concurrency` / `stream` keywords:

```python
wma   = tacuda.wma_batch(matrix, 10)
adx   = tacuda.adx_batch(highs, lows, closes, 14)
mfi   = tacuda.mfi_batch(highs, lows, closes, volumes, 14)
m, s, h = tacuda.macdext_batch(matrix, 12, 1, 26, 1, 9, 1)
down, up = tacuda.aroon_batch(highs, lows, 14)
tr    = tacuda.trange_batch(highs, lows, closes)
```

### Python with device-resident PyTorch tensors

The low-level `tacuda.sma_batch_d` wrapper accepts integer GPU addresses;
it does not copy or allocate outputs. Keep contiguous CUDA float32 tensors
alive until the supplied stream has completed. Use the same CUDA device
and stream as the tensor producer:

```python
import torch
import tacuda

with torch.cuda.device(0):
    x = torch.arange(30, dtype=torch.float32, device="cuda:0").reshape(3, 10)
    y = torch.empty_like(x)
    stream = torch.cuda.current_stream()
    tacuda.sma_batch_d(x.data_ptr(), y.data_ptr(), 10, 3, 5, 4,
                       stream.cuda_stream)
    stream.synchronize()
    print(y.cpu())  # first 4 columns NaN; first row then 2, 3, 4, 5, 6, 7
```

Python returns `None` on success and raises on failure. `period=20` with
only 10 bars correctly produces all NaNs: there is no complete window.
This is the Python equivalent of Node's `device.smaBatch`, not the host
array helper `sma_batch`. The example requires PyTorch with CUDA installed.

Which stream handle to pass:

- PyTorch's default stream *is* the legacy default stream, so on it
  `cuda_stream == 0` (CuPy's null stream likewise). That is the example
  above: the batch waits for the default stream, overlaps series across
  the pool (`maxConcurrency` applies) and returns when done.
- On a side stream (`torch.cuda.Stream()`) pass its handle: the batch is
  ordered on it, `maxConcurrency` is ignored and the call returns
  immediately. Passing `0` from inside `with torch.cuda.stream(side):` is a
  race — PyTorch side streams are non-blocking, the legacy stream does not
  wait for them.
- Tensors must be contiguous `float32`; `data_ptr()` is only the address of
  the first element, nothing checks dtype or strides.
- 0.7.x did **not** make the pool wait for the legacy stream, so with
  stream `0` a producer kernel still in flight raced the batch: stale rows,
  and all-NaN rows for indicators that NaN-fill their output before
  computing it (SUM, SMA). Depending on the family only the first series
  or every series was hit. Workaround on 0.7.x:
  `torch.cuda.current_stream().synchronize()` before every `*_batch_d` call
  with stream `0`.

## Node.js

```js
const tacuda = require('tacuda');

// Array of per-symbol arrays (equal lengths)...
const rsiRows = tacuda.rsiBatch(rows, 14, { maxConcurrency: 4 });

// ...or a flat Float32Array + explicit seriesLen
const smaRows = tacuda.smaBatch(flat, 20, { seriesLen: 4096 });

const { macd, signal, hist } = tacuda.macdBatch(rows);
const { upper, middle, lower } = tacuda.bbandsBatch(rows, 20);
const { k, d } = tacuda.stochasticBatch(highRows, lowRows, closeRows, 14, 3);

// OHLCV containers work directly:
const atrRows = tacuda.atrBatch(ohlcvArray, 14, { maxConcurrency: 2 });

// GPU-resident matrices:
tacuda.device.smaBatch(dIn, dOut, seriesLen, numSeries, 20, 4);
```

Results come back as arrays of `Float32Array` rows (subarray views over one
allocation — copy if you need to detach). Generated families follow the same
shape; multi-output families return keyed objects and OHLCV-column families
accept an `Ohlcv[]` directly:

```js
const wma = tacuda.wmaBatch(rows, 10);
const adx = tacuda.adxBatch(ohlcvArray, 14);          // or (highs, lows, closes, 14)
const { down, up } = tacuda.aroonBatch(highRows, lowRows, 14);
const { macd, signal, hist } = tacuda.macdextBatch(rows, 12, 1, 26, 1, 9, 1);
tacuda.device.rocBatch(dIn, dOut, seriesLen, numSeries, 10, 4);
```

## .NET

```csharp
using Tacuda.Bindings;

float[][] rsi = TacudaBatch.Rsi(rows, period: 14, maxConcurrency: 4);
float[][] sma = TacudaBatch.Sma(rows, period: 20);
var (macd, signal, hist) = TacudaBatch.Macd(rows);
var (upper, middle, lower) = TacudaBatch.Bbands(rows, period: 20);

OhlcvSeries[] portfolio = LoadPortfolio();
float[][] atr = TacudaBatch.Atr(portfolio, period: 14);
var (k, d) = TacudaBatch.Stochastic(portfolio, kPeriod: 14, dPeriod: 3);
```

Raw matrices are also reachable via `NativeMethods.ct_*_batch(...)`.
Generated families take jagged `float[][]` matrices per input and return
tuples for multi-output indicators:

```csharp
float[][] wma = TacudaBatch.Wma(rows, 10);
float[][] adx = TacudaBatch.Adx(highRows, lowRows, closeRows, 14);
var (down, up) = TacudaBatch.Aroon(highRows, lowRows, 14);
var (macd2, signal2, hist2) = TacudaBatch.Macdext(rows, 12, 1, 26, 1, 9, 1);
```

## Relationship to `TACUDA_BATCH_COUNT`

The legacy environment knobs (`TACUDA_BATCH_COUNT`, `TACUDA_SERIES_LEN`)
reinterpret a single concatenated input inside select kernels (RSI, MACD,
ATR, NATR, BBANDS, Stochastic, MAMA). They still work, but the explicit
`*_batch` API is preferred: no global state, per-call concurrency control,
works for every indicator wired into the dispatchers, and composes with
user streams.

## Performance notes

- Series of ≥ 64K elements: single-series kernels already saturate the GPU;
  batching mainly saves call overhead — use moderate `maxConcurrency` (2-4).
- Many short series (≤ 16K elements): stream overlap matters; benchmark
  `benchmarks/benchmark_batch_api.py` measures the sweet spot on your GPU.
- `*_batch_d` + one user stream per portfolio is the lowest-overhead path
  for pipelines that already keep data on the GPU.
