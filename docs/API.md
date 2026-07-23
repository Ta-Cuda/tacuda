# API Reference

tacuda exposes one flat C ABI: `include/tacuda.h` (556 host entry points)
plus a device-pointer header `include/tacuda_device.h` (200 `ct_*_d`
functions) — 756 exported symbols in total. All language bindings are thin
wrappers over this surface, so the conventions below apply to C++, Python,
Node.js, and .NET alike.

Related: [DEVICE_API.md](DEVICE_API.md) (zero-copy pipelines),
[BATCH_API.md](BATCH_API.md) (multi-symbol execution),
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) (TA-Lib parity corner
cases), [CONTRIBUTING.md](CONTRIBUTING.md#abi-stability-policy) (ABI
stability guarantees).

## Conventions

### Status codes and errors

Every `ct_*` function returns `ctStatus_t`:

| Code | Meaning |
|------|---------|
| `CT_STATUS_SUCCESS` (0) | OK |
| `CT_STATUS_ALLOC_FAILED` | device memory acquisition failed |
| `CT_STATUS_COPY_FAILED` | H2D/D2H transfer failed |
| `CT_STATUS_KERNEL_FAILED` | kernel launch/execution failed |
| `CT_STATUS_INVALID_ARGUMENT` | null pointer, non-positive size, bad period/parameter |
| `CT_STATUS_DEVICE_ERROR` | other CUDA error |
| `CT_STATUS_UNSUPPORTED` | unsupported storage/compute dtype combination |

`ct_status_string(status)` renders the enum name; `ct_get_last_error()`
returns a thread-local human-readable detail message. Every entry point
clears the message on entry and sets it on any non-success path, so it
always describes the *last* call. Bindings convert non-zero statuses into
exceptions (`RuntimeError` / `Error` / `TacudaException`) carrying both
strings.

### Output alignment (warmup NaN)

Outputs are always full-length (`size` elements). Elements before an
indicator's first computable index are `NaN` — the "warmup" region — and
inputs shorter than the lookback succeed with an all-NaN line, mirroring
TA-Lib's empty output range. Alignment follows TA-Lib exactly (verified per
function by `benchmarks/differential_corpus.py`). Candlestick signals and
HT_TRENDMODE warm up with `0` like TA-Lib's int outputs; MAXINDEX/MININDEX
report `NaN` in the warm-up region (float outputs — a documented delta from
TA-Lib's int `0`). The remaining corner cases are listed in
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

### Memory contracts

- Host API: caller owns all buffers; functions never allocate host memory.
  Input and output buffers must not alias unless documented.
- Device API (`ct_*_d`): pointers must be device pointers of `float`
  (`ct_*_ex_d` for other dtypes); allocate via `ct_device_alloc` /
  `ct_device_alloc_ex` or any CUDA allocator.
- Internal scratch comes from a process-wide `DeviceBufferPool`
  (`ct_device_buffer_pool_cleanup`, `_allocation_count`,
  `_set_max_cache_bytes` to control it).
- Batch API host matrices ≥ 1 MB are temporarily page-locked; buffers must
  stay valid until the call returns (calls are synchronous).

### Streams

The trailing `cudaStream_t stream` parameter (default `0`) selects the CUDA
stream:

- Host API: work is enqueued on `stream`; the call synchronizes before
  returning, so results are ready.
- Device API: asynchronous with respect to the host — synchronize the
  stream (`ct_device_sync` or `cudaStreamSynchronize`) before reading
  results.
- Batch API: passing a non-null stream serializes the whole batch on it;
  passing null lets the internal stream pool overlap series
  (`maxConcurrency` caps the pool).

### Data types

Default element type is `float32`. `ct_*_ex` / `ct_*_ex_d` variants accept a
`ctDataType_t` (`CT_DTYPE_F16` = 0, `CT_DTYPE_F32` = 1, `CT_DTYPE_F64` = 2)
applied to all data buffers of the call. Half precision requires compute
capability ≥ 5.3. In Python precision is selected per call
(`tacuda.sma(data, 20, dtype="float64")`); Node.js infers it from the
TypedArray type in `*Ex` wrappers; C# exposes the raw `_ex` signatures.

### Naming

- C symbols: `ct_` + lower_snake indicator name (+ `_d` device, `_ex`
  dtype-aware, `_batch` multi-series).
- Candlestick patterns: `ct_cdl_<pattern>` returning ±100/0 signals as
  floats.
- Python keeps snake_case; Node.js uses camelCase; .NET exposes raw
  `NativeMethods.ct_*` plus PascalCase helpers.

### Versioning

`CT_VERSION_MAJOR/MINOR/PATCH` macros and `ct_version()` (string
"MAJOR.MINOR.PATCH") identify the ABI. Within a MAJOR version the C ABI is
append-only — see the
[ABI stability policy](CONTRIBUTING.md#abi-stability-policy).

## Function categories (host API)

| Category | Examples | Count |
|----------|----------|------:|
| Moving averages | `ct_sma`, `ct_ema`, `ct_wma`, `ct_dema`, `ct_tema`, `ct_trima`, `ct_kama`, `ct_mama`, `ct_t3`, `ct_ma`, `ct_mavp` | 12 |
| Momentum | `ct_rsi`, `ct_macd`, `ct_macdext`, `ct_stochastic`, `ct_willr`, `ct_cci`, `ct_cmo`, `ct_roc*`, `ct_mom`, `ct_apo`, `ct_ppo`, `ct_ultosc`, `ct_imi` | 25+ |
| Directional movement | `ct_adx`, `ct_adxr`, `ct_dx`, `ct_plus_di`, `ct_minus_di`, `ct_plus_dm`, `ct_minus_dm`, `ct_aroon*` | 9 |
| Volatility | `ct_atr`, `ct_natr`, `ct_trange`, `ct_stddev`, `ct_var`, `ct_avgdev`, `ct_bbands`, `ct_accbands` | 8 |
| Volume | `ct_obv`, `ct_ad`, `ct_adosc`, `ct_mfi`, `ct_nvi`, `ct_pvi` | 6 |
| Price transforms | `ct_avgprice`, `ct_medprice`, `ct_typprice`, `ct_wclprice`, `ct_midprice`, `ct_midpoint`, `ct_bop` | 7 |
| Statistics / regression | `ct_linearreg*`, `ct_tsf`, `ct_beta`, `ct_correl`, `ct_max/min*`, `ct_sum` | 14 |
| Hilbert transform | `ct_ht_dcperiod`, `ct_ht_dcphase`, `ct_ht_phasor`, `ct_ht_sine`, `ct_ht_trendline`, `ct_ht_trendmode` | 6 |
| SAR | `ct_sar`, `ct_sarext` (hybrid CPU path for long single series) | 2 |
| Candlestick patterns | `ct_cdl_doji`, `ct_cdl_engulfing`, ... (`ct_cdl_*`) | 63 |
| Vector math | `ct_add`, `ct_sub`, `ct_mult`, `ct_div`, `ct_sin`, `ct_cos`, `ct_sqrt`, `ct_ln`, ... | 19 |
| STAD club pack | `ct_asi`, `ct_obtr`, `ct_wobv`, `ct_hpi`, `ct_xaverage`, `ct_rolling_*`, ... | 17 |
| Batch dispatchers | `ct_*_batch[_d]`, 166 indicator families | 332 |
| Precision variants | `ct_*_ex` with `ctDataType_t` (float16/32/64) | 30+ |
| Infrastructure | `ct_version`, `ct_status_string`, `ct_get_last_error`, `ct_device_*`, `ct_device_buffer_pool_*` | 15+ |

Every host function has a `ct_*_d` device twin unless it is CPU-only by
design (SAR hybrid path still exposes `ct_sar_d`).

## Bindings

| Metric | Python | Node.js | C# (.NET) |
|--------|--------|---------|-----------|
| Native entry points (audited) | 756 / 756 | 756 / 756 | 756 / 756 |
| FFI mechanism | `ctypes` (generated) | `koffi` (hand-written + generated) | P/Invoke (generated, public) |
| Return style | numpy arrays; multi-output → tuple | `Float32Array`; multi-output → object | status code; caller allocates outputs |
| Errors | `RuntimeError` | `Error` | `ctStatus_t` (+ `TacudaException` in helpers) |
| OHLCV container | `tacuda.OHLCV` | `tacuda.Ohlcv` / plain object | `OhlcvSeries` + extension methods |
| Batch API | `*_batch` (2D NumPy) | `*Batch` + `device.*Batch` | `TacudaBatch.*` |
| Package | `tacuda` (PyPI) | `tacuda` (npm) | `Tacuda` (NuGet) |

Generated bindings (Python, C#) are produced by
`bindings/generate_bindings.py` from the two public headers. Node.js is
maintained in `bindings/nodejs/index.js` plus the generated
`batch_generated.js`. `bindings/audit_bindings.py` verifies all three cover
every header symbol and exits non-zero on any gap (runs in CI).

Point the bindings at your build of the native library (PyPI wheels and the
NuGet package bundle natives; npm loads via the environment):

```bash
export TACUDA_LIBRARY=$HOME/tacuda/build/libtacuda.so     # Linux
```

```bat
set TACUDA_LIBRARY=C:\path\to\tacuda\build-win\tacuda.dll  :: Windows
```

Python additionally accepts a directory via `TACUDA_LIBRARY_PATH`.

### Host API per language

```python
# Python — returns numpy array, raises RuntimeError on failure
result = tacuda.sma(data, period=20)
macd, signal, hist = tacuda.macdext(data, 12, 0, 26, 0, 9, 0)  # MA types per line
```

```javascript
// Node.js — returns Float32Array (or object for multi-output)
const result = tacuda.sma(data, 20);
const { macd, signal, hist } = tacuda.macdext(data, 12, 0, 26, 0, 9, 0);
```

```csharp
// C# — status code, output via parameter
int status = NativeMethods.ct_sma(input, output, size, period: 20, IntPtr.Zero);
```

Device-pointer style per language is shown in
[DEVICE_API.md](DEVICE_API.md#example-3-indicator-pipeline-2-transfers).

## OHLCV containers

Every binding accepts a columnar OHLCV container in place of separate
`open/high/low/close/volume` arrays; manual packing keeps working (the
container path is additive). Container-aware functions: everything whose
host inputs are all OHLCV columns — volatility (ATR/NATR/TRANGE),
directional movement (ADX/ADXR/DX/±DI/±DM), oscillators (Stochastic, CCI,
WILLR, ULTOSC, IMI), volume (OBV, MFI, AD, ADOSC, NVI, PVI), SAR/SAREXT,
Aroon, price transforms and all 63 candlestick patterns. Single-input
functions (`sma`, `rsi`, ...) take an explicit column
(`tacuda.rsi(ohlcv.close, 14)`).

```python
ohlcv = tacuda.OHLCV.from_columns(o, h, l, c, v)   # or .from_rows(rows)
atr = tacuda.atr(ohlcv, period=14)
k, d = tacuda.stochastic(ohlcv, kPeriod=14, dPeriod=3)
doji = tacuda.cdl_doji(ohlcv)
```

```js
const { Ohlcv, atr, stochastic } = require('tacuda');
const series = Ohlcv.fromColumns(open, high, low, close, volume);
// or simply: const series = { open, high, low, close, volume };
const atrOut = atr(series, 14);
const { k, d } = stochastic(series, 14, 3);
```

```csharp
var candles = new OhlcvSeries(open, high, low, close, volume);
float[] atr = candles.Atr(14);              // extension methods allocate + throw
var (k, d) = candles.Stochastic(14, 3);
```

```cpp
#include <tacuda/OHLCVSeries.h>               // header-only
tacuda::OHLCVSeries s(open.data(), high.data(), low.data(), close.data(),
                      volume.data(), n);
ct_atr(s.high_data(), s.low_data(), s.close_data(), atr.data(),
       (int)s.size(), 14, 0.0f);
```

Containers validate column lengths at construction; passing a container to
a function that does not accept one raises a type error — nothing degrades
silently. Batch entry points take arrays of containers where it makes
sense (`tacuda.atrBatch(ohlcvArray, 14)`, `TacudaBatch.Atr(portfolio, 14)`).
