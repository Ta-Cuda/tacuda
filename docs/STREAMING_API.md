# Streaming (Incremental Update) API

Added in v0.8.0. A stream carries indicator state across calls, so a newly
closed bar costs a state update instead of a full-series recompute. This is
the API for live and paper trading; the host, device and batch APIs remain
the right choice for research over whole histories.

Design rationale and the plan for batch streams: [DESIGN_STREAMING_API.md](DESIGN_STREAMING_API.md).

## Why it exists

Recomputing the whole history on every tick is O(n) per bar, and on the GPU
it also pays a PCIe roundtrip and a kernel launch. Measured on the RTX
3090 Ti (RSI(14), `benchmarks/bench_stream_latency.cpp`):

| History | Full recompute per tick (`ct_rsi`) | Stream append per tick | Ratio |
| ------- | ---------------------------------- | ---------------------- | ----- |
| 100 000 bars | p50 142 µs, p99 154 µs | p50 **0.04 µs**, p99 0.06 µs | ~3 500x |
| 1 000 000 bars | p50 1 234 µs, p99 1 266 µs | p50 **0.04 µs**, p99 0.05 µs | ~31 000x |

The stream cost is flat: it does not grow with history length, which is the
whole point. Replaying history through a stream runs at ~55 M bars/s on one
CPU core.

Single-series streams therefore run **on the CPU** with double-precision
state. A one-bar update is a handful of floating-point operations, while a
GPU roundtrip has a floor of tens of microseconds no matter how little work
it carries. The GPU's place in a streaming workload is whole-portfolio
appends — see batch streams below, where the same measurement puts the
crossover at about 2048 series.

## Covered families (v0.8.0, wave 1)

| Family | Create | Inputs | Outputs | Lookback |
| ------ | ------ | ------ | ------- | -------- |
| SMA | `ct_sma_stream_create(period)` | close | 1 | `period-1` |
| EMA | `ct_ema_stream_create(period)` | close | 1 | `period-1` |
| WMA | `ct_wma_stream_create(period)` | close | 1 | `period-1` |
| RSI | `ct_rsi_stream_create(period)` | close | 1 | `period` |
| ATR | `ct_atr_stream_create(period)` | high, low, close | 1 | `period` |
| NATR | `ct_natr_stream_create(period)` | high, low, close | 1 | `period` |
| OBV | `ct_obv_stream_create()` | close, volume | 1 | 0 |
| AD | `ct_ad_stream_create()` | high, low, close, volume | 1 | 0 |
| MACD | `ct_macd_stream_create(fast, slow, signal)` | close | 3 (macd, signal, hist) | `slow-1 + signal-1` |
| BBANDS | `ct_bbands_stream_create(period, upMul, dnMul)` | close | 3 (upper, middle, lower) | `period-1` |

MACD streams implement the classic all-EMA MACD, i.e. `ct_macd` with
`maType = CT_MA_EMA`. Like TA-Lib, all three MACD lines share one
`outBegIdx`: the line is NaN until the signal lookback, not just the signal.

## Batch (portfolio) streams

One handle carries `numSeries` independent series of one indicator, and one
append advances all of them. It is the *same* state machine as the
single-series path, so a batch of N series is bit-for-bit what N separate
streams produce — the choice is about call overhead and where the work runs,
never about results. That equivalence is asserted per family, per execution
path, in `tests/cpp/test_stream_batch.cpp`.

```c
ctStreamBatch_t *rsi = NULL;
ct_rsi_stream_batch_create(14, numSymbols, &rsi);

/* closes is series-major: symbol s's n bars at [s*n .. s*n+n-1] */
ct_rsi_stream_batch_append(rsi, closes, n, out);

ct_stream_batch_destroy(rsi);
```

Same ten families as above, with `_batch` in the name, plus
`ct_stream_batch_{destroy,reset,lookback,series,outputs,bars,set_exec}`.

### Where the work runs

`ctStreamExec_t` selects the execution path; `CT_STREAM_EXEC_AUTO` (the
default) switches at the measured crossover of **2048 series**. Per-tick
latency of the 7-indicator pipeline (SMA, EMA, RSI, ATR, MACD, BBANDS, OBV),
one closed bar per symbol, i3-12100F + RTX 3090 Ti
(`benchmarks/bench_stream_portfolio.cpp`):

| Symbols | CPU p50 / p99 | GPU p50 / p99 | AUTO picks |
| ------- | ------------- | ------------- | ---------- |
| 64 | 4.6 / 5.3 µs | 113 / 125 µs | CPU |
| **256** | **16.8 / 19.5 µs** | 119 / 134 µs | CPU |
| 1 024 | 67.6 / 77.8 µs | 124 / 142 µs | CPU |
| 2 048 | 141 / 155 µs | 139 / 156 µs | GPU (crossover) |
| 4 096 | 300 / 324 µs | 161 / 177 µs | GPU |
| 16 384 | 1 215 / 1 325 µs | 306 / 326 µs | GPU |
| 65 536 | 4 721 / 5 202 µs | 764 / 794 µs | GPU |

The shape is the same one that drove the single-series design: the GPU has a
fixed floor per append (two transfers, a launch, a sync) that a small
portfolio never earns back, and a throughput advantage that a large one does.
AUTO keeps p99 under 1 ms all the way to 65 536 symbols.

There is deliberately **no library-managed thread pool**. A first cut spawned
worker threads per append and measured 595 µs at *every* portfolio size —
thread creation swamped the work it was parallelizing. Since series are fully
independent, a caller who wants CPU parallelism can shard the portfolio
across several batch handles and drive them from its own threads, without the
library making that decision for them.

Policy may change between appends; state migrates with it (asserted by a test
that runs one series through CPU, then GPU, then CPU again and compares
against a single-series reference).

## Usage

```c
#include <tacuda_stream.h>

ctStream_t *rsi = NULL;
if (ct_rsi_stream_create(14, &rsi) != CT_STATUS_SUCCESS) {
    fprintf(stderr, "%s\n", ct_get_last_error());
    return 1;
}

/* Warm-up replay is an ordinary append: no separate priming call. */
float *warm = malloc(sizeof(float) * historyLen);
ct_rsi_stream_append(rsi, history, historyLen, warm);

/* Then one call per closed bar. */
float value;
ct_rsi_stream_append(rsi, &newClose, 1, &value);

ct_stream_destroy(rsi);
```

Every append consumes `n` bars and writes exactly `n` values per output
line, aligned one-to-one with the appended bars. Bars inside the warm-up
window produce NaN, exactly like the leading NaNs of a full-series call —
so the output of `append(history, n)` is bit-comparable with the head of
`ct_rsi(history, n)`.

Multi-output families take one array per line:

```c
ctStream_t *macd = NULL;
ct_macd_stream_create(12, 26, 9, &macd);
ct_macd_stream_append(macd, closes, n, outMacd, outSignal, outHist);
```

### Lifetime and introspection

| Call | Purpose |
| ---- | ------- |
| `ct_stream_destroy(s)` | Release. NULL-safe. Reusing a destroyed handle is rejected, not undefined. |
| `ct_stream_reset(s)` | Drop state, keep parameters — equivalent to a fresh handle. |
| `ct_stream_lookback(s, &lb)` | Bars needed before the first non-NaN output. |
| `ct_stream_bars(s, &n)` | Bars appended since creation or last reset. |
| `ct_stream_outputs(s, &k)` | Output lines the matching append writes (1 or 3). |

Handles are **not** thread-safe: use one stream per producer thread, or
synchronize externally. Distinct handles are fully independent, so a
portfolio of N symbols is N handles.

### Error contract

Identical to the host API: a thread-local message retrievable with
`ct_get_last_error()`, cleared on entry to every call, and no C++ exception
ever crosses the C boundary. `CT_STATUS_INVALID_ARGUMENT` covers null
pointers, `n <= 0`, out-of-range periods (TA-Lib's own BAD_PARAM ranges),
destroyed or foreign handles, and appending through the wrong family's
function (an SMA append on an ATR handle is rejected, not silently
misinterpreted).

## Numerical relationship to the full-series API

Stream state is double precision, matching TA-Lib's own accumulators.
Several full-series kernels carry their recursion in float (the Wilder
affine scan, for one), so stream values are **at least as accurate** as the
full-series path but not bit-identical to it. Measured worst-case relative
deviation over a 4096-bar random walk with a flat stretch and a gap
(`tests/cpp/test_stream_api.cpp`, oracle `|a-b| / max(1,|a|,|b|)`):

| Family | Worst deviation vs full series |
| ------ | ------------------------------ |
| MACD (all three lines), OBV, AD, SMA | 0 (bit-identical) |
| BBANDS upper/lower | 0; middle 1.2e-07 |
| EMA | 7.8e-08 |
| WMA | 1.2e-07 |
| RSI | 4.6e-07 |
| ATR | 6.4e-07 |
| NATR | 7.4e-07 |

NaN masks match **exactly** — that part is asserted, not toleranced. NaN
inputs propagate with the same semantics as the full-series path (a NaN
price poisons the Wilder recursion rather than counting as a zero change).

Chunking is irrelevant to the result: appending 4096 bars at once, in
chunks of 16, of 3, or one at a time all produce identical output. That
equivalence is asserted for every family in the test suite.

## Cost model

| Family group | Per-bar cost |
| ------------ | ------------ |
| EMA, RSI, ATR, NATR, OBV, AD, MACD | O(1) — a fixed number of FLOPs |
| SMA, WMA, BBANDS | O(period) — the window is re-summed from a ring buffer |

The window families deliberately re-sum instead of keeping a rolling
add/subtract accumulator: a rolling sum drifts away from the full-series
kernels (which sum each window from scratch) over long histories. O(period)
is bounded by the parameter, not by history length, so a 200-period SMA
stream is still thousands of times cheaper per tick than recomputing 1M
bars.

## Language bindings

The C ABI is wrapped by hand in each language rather than generated: the
binding generator models array-in/array-out calls, while a stream is a handle
with a lifetime. Coverage is still gated — `bindings/audit_bindings.py` checks
all 52 streaming symbols in all three languages and fails CI on any gap.

**Python** — `tacuda.stream`, context managers:

```python
import numpy as np, tacuda

with tacuda.stream.Rsi(14) as rsi:
    warm = rsi.append(history)          # ndarray in, ndarray out
    value = rsi.append(new_close)[-1]

with tacuda.stream.MacdBatch(12, 26, 9, num_series=256) as macd:
    macd.set_exec(tacuda.stream.EXEC_GPU)
    line, signal, hist = macd.append(closes)   # (256, n) matrices
```

**Node.js** — classes, `destroy()` (and `Symbol.dispose` where available):

```js
const tacuda = require('tacuda');
const rsi = new tacuda.RsiStream(14);
const warm = rsi.append(history);            // Float32Array
rsi.destroy();

const batch = new tacuda.RsiStreamBatch(14, 256);
batch.setExec(tacuda.stream.EXEC_GPU);
const out = batch.append(flatSeriesMajor);   // 256 * n elements
```

**C#** — `IDisposable`, tuple returns for multi-line families:

```csharp
using var rsi = new RsiStream(14);
float[] warm = rsi.Append(history);

using var macd = new MacdStream(12, 26, 9);
var (line, signal, hist) = macd.Append(closes);

using var batch = new RsiStreamBatch(14, numSeries: 256);
batch.SetExec(StreamExec.Gpu);
```

Using a released handle throws in every language (`ObjectDisposedException`
in C#, an error in Python and Node) instead of reaching freed native state.

## Limits in v0.8.0

- No persistent batch plans yet. Each indicator's batch append moves its own
  inputs, so a 7-indicator GPU pipeline uploads the same close matrix seven
  times per tick. A plan would upload OHLCV once and run the pipeline over
  the resident copy — worth roughly half the GPU tick cost at tens of
  thousands of symbols. It is not built yet because the acceptance target is
  met without it (see the table above); the measurement, not a guess, is what
  schedules it.
- float32 inputs and outputs only; there is no `_ex` dtype variant yet.
- Ten families, not all 161. The set covers what live strategies actually
  poll; the rest is additive work behind the same handle contract.
- No state serialization: a process restart replays history. Replay runs at
  ~55 M bars/s, so a 1M-bar warm-up costs ~18 ms.
- Language bindings (Python, Node, C#) are not generated yet — the C ABI is
  stable and callable through FFI today.

## Testing

`benchmarks/streaming_corpus.py` sweeps all ten families over the seven
differential-corpus datasets (random walks, trend, flat, zero-crossing,
short) under the same strict oracle, checking three things per cell:
appending in chunks of 1/3/16/all reproduces the full-series tacuda call,
the chunkings are bit-identical to each other, and the result matches the
installed **TA-Lib**. Batch streams are checked row-by-row against single
streams on both execution paths. Result: **699/699 cells pass**. The one
exclusion is NATR on the zero-crossing dataset, where TA-Lib's own
`ta_NATR.c` corrupts its first output bar — the same exclusion the
differential corpus makes, for the same reason.

`tests/cpp/test_stream_batch.cpp` (8 cases) asserts that a batch equals N
single streams bit for bit — per family, per execution path, at chunk sizes
1/7/whole — plus introspection, reset (including invalidation of resident
device state), policy changes mid-stream, and the error contract.

`tests/cpp/test_stream_api.cpp` (18 cases) asserts, for every family:
stream-vs-full-series parity under the corpus oracle at four chunk sizes,
exact NaN-mask equality, lookback correctness, reset-equals-fresh,
warm-up-replay-then-tick-by-tick equivalence, handle independence, NaN
propagation, and the full error contract including use-after-destroy.

Run them with the rest of the suite:

```sh
ctest --output-on-failure -R TacudaStream
```

Both suites are part of the standard run; the full suite is green at 417/417
on the RTX 3090 Ti.
