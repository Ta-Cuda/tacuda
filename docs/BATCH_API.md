# Batched Execution API (Multi-Symbol / Portfolio)

Added in v0.2.0, extended to 52 indicator families in v0.3.0 and to the
full indicator surface (166 families, including the 63 candlestick
patterns and the STAD club pack) in v0.4.0. The batch API computes the
same indicator over many independent series (symbols) in one call,
using a fused kernel for eligible legacy workloads and a stream pool for
their fallback. Release 0.9.1 provides **246 batch families**; its new
reference-backed families use a shared CUDA kernel with caller-stream
scratch. See the current coverage and capture limits below.

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
SAR_EL, MAMA, HT_*) still run one GPU thread over a series; their
throughput comes from batching many series, not from parallelizing one.

The dispatcher also supports [row-batched families](#row-batched-legacy-families) and a [short-series path](#short-series-path). Remaining workloads use the per-series dispatcher.

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

## TA-Lib 0.8.3 families and typed outputs

Release 0.9.1 covers 237 pinned TA-Lib functions. The
[compatibility guide](TA_LIB_COMPATIBILITY.md) list the additions,
compatible aliases and verification. New families have
single-series host/device and series-major host/device batch APIs. Their
CUDA backend processes all rows in one kernel with private double scratch;
`maxConcurrency` is ignored. `TACUDA_BATCH_FUSED` controls the legacy dispatcher
only and does not disable this backend. Scratch capture requires graph memory
nodes; these new graphs are not restricted to one node.

Inputs remain float32. Real outputs are float32 and new integer outputs are
signed int32: FRACTAL flags, SUPERTREND direction, SQZMOM levels, and ZIGZAG
direction/pivot index. Integer warmup is zero; real warmup is NaN. Each output
is a separate buffer, with the same row layout. HA and ICHIMOKU have four
outputs. Legacy float candle/index buffers retain their original types.

Python returns `np.int32`, Node returns `Int32Array`, and C# returns
`int[]`/`int[][]` for the new integer outputs, including mixed tuples/objects:

```python
line, trend = tacuda.supertrend_batch(highs, lows, closes, 10, 3.0)
assert line.dtype == np.float32 and trend.dtype == np.int32
ha_open, ha_high, ha_low, ha_close = tacuda.ha_batch(opens, highs, lows, closes)
volume_osc = tacuda.pvo_ma_batch(volumes, 12, 26, 13)  # RMA smoothing
```

New `MAType` values are 9 HMA, 10 DISABLED, 11 DEFAULT, 12 ZLEMA, 13 RMA,
14 VIDYA, 15 ALMA. DEFAULT follows the enclosing function; ordinary MACD
uses EMA. Selectable smoothing is exposed by `pvo_ma`, `bbands_ma`,
`stochastic_ma`, `stochf_ma` and `stochrsi_ma`, plus the existing APO/PPO
`_ma` names. Scalar options are explicit; the pinned catalog records oracle
defaults. Use `asi_wilder` and `kurtosis_unbiased` for the upstream formulas,
preserving legacy club `asi` and population `kurtosis`.

The baseline reserves approximately 512 bytes per bar per active worker,
with a 256 MiB worker budget (one long row can exceed it). It establishes
correctness; long sequential histories and large selection windows can be
slow. Benchmark the intended shapes before treating it as a performance path.

## Data layout

All batch entry points take dense **series-major** matrices:
`numSeries` rows of `seriesLen` float32 elements, row `i` = series `i`,
rows contiguous in memory. Every series is processed independently with
identical parameters. The fused path uses double recurrences and can differ
in the last float32 bits from the original single-series kernels; see the
numerical contract below.

## Fused execution

Added in 0.9.0. A supported `*_batch_d` call launches **one kernel for the entire matrix**,
without scratch allocations or memory nodes. Host variants upload and
download the full matrices around that same kernel. The C ABI and all binding
signatures are unchanged. `maxConcurrency` is ignored on this path.

| Families | Work per thread | Automatic selection |
|----------|-----------------|---------------------|
| SMA, WMA, SUM, MIN, MAX, STDDEV, BBANDS | one output element, direct window in double (MIN/MAX use float comparisons) | period 1..64 |
| STOCHASTIC | one output element, fused FASTK/SLOWK/SLOWD | K and D periods 1..64 and `(2*D-1)*K + D*D <= 1024` |
| ATR, NATR, RSI, EMA, OBV, AD, ADOSC | one series, sequential recurrence in double | `seriesLen <= 4*numSeries` |
| MFI | one series, double flow ring and rolling totals | same shape rule; period 2..64 |
| MACD | one series, EMA recurrences in double | same shape rule; `maType == CT_MA_EMA` |

The window limits bound redundant work; the shape rule avoids serializing a
long history on one GPU thread. Unsupported periods, MACD types and families
keep the original implementation and its argument validation.

`TACUDA_BATCH_FUSED` selects the dispatcher:

- `auto` (default, also used for unrecognized values): the table above.
- `0`: original per-series dispatcher for legacy families. New reference-backed
  functions keep their shared backend.
- `1`: force fused execution for supported families, overriding the recurrent
  shape rule. Window limits and supported MACD types still apply. This can be
  much slower for long histories with few series.

The variable is read on each call. Set it before submitting work; changing
process environment concurrently with library calls is the caller's
responsibility. A captured graph retains its chosen kernel regardless of
later environment changes.

### Numerical contract and migration

Recurrent fused kernels evaluate the TA-Lib recurrence in double, without
FMA contraction, then round each public output to float32. ATR/NATR compute
true range in double; MACD keeps its internal MACD and signal lines in double
until output. This improves agreement with TA-Lib, but changes the last bits
of results from 0.8, notably ATR and RSI. Existing streaming handles retain
their original rounding contract; the new recurrences are separate reusable
states in `utils/StreamStates.h`.

Period 1 retains EMA's copy-through behavior. NATR period 1 follows TA-Lib's
special case: unnormalized TRANGE, with a NaN at the first bar. This boundary
case is corrected on the original and streaming ATR/NATR paths as well;
period-one true ranges recover after a NaN bar. Normal-period streaming
rounding remains unchanged.

Every **fused** row is bit-independent of row order, other rows, batch size
and future bars. This does not imply bit equality across dispatcher paths:
in `auto`, reducing the number of series can select the original float-based
recurrent kernels. Use `1` when that invariant must hold even for a single
row, or `0` to keep the original per-series numerical path.

Windows sum their local float32 inputs in double. STDDEV/BBANDS use population
variance `E[x*x] - E[x]*E[x]`, clamped at zero. They avoid subtracting long
prefix sums but still have the usual cancellation limit for high-offset,
low-variance data. MFI and AD preserve the library's non-finite contribution
guards. NaNs cannot cross rows; direct windows can recover after a NaN leaves
the window. STOCH preserves the shipped float arithmetic and summation order,
so it is tolerance-compatible with TA-Lib rather than universally bit-equal.
No universal bit-equality claim is made for TA-Lib's rolling double windows.
MFI re-sums its stored flows if cancellation produces a negative window total.
This removes tiny negative residuals for positive prices/volumes while retaining
genuine signed-volume behavior; tiny TA-Lib residuals near zero can differ.

Separately, single-series MFI, AD, ADOSC and STDDEV use a fixed-tree scan in
place of a scheduling-dependent device-wide floating-point scan. Repeated
eager calls and graph replay are deterministic for the same build/device,
including million-element inputs. This fix also applies with the fused
dispatcher disabled: `0` cannot reproduce the old scan's nondeterministic
choice of rounding. See the GPU validation commands in
[the performance guide](PERFORMANCE.md#fused-batch-validation).

## Row-batched legacy families

Unreleased (after 0.9.1). 118 more families compute a batch call in a fixed
number of operations, independent of `numSeries`, **with their own
single-series kernels**. Every row is bit-identical to the per-series
dispatcher in the same build, including NaN payloads.

| Plan | Families | How |
| ---- | -------- | --- |
| Flat (103) | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, BOP, TRANGE; MOMENTUM, ROC, ROCP, ROCR, ROCR100, CHANGE; WILLR, CCI, VAR, AVGDEV, LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE, TSF, MIDPRICE, MINMAX, AROON, AROONOSC, ULTOSC, STOCHF, TRIMA, ACCBANDS; all 63 candlestick patterns; club MARKET_STRENGTH, TRENDSCORE, REFDEV, SKEW, KURTOSIS, PIVOT_HIGH, PIVOT_LOW, ROLLING_MEAN/MAX/MIN/STD | The family's `ct_*_d` runs over the flattened `numSeries * seriesLen` matrix as if it were one series, in blocks of whole rows of up to 2^22 elements (one row if a row is longer). Its kernels read only the window that ends at the element they write, so each element at or past its row's first valid index sees only its own row. One more kernel per block copies the block's first row's leading warm-up elements over those the other rows computed from the previous row's tail. |
| Rows (14) | SAR, SAREXT, SAR_EL, MAXINDEX, MININDEX, MINMAXINDEX, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE, MAMA, MACDFIX | One GPU thread per row runs the family's sequential single-series state machine: the same device function as the single-thread kernel, in the same translation unit, with the same compile flags. The SAR and index families branch on the data (trend flips, window rescans), and rows sharing a warp would run those branches one after another, so while rows are few each row gets a warp of its own: up to the device's resident warps for SAR_EL and the index families, up to 6 rows per SM for SAR and SAREXT, whose double arithmetic occupies the FP64 pipeline for a full warp either way. More rows, and the other families, pack 64 rows per block. |
| Window (1) | MIDPOINT | One thread per output, with row-local window bounds and NaN warm-up. |

The warm-up copy preserves the flattened paths' bits: depending
on the family the per-series path leaves `0xFFFFFFFF` NaNs from a memset,
`nanf("")` NaNs written by a kernel, or zeros (candlesticks). A block's first
row holds exactly those values because warm-up does not depend on the data.
The block size bounds the scratch that the stacked single-series wrappers
allocate (up to three `float` matrices of the block), which on the
per-series path was a few kilobytes per row.

A call takes the batched path when all of these hold; otherwise it keeps the
per-series dispatcher, which also reports invalid parameters with the
original status and message:

- `numSeries >= 2` and `TACUDA_BATCH_FUSED` is not `0`;
- the parameters are valid for the plan (positive periods, MACDFIX
  `signalPeriod > 0`);
- flat plans: every row has a valid element, i.e. `seriesLen` exceeds the
  first valid index (`period - 1` for WILLR, `period` for ROC, the pattern's
  lookback for candlesticks), and `seriesLen <= INT_MAX / 4`, because the
  single-series kernels index with `int`;
- flat plans: `numSeries >= 64` or `seriesLen < 65536` (16384 for ACCBANDS
  and TRIMA). Flattening only removes launches; a few long rows already fill
  the GPU per series, and the flat call would add its warm-up copy (up to 7%
  slower at 8 × 250000 on an RTX 3090 Ti). The SMA composites also lose the
  per-series stream overlap of their passes (ACCBANDS 13% slower at
  32 × 62500);
- TRIMA and ACCBANDS: every SMA pass uses a direct window kernel (by default
  periods below 96, see `TACUDA_SMA_PREFIX_T`); the prefix-sum SMA would
  scan across rows and round differently.

Streams behave as for fused calls: `maxConcurrency` is ignored, stream `0`
runs on the legacy default stream and returns when done, a caller's stream
only receives enqueued work and can be captured. Up to 2^22 elements, a
captured call has 1 to 3 nodes for most families; each further block of
rows repeats a flat family's nodes. STOCHF, TRIMA, ACCBANDS, AROON and
MINMAX reuse single-series wrappers that allocate scratch and copy stacked
outputs (7 to 18 nodes per block), so like their per-series path they need
graph memory nodes to be captured. The row-thread families write their outputs directly and need no
scratch, unlike the stacked per-series MAMA, MACDFIX, MINMAXINDEX, HT_PHASOR
and HT_SINE wrappers.

Plans live in `scripts/generate_batch_api.py` (`ROWS`). Candlestick warm-ups
are read from each kernel's `lookback` constant, so changing one regenerates
the plan; CTest `batch_generated` fails until the generated file is current.
`tests/python/batch_rows.py` (CTest `batch_rows`, and
`batch_rows_without_memory_pools` on Linux) checks every plan against the
per-series dispatcher and direct single-series calls across shapes,
parameters and non-finite data, the path choice, graph capture and replay,
the host and stream-0 entry points and error statuses; with `--ref-lib` it
also compares against another build. Measurements are in
[the performance guide](performance/README.md#row-batched-families).

## Short-series path

DEMA, TEMA, T3, KAMA, TRIX, CMO, ADX, ADXR, DX, PLUS_DI, MINUS_DI,
PLUS_DM, MINUS_DM, APO, PPO, PVO, MA, MACD_LINE, STOCHRSI, IMI, NVI,
PVI, CORREL, BETA, ASI, OBTR, WOBV, HPI and XAVERAGE can each submit
one kernel for the batch. A thread owns a row for recurrences.
CORREL and windowed MA distribute independent windows among threads.
Calls use no graph allocation nodes on this path.

In `auto`, this path requires at least 32 rows, at most 1024 bars per
row, and `seriesLen <= 4*numSeries`. Mode `1` allows smaller batches
with at least two rows. The 1024-bar and parameter limits still apply.
ADXR, BETA, CORREL, IMI, KAMA, and MA limit their window to 64.
STOCHRSI limits its K and D windows to 64. MA type MAMA uses the
existing implementation; extended MA types use the reference backend.
CMO and PLUS_DM/MINUS_DM keep the original float scan when
`period < 2` or `seriesLen > 80*(period-1)`. In those cases the scan
and recurrence can round differently after extended flat or non-finite
inputs, even if their update equations agree.
The original dispatcher handles lengths and parameters outside these
conditions and remains available through `TACUDA_BATCH_FUSED=0`.

The short-series kernels use the existing formulas with a different
arithmetic order. Compare numeric results with tolerance. For supported
short STOCHRSI, single-series and batch calls share the same RSI recurrence
so the dispatcher agrees even when the stochastic range is nearly zero.
This can change the historical signal on those windows, including in
mode `0`. The short path keeps each row's output independent of its
position and its neighbors.

CTest `batch_short` checks all 29 families against the per-series
dispatcher on finite, flat and non-finite fixtures. It verifies host
results and single-node graph replay. The Linux capability shim also
runs the allocation-free graph cases.

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

- Fused calls use the supplied stream directly; `maxConcurrency` is ignored.
  With stream `0`, device variants synchronize that stream before returning.
  On a non-null stream they only enqueue one kernel and stay asynchronous.
- On the original dispatcher, `maxConcurrency = 0` — auto: `min(numSeries, 4)` internal non-blocking
  streams (process-wide pool, reused across calls).
- On the original dispatcher, `maxConcurrency = N` — cap the pool at N streams.
- On the original dispatcher, `stream == NULL` is the legacy default stream:
  `*_batch_d` makes the pool wait for work already queued there, then
  blocks until the batch is done.
- `stream != NULL` — everything runs ordered on the caller's stream;
  `maxConcurrency` is ignored and `*_batch_d` variants stay asynchronous
  (host variants still synchronize before returning, matching host-API
  semantics). The call only enqueues work: nothing inside it waits for the
  stream or the device, and it can be captured into a CUDA graph (below).
- No `*_d` call, or `*_batch_d` call on a non-null stream, synchronizes internally. Before this was
  fixed, every Thrust scan ran with `thrust::cuda::par` — `cudaMalloc`,
  `cudaStreamSynchronize`, `cudaFree` per scan — and a few seeds were read
  back to the host, so families built on scans (MFI, RSI, ATR, ADOSC,
  STDDEV, BBANDS, SUM, OBV, AD, ADX, MACDEXT, KAMA, ...) blocked the host once
  or twice per series: `maxConcurrency` changed nothing and the
  "asynchronous" user-stream call returned only when its work had finished.
  Library code passes `tacuda::detail::thrust_on(stream)`
  (`include/utils/ThrustPolicy.cuh`) to Thrust; it is `par_nosync` with
  stream-ordered temporary storage. Thrust algorithms that return a value
  to the host (`reduce`, `transform_reduce`, ...) synchronize under any
  policy and must not be used in device paths
  (`include/utils/DeviceSeed.cuh` computes such seeds on the device).
- Host matrices ≥ 1 MB are temporarily page-locked (`cudaHostRegister`)
  to enable truly async copies; registration failure silently degrades
  to pageable copies.
- Fan-out/fan-in is event-based: transfers on the base stream, kernels
  round-robin across the pool, downloads after all kernels complete.

### CUDA graph capture

A `*_batch_d` call on a non-null stream can be captured into a CUDA graph
(any capture mode, `cudaStreamCaptureModeGlobal` included) and replayed.
Scratch-backed paths require graph memory-node support: CUDA runtime/driver
11.4+ and a nonzero `cudaDevAttrMemoryPoolsSupported` device attribute. If this
capability is absent, a call that needs captured scratch returns
`CT_STATUS_UNSUPPORTED` with an explanation from `ct_get_last_error()`.
Allocation-free paths, including supported fused calls, do not require memory
pools. A toolkit version alone does not establish device capability.
For a fixed pipeline over buffers that never move, a whole pass of many
calls becomes one graph launch:

```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):          # warm-up outside the capture
    tacuda.mfi_batch_d(h.data_ptr(), l.data_ptr(), c.data_ptr(), v.data_ptr(),
                       out.data_ptr(), T, S, 14, 0, s.cuda_stream)
s.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    tacuda.mfi_batch_d(h.data_ptr(), l.data_ptr(), c.data_ptr(), v.data_ptr(),
                       out.data_ptr(), T, S, 14, 0, s.cuda_stream)
g.replay()                           # recomputes `out` from the current h/l/c/v
```

- Capture only a call with a non-null stream. With `stream == 0` the call
  fans out to the internal stream pool and blocks until it is done, which
  is not capturable.
- Pointers and scalars are frozen into the graph: a replay reads the same
  input buffers and writes the same outputs.
- Scratch memory a call needs is allocated during capture with
  `cudaMallocAsync`/`cudaFreeAsync`, i.e. as memory nodes owned by the
  graph; outside capture it comes from the library's stream-ordered pool
  (see "Memory contracts" in `docs/api/conventions.md`). CUDA allows at most
  one executable instance of a graph with memory nodes at a time.
- After any failed library call during capture, finish/discard that capture
  and its partial graph; do not launch it. A capability refusal does not call
  an unsupported CUDA allocator or leave that allocator's runtime error for
  the next call. The capability check does not clear caller-owned CUDA errors.
- Outside capture, devices without memory-pool support use the event-delayed
  ordinary buffer pool. Warming that pool cannot enable graph memory nodes;
  reusing its buffers inside a replayable graph would violate their lifetime.
- Make one eager call first (as above) so kernel loading and pool growth do
  not happen inside the capture.
- Replay removes the host cost of submitting the work. Legacy fallback
  paths still issue their per-series kernels; fused and new reference-backed
  paths process rows in one kernel.

`tests/python/batch_graph_capture_sweep.py` (CTest
`batch_graph_capture_sweep`) captures every `*_batch_d` family in global
mode, replays it and requires the eager result bit for bit; with
`--ref-lib` it also compares against another build of the library.
On devices without graph memory support it separately counts validated
`CT_STATUS_UNSUPPORTED` refusals, while continuing to replay allocation-free
cases. Linux CTest also exercises this capability with a test-only CUDA
interposer (`scratch_capture_without_memory_pools`,
`batch_graph_capture_without_memory_pools`, `batch_fused_without_memory_pools`).
This is capability-path coverage on the available GPU, not physical Maxwell
hardware acceptance.

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
  above: the batch waits for preceding default-stream work and returns
  when done. Fused calls run there directly; the fallback dispatcher uses
  the stream pool (`maxConcurrency` applies only to that fallback).
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

- The measurements below describe the original dispatcher (`TACUDA_BATCH_FUSED=0`),
  before fused execution. Current fused measurements and reproduction commands
  are recorded in [the performance guide](PERFORMANCE.md#fused-batch-validation).
- Series of ≥ 64K elements: single-series kernels already saturate the GPU;
  batching mainly saves call overhead — use moderate `maxConcurrency` (2-4).
- Many short series (≤ 16K elements): stream overlap matters; benchmark
  `benchmarks/benchmark_batch_api.py` measures the sweet spot on your GPU.
- Thousands of short series: every series costs its own launches, so the
  host's launch rate is the limit, not arithmetic. RTX 3090 Ti, 4 000 series
  × 234 bars, one call each of MFI, ATR, RSI, ADOSC, STDDEV, BBANDS, SUM,
  OBV, AD, STOCH, SMA, MIN, MAX (the pass from issue #6): 1.11 s with the
  per-scan synchronization, 0.44 s without it (`stream = 0`,
  `maxConcurrency` 0 or 4); MFI alone 143 → 50 ms. `maxConcurrency = 1`
  is 1.7–3.7× slower than 4 for BBANDS, AD and ADOSC — before the fix every
  value gave the same time.
- On a user stream the whole batch is ordered on that one stream, so
  kernels of different series never overlap on the device: the same pass
  takes 0.71–0.74 s, and its graph replay 0.56 s (the device runs ~2.5 µs
  kernels one after another). For many short series `stream = 0` with the
  default pool is the faster choice; a user stream buys ordering with the
  caller's work and graph capture.
- Scratch memory no longer grows with the number of series in flight: the
  stream-ordered pool reuses a block on its stream as soon as the work
  that freed it is queued. 64 series × 200 000 bars of MFI reserve 34 MiB
  whether the pool or a user stream runs them. (With event-based reuse, as
  `DeviceBufferPool` does, the same batch took 44–62 MiB and grew with how
  far the host ran ahead of the GPU.)
