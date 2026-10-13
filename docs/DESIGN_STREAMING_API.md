# Design: Streaming (Incremental Update) API

Status: **M1 and the batch-stream half of M2 implemented** (2026-08-02);
persistent batch plans are still design-only. Shipped in 0.8.0 — see
[ROADMAP.md](ROADMAP.md). The shipped user-facing contract is documented in
[STREAMING_API.md](STREAMING_API.md); this file keeps the reasoning and the
open questions.

The single-series contract below shipped essentially as drafted, with two
changes made during implementation: `ct_stream_bars` / `ct_stream_outputs`
were added to the introspection set, and period validation follows TA-Lib's
per-function BAD_PARAM ranges (most families start at 2; ATR/NATR and the
MACD signal accept 1, where the indicator degenerates gracefully).

## Goal

Let a live-trading process append newly closed bars and receive updated
indicator values without recomputing the whole series:

```c
ctStream_t *rsi;
ct_rsi_stream_create(14, &rsi);
ct_rsi_stream_append(rsi, history, historyLen, out);   /* warm-up replay */
/* ... per tick / per closed bar: */
ct_rsi_stream_append(rsi, &newBar, 1, &outValue);      /* O(1), not O(n) */
ct_stream_destroy(rsi);
```

The defining contract: **appending bars one at a time (or in chunks)
produces exactly the tail of a full-series recompute** — same values, same
NaN warm-up, same TA-Lib-aligned semantics the differential corpus already
enforces for the full-series API.

## Motivation

- A full recompute per tick is O(n) per bar and grows with history; a
  state-carrying update is O(1). At 1M bars of history even the fastest
  full-series path wastes ~milliseconds per tick per series.
- TA-Lib has no streaming mode; users hand-roll incremental EMAs and get
  drift. A corpus-verified incremental path is a capability gap we can own.
- The strategy-optimizer work already produced the evidence for the
  architecture split below (GPU share of latency-bound workloads is small;
  PCIe dominates single-series roundtrips — see
  [EXPERIMENTS.md](EXPERIMENTS.md) and the dropped `ct_*_param_sweep_d`
  decision in [ROADMAP.md](ROADMAP.md)).

## Non-goals

- Signal generation, order logic, backtesting — out of core (non-goals in
  [ROADMAP.md](ROADMAP.md)); `tacuda-opt` may consume this API later.
- Tick aggregation / bar building. Input is closed bars, same as the
  full-series API.
- Changing parameters mid-stream. Create a new stream instead.
- Removing or rewriting history (no "un-append"). `ct_stream_reset` +
  replay covers repaints.

## Architecture decisions

### D1. Single-series streams run on the CPU

A one-bar update of any wave-1 indicator is a handful of double-precision
operations — microseconds on a CPU core. A GPU roundtrip pays two PCIe
latencies plus a kernel launch (~10–20 µs floor) to do the same work. The
measured precedent: SAR/SAREXT already use a hybrid CPU path in the host
API for exactly this reason, and the optimizer measurements showed
latency-bound request sets are PCIe-bound, not compute-bound.

**Confirmed by measurement** (RTX 3090 Ti, RSI(14),
`benchmarks/bench_stream_latency.cpp`): a CPU stream append costs p50
0.04 µs and does not grow with history, while a full `ct_rsi` recompute
costs p50 142 µs over 100K bars and 1 234 µs over 1M. Even a hypothetical
zero-cost GPU kernel could not beat 0.04 µs through the driver, so routing
single-bar updates to the device would have been strictly worse.

Consequence: single-series stream state lives in host memory, in
**double precision** (matching TA-Lib's internal accumulators, which is
what makes bit-stable long-running recursions possible). No CUDA calls on
the append path.

### D2. The GPU serves batch streams (portfolio append)

Where the GPU wins is one call appending a bar (or chunk) to many series at
once: state matrices stay resident on the device, one kernel updates all
series, outputs come back in one transfer. This is the per-tick analogue of
the existing batch API and reuses its conventions (series-major matrices).

**Measured, and the threshold is higher than "hundreds or thousands" implied
above** (7-indicator pipeline, one bar per tick,
`benchmarks/bench_stream_portfolio.cpp`): the CPU path wins below ~2048
series (256 symbols: 16.8 µs vs 119 µs on the GPU) and the GPU wins above it
(65 536 symbols: 764 µs vs 4 721 µs). The crossover at 2048 is where the two
meet within noise, and `CT_STREAM_EXEC_AUTO` switches there. The state
machines are the same POD structs on both paths (`utils/StreamStates.h`), so
the two agree bit for bit and the policy is purely about speed.

**Rejected: a library-managed CPU thread pool.** The first implementation
spawned workers per append and cost 595 µs at every portfolio size — thread
creation dominated the work. Since series are independent, callers can shard
across several batch handles with their own threads; the library does not
need to own that decision, and the GPU path already covers the large-portfolio
regime inside the latency budget.

### D3. Persistent batch plans

Per-tick calls must not pay validation + allocation every time.
A **plan** pins shape (numSeries, seriesLen/capacity), parameters and
preallocated device buffers at creation; `execute`/`append` then only
launches work. CUDA Graphs capture of the repeated pipeline is an
evaluation item, not a commitment.

**Not built yet, and the reason is on record.** Batch streams already keep
their device state *and* their I/O buffers resident across appends, so the
allocation half of the rationale is gone. What remains is real but narrower:
each indicator's append moves its own inputs, so a 7-indicator GPU pipeline
uploads the same close matrix seven times per tick and issues seven
launch/sync pairs. At 65 536 symbols that is most of the 764 µs; a plan that
uploads OHLCV once and runs the pipeline over the resident copy should take
a large bite out of it. But the acceptance target (p99 < 1 ms at 256
symbols) is met at 19.5 µs without any of it, and AUTO stays under 1 ms up
to 65 536 symbols — so plans are scheduled on evidence rather than built on
spec, the same way `ct_*_param_sweep_d` was dropped.

### D4. ABI style

Same flat, generator-friendly style as the rest of the C ABI (append-only
within a MAJOR, existing entry points immutable): per-indicator `create`/`append`
functions (inputs differ per family: close-only vs HLC vs OHLCV),
shared lifetime functions on an opaque handle. All additions are
append-only symbols.

## C contract skeleton

### Opaque handles and lifetime

```c
typedef struct ctStream ctStream_t;             /* single series  */
typedef struct ctStreamBatch ctStreamBatch_t;   /* portfolio      */
typedef struct ctBatchPlan ctBatchPlan_t;       /* per-tick plans */

/* Shared lifetime — valid for any stream kind. Destroy is NULL-safe.
 * Reset keeps parameters, drops all state (as if freshly created). */
CTAPI_EXPORT ctStatus_t ct_stream_destroy(ctStream_t *stream);
CTAPI_EXPORT ctStatus_t ct_stream_reset(ctStream_t *stream);
CTAPI_EXPORT ctStatus_t ct_stream_batch_destroy(ctStreamBatch_t *streams);
CTAPI_EXPORT ctStatus_t ct_stream_batch_reset(ctStreamBatch_t *streams);

/* Introspection: how many bars must be appended before the first
 * non-NaN output (TA-Lib lookback of the configured indicator). */
CTAPI_EXPORT ctStatus_t ct_stream_lookback(const ctStream_t *stream,
                                           int *lookback);
```

### Single-series streams (CPU state, D1)

```c
/* One create per family; parameters mirror the full-series entry point. */
CTAPI_EXPORT ctStatus_t ct_ema_stream_create(int period, ctStream_t **out);
CTAPI_EXPORT ctStatus_t ct_rsi_stream_create(int period, ctStream_t **out);
CTAPI_EXPORT ctStatus_t ct_macd_stream_create(int fastPeriod, int slowPeriod,
                                              int signalPeriod,
                                              ctStream_t **out);
CTAPI_EXPORT ctStatus_t ct_atr_stream_create(int period, ctStream_t **out);
/* ... wave 1: sma, wma, natr, adx family, bbands, stoch/stochf, obv, ad */

/* Append n closed bars; writes exactly n output values per line, aligned
 * to the appended bars (warm-up positions are NaN, same as the tail of a
 * full recompute). n >= 1; inputs/outputs are host float32 like the host
 * API; internal state stays double. */
CTAPI_EXPORT ctStatus_t ct_ema_stream_append(ctStream_t *stream,
                                             const float *close, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_atr_stream_append(ctStream_t *stream,
                                             const float *high,
                                             const float *low,
                                             const float *close, int n,
                                             float *out);
CTAPI_EXPORT ctStatus_t ct_macd_stream_append(ctStream_t *stream,
                                              const float *close, int n,
                                              float *outMacd,
                                              float *outSignal,
                                              float *outHist);
```

Notes:

- Warm-up replay is just `append(history, historyLen, out)` — no separate
  "prime" call, one code path, one semantics.
- Handles are **not** thread-safe; one stream per producer thread, or
  external synchronization. Different handles are independent.
- Error contract matches the host API: `api_guard`, thread-local
  `ct_get_last_error()`, `CT_STATUS_INVALID_ARGUMENT` on NULL/negative
  inputs, no exception crosses the C boundary. Appending to a destroyed
  or foreign handle is rejected via a handle magic/ownership check (the
  cross-context precedent from the Context lifetime work).

### Batch streams (GPU state, D2)

```c
/* State for numSeries independent series of one indicator; state
 * matrices live on the device chosen at create time. */
CTAPI_EXPORT ctStatus_t ct_ema_stream_batch_create(int period, int numSeries,
                                                   ctStreamBatch_t **out);

/* Append n bars to every series in one call. Matrices are series-major
 * like the batch API: each series' n bars contiguous, numSeries * n
 * elements total. Host variant transfers in/out; _d variant takes device
 * pointers and leaves outputs resident (zero-copy pipelines). */
CTAPI_EXPORT ctStatus_t ct_ema_stream_batch_append(
    ctStreamBatch_t *streams, const float *closes /* [numSeries x n] */,
    int n, float *outs /* [numSeries x n] */,
    cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_ema_stream_batch_append_d(
    ctStreamBatch_t *streams, const float *d_closes, int n, float *d_outs,
    cudaStream_t stream CTAPI_STREAM_DEFAULT);
```

- Ragged portfolios (series at different lengths) are supported the same
  way the batch API handles them today: NaN-pad and let NaN semantics
  carry, or run several batch streams. No per-series lengths in v1.
- Indicators whose state includes a rolling window (MINMAX-based: STOCH,
  WILLR, BBANDS middle band, ...) keep an O(lookback) ring buffer per
  series inside the handle; memory is `numSeries * O(maxLookback)`,
  reported at create time via `ct_get_last_error`-adjacent introspection
  if needed (open question Q3).

### Persistent batch plans (D3) — sketch, refined in M2

```c
typedef struct ctBatchPlanDesc {
  int numSeries;
  int maxAppend;        /* max n per execute; buffers sized once */
  int maxConcurrency;   /* stream pool, same semantics as batch API */
  int deviceId;         /* reserved; must be 0 until multi-GPU (0.9.0) */
} ctBatchPlanDesc_t;

CTAPI_EXPORT ctStatus_t ct_batch_plan_create(const ctBatchPlanDesc_t *desc,
                                             ctBatchPlan_t **out);
CTAPI_EXPORT ctStatus_t ct_batch_plan_add_ema(ctBatchPlan_t *plan,
                                              int period, int *slotOut);
/* ... one add_* per wave-1 family; slots identify output columns */
CTAPI_EXPORT ctStatus_t ct_batch_plan_finalize(ctBatchPlan_t *plan);
CTAPI_EXPORT ctStatus_t ct_batch_plan_append(ctBatchPlan_t *plan,
                                             const float *ohlcv /* fixed layout */,
                                             int n, float *outs /* per slot */,
                                             cudaStream_t stream CTAPI_STREAM_DEFAULT);
CTAPI_EXPORT ctStatus_t ct_batch_plan_destroy(ctBatchPlan_t *plan);
```

Acceptance metric (from the roadmap): 256 symbols × 7-indicator plan,
1-bar append, **p99 < 1 ms** end-to-end on the RTX 3090 Ti.

## Parity oracle

The corpus is the contract. New streaming mode in
`benchmarks/differential_corpus.py` (wired into the GPU CI gate):

1. For every wave-1 spec and every corpus dataset, split the series into a
   prefix and a tail; create a stream, append the prefix, then append the
   tail in chunks of 1, 3 and 16 bars.
2. Compare the concatenated stream outputs against the full-series host
   call on the whole dataset under the strict oracle (shape, NaN masks,
   ±Inf, mixed abs/rel tolerance).
3. Target for double-state CPU streams: **bit-equal** to the full-series
   result where the full-series path is itself the double state machine;
   corpus tolerances elsewhere (documented per family, like
   [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) does today).
4. NaN inputs must poison/propagate exactly as the full-series semantics
   do (RSI Wilder poisoning, OBV NaN handling, ...) — the corpus datasets
   already exercise this.

Lifetime tests follow the `tests/cpp/test_context_lifetime.cpp` pattern:
destroyed-handle rejection, foreign-handle rejection, reset-equals-new,
independence of parallel handles.

## Bindings sketch

- Python: `s = tacuda.stream.Rsi(14)`; `s.append(arr) -> np.ndarray`;
  context-manager `__exit__` destroys. Batch: `tacuda.stream.RsiBatch`.
- Node: `const s = new tacuda.RsiStream(14); s.append(f32array)`;
  `Symbol.dispose` support.
- C#: `using var s = new Tacuda.RsiStream(14);` over `SafeHandle`.
- `bindings/audit_bindings.py` grows a streaming section; any gap fails CI.

## Open questions (to close before M1 exit)

- **Q1 — state serialization**: `ct_stream_save/restore` for process
  restarts without replaying history. Wave 2 unless a design blocker
  appears (restore must include the exact double state and window rings).
- **Q2 — max n per append** for batch streams/plans: fixed at create
  (`maxAppend`) vs reallocating. Skeleton says fixed — simpler and
  matches the latency goal.
- **Q3 — memory introspection**: report per-handle state bytes at create?
- **Q4 — multi-output alignment**: MACD/BBANDS/STOCH lines share one
  lookback convention in the full API (shared outBegIdx); streams must
  reproduce it — verify per family in the corpus, not by assumption.
- **Q5 — plans vs batch streams overlap**: a plan is essentially N batch
  streams + fused I/O. Decide in M2 whether plans wrap batch-stream
  handles internally or duplicate the state layout.

## Milestones

Tracked in [ROADMAP.md](ROADMAP.md) v0.8.0: M1 (this design + wave-1 CPU
streams + oracle), M2 (batch streams + plans + latency target), M3
(bindings + CI), M4 (docs/site/article + release).
