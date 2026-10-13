Changelog
=========

Unreleased (0.8.0 development)
------------------------------
Release preparation, 2026-10-14:

- Packaging: the Windows wheel no longer inherits the Linux `libtacuda.so`.
  setuptools never prunes `build/lib`, so the second platform wheel carried
  both natives. The validation step missed it because bash `set -e` ignores
  a failing `! cmd`; it now checks the exact native set of every wheel and a
  source-only sdist.
- Publishing moved from the three package jobs into one `publish` job that
  runs after all packages validate, checks secrets and the npm token first,
  and skips versions a registry already has, so a failed push is completed
  by a rerun instead of leaving a partial release.
- Linux GPU validation of `4e8edd1` (RTX 3090 Ti, CUDA 12.8): CTest 425/425,
  corpus 165/165, streaming corpus 699/699, Python/Node/C# pipelines green.
  Evidence in `docs/validation/2026-10-14/`, details in OPERATIONS.md.

Release preparation, 2026-10-07:

- Replace dereferenced raw stream handles with non-reused registry tokens;
  destroyed handles cannot alias new allocations. Covers single and batch
  APIs, with regression checks for allocation churn and wrong handle kinds.
- Include string-returning introspection in the binding audit: 811 declared
  symbols, not 808. Fix Python error-message return types and expose native
  status/last-error helpers in Python and Node.
- Align native/package versions to 0.8.0. Publication remains gated on a
  numbered release section, an exact matching tag and GPU correctness.
- Refresh CUDA installation, binding/test workflow triggers and the pinned
  TA-Lib reference. Validate actual PE/ELF exports instead of binary strings.
- Correct documentation counts, per-indicator warm-up, and benchmark baselines.
  Coverage means native TA-Lib 0.7.1 (161 functions); 0.8.1 migration remains
  open (40 missing corpus specs and IMI/WILLR edge semantics).

Streaming (incremental update) API, milestones M1, M2 and M3 — single-series
streams, batch (portfolio) streams, and language bindings with corpus-level
parity verification. User guide: `docs/STREAMING_API.md`, rationale:
`docs/DESIGN_STREAMING_API.md`.

M3 — bindings and the streaming corpus:

- Python (`tacuda.stream`), Node.js (`tacuda.RsiStream`, ...) and C#
  (`RsiStream`, ...) wrappers for all 20 stream types. Hand-written rather
  than generated: the binding generator models array-in/array-out calls and
  cannot express a handle with a lifetime. Idiomatic per language — context
  managers in Python, `destroy()`/`Symbol.dispose` in Node, `IDisposable`
  with tuple returns in C#; a released handle is rejected everywhere instead
  of reaching freed native state.
- `bindings/audit_bindings.py` grew a streaming bucket: all 52 streaming
  symbols must be reachable from every language, expanding the family tables
  the Python and Node wrappers are built from rather than trusting a literal
  name scan. Verified to actually fail (exit 1, 4 gaps) when a family is
  removed. Wired into ctest as `bindings_coverage`.
- `bindings/generate_stream_bindings.py` emits the repetitive C# per-family
  wrappers with a `--check` staleness gate, wired into ctest as
  `stream_bindings_up_to_date`.
- New `benchmarks/streaming_corpus.py`: all ten families over the seven
  differential-corpus datasets under the same strict oracle, asserting that
  chunked appends (1/3/16/all) reproduce the full-series tacuda call, that
  the chunkings are bit-identical to each other, that the result matches the
  installed **TA-Lib**, and that batch rows equal single streams on both
  execution paths. **699/699 cells pass.** The single exclusion is NATR on
  the zero-crossing dataset, where TA-Lib's own ta_NATR.c corrupts its first
  output bar — the same exclusion the differential corpus already makes.
- Streaming cases added to all three binding pipelines: Python 34/34,
  Node 201/201, C# 36/36. C++ suite plus binding gates: 419/419.

M2 — batch (portfolio) streams:

- 27 more appended C ABI symbols: `ctStreamBatch_t` carries `numSeries`
  independent series of one indicator, one append advances all of them
  (`ct_<family>_stream_batch_create/append` for the same ten families, plus
  `ct_stream_batch_{destroy,reset,lookback,series,outputs,bars,set_exec}`).
  Matrices are series-major like the batch API.
- The state machines moved to `include/utils/StreamStates.h` as POD structs
  with `__host__ __device__` step functions, so the single-series path, the
  CPU batch path and the CUDA kernel run *the same* recurrence. A batch of N
  series is therefore bit-identical to N separate streams — asserted per
  family, per execution path, in `tests/cpp/test_stream_batch.cpp` (8 cases,
  including reset invalidating resident device state and policy changes
  mid-stream migrating state rather than restarting it).
- `ctStreamExec_t` selects CPU or GPU; `CT_STREAM_EXEC_AUTO` switches at the
  measured crossover of 2048 series. 7-indicator pipeline, one bar per tick,
  i3-12100F + RTX 3090 Ti (`benchmarks/bench_stream_portfolio.cpp`):
  **256 symbols p99 19.5 µs** against the 1 ms acceptance target; CPU wins to
  1024 symbols (67.6 µs vs 124 µs), the paths meet at 2048 (141 vs 139 µs),
  and the GPU wins above (65 536 symbols: 764 µs vs 4 721 µs). AUTO keeps p99
  under 1 ms through 65 536 symbols.
- Dropped before shipping: a library-managed CPU thread pool. Spawning
  workers per append measured 595 µs at *every* portfolio size — thread
  creation swamped the work. Series are independent, so callers who want CPU
  parallelism shard across several handles with their own threads.
- Persistent batch plans are deliberately not built yet. Batch streams
  already keep device state and I/O buffers resident, so what a plan would
  still buy is one OHLCV upload per tick instead of one per indicator; that
  is worth much of the GPU tick cost at tens of thousands of symbols, but the
  acceptance target is met without it. Rationale recorded in
  DESIGN_STREAMING_API.md D3.

M1 — single-series streams:

- New public header `include/tacuda_stream.h` and 25 appended C ABI
  symbols: an opaque `ctStream_t` handle with shared lifetime calls
  (`ct_stream_destroy`/`reset`/`lookback`/`bars`/`outputs`) plus
  `create`/`append` pairs for ten families — SMA, EMA, WMA, RSI, ATR,
  NATR, OBV, AD, MACD (3 lines) and BBANDS (3 lines). Append-only, so the
  existing ABI is untouched.
- Streams carry their state on the host in double precision. Measured on
  the RTX 3090 Ti (RSI(14), `benchmarks/bench_stream_latency.cpp`): an
  append costs p50 0.04 µs regardless of history length, against p50
  142 µs (100K bars) and 1 234 µs (1M bars) for a full `ct_rsi`
  recompute — roughly 3 500x and 31 000x per tick. History replay runs at
  ~55 M bars/s. This measurement is what settled the CPU-vs-GPU choice for
  single-series updates; portfolio-wide appends stay the GPU's job and are
  the next milestone.
- Each state machine mirrors the exact recurrence of its full-series
  counterpart, including the load-bearing details: TA-Lib's flat-window
  RSI convention and NaN poisoning, the ATR seed as the mean of
  TR[1..period] landing at index `period`, NATR's `TA_IS_ZERO` close
  guard, MACD's two EMA seeds taken at `slowPeriod-1` (the fast one is the
  fast-length SMA *ending* there) with the signal EMA consuming the
  float-rounded MACD line and all three lines sharing one `outBegIdx`, and
  BBANDS' population variance clamped at zero.
- Parity is verified against the full-series host API, not asserted:
  `tests/cpp/test_stream_api.cpp` (18 cases) compares every family at four
  chunk sizes (1, 3, 16, whole series) under the corpus oracle. NaN masks
  match exactly; MACD, OBV, AD and SMA are bit-identical, the rest agree
  within 1.2e-07 (EMA/WMA/BBANDS) to 7.4e-07 (NATR) relative — streams are
  double where some full-series kernels recurse in float, so they are at
  least as accurate, not bit-equal.
- Lifetime and error contract match the host API: `api_guard` equivalent
  on every entry point, thread-local `ct_get_last_error()`, NULL-safe
  destroy, use-after-destroy rejected with `CT_STATUS_INVALID_ARGUMENT`
  instead of undefined behavior, and an append through the wrong family's
  function rejected rather than silently misread.
- Window families (SMA/WMA/BBANDS) re-sum their ring buffer per bar
  (O(period), independent of history) rather than keeping a rolling
  accumulator, which would drift away from the full-series kernels.

Not yet: persistent batch plans, `_ex` dtype variants, and state serialization.

0.7.1 — 2026-07-13
------------------
Third-round audit fixes: the last lifetime hole, full device error
contract, Node batch shape checks and stricter verification tooling.

CUDA lifetime, closed for ALL buffer types:
- `DeviceBufferBase` (dtype-aware) now carries the same stream guard as
  `DeviceBuffer`; `Context::upload/allocate(..., dtype)` pass the control
  block, so those buffers can also safely outlive their Context.
- Cross-context rejection: every Context method that consumes a
  DeviceBuffer verifies the buffer was created by THIS Context (buffers on
  raw user-managed streams pass through). Running Context B's kernels over
  Context A's buffer would order the release on the wrong stream — now it
  throws `std::invalid_argument` instead of risking an early pool return.
- New lifetime tests (`tests/cpp/test_context_lifetime.cpp`): float and
  dtype-aware buffers outliving their Context, cross-context rejection for
  single- and multi-input methods and download, view guard inheritance.

Windows build:
- `tools/optimizer/src/portfolio/Portfolio.cpp` missing `<iterator>`
  (std::back_inserter) broke the default `TACUDA_BUILD_OPTIMIZER=ON`
  configuration under MSVC. CI now builds ALL targets on both OSes, not
  just the tacuda library, so optimizer/example_app breakage is visible.

Device C API error contract, complete:
- Every `ctStatus_t` entry point in `api_device.cu` (manual, `_ex_d` and
  all 114 generated `ct_*_d`) calls `clear_last_error()` on entry — a
  successful call no longer leaves a stale message in
  `ct_get_last_error()`.
- All remaining silent INVALID_ARGUMENT / UNSUPPORTED / COPY_FAILED
  branches now set a message (generator emits them; one-line manual checks
  braced with `set_last_error`).
- `ct_device_alloc_ex` reports cudaMalloc failures via the same channel.

Bindings:
- Node batch wrappers compare BOTH dimensions (`seriesLen` x `numSeries`),
  not just the element count: a 2x10 vs 4x5 mix now throws RangeError
  (hand-written atr/stochastic and all generated multi-input families).
  Negative test added to the Node pipeline.
- Node now covers the full C ABI: the generator emits koffi declarations
  and `device.*` wrappers for all 114 generated single-series `ct_*_d`
  functions (`bindings/audit_bindings.py` reports 756/756 and exits
  non-zero on any future gap).

Semantics:
- IMI validates period in [2, 100000] like TA-Lib BAD_PARAM (period=1 was
  accepted before); ADOSC periods bounded to [2, 100000] as well.

Verification tooling:
- Corpus coverage check is EXACT (talib_name recorded on each spec by
  _maybe_add) and failing coverage fails the run; summary wording is
  "N/M specs" (161 TA-Lib functions, some covered by multiple specs).
- Candle cells record referenceSignalCount / positive / negative, the
  summary lists patterns that never fire on the corpus (vacuous exact
  pass) and the JSON keeps per-spec signal totals.
- `bindings/audit_bindings.py` also scans `batch_generated.js`, exits 1 on
  any gap/stale decl, and runs in the native-build workflow.
- NuGet smoke only tolerates DllNotFoundException whose message names a
  CUDA runtime library; a load failure on tacuda itself (broken RID or
  probing) fails the job.

0.7.0 — 2026-07-13
------------------
Second-round audit fixes: CUDA lifetime hardening, remaining TA-Lib
discrepancies, exact candle parity and an honest release pipeline.

CUDA lifetime:
- The Context stream now lives in a shared control block; every
  DeviceBuffer created by a Context holds a guard, so buffers may safely
  outlive the Context (the stream is synchronized and destroyed only after
  the last user is gone).
- `DeviceBufferPool::releaseAsync`: when event recording or the fallback
  stream synchronize FAILS, the pointer is freed (leak-safe) and never
  re-enters the cache — a failed sync no longer risks handing out memory
  that may still be written by a dying stream.

Semantics:
- ADOSC keeps the period ROLES exactly like TA-Lib: ADOSC(10,3) =
  EMA(10) - EMA(3); only the lookback uses the slowest period (the 0.6.0
  sorting normalization produced a flipped sign for reversed periods).
  Minimum period is 2, like TA-Lib's BAD_PARAM contract.
- IMI is end-of-window aligned with a leading warm-up of period-1 NaNs
  (unit test updated — it previously froze the start-of-window layout).
- HT_TRENDMODE warm-up is 0 in ALL paths (host CPU, device kernel, batch),
  matching TA-Lib's int-output convention.

Candle parity, now exact:
- All 61 pattern predicates run in DOUBLE like TA-Lib (double
  TA_CANDLEAVERAGE, real bodies and shadows), so borderline candles resolve
  identically on the float32-quantized inputs.
- The remaining 31 patterns are literal ports too (DOJI/HAMMER/ENGULFING
  incl. the +-80 touching-edges convention, HARAMI/HARAMICROSS +-80,
  KICKING/KICKINGBYLENGTH with shadow-inclusive gaps, BREAKAWAY, INNECK,
  LADDERBOTTOM, TASUKIGAP, 3LINESTRIKE, 3OUTSIDE, RISEFALL3METHODS, ...).
- The corpus oracle for candles demands EXACT signal equality: no head
  mask, no flip budget; mismatch rates are reported over non-zero signals.

Device C API / bindings:
- All 51 hand-written `ct_*_d` validation sites return
  CT_STATUS_INVALID_ARGUMENT with a fresh error message (previously
  KERNEL_FAILED with a stale one).
- `element_size()` returns 0 for unknown dtypes and every consumer
  validates (alloc/upload/download/convert reject instead of silently
  treating garbage enums as float32).
- Node: mfi/ad/adosc validate all four arrays against each other, wobv
  validates volume, `_ex` wrappers require matching TypedArray dtypes and
  lengths (`assertSameTyped`).

Corpus coverage and honesty:
- Added specs: MA(SMA/EMA), LINEARREG, LINEARREG_SLOPE/INTERCEPT/ANGLE,
  TSF. The runner prints a warning listing any installed TA-Lib function
  without a spec.
- Removed the no-op `slice_valid` shims from RSI/ADX/ADXR/DX/±DI specs;
  the only remaining use aligns the ct_macd_line EXTENSION and is
  documented as such.

Release pipeline:
- npm packaging now `needs: [natives]` — it cannot publish before the
  (tag-mandatory) GPU gate passes.
- Python wheel smoke asserts the import resolves from site-packages (not
  the source tree); npm smoke installs the packed .tgz into a scratch
  project and requires the INSTALLED package; a new NuGet smoke builds a
  console consumer against the local nupkg (SDK 8 + 9) and calls
  `NativeMethods.ct_version()`.
- `example_app` rewritten onto the public C API (host + device calls) —
  it links again with the restricted export table.
- Linux .so builds with hidden visibility presets: only CTAPI_EXPORT /
  TACUDA_API symbols remain in the dynamic symbol table.
- Root CHANGELOG.md points at docs/CHANGELOG.md as canonical;
  package-lock.json version is kept in sync.

0.6.0 — 2026-07-13
------------------
Step 7 of the audit plan: the remaining tracked deviations from TA-Lib are
closed or corpus-verified. Differential corpus result on the RTX 3090 Ti:
**1098/1098 cells, 157/157 specs green** (88 numeric + APO/PPO-SMA + 6
Hilbert + 61 candlestick patterns over 7 datasets; report committed as
`benchmarks/corpus_rtx3090ti_0.6.0.json`). Full C++ suite 387/387; binding
pipelines Python 26/26 and Node 194/194 on both Linux and Windows.

- **APO/PPO `matype`**: new appended C-ABI symbols `ct_apo_ma`/`ct_ppo_ma`
  with a selectable MA type (TA-Lib's `optInMAType`; TA-Lib default is
  SMA). The plain `ct_apo`/`ct_ppo` keep their historical EMA behavior.
  APO/PPO/MACD/MACDEXT now normalize reversed periods like TA-Lib (slow =
  max; MACDEXT swaps the MA types together with the periods) instead of
  rejecting them. Bindings: `tacuda.apo_ma/ppo_ma` (Python, generated),
  `apoMa/ppoMa` (Node), `ct_apo_ma/ct_ppo_ma` (C#). Corpus specs compare
  APO/PPO with matype=0 against TA-Lib defaults.
- **Hilbert family**: literal-port fixes against ta_HT_*.c — the dominant
  cycle phase is now STATEFUL across bars exactly like TA-Lib (an exact
  zero imaginary part adjusts the previous phase by ±90° instead of
  resetting it; offsets stack on the carried value), it is updated on every
  bar including warm-in, DCPeriodInt is no longer clamped, and the
  360/smoothPeriod lag compensation is unconditional. HT_TRENDMODE keeps
  its sine/leadSine/DCPhase state in TA-Lib's exact update order and its
  warm-up is NaN. All six HT functions added to the differential corpus.
- **Candlestick patterns**: all 61 TA-Lib patterns added to the
  differential corpus as discrete signals — NaN masks must match and the
  fraction of flipped bars (float32-vs-float64 boundary rounding) must stay
  under 0.2% per dataset; a logic difference fails instantly. Dead
  fixed-percentage helpers (`is_long_body`/`is_short_body`/`is_downtrend`/
  `is_uptrend`) removed from CandleUtils.h — every pattern uses the TA-Lib
  candle-settings machinery (rolling averages over the settings periods).
- **Short inputs**: 60+ indicators now return success with an all-NaN line
  when the input is shorter than the lookback (TA-Lib returns success with
  zero output elements), instead of CT_STATUS_INVALID_ARGUMENT. Parameter
  validation (non-positive periods, null pointers) still errors. Covers
  the SMA/EMA/WMA/MA family, Wilder family (ATR/ADX/DX/±DI/±DM), ROC
  family, stochastics (shared K/D lookback), AROON, MINMAX/INDEX family,
  DEMA (3*p-2 class), RSI, ULTOSC, MACD/MACDEXT and the rest of the
  windowed families.

0.5.0 — 2026-07-13
------------------
Hardening release driven by an external production-readiness audit: memory
lifetime, C ABI safety, honest TA-Lib parity checking, and a set of breaking
semantic alignments with TA-Lib.

Memory / runtime safety (P0):
- `DeviceBufferPool` rewritten: allocations are cached **per device**, a
  buffer released while stream work is still in flight is parked on a CUDA
  event (`releaseAsync`) and reused only after the event fires, and
  `cleanup()` now frees only cached/pending buffers — never allocations
  still held by callers (previous behavior could use-after-free both ways).
  `DeviceBuffer`/`DeviceBufferBase`/api-internal buffers all release
  stream-aware now.
- C ABI guard: all 183 host entry points are wrapped in a `noexcept`
  `api_guard` — thread-local error cleared on entry, every pointer/size
  validated (`CT_STATUS_INVALID_ARGUMENT` instead of UB on nulls, negative
  sizes, or `std::vector` length overflow), and no C++ exception (including
  indicator constructors like MAVP's) can cross the C boundary anymore.
  Multi-output temp buffers use `size_t` arithmetic (no `int` overflow).
- Batch API: `cudaStreamSynchronize` failures now propagate
  (`CT_STATUS_DEVICE_ERROR` instead of silent `SUCCESS` after an async
  kernel fault); `DeviceMatrices::acquire` no longer leaks pool matrices on
  partially failed allocation.
- WMA: shared mutable weight caches removed entirely (weights 1..P are
  computed in-register — the caches were racy across streams and wrong
  across devices); the vectorized path no longer performs misaligned
  `float4` loads.
- `Context` move semantics implemented (stream ownership transfers; no
  double `cudaStreamDestroy`); all multi-input `Context` methods validate
  equal buffer sizes before launching kernels.

Breaking: TA-Lib semantic alignment (outputs change):
- `SUM`, `MINMAX`: output is now end-of-window aligned with a leading
  warm-up of `period-1` NaNs (TA-Lib convention) instead of
  start-of-window with trailing NaNs.
- `TRANGE`: bar 0 is NaN (TA-Lib lookback 1), not `high-low`.
- `RSI`: a fully flat window yields `0` (TA-Lib `100*g/(g+l)` with zero
  sums), not `50`; NaN inputs poison the Wilder recursion instead of being
  treated as zero change.
- `ROC`/`ROCP`/`ROCR`/`ROCR100`: zero denominator yields `0` (TA-Lib), not
  NaN.
- `ADOSC`: lookback is `slowPeriod-1` (one bar earlier, matching TA-Lib
  `EMA` lookback); reversed periods are normalized (fastest=min,
  slowest=max) instead of rejected.
- `AROON`/`AROONOSC`/`MAXINDEX`/`MININDEX`/`MINMAXINDEX`: ties now pick the
  most recent bar (TA-Lib sliding-update semantics).
- `ULTOSC`: the three periods are sorted internally like TA-Lib instead of
  requiring pre-sorted arguments.
- Hilbert family (`HT_*`): outputs are NaN-prefilled, so short inputs
  return NaN instead of untouched memory.
- The optimizer's TacudaBackend/TacudaBatchBackend TRANGE warm-up shims and
  the benchmark SUM/MINMAX alignment shims were removed (kernels are
  natively aligned now).

Parity verification:
- `compare_arrays` is a strict oracle: shape, NaN masks, ±Inf masks/signs
  and a mixed absolute/relative tolerance (`|a-b|/max(1,|a|,|b|)`) must all
  pass — `nan_mismatch` can no longer be ignored (the old oracle passed
  `[2,3]` vs `[NaN,3]` and even values-vs-all-NaN).
- New `benchmarks/differential_corpus.py`: every binding-level function
  runs against the installed TA-Lib over random walks (3 seeds), a trend,
  a flat series, zero-crossing data and a short (32-bar) input, under the
  strict oracle; JSON report; nonzero exit on any failure. Wired into the
  GPU CI job. Result on the RTX 3090 Ti: **615/615 cells, 88/88 specs
  green** (one cell skipped: TA-Lib's own ta_NATR.c bug writes the
  zero-close guard value to outReal[0] instead of outReal[outIdx]).
- The corpus immediately caught and led to fixing real bugs the old
  comparison never saw:
  - KAMA block-scan produced garbage for tails in (256, 4096] — only the
    first CUDA block was ever scanned; plus the efficiency ratio now
    matches TA-Lib exactly (signed periodROC comparison, er=1 on zero
    volatility).
  - MAMA is now a literal double-precision TA-Lib port: plain `atan`
    phase/period (atan2 flipped the phase by 180° for negative I1 and
    locked the adaptive alpha onto different trajectories), double WMA(4)
    smoothing, TA-Lib's zero seeds and same-bar smoothing order. The old
    comparison hid this behind an absolute tolerance of 70.0.
  - MAVP+EMA seeds each per-period EMA at the MAVP start index the way
    TA-Lib's TA_MA(startIdx, ...) does, instead of running whole-series
    EMAs (visible drift for thousands of bars); the old spec masked the
    first 60 bars and allowed an absolute tolerance of 2.0.
  - CCI accumulates in double and applies TA-Lib's both-nonzero guard
    (float32 rounding on constant windows used to turn 0/0 noise into
    ±66.67).
  - CORREL emits 0 for degenerate windows (TA_IS_ZERO_OR_NEG semantics,
    with a relative clamp against prefix-sum cancellation noise).
  - Stochastic %K emits 0 on a zero range (two kernels said 100), and
    STOCHF/STOCHRSI publish %K/%D with TA-Lib's shared outBegIdx.
  - MAXINDEX/MININDEX/MINMAXINDEX reproduce TA-Lib's path-dependent tie
    handling exactly via sequential state-machine kernels.
  - TEMA/TRIX return all-NaN success for inputs shorter than the lookback
    (TA-Lib returns success with zero elements) instead of erroring.
  - NATR guards the close denominator with TA-Lib's 1e-14 epsilon.

Bindings contracts:
- Node.js: all multi-array wrappers validate equal input lengths
  (`assertSameLength`) before calling native code.
- C#: `void*` precision (`_ex`) parameters marshal as `IntPtr` instead of
  `float[]` (F16/F64 buffers are no longer corrupted by the marshaller).
- Python: `_ex` wrappers honor `storageType` — buffers convert to and
  return in the selected dtype (float16/float32/float64) instead of being
  silently cast to float32.

Build / CI / release:
- `WINDOWS_EXPORT_ALL_SYMBOLS` removed: the DLL exports only the public
  C ABI (`CTAPI_EXPORT`) and the C++ `Context` API (`TACUDA_API`), not
  ~5.9k internal C++/Thrust/CUB symbols.
- GPU correctness job (C++ suite + binding pipelines + differential
  corpus) is now mandatory for version tags — packaging waits for it.
- Package smoke tests are real: Python wheels are validated for per-
  platform contents and importability (no more `find_spec`-only), npm
  load errors are no longer swallowed by `|| true`.
- Python wheels are built per platform (`manylinux_2_35_x86_64` with the
  .so, `win_amd64` with the .dll) — no more `py3-none-any` wheel carrying
  both natives.
- README claims corrected: "161/161" now means full API surface with
  corpus-tracked numerical parity (deviations documented), and prebuilt
  binaries state CC ≥ 7.5 (source builds cover older archs).
- Repo hygiene: 629 tracked build artifacts (node_modules, obj/bin,
  __pycache__, dist tarballs/nupkgs) removed from git and ignored.

0.4.1 — 2026-07-08
------------------
- Club-pack recursions re-formulated as parallel scans (`ClubPack.cu`):
  OBTR/WOBV/ASI are prefix sums of per-bar terms (thrust inclusive scan,
  double accumulation), HPI an `AffineScan` first-order recurrence like
  EMA, XAverage a seed-aware affine monoid scan whose NaN inputs are
  identities (exact sequential seeding semantics preserved; all ClubPack
  parity tests unchanged and green). One series now saturates the GPU on
  its own: `exp06_asi` indicator phase 2.03 s → 0.17 s at 16 sym × 500 k
  (×11.7). `SAR_EL` stays the documented single-thread exception (trend
  flips + acceleration clamps do not associate).
- Optimizer: multi-threaded CPU indicator baseline —
  `PerSymbolBatchBackend` takes a thread count (work-stealing over
  symbols, per-symbol backend instances stay single-owner), wired to
  `--threads` for the portfolio TA-Lib path (0 = all cores, unchanged
  default). `bench_experiments_batch.py` now measures four
  configurations (batch, GPU per-symbol loop, TA-Lib 1 thread, TA-Lib
  all cores) and takes `--strategies` for subset/scaling runs.
- TA-Lib comparison round on the RTX 3090 Ti + i3-12100F box (summaries
  committed under `tools/optimizer/bench/exp_v3_*`): all 94 systems
  ×1.18 (16 sym × 500 k) and ×1.63 (64 × 500 k) vs single-threaded
  TA-Lib with identical trades; scaling sweep 4 → 1024 symbols shows the
  windowed kernels beating even all 8 CPU threads (`exp10_ref_dev` ×10.1
  vs 1 thread / ×2.5 vs 8 at 1024 sym × 100 k), while trivial
  request sets stay PCIe-bound — analysis in docs/EXPERIMENTS.md.
- Site: new "Strategy workloads" section in `site/benchmark.html`
  (full-pack table, kernel-category table, two log-log scaling charts,
  TA-Lib capability matrix), generated from the committed summaries by
  `scripts/gen_talib_comparison.py`; linked from `experiments.html`.
- No C ABI signature changes (kernel-internal rewrite only); version
  0.4.1 across native + Python/Node/.NET packages.

0.4.0 — 2026-07-08
------------------
- New: full batch coverage of the indicator surface — every C-ABI family
  that lacked a `_batch` variant got generated `ct_*_d` device wrappers
  and `ct_*_batch`/`ct_*_batch_d` entry points: SAR/SAREXT, MAMA,
  MACDFIX, STOCHRSI, TRIX, NVI/PVI, six `ct_ht_*`, APO/PPO/PVO, BOP,
  IMI, AVGDEV, MIDPOINT/MIDPRICE, ±DM, MAXINDEX/MININDEX/
  MINMAX(INDEX), CHANGE, MACD_LINE, MA/MAVP, CORREL/BETA and all 63
  candlestick patterns. 114 new device wrappers are thin generated
  shims over the existing CUDA indicator classes (multi-output classes
  get pooled stacked buffers + D2D split); `scripts/generate_batch_api.py`
  now also emits the device header region, `api_device_generated.inc`
  and its include list. Batch runner I/O limit raised to 5 matrices
  (OHLCV). 166 batch families total.
- New: STAD "club pack" indicators as native families (`ClubPack.cu`,
  semantics = the EXPERIMENTS pack reference implementations):
  `ct_asi`, `ct_sar_el` (EasyLanguage-seeded parabolic), `ct_obtr`,
  `ct_wobv`, `ct_hpi`, `ct_market_strength`, `ct_trendscore`,
  `ct_refdev`, `ct_skew`, `ct_kurtosis`, `ct_pivot_high`/`ct_pivot_low`,
  `ct_xaverage` and NaN-strict `ct_rolling_{mean,max,min,std}` — host +
  device + batch each (host single-series entries are generated over the
  batch runner with numSeries=1). Sequential recursions run one GPU
  thread per series (the SAR precedent); windowed ones one thread per
  element. C++ parity tests (`ClubPack.*`) against the pack references.
- New (optimizer): derived-series indicator requests —
  `IndicatorRequest.chain` lets NaN-aware club families consume another
  request's output line ("XAverage of OBTR"); all four backends compute
  chains natively (GPU: device-side, batch backend on whole matrices,
  no host round-trips; CPU backends via the shared `ClubCpu.h` port),
  the indicator cache folds chains into its key, `validate_chain()`
  guards the contract. `Family` grew by 17 club members. Nine
  EXPERIMENTS systems moved their custom CPU math into requests
  (exp03_obtr, exp04_hpi_breakout, exp04_price_dist, exp06_asi,
  exp06_parabolic_vol, exp09_obv_revisited, exp10_ref_dev,
  exp10_superman, exp11_trendscore).
- Bindings: Python/ctypes and C# P/Invoke pick the ~360 new symbols up
  from the regenerated `_generated.py` / `Tacuda.Generated.cs`; Node.js
  gains hand-written club wrappers (`asi`, `sarEl`, `obtr`, `wobv`,
  `hpi`, `marketStrength`, `trendscore`, `refdev`, `skew`, `kurtosis`,
  `pivotHigh/Low`, `xaverage`, `rollingMean/Max/Min/Std`) plus the
  generated `*Batch` families; C# `TacudaBatch` partial regenerated.
- Benchmarks (94 EXPERIMENTS systems, indicator phase, RTX 3090 Ti):
  the batch API now beats the per-symbol device loop ×1.64 at
  16 sym × 500 k bars and ×1.57 at 256 × 20 k (was ×0.85/×0.99 before
  the rewire) — chained club systems gain up to ×3.7 (`exp06_asi`
  2.03 s vs 7.51 s). Against TA-Lib: windowed families win big
  (`exp10_ref_dev` ×4.6, `exp04_price_dist` ×3.8), single-thread GPU
  recursions on few series still lose (`exp03_obtr` 1.13 s vs 0.21 s) —
  scan-based reformulation is the known next lever. Full numbers in
  docs/EXPERIMENTS.md.
- Docs: BATCH_API.md family table (0.4.0), EXPERIMENTS.md "gaps closed"
  section, STRATEGY_OPTIMIZER.md.
- EXPERIMENTS × batch API: all 94 pack systems run multi-symbol through
  `tacuda-opt portfolio` on three interchangeable backends — `tacuda`
  (one `ct_*_batch_d` per indicator request), the new `tacuda-device`
  (same GPU kernels, one single-series `ct_*_d` call per symbol — the
  benchmark baseline) and `talib`. Portfolio reports now split timing
  into the indicator phase (uploads + kernels + downloads) and the
  signals+simulation phase. New driver
  `scripts/bench_experiments_batch.py` (94 systems × 3 backends with
  net-profit parity checks): on the RTX 3090 Ti box the indicator phase
  over all systems is batch 16.6 s / device-loop 14.1 s / TA-Lib 18.2 s
  at 16 sym × 500 k bars, and 14.8 / 14.7 / 10.9 s at 256 × 20 k — the
  EXPERIMENTS workload is PCIe/latency-bound (median ~3 requested lines,
  CPU simulation), so the batch API pays off only on kernel-heavy request
  sets (`exp06_cci_avg` ×2.0 vs TA-Lib); numbers and the inventory of
  what is still missing from the batch API (unbatched C-ABI functions,
  CPU-only club indicators, derived-series chaining) are documented in
  `docs/EXPERIMENTS.md`. Fixed in the optimizer batch backend: column
  uploads are synchronized before the first `ct_*_batch_d` call (batch
  entry points dispatch onto non-blocking streams that do not order
  against the legacy-stream upload — a real race), outputs download as
  one matrix-sized D2H per line into a reused staging buffer (was
  `numSeries` small copies + a fresh device alloc per call). Backend
  parity tests now cover all 22 requested families on tacuda/TA-Lib/
  reference with per-family tolerances; optimizer suite is 129 tests.
  Scan-based GPU kernels (RSI/BBANDS/EMA/ATR…) are documented as
  ULP-reproducible only (CUB decoupled look-back), so the bench compares
  GPU paths within 0.1% of net profit. No C ABI / bindings / package
  changes.
- EXPERIMENTS strategy pack: 94 classic trading systems from the Omega
  Research System Trading & Development Club volumes 1–13 (1997–2000),
  ported from TradeStation EasyLanguage to `tacuda-opt` strategies
  (`tools/optimizer/src/strategy/ExpVol01..13.cpp`, shared utilities in
  `ExpHelpers.h`, names `exp01_*`…`exp13_*`). Uniform adaptation
  conventions: close-confirmed stop/limit fills at the next open, one unit
  per side (no pyramiding), dollar stops as ATR multiples; club-specific
  indicators (Accumulation Swing Index, modified parabolic, On Balance
  True Range, Herrick-Payoff proxy, MarketStrength, TrendScore, Reference
  Deviation, rolling kurtosis/skew, swing pivots) implemented CPU-side.
  The `Family` request set grew by 17 indicator families (WMA, TRIMA,
  MACD, STOCH, CCI, ADX, +DI/-DI, OBV, MOM, ROC, STDDEV, LINEARREG, MAX,
  MIN, TRANGE, …) across all three backends. New tests
  (`test_experiments.cpp`): registry, declarations, full-pack smoke run,
  signal-range and helper units — optimizer suite now 127 tests. Benchmark
  driver `scripts/bench_experiments.py` (backtest + genetic + walk-forward
  per system; 94 systems × 3 passes ≈ 47 s on the RTX 3090 Ti box) and
  site generator `scripts/gen_experiment_pages.py`: hub
  `site/experiments.html` + 94 per-system pages with descriptions,
  parameters, reproduce commands, equity SVG, GA top-5 and WFO tables.
  Docs: `docs/EXPERIMENTS.md`. No C ABI / bindings / package changes.
- New tool: `tacuda-opt` strategy optimizer (`tools/optimizer/`, CMake option
  `TACUDA_BUILD_OPTIMIZER=ON`) — console backtester + exhaustive grid search
  in the MultiCharts spirit. GPU indicators through the device API (single
  OHLCV upload, per-request `ct_*_d` kernels, host-side indicator cache with
  cross-variant deduplication), CPU trade simulation (next-bar-open fills,
  commission/slippage model, double-precision PnL) parallelized over
  variants, metrics (PF, MaxDD, Sharpe, Sortino, ...) with a `--fitness`
  expression parser, CSV/JSON reports. TA-Lib CPU fallback backend loaded
  dynamically at runtime (`--backend auto|tacuda|talib`) — no build-time
  TA-Lib dependency. Three built-in strategies: `ma_cross`, `rsi_reversion`,
  `bbands_breakout`. Measured on RTX 3090 Ti: 10 000-variant EMA-cross grid
  on 1M bars in 21.5 s end-to-end; 1M-bar backtest in 0.17 s.
  Docs: `docs/STRATEGY_OPTIMIZER.md`; tests: `tacuda_opt_tests`
  (ctest label `optimizer`). No C ABI / bindings / package changes.
- `tacuda-opt genetic` (optimizer M2): genetic search over the same discrete
  space as `grid` (genes are indices into `--grid` axes) — tournament
  selection, uniform crossover, elitism, gaussian mutation scaled in axis
  steps, early stop on stagnation, genome deduplication (no variant is ever
  simulated twice), deterministic for a fixed `--ga-seed`. On the M1
  10 000-variant space × 1M bars it found the same best variant with 294
  unique evaluations (2.9%) in 1.16 s vs 21.5 s exhaustive; the acceptance
  test (≥95% of grid best at ≤10% of evaluations) is part of the suite.
- `tacuda-opt wfo` (optimizer M3): walk-forward optimization — K equal
  rolling windows with an `--is` in-sample share, per-window re-optimization
  (`--optimizer grid|genetic`), out-of-sample evaluation of the chosen
  parameters, stitched OOS equity with overall metrics and WFE
  (per-bar OOS/IS net-profit ratio), console table + `--out-json` report.
  Indicators are computed once over the full series and shared between
  windows through the cache (causal — a no-leak test guards it); ranged
  simulation (`SimRange`) keeps trading inside each segment. 6 windows over
  1M bars: 1.53 s (grid, 230 variants/window) / 2.33 s (genetic over the
  10K space). Optimizer test suite is now 90 tests.
- `tacuda-opt portfolio` (optimizer M4): multi-symbol backtest with shared
  parameters — the optimizer's first consumer of the batch API. Per used
  OHLCV column one `numSeries × seriesLen` matrix is uploaded once and each
  indicator request is a single `ct_*_batch_d` call (per-symbol TA-Lib on
  the CPU path). Series are aligned to the timestamp intersection (or by
  index when equal-length and timestamp-less), capital is split equally,
  positions/trades stay per-symbol, equity is aggregated with portfolio
  metrics and per-symbol console/JSON tables. 8 symbols × 1M bars in 1.28 s,
  16 × 1M in 1.74 s (RTX 3090 Ti); tacuda and TA-Lib backends agree to the
  cent.
- Monte-Carlo robustness (optimizer M4): `--mc N` on `backtest` and
  `portfolio` randomizes the closed-trade sequence (`--mc-mode permute` for
  order permutations, `resample` for bootstrap with replacement,
  `--mc-seed` deterministic) and reports P5/P25/P50/P75/P95/worst of
  netProfit and maxDD, P(net loss) and P(maxDD > original), in the console
  and as a `monteCarlo` JSON section. 10 000 permutations of a 10 458-trade
  backtest add ~0.4 s. Optimizer test suite is now 115 tests. Decision
  recorded: `ct_*_param_sweep_d` kernels are dropped — the GPU share of an
  optimization loop is < 20% (CPU simulation dominates), measured
  tacuda-vs-TA-Lib end-to-end grid times match within 1%.

0.3.0 — 2026-07-05
------------------
- New: batched execution extended to every indicator family with a device (`ct_*_d`) variant —
  45 additional families, 90 new exported symbols (`ct_*_batch` + `ct_*_batch_d`), 395 native
  entrypoints total. Newly batched: WMA, DEMA, TEMA, TRIMA, KAMA, T3, MOMENTUM, ROC/ROCP/ROCR/
  ROCR100, CMO, STDDEV, VAR, SUM, MAX, MIN, LINEARREG (+slope/intercept/angle), TSF, NATR,
  TRANGE, ADX, ADXR, DX, +DI, -DI, WILLR, CCI, ULTOSC, TYPPRICE, WCLPRICE, MEDPRICE, AVGPRICE,
  AROON, AROONOSC, STOCHF, MACDEXT, ACCBANDS, OBV, MFI, AD, ADOSC.
  Same matrix layout (`numSeries x seriesLen`, series-major) and stream scheduling as 0.2.0.
- Vector-math functions are intentionally not batched: element-wise kernels are shape-agnostic,
  so calling e.g. `ct_sin` on the flattened matrix is already the optimal batch.
- New: `scripts/generate_batch_api.py` — single family table generating the C declarations,
  CUDA dispatchers, Node.js wrappers and C# `TacudaBatch` helpers; Python wrappers are built
  at import time from `_generated.py` specs.
- Bindings: Python `*_batch` (2D NumPy in/out, keyword or positional scalars), Node.js `*Batch`
  (arrays of series or flat matrices; OHLCV-column families also accept `Ohlcv[]`; device
  variants under `tacuda.device.*Batch`), .NET `TacudaBatch.*` (jagged arrays, tuples for
  multi-output families).
- Tests: C++ `BatchAPIGenerated` sweep (every new family vs per-series `ct_*_d` loop, device
  vs host batch), pipeline additions in Python/Node.js/C#.
- Fixed: `bindings/generate_bindings.py` wrote the Python spec to a stale duplicate
  (`bindings/python/_generated.py`); it now targets `bindings/python/tacuda/_generated.py`.

0.2.0 — 2026-07-05
------------------
- New: batched (multi-symbol) execution API — `ct_{sma,ema,rsi,atr,macd,bbands,stochastic}_batch`
  host variants and `*_batch_d` device variants. Series-major matrices
  (`numSeries x seriesLen`), per-call `maxConcurrency` controlling an internal pool of
  non-blocking CUDA streams (auto = min(numSeries, 4)), event-based fan-out/fan-in,
  optional `cudaHostRegister` pinning for matrices ≥ 1 MB, user-stream composition.
  See `docs/BATCH_API.md`.
- New: structured OHLCV inputs across all bindings (`docs/OHLCV_MIGRATION_GUIDE.md`):
  - Python: `tacuda.OHLCV` accepted by every H/L/C(/V)-family function (`tacuda.atr(ohlcv, 14)`).
  - Node.js: `tacuda.Ohlcv` class + plain `{open, high, low, close, volume}` objects accepted
    by ~40 indicator wrappers and all candlestick patterns.
  - .NET: extension methods on `OhlcvSeries` (`candles.Atr(14)`, `candles.Stochastic(14, 3)`, ...)
    plus `TacudaBatch` portfolio helpers; `NativeMethods`/`ctDataType_t` are now `public`.
- New: `ct_version()` + `CT_VERSION_*` macros; exposed as `tacuda.version()` (Python/Node.js)
  and `NativeMethods.ct_version()` (.NET).
- Batch wrappers in bindings: `*_batch` (Python, 2D NumPy in/out), `*Batch` (Node.js, arrays of
  series or flat matrices, incl. `tacuda.device.*Batch`), `TacudaBatch.*` (.NET, jagged arrays
  or `OhlcvSeries[]`).
- Tests: C++ `test_batch_api.cpp` (batch vs single-series, device batch, invalid args, version),
  Python/Node.js/C# pipelines extended with OHLCV-container and batch-parity checks.
- Benchmarks: `benchmarks/benchmark_batch_api.py` — loop vs batch at several concurrency levels
  plus stress phase (result stability, buffer-pool growth, invalid-input rejection).

0.1.2 — 2026-07-05
------------------
- SAREXT: exact TA-Lib algorithm port (start direction via +DM/-DM, signed `startValue`,
  offset-on-reverse, SAR clamping); compare tolerance tightened from 450 to 1e-3.
- SAR/SAREXT host API: hybrid CPU path (double precision, no PCIe transfers) — 1M bars now
  computes in single-digit ms instead of ~400 ms; device API still runs on GPU.
- New: `ct_macdext`/`ct_macdext_d` with independent MA types for fast/slow/signal lines
  (all `ctMaType_t` supported, TA-Lib period-swap semantics).
- New: 19 element-wise vector math functions (host + device API): ACOS, ADD, ASIN, ATAN, CEIL,
  COS, COSH, DIV, EXP, FLOOR, LN, LOG10, MULT, SIN, SINH, SQRT, SUB, TAN, TANH.
  Device API is 43–1167x faster than TA-Lib on 10M elements (RTX 3090 Ti).
  TA-Lib function coverage is now 161/161.
- Bindings synchronized to the full native surface (291 functions each):
  - Node.js: added vector math, `macdext`, `_ex` precision variants, `tacuda.device.*`
    zero-copy namespace, `tacuda.bufferPool` controls, legacy `ct_*_device` wrappers.
  - Python/C#: exposed `ct_device_alloc_ex` (generator now parses `void*` returns).
  - C#: platform-neutral `DllImport("tacuda")` so one nupkg works on win-x64 and linux-x64.
- Python bindings: fixed stale `tacuda/_generated.py` (package copy now synced with generator
  output) and added `ctDataType_t` scalar mapping.
- Packages rebuilt: PyPI wheel/sdist, npm tgz, NuGet nupkg (win-x64 DLL + linux-x64 .so),
  all versioned 0.1.2.
- EMA seeding and warmup aligned with TA-Lib across EMA-dependent indicators.
- ATR/NATR: fixed initial average (TR[0..p-1]) and indexing (first at p-1), NATR normalizes per-step.
- BBANDS: switched to double prefix sums; window-aligned outputs; tail `NaN`s.
- Beta: regression slope computed over price window [i-p+1..i]; warmup `NaN`s.
- KAMA: ER via prefix sums of |Δ| in double; outputs aligned to window start; tail `NaN`s.
- RSI: deterministic accumulation variant for large periods to satisfy strict tolerance.
- MACD/APO/PPO: warmup seeded to match TA-Lib (SMA seed for signal) while retaining GPU kernels for bulk work.

0.1.1 — 2026-05-03
------------------
- Unified packaging/release template for NuGet, PyPI and npm; consistent
  0.1.1 metadata across package manifests.
- Auto-generation of `ctDataType_t` in the C# bindings via
  `bindings/generate_bindings.py`; typed entrypoint metadata no longer
  depends on manual edits in `Tacuda.Generated.cs`.
- Packaging READMEs reference a single canonical changelog.

0.1.0 — 2026-03-05
------------------
- Initial public package scaffolding for C# (NuGet), Python (PyPI) and
  Node.js (npm).
- Generated bindings for the host/device API surface.
- Core indicator wrappers and packaging docs per language binding.
