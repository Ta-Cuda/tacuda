# TaCuda Roadmap

Last reviewed: 2026-08-02 (streaming API M1-M3 on the 0.8.0 line).
This document tracks where the project stands and what ships next.
Order inside a milestone is priority order; items move, versions don't slip silently.

## Where the project stands (v0.7.1)

| Area | State |
|------|-------|
| TA-Lib coverage | 161/161 functions; strict differential corpus 165/165 specs green (exact-name coverage enforcement, exact candlestick parity incl. activation fixtures for the rare patterns) |
| Native ABI | 808 `extern "C"` entrypoints: 556 host (incl. `_ex` precision and `_batch`) + 200 `_d` device + 52 streaming; append-only policy ([CONTRIBUTING.md](CONTRIBUTING.md#abi-stability-policy)) |
| Batch API | 166 indicator families (incl. all candlestick patterns and the STAD club pack), stream-pool scheduling, host + device variants ([BATCH_API.md](BATCH_API.md)) |
| Bindings | Python (ctypes), Node.js (koffi), C#/.NET (P/Invoke) — full 808-symbol surface incl. streaming handles, audited in CI (`bindings/audit_bindings.py`) |
| Streaming | Incremental append for 10 families, single-series and portfolio-wide ([STREAMING_API.md](STREAMING_API.md)); parity swept by `benchmarks/streaming_corpus.py` |
| Memory safety | Stream-ordered buffer pool; buffers may outlive their Context; cross-Context use rejected; lifetime tests in ctest |
| Packages | PyPI platform wheels, npm tgz, NuGet nupkg (win-x64 + linux-x64) built by the `release-packages` workflow with GPU-gated publishing |
| CI | `native-build` (Linux + Windows, all targets), GPU test job (ctest + corpus + binding pipelines) mandatory for version tags |
| Docs | User guide / API / performance / operations tree, implementation notes, changelog, release criteria, adoption checklist |
| Site | https://tacuda.com (maintained separately; EXPERIMENTS pages generated from this repo) |
| Benchmarks | Pipeline 15.4x vs TA-Lib @1M bars, vector math up to 1167x, batch up to 7.5x vs loops (RTX 3090 Ti) |
| Tools | `tacuda-opt` strategy optimizer (M0–M4 shipped) with the 94-system EXPERIMENTS pack |

Known weak spots, remeasured 2026-08-02 on the RTX 3090 Ti + i3-12100F
(`benchmarks/bench_host_vs_talib.py`, 500K bars, median of 7). The scan
rewrites landed: CMO 5.53x, TRIX 6.22x, MACD 3.78x, DX 3.14x, ±DI 2.4x,
OBV 2.21x, KAMA 1.41x — 14 of 24 sampled functions now beat TA-Lib through
the plain host API. What is still slower splits in two, with different
answers:

- **Transfer-bound trivial work** (TYPPRICE 0.42x, SMA 0.59x, WMA/SUM 0.76x,
  ROCR100 0.82x): TA-Lib computes `(h+l+c)/3` over 500K bars in 0.40 ms,
  about what moving 2 MB each way costs before any arithmetic. Measured at
  2M bars too — the ratios improve as the launch cost amortizes but never
  cross 1.0x. Use the device or batch API; for a genuine one-off call, CPU
  TA-Lib is the right tool.
- **Sequential by design** (SAR 0.93x, HT family 0.80x at 500K and 0.26x at
  2M, MAXINDEX 0.05x): one GPU thread is slower than one CPU thread, so the
  Hilbert family degrades as the sequential work grows. MAXINDEX stays worst
  because TA-Lib resolves ties path-dependently and the corpus demands that
  exactly; a parallel argmax would be fast and would return different
  indices. Parallelism here comes from batching many series, not from
  splitting one.

Details in [PERFORMANCE.md](PERFORMANCE.md).

## v0.7.x — public release line

- [x] Docs cleanup: internal session reports and third-party doc copies
      removed; `docs/` holds only current user/developer documentation.
- [x] Legacy `ct_*_device` aliases hidden from the documented surface
      (still exported for ABI compatibility).
- [ ] Publish 0.7.x packages to the public registries (PyPI, npm, NuGet)
      via `release-packages` with `publish=true`; verify install-from-registry
      on a clean machine (gates in [CONTRIBUTING.md](CONTRIBUTING.md#release-criteria)).
- [ ] Tag the release on GitHub with notes from [CHANGELOG.md](CHANGELOG.md);
      attach native binaries with checksums (automated by the workflow).
- [ ] Register a self-hosted GPU runner (`gpu` label) on the public repo —
      tagged releases are intentionally blocked without it.

## Shipped: performance milestone (sequential indicators)

Goal was "no indicator slower than TA-Lib through the host API at ≥500K
bars". Met for the algorithmic cases it targeted; closed 2026-08-02 with the
remaining two groups documented rather than left as an open target.

- [x] Wilder family (ATR/NATR/ADX/ADXR/DX/±DI/CMO/RSI) reworked as
      scan-based smoothing — CMO 0.03x → 5.53x, DX 0.03x → 3.14x,
      ±DI 0.03x → 2.4x.
- [x] KAMA / Variance / OBV as prefix-scan formulations — OBV 0.04x → 2.21x,
      KAMA 0.01x → 1.41x, VAR 0.00x → parity.
- [x] Hilbert family profiled and accepted as sequential-by-design; it
      degrades with size (0.80x at 500K, 0.26x at 2M) because a single GPU
      thread carries the recurrence.
- [x] Host API remeasured (`benchmarks/bench_host_vs_talib.py`) and the
      claims in this file and [PERFORMANCE.md](PERFORMANCE.md) rewritten
      around the result.
- [x] Rejected as unreachable: 100% of functions ≥ TA-Lib through the host
      API. Trivially cheap operations cannot outrun their own PCIe transfer,
      and MAXINDEX cannot be parallelized without changing TA-Lib's
      path-dependent tie handling.

## v0.8.0 — streaming & live trading

Goal: append newly closed bars and get updated indicator values without
recomputing the whole series — the capability TA-Lib does not have.
User guide: [STREAMING_API.md](STREAMING_API.md); rationale and the
measurements behind each decision: [DESIGN_STREAMING_API.md](DESIGN_STREAMING_API.md).

- [x] **M1 — single-series streams.** Handle-based API for SMA, EMA, WMA,
      RSI, ATR, NATR, OBV, AD, MACD and BBANDS. An append costs p50 0.04 µs
      and is flat in history length, against 142 µs (100K bars) / 1 234 µs
      (1M) for a full recompute. Parity against the full-series API is
      asserted per family at four chunk sizes.
- [x] **M2 — batch (portfolio) streams.** One call advances every series;
      the state machines are shared with the single-series path, so a batch
      of N is bit-identical to N separate streams. 256 symbols × 7
      indicators, one bar per tick: **p99 19.5 µs**. `CT_STREAM_EXEC_AUTO`
      switches CPU→GPU at the measured crossover of 2048 series and holds
      p99 under 1 ms through 65 536 symbols.
- [x] **M3 — bindings and corpus.** Python, Node.js and C# wrappers for all
      20 stream types, gated by `bindings/audit_bindings.py`;
      `benchmarks/streaming_corpus.py` sweeps 10 families × 7 datasets
      (699/699 cells) checking chunked appends against both the full-series
      call and the installed TA-Lib.
- [ ] **M4 — release.** Site section, a live-trading article, and the 0.8.0
      release itself.
- [ ] Persistent batch plans — deferred on measurement, not dropped. Batch
      streams already keep device state and I/O buffers resident, so what a
      plan would still buy is one OHLCV upload per tick instead of one per
      indicator. Worth much of the GPU tick cost at tens of thousands of
      symbols; no current target needs it.
- [ ] Remaining families: ADX group, STOCH/STOCHF.

## v0.9.0 — scale & distribution

Deferred until 0.8.0 shows where demand actually is.

- [ ] Multi-GPU dispatch for batch API (`deviceId` in batch options; one
      stream pool per device).
- [ ] float64 end-to-end validation for the `_ex` path (storage and compute),
      documented tolerances vs TA-Lib float64.
- [ ] Conda package + Docker images (runtime and dev) for the Python stack.
- [ ] aarch64/Jetson build in CI (compute 7.2/8.7), publish as extra wheel tag.
- ~~Parameter-sweep batch kernels (`ct_*_param_sweep_d`: one series × N
  parameter sets)~~ — **dropped 2026-07-06** by the strategy-optimizer
  M1/M4 measurements: the GPU share of an optimization loop is well under
  20% (CPU simulations dominate), so parameter-major kernels would not pay
  off.

## Shipped: `tacuda-opt` strategy optimizer

A console C++ strategy optimizer in the spirit of MultiCharts lives in
`tools/optimizer/` (targets `tacuda-opt`, `tacuda_opt_tests`). M0–M4 shipped
2026-07-06: single backtest, exhaustive GPU grid search, genetic search,
walk-forward analysis, multi-symbol portfolio backtest over the batch API,
Monte-Carlo robustness checks, indicator cache, TA-Lib CPU fallback backend,
CSV/JSON reports — user guide in [STRATEGY_OPTIMIZER.md](STRATEGY_OPTIMIZER.md),
plus the 94-system EXPERIMENTS pack ([EXPERIMENTS.md](EXPERIMENTS.md)).
Measured on the RTX 3090 Ti: 10K variants × 1M bars in 21.5 s; the genetic
run found the same best variant with 2.9% of the evaluations in 1.16 s;
6-window walk-forward over 1M bars in 1.5–2.3 s; an 8-symbol × 1M-bar
portfolio plus 1000 Monte-Carlo permutations of 85 600 trades in 1.3 s.

It doubles as the first real downstream consumer of the device and batch
APIs and feeds requirements back into the core (derived-series chaining,
indicator caching, the club-pack scan kernels).

## v1.0.0 — stability contract

Ships only when the streaming surface has soaked for a full minor cycle.

- [ ] ABI freeze at MAJOR=1: current entrypoints immutable, additions only
      (policy already drafted in [CONTRIBUTING.md](CONTRIBUTING.md#abi-stability-policy)).
- [ ] Signed releases end-to-end (sigstore for wheels/tgz/nupkg, SBOM attached)
      per [CONTRIBUTING.md](CONTRIBUTING.md#release-criteria).
- [ ] GPU test matrix in CI: at least one Pascal (CC 6.x) and one Ada/Hopper
      runner executing the full ctest + binding pipelines per release.
- [ ] LTS branch policy: security and correctness fixes backported for the
      lifetime of 1.x.

## Non-goals

Explicitly out of scope, so contributors don't burn time on them:

- ROCm / Metal / SYCL ports — TaCuda is CUDA-first by definition.
- A pandas/Polars DataFrame layer inside the core — belongs in downstream
  wrappers; the bindings stay thin over the C ABI.
- Strategy/backtesting logic inside the core library — the core computes
  indicators; everything above them lives in `tools/optimizer`.
- GUI/charting.

## How this file is maintained

Update when a milestone ships or scope genuinely changes, in the same commit
as the change that makes it true. Checked boxes stay in place until the
milestone section is archived into the changelog.
