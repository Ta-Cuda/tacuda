# TaCuda Roadmap

Last reviewed: 2026-07-23 (v0.7.1, first public release).
This document tracks where the project stands and what ships next.
Order inside a milestone is priority order; items move, versions don't slip silently.

## Where the project stands (v0.7.1)

| Area | State |
|------|-------|
| TA-Lib coverage | 161/161 functions; strict differential corpus 165/165 specs green (exact-name coverage enforcement, exact candlestick parity incl. activation fixtures for the rare patterns) |
| Native ABI | 756 `extern "C"` entrypoints: 556 host (incl. `_ex` precision and `_batch`) + 200 `_d` device; append-only policy ([CONTRIBUTING.md](CONTRIBUTING.md#abi-stability-policy)) |
| Batch API | 166 indicator families (incl. all candlestick patterns and the STAD club pack), stream-pool scheduling, host + device variants ([BATCH_API.md](BATCH_API.md)) |
| Bindings | Python (ctypes), Node.js (koffi), C#/.NET (P/Invoke) — full 756-symbol surface, audited in CI (`bindings/audit_bindings.py`) |
| Memory safety | Stream-ordered buffer pool; buffers may outlive their Context; cross-Context use rejected; lifetime tests in ctest |
| Packages | PyPI platform wheels, npm tgz, NuGet nupkg (win-x64 + linux-x64) built by the `release-packages` workflow with GPU-gated publishing |
| CI | `native-build` (Linux + Windows, all targets), GPU test job (ctest + corpus + binding pipelines) mandatory for version tags |
| Docs | User guide / API / performance / operations tree, implementation notes, changelog, release criteria, adoption checklist |
| Site | https://tacuda.com (maintained separately; EXPERIMENTS pages generated from this repo) |
| Benchmarks | Pipeline 15.4x vs TA-Lib @1M bars, vector math up to 1167x, batch up to 7.5x vs loops (RTX 3090 Ti) |
| Tools | `tacuda-opt` strategy optimizer (M0–M4 shipped) with the 94-system EXPERIMENTS pack |

Known weak spots, measured on the RTX 3090 Ti:
a group of state-dependent indicators runs as single-thread GPU kernels and is
slower than TA-Lib through the host API (SAR ≈0.02x, KAMA ≈0.01x, CMO / DX /
±DI ≈0.03x, OBV ≈0.04x, HT family). They are numerically correct; they are just
not parallelized yet. The device/batch paths amortize most of the gap in real
pipelines (SAR/SAREXT additionally use a fast CPU path through the host API),
but the host-API numbers are honest and we want them green.

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

## v0.8.0 — performance milestone (sequential indicators)

Goal: no indicator slower than TA-Lib through the host API at ≥500K bars.

- [ ] Wilder-family rework (ATR/NATR/ADX/ADXR/DX/±DI/CMO/RSI long-period):
      scan-based smoothing (first-order IIR as an associative scan).
      Target: 0.03x → >2x host API.
- [ ] KAMA / Variance / OBV: prefix-scan formulations (cumulative sums are
      textbook scans; OBV is a signed prefix sum; the club pack already
      ships scan-based recursions to crib from).
- [ ] Hilbert transform family: parallel IIR via block-scan or accept and
      document as sequential-by-design; decide after profiling.
- [ ] Re-run the full 161-function comparison, refresh the site and README
      numbers (target: 100% of functions ≥ TA-Lib at 500K bars).

## v0.9.0 — streaming & scale

Goal: serve live-trading workloads, not only batch research.

- [ ] Incremental update API: append N new bars and recompute only affected
      tails (`ct_*_update` family) instead of full-series recompute.
- [ ] Persistent batch plans: reusable descriptor (series count, lengths,
      params) so per-tick batch calls skip validation/allocation.
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

Ships only when the 0.8/0.9 surface has soaked for a full minor cycle.

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
