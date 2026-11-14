# Performance Guide

Measured baselines: RTX 3090 Ti, CUDA 12.8, float32, host API timings
include PCIe transfers (headline tables in the [project README](../README.md#-performance-benchmarks);
reproduce with the scripts in `benchmarks/`).

Historical timings below were recorded before the 0.8.0 handle-registry fix.
Use dated validation results to assess current streaming latency. Vector-math
speedups must name the CPU baseline; `benchmark_vector_math.py --json report.json`
records both baselines, versions, sizes and timing policy.

## What is fast where

| Workload | Recommended path | Typical gain vs TA-Lib (CPU) |
|----------|------------------|------------------------------|
| Single indicator, single series | host API | 0.4–6.2x depending on the indicator (see below) |
| Multi-indicator pipeline, one upload | device API (`ct_*_d`) | 10–15x+ |
| Element-wise math on big arrays | device vector math | separate NumPy/TA-Lib ratios in the README table |
| Portfolio (many symbols, same indicator) | batch API | linear in symbols vs loop, stream overlap on top |
| SAR/SAREXT single long series | automatic hybrid CPU path | parity with TA-Lib |
| Live tick updates (one closed bar) | streaming API | p50 0.04 µs vs 142 µs–1.2 ms for a recompute |

## Host API, per indicator

Remeasured 2026-08-02 (RTX 3090 Ti + i3-12100F, 500K bars, median of 7;
reproduce with `python benchmarks/bench_host_vs_talib.py`). The host API is
one call, one series, one PCIe round trip — the least favourable shape for a
GPU library, so these are floor numbers, not headlines.

Fastest through the host API: TRIX 6.22x, CMO 5.53x, RSI 4.42x, MACD 3.78x,
ATR 3.29x, DX 3.14x, ADX 3.05x, ±DI 2.4x, OBV 2.21x, STDDEV 1.93x,
BBANDS 1.64x, KAMA 1.41x, EMA 1.33x.

Slower, and worth knowing why:

| Indicator | 500K bars | 2M bars | Why |
| --------- | --------- | ------- | --- |
| TYPPRICE | 0.42x | 0.54x | arithmetic cheaper than its own transfer |
| SMA(14) | 0.59x | 0.72x | same |
| WMA(10) / SUM(14) | 0.76x | 0.85x | same |
| ROCR100(10) | 0.82x | 0.90x | same |
| VAR(14) | 0.95x | 0.90x | same |
| SAR | 0.93x | 0.97x | sequential; already on the hybrid CPU path |
| HT_TRENDLINE / HT_DCPERIOD | 0.80x | 0.26x | recurrent IIR on one GPU thread |
| MAXINDEX(30) | 0.05x | 0.04x | TA-Lib tie handling forces a state machine |

The first group is transfer-bound: the ratios improve with size as the launch
cost amortizes, but never cross 1.0x, because the transfer grows with the
data. Inside a device pipeline that already uploaded the series they are
effectively free. The second group is sequential by design — parallelism
comes from batching many series, not from splitting one.

## Rules of thumb

1. **Amortize transfers.** One H2D upload + N kernels + one D2H download
   beats N host-API calls. The host API pays ~2 PCIe crossings per call.
2. **Batch portfolios.** `ct_*_batch` uploads the whole matrix once and
   fans kernels out across streams. For ≤ 16K-bar series the stream overlap
   matters; for ≥ 256K-bar series a single kernel already saturates the GPU
   and `maxConcurrency` ≈ 2–4 is enough.
3. **Reuse buffers.** The internal `DeviceBufferPool` caches allocations;
   steady-state workloads should show a flat
   `ct_device_buffer_pool_allocation_count()`. Cap the cache with
   `ct_device_buffer_pool_set_max_cache_bytes` on shared GPUs.
4. **Prefer float32.** `_ex` float64 halves effective bandwidth; float16
   doubles it but check indicator tolerance first.
5. **Pin big host buffers.** Batch calls do this automatically (≥ 1 MB).
   For custom pipelines, allocate pinned memory yourself and pass device
   pointers.
6. **Use streams for concurrency.** Independent indicator chains on
   separate `cudaStream_t` values overlap; everything on the default stream
   serializes.

## Measuring

Reproducible datasets: `benchmarks/datasets.py` (frozen seed). Key
harnesses:

```bash
python benchmarks/benchmark_pipeline.py        # host API vs TA-Lib
python benchmarks/benchmark_device_api.py      # device pipelines
python benchmarks/benchmark_batch_api.py       # batch concurrency sweep + stress
python benchmarks/compare_with_talib.py        # correctness within tolerance
```

Report GPU model, driver, CUDA version alongside numbers; host-API results
depend heavily on PCIe generation.

## Known cost cliffs

- First call pays CUDA context + module load (~100 ms); warm up before
  measuring.
- `maxConcurrency=1` forces sequential streams — usually only useful for
  debugging ordering issues.
- Very small inputs (< 4K bars) are launch-latency dominated; CPU TA-Lib
  wins below roughly 16K bars for single calls. Batch small series instead.
- OBV on huge series accumulates in float32; relative error stays ~4e-8 but
  absolute values drift — use `_ex` float64 if you need exact sums.

## Fused batch validation

Measured on 2026-11-14 on the Linux GPU validation host: RTX 3090 Ti, driver 590.48.01,
CUDA 12.8, Ubuntu 24.04.4, Release build for SM86. The baseline is the earlier per-series GPU
dispatcher; the new implementation was measured during 0.9.0 preparation.
Both run on identical GPU-resident float32 inputs. Existing GPU services
were left running; GPU utilization was 0% before timing.

The reporter's pass has 13 calls: MFI, ATR, RSI, ADOSC, STDDEV, BBANDS, SUM,
OBV, AD, STOCHASTIC, SMA, MIN and MAX. Times are median synchronized wall
time for the entire pass, after warm-up, excluding transfers. Three timed
passes for eager execution; ten for graph replay.

| Series × bars | Stage 0, ms | Fused auto, ms | Gain |
|---------------|-------------|----------------|------|
| 4000 × 234 | 431.09 | 2.615 | 165× |
| 20000 × 234 | 2151.44 | 13.60 | 158× |
| 100000 × 64 | 10613.90 | 23.28 | 456× |
| 4 × 234 | 0.632 | 0.368 | 1.7× |
| 8 × 1000000 | 17.79 | 16.24 | 1.1× |

At 4000 × 234, the new pass takes 2.54 ms on a non-blocking caller stream,
or 2.53 ms replaying a graph containing exactly 13 kernel nodes. MFI alone
takes 0.289 ms rather than 49.62 ms. At small series counts the dispatcher
keeps long recurrences on their original parallel path; forcing one thread
per series there is intentionally optional. MFI uses a double flow ring
rather than recomputing its price/volume window for every output.

Correctness evidence:

- `batch_fused`: all 17 first-wave families, TA-Lib comparisons, one kernel
  node per supported call, poisoned-output graph replay, host/default/user
  stream agreement, row permutation, batch size, prefixes, NaN isolation,
  dispatch boundaries, unsupported parameters and error classification.
- Reporter's original 4000 × 234 inputs were checked against float32-rounded
  TA-Lib. STOCH retains the shipped float arithmetic and
  its 663242 differing elements. The additional EMA/WMA/NATR/MACD regression
  fixtures are also bit-equal at their normal periods. This is evidence for
  these fixtures, not a universal bit-equality guarantee for every input.
- MFI checks include positive-volume range boundaries and signed-volume
  controls. Re-summing a rounded zero window can remove tiny TA-Lib residuals.
- Native CTest 327/327 (library targets; optimizer disabled for this run),
  plus 180 graph-capture cases over all 166 batch families
  in both forced-on and forced-off modes; full differential corpus
  1319/1319 cells across 165 specs (one documented deviation skipped).
- Single-series MFI, AD, ADOSC and STDDEV: 12 eager repetitions plus 12 graph
  replays at lengths 1048577 and 8000001, all finite bits identical. The
  baseline failed the million-element MFI/ADOSC/STDDEV repetition test.

Oracle: TA-Lib Python 0.7.0, linked C library 0.6.4. Input float32 values
are widened to float64 for TA-Lib and reference outputs rounded to float32.
EMA period 1, NATR/ATR period 1 and MACD signal period 1 have explicit
boundary regressions. The installed TA-Lib C MACD signal-1 path returns
uninitialized values, so that case uses known constant-line expected values
instead of that oracle.

Reproduce in a GPU build environment with NumPy and TA-Lib installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j6
ctest --test-dir build --output-on-failure
python tests/python/batch_fused.py --lib build/libtacuda.so --cudart-dir "$CUDA_HOME/lib64"
python tests/python/scan_determinism.py --lib build/libtacuda.so --cudart-dir "$CUDA_HOME/lib64" --length 8000001
python benchmarks/benchmark_batch_fused.py --lib build/libtacuda.so --ref-lib /path/to/stage0/libtacuda.so --cudart-dir "$CUDA_HOME/lib64" --json fused-benchmark.json
```

Dispatch and migration/rollback controls are in
[BATCH_API.md](BATCH_API.md#fused-execution). In particular,
`auto` can change the last bits when a shape crosses into the original
recurrent path; force `1` to keep the fused row contract at every batch size.
Fixed-tree scans fix reproducibility, but do not eliminate the cancellation
limit of long-prefix differences on the original STDDEV path.
