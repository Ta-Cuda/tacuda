# Performance Guide

Measured baselines: RTX 3090 Ti, CUDA 12.8, float32, host API timings
include PCIe transfers (headline tables in the [project README](../README.md#-performance-benchmarks);
reproduce with the scripts in `benchmarks/`).

## What is fast where

| Workload | Recommended path | Typical gain vs TA-Lib (CPU) |
|----------|------------------|------------------------------|
| Single indicator, single series | host API | 0.4–6.2x depending on the indicator (see below) |
| Multi-indicator pipeline, one upload | device API (`ct_*_d`) | 10–15x+ |
| Element-wise math on big arrays | device vector math | 43–1167x on 10M elements |
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
