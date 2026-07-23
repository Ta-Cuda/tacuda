# Performance Guide

Measured baselines: RTX 3090 Ti, CUDA 12.8, float32, host API timings
include PCIe transfers (headline tables in the [project README](../README.md#-performance-benchmarks);
reproduce with the scripts in `benchmarks/`).

## What is fast where

| Workload | Recommended path | Typical gain vs TA-Lib (CPU) |
|----------|------------------|------------------------------|
| Single indicator, single series | host API | 1.3–2.6x on 1M bars (PCIe-bound) |
| Multi-indicator pipeline, one upload | device API (`ct_*_d`) | 10–15x+ |
| Element-wise math on big arrays | device vector math | 43–1167x on 10M elements |
| Portfolio (many symbols, same indicator) | batch API | linear in symbols vs loop, stream overlap on top |
| SAR/SAREXT single long series | automatic hybrid CPU path | parity with TA-Lib |

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
