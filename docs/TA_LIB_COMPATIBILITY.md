# TA-Lib 0.8.3 compatibility in TaCuda 0.9.1

TaCuda 0.9.1 targets **237 functions from TA-Lib 0.8.3**. The native and
Python oracle versions are pinned in `tests/requirements-reference.txt` and
`tests/reference/talib-0.8.3.json`, against upstream commit
[`62021c1d14ef5499505af5547384cab509adf3fc`](https://github.com/TA-Lib/ta-lib/tree/62021c1d14ef5499505af5547384cab509adf3fc).
An older oracle fails the version gate; it cannot silently shrink coverage.
This includes the 201 functions in 0.8.1 and its 36 further additions in 0.8.3.
Unreleased functions on upstream `main` are outside this target.

## APIs and types

New indicators have host, device, host-batch and device-batch entry points.
The complete C surface has **1129 exports**, including 52 streaming exports.
All 811 preceding public symbols and their signatures are retained.
Use the native library and bindings from the same release.

Inputs and real outputs use float32. New integer outputs use signed int32:
FRACTAL flags, SUPERTREND direction, SQZMOM levels, and ZIGZAG direction/pivot
index. Python returns `np.int32`, Node returns `Int32Array`, and C# uses
`int[]`/`int[][]`. Historical float candle/index APIs keep their original
types. HA and ICHIMOKU each have four separate output arrays.

```python
import numpy as np
import tacuda

rma = tacuda.rma(close, 14)
hma = tacuda.hma(close, 14)
zlema = tacuda.zlema(close, 14)
line, direction = tacuda.supertrend_batch(high_rows, low_rows, close_rows, 10, 3.0)
assert line.dtype == np.float32 and direction.dtype == np.int32
```

`MAType` values 0-8 retain SMA through T3; the extension is:

| Value | Type | Meaning |
| ---: | --- | --- |
| 9 | HMA | Hull moving average |
| 10 | DISABLED | Identity, no smoothing |
| 11 | DEFAULT | Enclosing function's default; ordinary MACD uses EMA |
| 12 | ZLEMA | Zero-lag EMA |
| 13 | RMA | Wilder smoothing |
| 14 | VIDYA | Variable-index dynamic average |
| 15 | ALMA | Arnaud Legoux moving average |

MA/MAVP/MACDEXT and selectable-MA composites support these modes.
`pvo_ma`, `bbands_ma`, `stochastic_ma`, `stochf_ma` and `stochrsi_ma` add
selectable smoothing; APO/PPO retain their existing `_ma` names. Standalone
ALMA/VIDYA options are not encoded in MAType; fixed upstream defaults apply
there. Scalar arguments are explicit in the generated TaCuda APIs.

Names with different historical definitions remain separate:

| TA-Lib definition | Compatible TaCuda API | Existing API preserved |
| --- | --- | --- |
| ASI with limit-move scaling | `asi_wilder(..., limitMove)` | Club `asi` |
| Sample-adjusted Fisher kurtosis | `kurtosis_unbiased(..., period)` | Population `kurtosis` |
| Selectable-MA volume oscillator | `pvo_ma(..., fastPeriod, slowPeriod, maType)` | EMA `pvo` |

## Numerical migration from 0.9.0

Flat RSI and IMI now return neutral **50**. WILLR preserves NaNs and clips
to [-100, 0]. Equal extrema select the newest index. NATR uses an exact-zero
close guard, including very small prices. NVI/PVI align leading NaNs while
retaining deterministic double product scans. The release also corrects
STOCH's minimum sufficient length, PVO warmup, CORREL's long-offset accuracy,
and BBANDS/BETA/Hilbert reproducibility and precision.

New reference-backed formulas align leading NaNs like the NumPy oracle;
middle NaNs follow each formula's own propagation/recovery rules. Real warmup
is NaN; new integer warmup is zero. ZIGZAG tracks causal state and its pivot
indices refer to the original row. FRACTAL emits confirmation flags, not
pivot prices. Display shifts do not physically shift output buffers.

These corrections also apply with `TACUDA_BATCH_FUSED=0`. Pin 0.9.0 to
reproduce its historical outputs. Tolerance-based oracle acceptance does not
claim bitwise identity for every function or parameter combination.

## GPU execution and limits

The new backend compiles licensed upstream batch formulas as CUDA device
functions, with double intermediates and independent per-row state. One
kernel processes rows using private caller-stream scratch. It ignores
`maxConcurrency` and is independent of the legacy fused dispatcher switch.
Existing optimized ordinary MACD and NVI/PVI paths remain available.

Scratch reserves 64 doubles per bar plus 512 doubles per active worker.
Workers share a 256 MiB budget; one longer row can exceed it (about 488 MiB
for a million bars). This is a correctness baseline. Long sequential rows,
large selection windows and unusual recursive MA combinations may be slow.
Historical speedups apply to their measured workloads, not to every new
indicator. Graphs using scratch include memory nodes and require memory-pool
support; unsupported devices receive `CT_STATUS_UNSUPPORTED` during capture.
Eager execution remains available. Allocation-free legacy fused graphs keep
their one-node contract. See [batch API](BATCH_API.md).

Full function coverage does not extend the public incremental API beyond its
existing ten families. Automatic stabilization and display-shift query APIs
are separate compatibility surfaces. Ordinary unstable periods are zero.
The curated upstream source and generated formulas retain their BSD notices;
packaged notices are in `THIRD_PARTY_NOTICES.txt`.

## Validation

The expanded corpus checks 237 names with 241 parameter specs over eight
datasets: **1928/1928 cells passed**, with no exclusions and all candle
patterns activated. New/changed API checks passed **1544/1544** at both 512
and 32 bars. Integer outputs are compared exactly; new real values use
`rtol=atol=2e-5`, including NaN/Inf behavior. Legacy specs retain their
per-formula tolerances and exact candle signal comparisons.

Typed host/device/batch, row permutation/count, buffer guards and graph replay
passed for 85 families. Compute Sanitizer memcheck/initcheck reported zero
errors for the 80 reference-backed families on Linux/RTX 3090 Ti/CUDA 12.8.
This sanitizer statement does not cover every legacy CUB path.
The existing streaming corpus passed 839/839 cells.

Run the pinned oracle, `benchmarks/differential_corpus.py`,
`tests/python/talib08_corpus.py`, `tests/python/talib08_batch_contract.py` and
CTest to reproduce. Release-specific platform and package receipts are
recorded in [operations](OPERATIONS.md).
