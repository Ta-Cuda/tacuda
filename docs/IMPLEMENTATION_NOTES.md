# Implementation Notes

Release **0.9.1** targets **237 functions from native/Python TA-Lib 0.8.3**.
The [compatibility guide](TA_LIB_COMPATIBILITY.md) defines new output types,
MA modes, numerical migration and scratch/performance limits. The expanded
oracle passes 1928/1928 cells with no exclusions; exact integer/candle checks
and per-formula real tolerances remain distinct from bitwise identity.

Older release notes and implementation details below retain their historical
context. The former NATR zero-close oracle exclusion is removed in 0.9.1.

## EMA seeding and warmup (TA-Lib compatibility)

- All EMA-based indicators follow TA-Lib's seeding and warmup rules:
  - Seed: the first valid EMA is produced after `period-1` samples using a seed
    compatible with TA-Lib. Our GPU path aligns MACD/APO/PPO via a warmup helper
    (SMA seed for signal; EMA buffers aligned to emit NaN for indices
    `< period-1`).
  - Warmup: results are `NaN` for indices `[0 .. period-2]` for standalone EMA.
    Indicators combining multiple EMAs (e.g., MACD) propagate the maximum warmup
    among constituents.

## Indexing conventions (all TA-Lib aligned since 0.5.0)

- Windowed functions (SMA/SUM/MIN/MAX/MINMAX/BBANDS/VAR/StdDev/Correl):
  the window `[end-period+1, end]` is reported at `end`, with a leading
  warm-up of `period-1` NaNs. (Before 0.5.0, SUM and MINMAX were
  start-of-window aligned — that deviation is gone.)
- `TRANGE` has a lookback of 1: bar 0 is NaN (no previous close).
- Wilder lookbacks depend on the family: RSI, ATR/NATR and DI/DX first
  output at `period` for ordinary periods; ADX at `2*period-1`, ADXR at
  `3*period-2`. Directional movement (PLUS_DM/MINUS_DM) uses `period-1`.
  Period-one special cases follow their individual TA-Lib contracts.
  Subsequent values use Wilder's smoothing.
- MACD/APO/PPO: EMAs computed with warmup NaNs; the published MACD outputs
  are aligned to `slowPeriod-1 + signalPeriod-1` like TA-Lib.
  (`ct_macd_line` is a tacuda extension and exposes the raw line from
  `slowPeriod-1`.)
- STOCH/STOCHF/STOCHRSI publish %K and %D with a **common** first valid
  index, exactly like TA-Lib's shared `outBegIdx` (the K line loses the D
  warm-up as well).
- `ADOSC` lookback is `slowPeriod-1`; reversed periods are normalized
  (fastest = min, slowest = max) instead of rejected. `ULTOSC` sorts its
  three periods internally.

## Value conventions on degenerate inputs (TA-Lib aligned)

- Flat `RSI` and `IMI` return neutral **50**, following TA-Lib 0.8.3.
  Releases through 0.9.0 followed the older zero/NaN definitions. Middle
  NaNs poison the RSI recurrence instead of counting as zero change.
- `ROC`/`ROCP`/`ROCR`/`ROCR100` emit `0` when the denominator is zero.
- Stochastic %K emits `0` when the high-low range is zero.
- `CCI` emits `0` unless both the numerator and the mean deviation are
  nonzero; the kernel accumulates in double so a constant window produces
  an exact zero deviation like TA-Lib.
- `CORREL` emits `0` for a degenerate (zero-variance) window.
- `KAMA`'s efficiency ratio is forced to 1 when the volatility sum is zero
  or does not exceed the signed direction, exactly like TA-Lib.

## Non-finite inputs in window indicators

A NaN or Inf affects exactly the windows containing it; the output recovers
once the value has left the window. This holds for SMA, WMA and BBANDS on every
internal algorithm and in streaming, so choosing a period never changes what a
gap means.

This differs from the corresponding TA-Lib running-total paths, which can
retain a middle NaN after it leaves the window. We chose window semantics
because the short-period kernels and streaming already behaved that way,
and because device pipelines routinely feed
one indicator's warm-up NaNs into another.

Prefix-sum paths stay O(N) on clean data at the cost of one branch per element.
A window whose prefix difference is not finite is recomputed directly from the
input, which is O(period) for those windows only. A series with leading NaNs
pays that for every window after them; the alternative was a second scan on all
data, including the clean majority.

Legacy prefix-based `SUM`, `STDDEV`, `VAR` and `BETA` paths do not
recover after a non-finite input. CORREL preserves its zero-valued suffix
after the first non-finite input. Eligible fused `SUM`/`STDDEV` instead use
direct windows and can recover once the bad value leaves the window. Their
gap behavior can therefore change with the period or `TACUDA_BATCH_FUSED`.
A uniform window policy across these paths remains open work.

Recursive indicators (EMA, RSI, ATR, MACD) are not window indicators: a
middle NaN poisons their recurrence. Leading-NaN alignment follows the
individual API contract; the new backend skips the NumPy-wrapper prefix.

## Tie-breaking in rolling extrema

- `MAXINDEX`/`MININDEX`/`MINMAXINDEX` choose the newest equal extremum,
  including when a stale window is recomputed, following TA-Lib 0.8.3.
  Their implementation still walks each row sequentially; batch calls
  parallelize across rows. The older reference used a path-dependent tie
  rule, which remains relevant only to historical measurements.
- `AROON`/`AROONOSC` use TA-Lib's `>=`/`<=` sliding updates, which our
  per-window kernels reproduce statelessly (most recent tie wins).

## Precision and performance

- Storage precision is float32 by default (`_ex` variants offer
  float16/float64). Comparisons against TA-Lib's double math use a mixed
  absolute/relative criterion: `|a-b| / max(1, |a|, |b|) <= tol`.
- Precision paths preserve each indicator's seeds and lookbacks. Ordinary
  MACD retains its optimized implementation; MACDEXT and MAVP use the
  reference-backed CUDA formulas for the expanded MA modes and exact seed/NaN
  rules. These paths compute in double with float32 public real outputs and
  can regress in throughput or scratch use on long rows. Precision tests use
  random walks because a linear ramp can hide a misplaced EMA seed.
- Windowed statistics use double-precision prefix sums to minimize
  accumulation error; CCI accumulates its window in double; MAMA and the
  MAVP per-period EMA/KAMA state machines run fully in double.
- Ratio-of-near-zero outputs (`TRIX`, `PPO`, `BOP`, `ADOSC`) amplify
  float32 rounding without bound when their denominators cross zero; the
  corpus uses relaxed scaled tolerances there (5e-2..2e-3, see
  `benchmarks/talib_compare_helpers.py`).

## Corrections in 0.9.1

The older NATR zero-close oracle bug is fixed in TA-Lib 0.8.3, so the
corresponding corpus dataset is no longer excluded. Selectable MA variants
are now available for BBANDS/STOCH/STOCHF/STOCHRSI. GPU Hilbert functions use
build-host libm coefficients and separate multiply/add operations; this
corrects the former flat/candle phase deviations on the accepted platforms.
See [the versioned compatibility contract](TA_LIB_COMPATIBILITY.md).

### Historical corrections

Closed in 0.7.1:
- Buffer/Context lifetime contract, all buffer types: `DeviceBuffer` AND
  `DeviceBufferBase` hold the Context's stream control block, so both may
  outlive the Context. A buffer created by Context A is rejected by
  Context B (`std::invalid_argument`): B's kernels would run on stream B
  while the buffer release is ordered on stream A, which could recycle
  pool memory early. Buffers wrapping raw user-managed streams are the
  caller's responsibility and pass through. Covered by
  `tests/cpp/test_context_lifetime.cpp`.
- Device C API `ct_get_last_error` contract: every entry point clears the
  message on entry and every non-success path sets one (manual, `_ex_d`
  and the 114 generated `ct_*_d`).
- IMI/ADOSC parameter bounds follow TA-Lib BAD_PARAM: periods in
  [2, 100000].

Closed in 0.7.0:
- Candlestick borderline bars — all 61 pattern predicates now run in
  DOUBLE like TA-Lib (double candle-settings averages, bodies and
  shadows), so borderline candles resolve identically; the corpus demands
  EXACT signal equality with no head mask and no flip budget.
- ADOSC period roles (ADOSC(10,3) = EMA(10) - EMA(3), lookback from the
  slowest period only), IMI end-of-window alignment with TA-Lib's
  unguarded division (flat window → NaN in that older reference; 0.9.1 uses neutral 50), HT_TRENDMODE zero warm-up in
  every path.

Closed in 0.6.0 (previously tracked here):
- APO/PPO `matype` — exposed via appended `ct_apo_ma`/`ct_ppo_ma`
  (older TA-Lib default SMA; 0.8.3 defaults to EMA).
- Candlestick heuristics — the dead fixed-percentage helpers are removed.
- Short inputs — indicators now return success with an all-NaN line when
  `size` is below the lookback, like TA-Lib's empty output range.
- Hilbert family — the dominant-cycle phase is stateful across bars
  exactly like ta_HT_*.c; all six HT functions are in the corpus.

(0.5.0 removed the previously listed MAMA deviation: the kernel is now a
literal double-precision TA-Lib port — plain `atan` phase, double WMA(4)
smoothing, zero seeds — and passes the strict corpus on every dataset.)
