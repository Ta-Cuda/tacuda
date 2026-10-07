# Implementation Notes

Semantics that TaCuda mirrors from TA-Lib, the precision decisions behind
them, and the **known deviations** that remain. The rules below are enforced
by the strict differential corpus (`benchmarks/differential_corpus.py`):
identical NaN/Inf masks and a mixed absolute/relative tolerance per function
across random walks, a trend, a flat series, zero-crossing data and a short
input. Compatibility here is scoped to native TA-Lib 0.7.1 (Python wrapper
0.7.0), pinned in `tests/requirements-reference.txt`. A run against TA-Lib
0.8.1 has 40 missing corpus specifications and differences on IMI/flat and
WILLR/zero-crossing inputs. These are migration gaps, not tolerance exceptions;
see [OPERATIONS.md](OPERATIONS.md) for the dated evidence.

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

- `RSI` of a fully flat window is `0` (TA-Lib computes `100*g/(g+l)` and
  guards the zero denominator with 0). NaN inputs poison the Wilder
  recursion instead of counting as zero change.
- `ROC`/`ROCP`/`ROCR`/`ROCR100` emit `0` when the denominator is zero.
- Stochastic %K emits `0` when the high-low range is zero.
- `CCI` emits `0` unless both the numerator and the mean deviation are
  nonzero; the kernel accumulates in double so a constant window produces
  an exact zero deviation like TA-Lib.
- `CORREL` emits `0` for a degenerate (zero-variance) window.
- `KAMA`'s efficiency ratio is forced to 1 when the volatility sum is zero
  or does not exceed the signed direction, exactly like TA-Lib.

## Tie-breaking in rolling extrema

- `MAXINDEX`/`MININDEX`/`MINMAXINDEX` reproduce TA-Lib's *stateful* tie
  handling exactly: a stale extremum triggers a left-to-right recompute
  where the oldest tie wins, while an incoming equal bar replaces the
  tracked extremum (newest wins). This is path-dependent, so these kernels
  walk each series sequentially (one thread per series; batch calls
  parallelize across series).
- `AROON`/`AROONOSC` use TA-Lib's `>=`/`<=` sliding updates, which our
  per-window kernels reproduce statelessly (most recent tie wins).

## Precision and performance

- Storage precision is float32 by default (`_ex` variants offer
  float16/float64). Comparisons against TA-Lib's double math use a mixed
  absolute/relative criterion: `|a-b| / max(1, |a|, |b|) <= tol`.
- Windowed statistics use double-precision prefix sums to minimize
  accumulation error; CCI accumulates its window in double; MAMA and the
  MAVP per-period EMA/KAMA state machines run fully in double.
- Ratio-of-near-zero outputs (`TRIX`, `PPO`, `BOP`, `ADOSC`) amplify
  float32 rounding without bound when their denominators cross zero; the
  corpus uses relaxed scaled tolerances there (5e-2..2e-3, see
  `benchmarks/talib_compare_helpers.py`).

## Known deviations from TA-Lib (tracked)

- **NATR on zero closes**: TA-Lib's `ta_NATR.c` contains a genuine bug —
  when a later close is exactly zero it writes the guard value `0.0` into
  `outReal[0]` instead of `outReal[outIdx]`, corrupting the *first* output
  bar. TaCuda emits 0 at the zero-close bar itself (the intended
  behavior); the differential corpus skips that one dataset for NATR
  instead of reproducing the out-of-place write.
- **`matype` beyond APO/PPO/MACDEXT**: BBANDS/STOCH/STOCHF/STOCHRSI expose
  TA-Lib's default MA types only (SMA smoothing); the selectable-matype
  variants of those functions are a planned API extension.

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
  unguarded division (flat window → NaN), HT_TRENDMODE zero warm-up in
  every path.

Closed in 0.6.0 (previously tracked here):
- APO/PPO `matype` — exposed via appended `ct_apo_ma`/`ct_ppo_ma`
  (TA-Lib default SMA verified by the corpus).
- Candlestick heuristics — the dead fixed-percentage helpers are removed.
- Short inputs — indicators now return success with an all-NaN line when
  `size` is below the lookback, like TA-Lib's empty output range.
- Hilbert family — the dominant-cycle phase is stateful across bars
  exactly like ta_HT_*.c; all six HT functions are in the corpus.

(0.5.0 removed the previously listed MAMA deviation: the kernel is now a
literal double-precision TA-Lib port — plain `atan` phase, double WMA(4)
smoothing, zero seeds — and passes the strict corpus on every dataset.)
