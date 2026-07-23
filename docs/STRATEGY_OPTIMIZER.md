# tacuda-opt — console strategy optimizer

Status: shipped with M0–M4 scope (single backtest, GPU grid search,
genetic optimization, walk-forward analysis, portfolio backtest,
Monte-Carlo robustness).
Source: `tools/optimizer/` (CMake targets `tacuda-opt`, `tacuda_opt_tests`).

`tacuda-opt` is a console optimizer of trading-strategy parameters in the
spirit of MultiCharts optimization: indicators are computed on the GPU
through the TaCuda device API, trade simulation and metrics run on CPU
threads. When no CUDA device is present the tool transparently falls back
to a dynamically loaded TA-Lib (`--backend auto`, the default).

## Measured performance (RTX 3090 Ti, Ubuntu 24.04)

| Scenario | Result |
|----------|--------|
| `backtest` ma_cross, 1M bars | 0.28 s end-to-end GPU (0.12 s TA-Lib — one-time CUDA init dominates here) |
| `grid` 10 000 variants (100×100 EMA cross), 1M bars, 8 threads | **22.0 s** end-to-end (M1 target: < 60 s) |
| Indicator work for that grid | 200 unique EMA lines computed once each (cache), 762 MB host cache, single chunk |
| `genetic` on the same 10 000-variant space | found the **same best variant** with **294 evaluations (2.9%)**, 17 generations, **1.16 s** (M2 target: ≥95% of grid best at ≤10% of evaluations) |
| `wfo` 6 windows, 230-variant grid per window, 1M bars | 1380 IS evaluations, **1.53 s**, WFE 0.60 |
| `wfo --optimizer genetic` 6 windows over the 10 000-variant space | 1768 IS evaluations (vs 60 000 exhaustive), **2.33 s**, WFE 0.60 |
| `portfolio` 8 symbols × 1M bars (`ct_*_batch_d`) + `--mc 1000` over 85 600 trades | **1.28 s** end-to-end |
| `portfolio` 16 symbols × 1M bars | **1.74 s** end-to-end |
| `--mc 10000` permutations of a 10 458-trade backtest | **~0.4 s** on top of the backtest |

The tacuda and TA-Lib backends produce identical results on the three
built-in strategies: the same 10 000-variant grid top table, the same WFO
equity/WFE and the same portfolio net profit to the cent (validated on
1M-bar synthetic runs and by the `Backends.*MatchesReference` unit tests).
End-to-end times match within 1% on `ma_cross`-style runs where CPU
simulation dominates; on the indicator-heavy `bbands_breakout` grid (900
unique BBANDS, 2700 lines over 1M bars) the GPU backend is ~1.4x faster
end-to-end (8.2 s vs 11.6 s).

## Build

The optimizer builds together with the library when
`TACUDA_BUILD_OPTIMIZER=ON` (default):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j8 --target tacuda-opt tacuda_opt_tests
```

The binary links `libtacuda` (and CUDA runtime); TA-Lib is **not** a build
dependency — it is looked up at runtime only when needed.

## Commands

```
tacuda-opt backtest   one backtest with fixed parameters
tacuda-opt grid       exhaustive grid search over parameters
tacuda-opt genetic    genetic search over the same parameter space as grid
tacuda-opt wfo        walk-forward optimization (rolling IS/OOS windows)
tacuda-opt portfolio  multi-symbol backtest (shared params, batch API)
tacuda-opt list       built-in strategies and their parameters
tacuda-opt --help     full option reference
```

### Data sources

- `--data PATH` — CSV, comma or semicolon separated. With a header the
  columns are matched by name (`time/date/timestamp`, `open`, `high`,
  `low`, `close`, `volume`); without one the layout
  `[ts,]open,high,low,close[,volume]` is inferred. Timestamps may be unix
  seconds/millis or `YYYY-MM-DD[ HH:MM[:SS]]`. Rows with `high < low` or
  non-numeric prices abort loading with the file:line in the message.
  The `portfolio` command takes one `--data` per symbol.
- `--synthetic N --seed S` — deterministic synthetic OHLCV (geometric
  random walk with a trend cycle), for demos and benchmarks. For
  `portfolio`, `--symbols K` (default 4) generates K series with seeds
  `S..S+K-1`.

### Built-in strategies

| Name | Parameters (defaults) | Logic |
|------|----------------------|-------|
| `ma_cross` | `fast=12`, `slow=48` | long while EMA(fast) is above EMA(slow), short below (stop-and-reverse) |
| `rsi_reversion` | `period=14`, `low=30`, `high=70` | RSI < low → long; RSI > high → flat. Long-only |
| `bbands_breakout` | `period=20`, `width=2.0` | close beyond the upper/lower Bollinger band enters; crossing back over the middle band exits |

Invalid combinations (e.g. `fast >= slow`) are skipped automatically and
show up in reports with the status `invalid parameter combination`.

On top of the three core strategies the **EXPERIMENTS pack** registers 94
classic systems (`exp01_*`…`exp13_*`) ported from the STAD Club volumes
1–13 — see [EXPERIMENTS.md](EXPERIMENTS.md) and the per-system pages on
the site. `tacuda-opt list` prints all 97 with their parameter spaces.

### Execution model

Signals are **target positions** (−1/0/+1) decided at each bar close and
filled at the **next bar open** — look-ahead is excluded by construction.
One position per instrument, size `--qty` (default 1). A position still
open on the last bar is force-closed at the final close so every trade
completes. Money math is `double` end-to-end (float32 only inside
indicators).

Costs per order side: `--commission-bps` (of notional) +
`--commission-fixed` (currency), slippage `--slippage-ticks × --tick-size`
against the fill. On a stop-and-reverse fill the order cost is split
per-unit between the closing and opening trades.

Trading is blocked until every requested indicator line leaves its NaN
warm-up (`maxLookback` across the variant's indicators).

### Metrics and fitness

NetProfit, GrossProfit/Loss, ProfitFactor (capped at 1e9 with no losers),
MaxDrawdown (absolute and % of peak), annualized Sharpe and Sortino
(`--bars-per-year`, default 252, population std over per-bar equity
diffs), WinRate, AvgTrade, trade count, exposure.

`--fitness` accepts an arithmetic expression over those metrics
(`+ - * / ( )`, unary minus, case-insensitive variables):

```
--fitness sharpe
--fitness "netprofit - 2*maxdd"
--fitness "pf * winrate - maxddpct * 100"
```

Variables: `netprofit grossprofit grossloss pf profitfactor maxdd maxddpct
sharpe sortino winrate trades numtrades avgtrade exposure`. Division by
zero yields 0.

### Backends

| `--backend` | Behaviour |
|-------------|-----------|
| `auto` (default) | tacuda when a CUDA device is available, otherwise the TA-Lib fallback |
| `tacuda` | force GPU; fails when no device is present |
| `tacuda-device` | GPU via the single-series device API even in `portfolio` (one `ct_*_d` call + its own uploads per symbol instead of one `ct_*_batch_d` call per request) — the baseline the batch API is benchmarked against. Identical to `tacuda` for single-series commands |
| `talib` | force CPU/TA-Lib; fails when the library cannot be loaded |

The TA-Lib shared library is loaded dynamically: first from the
`TACUDA_OPT_TALIB` / `TALIB_LIBRARY` environment variables (full path),
then by the platform-typical names (`ta-lib.dll`, `libta-lib.so[.0]`,
`libta_lib.so[.0]`). The bundled library inside a Python `ta-lib` wheel
works too, e.g.:

```bash
export TALIB_LIBRARY=$VENV/lib/python3.12/site-packages/ta_lib.libs/libta-lib-8d07fdfe.so.0.6.4
```

### GPU pipeline and the indicator cache

OHLCV columns are uploaded to the GPU once per run; each unique
`(family, column, params)` request runs as a `ct_*_d` kernel and only its
output lines are downloaded. Grid variants share overlapping requests
through a host-side cache — a 100×100 `ma_cross` grid computes 200 EMA
lines instead of 20 000.

Since 0.4.0 the request set covers 39 families (the 22 classic ones plus
the STAD club pack: ASI, SAR_EL, OBTR, WOBV, HPI, MARKET_STRENGTH,
TRENDSCORE, REFDEV, SKEW, KURTOSIS, PIVOT_HIGH/LOW, XAVERAGE and the
NaN-strict ROLLING_MEAN/MAX/MIN/STD), and a request can **chain** on
another request's output line (`IndicatorRequest::chain`, built with
`derived()`): "XAverage of OBTR" is one request whose whole chain is
computed inside the backend — on the GPU paths without any host
round-trip, in `portfolio` directly on the `numSeries × seriesLen`
matrices. Chaining is restricted to the NaN-aware club families over
single-output upstream requests (`validate_chain()`), and the indicator
cache key folds the chain in, so chained lines dedupe across variants
like any other request.

Variants are processed in deterministic memory-bounded chunks: per chunk
the unique requests are computed once, simulations run in parallel
(`--threads`, default all cores), then the cache is cleared. The budget is
`--cache-mb` (default 2048). Reports are byte-identical for any thread
count and chunking (verified by tests).

### Genetic optimization

`genetic` searches the **same discrete space** a `grid` run would
enumerate: each `--grid` axis becomes one gene whose value is an index
into the axis values, `--set` and defaults stay fixed. Results are
therefore directly comparable with the exhaustive grid on the same axes.

The algorithm is a classic generational GA: tournament selection
(`--tournament`, default 4), uniform crossover (`--crossover-prob` 0.9),
elitism (`--elite`, default 8 individuals survive unchanged) and gaussian
mutation that moves a gene by `N(0, --mutation-sigma)` **axis steps**
(default 2) with probability `--mutation-prob` (0.15 per gene). The run
stops after `--generations` (60) or when the best fitness has not improved
for `--stagnation` (12) generations.

Properties enforced by design and covered by tests:

- a genome is never simulated twice — duplicates reuse the cached result,
  so `evaluations` counts unique variants only;
- invalid parameter combinations get `-inf` fitness: they lose every
  tournament and never enter the elite, but do not crash the run;
- elitism makes the best fitness non-decreasing between generations;
- a fixed `--ga-seed` (default 42) makes the whole run deterministic.

The console/CSV/JSON reports are the grid ones over all evaluated
variants, plus a `generations N, unique variants evaluated M` line.

### Walk-forward optimization

`wfo` splits the series into `--windows` K rolling windows, each with an
in-sample (IS) share `--is` (default 0.75). Window lengths satisfy
`L_is + K·L_oos ≈ n`, window w covers
`[w·L_oos, w·L_oos + L_is + L_oos)`, so consecutive out-of-sample (OOS)
segments tile the timeline contiguously (up to `L_is + L_oos` leftover
bars at the tail stay unused and are reported).

Per window the optimizer (`--optimizer grid|genetic`) re-optimizes the
strategy on the IS segment, the best-by-fitness variant is then evaluated
on the OOS segment. Indicators are computed **once over the full series**
and shared between windows through the cache — this is causal (an EMA at
bar t only sees bars ≤ t), so nothing leaks from the future; a dedicated
test modifies data after a window's IS end and asserts the window's chosen
parameters do not change.

Trading inside a segment starts flat; the decision formed on the last IS
bar fills at the first OOS open. A position still open at the segment end
is force-closed at its last close. Windows where no variant is valid
(e.g. the IS segment is shorter than the indicator warm-up) are reported
as skipped and contribute flat equity.

The OOS equity curves are stitched by chaining per-window equity deltas
into one continuous curve; overall OOS metrics are computed over it. WFE
(walk-forward efficiency) is the per-bar OOS net profit divided by the
per-bar IS net profit of the chosen variants
(`(Σ oosNP / Σ oosBars) / (Σ isNP / Σ isBars)`, over optimized windows);
`n/a` when the IS profit is not positive.

### Portfolio backtest

`portfolio` runs one strategy with **fixed shared parameters** (`--set`,
defaults otherwise) across several symbols and aggregates the result:

- **Data.** One `--data CSV` per symbol (symbol names come from the file
  names), or `--synthetic N --symbols K` for a synthetic portfolio. Series
  are brought to a common bar grid first: when every file has usable
  timestamps they are aligned to the **timestamp intersection** (bars
  missing from any symbol are dropped and reported); without timestamps
  the series must already have equal lengths.
- **Batch API.** This is the optimizer's consumer of TaCuda's batch
  entrypoints: per used OHLCV column one `numSeries × seriesLen` matrix is
  uploaded to the GPU once, and each indicator request becomes a single
  `ct_*_batch_d` call that fans per-symbol kernels over TaCuda's internal
  stream pool. `--backend tacuda-device` keeps the same GPU kernels but
  loops the single-series device API per symbol (the benchmark baseline);
  on the CPU path (`--backend talib` or no GPU) each symbol is served by
  its own TA-Lib-backed backend. Every strategy — the core three and all
  94 EXPERIMENTS systems — runs unchanged on any of the three paths.
- **Timing.** The report and JSON split the run into the indicator phase
  (all backend calls: uploads + kernels + downloads) and the
  signals+simulation phase (per-symbol CPU work), so backends are compared
  on the part they actually own. A three-backend comparison over the whole
  EXPERIMENTS pack lives in `scripts/bench_experiments_batch.py`; the
  measured numbers and their analysis are in
  [EXPERIMENTS.md](EXPERIMENTS.md#batch-api-portfolio-path).
- **Accounting.** `--capital` is the total: each symbol trades an equal
  share (`capital / K`) with the same `--qty` and costs, one position per
  symbol, the usual next-bar-open fills. Portfolio equity is the sum of
  per-symbol equities; trades are merged chronologically; metrics
  (NetProfit, PF, MaxDD, Sharpe...) are computed over the aggregated curve
  and merged trade list. Exposure is reported as the mean of per-symbol
  exposures. A symbol whose data is too short for the indicator warm-up is
  reported as `SKIPPED` and contributes flat equity.

The console report shows a per-symbol table plus the aggregated block;
`--out-json` adds per-symbol metrics (and trade lists with
`--json-trades`). Exit code is 2 when no symbol could be simulated.

### Monte-Carlo robustness

`--mc N` (on `backtest` and `portfolio`) randomizes the realized trade
sequence N times and reports the distribution of the resulting equity
paths, MultiCharts-style:

- `--mc-mode permute` (default) — each iteration is a random permutation
  of the closed-trade PnLs. Net profit is order-invariant; what varies is
  the **drawdown path** — "how bad could the same trades have felt in a
  different order".
- `--mc-mode resample` — bootstrap: N trades drawn with replacement, so
  both the net-profit and the drawdown distributions vary.

The report shows P5/P25/P50/P75/P95 and the worst value for `netProfit`
and `maxDD`, the original chronological order's stats, `P(net loss)` and
`P(maxDD > original)`. The equity path is trade-level (capital plus
cumulative closed-trade PnL), so open-position excursions between entry
and exit are not part of the MC drawdown. `--mc-seed` (default 42) makes
the run deterministic; the section is appended to the JSON output as
`monteCarlo`.

## Examples

```bash
# one backtest, fixed parameters, with costs
tacuda-opt backtest --data ES_5min.csv --strategy ma_cross \
    --set fast=12 --set slow=48 --commission-bps 0.5 --slippage-ticks 1

# 10 000-variant EMA-cross grid on 1M synthetic bars
tacuda-opt grid --synthetic 1000000 --strategy ma_cross \
    --grid fast=2..101:1 --grid slow=102..201:1 \
    --fitness "netprofit - 2*maxdd" --top 20 --out-csv grid.csv

# RSI grid on real data, Sharpe fitness, JSON summary
tacuda-opt grid --data BTCUSDT_1h.csv --strategy rsi_reversion \
    --grid period=7..28:7 --grid low=20..35:5 --grid high=65..80:5 \
    --fitness sharpe --out-json rsi.json

# genetic search over the same 10 000-variant space (~3% of the work)
tacuda-opt genetic --synthetic 1000000 --strategy ma_cross \
    --grid fast=2..101:1 --grid slow=110..1100:10 \
    --population 128 --generations 60 --stagnation 12 --ga-seed 42

# walk-forward: 6 rolling windows, 75% in-sample, grid per window
tacuda-opt wfo --data ES_5min.csv --strategy ma_cross \
    --grid fast=2..20:2 --grid slow=30..250:10 \
    --windows 6 --is 0.75 --out-json wfo.json

# walk-forward with the genetic optimizer per window
tacuda-opt wfo --synthetic 1000000 --strategy ma_cross \
    --grid fast=2..101:1 --grid slow=110..1100:10 \
    --windows 6 --is 0.75 --optimizer genetic

# portfolio: 3 CSV symbols, shared params, Monte-Carlo on the merged trades
tacuda-opt portfolio --data ES.csv --data NQ.csv --data CL.csv \
    --strategy ma_cross --set fast=12 --set slow=48 \
    --capital 30000 --commission-bps 0.5 --mc 1000 --out-json pf.json

# synthetic 8-symbol portfolio through the GPU batch API
tacuda-opt portfolio --synthetic 1000000 --symbols 8 \
    --strategy ma_cross --set fast=12 --set slow=48 --backend tacuda

# Monte-Carlo on a single backtest: 10 000 bootstrap resamples
tacuda-opt backtest --synthetic 1000000 --strategy ma_cross \
    --set fast=12 --set slow=48 --mc 10000 --mc-mode resample

# force the CPU fallback (no GPU required)
tacuda-opt backtest --synthetic 100000 --strategy bbands_breakout --backend talib
```

Grid console output (top-N table):

```
=== Grid results: 10000 variants (0 skipped), fitness = netprofit - 2*maxdd ===
   #  params                     fitness    netProfit   maxDD%     PF  Sharpe  trades   win%
   1  fast=2 slow=103         17984.7937     18005.19      0.1  12.59    2.12   13083   25.2
   ...
```

`--out-csv` writes every variant (ranked, then skipped ones) with all
metrics; `--out-json` writes the run summary with the best variant and the
top-N array; `backtest --out-json --json-trades` includes the full trade
list.

Walk-forward console output:

```
=== Walk-forward: ma_cross, 6 windows, IS 75%, optimizer grid, fitness = netprofit ===
   #  IS bars             OOS bars            params                IS fit      OOS fit    OOS NP  trades
   1  0..333333           333333..444444      fast=2 slow=90    10568.3845    2439.7312   2439.73    1588
   ...
=== OOS summary (stitched equity, 666666 bars) ===
Net profit                    7528.53
...
WFE                             0.601
(IS evaluations 1380, unused tail bars 1)
```

`wfo --out-json` writes per-window ranges, chosen parameters, IS/OOS
metrics, the overall stitched OOS metrics and the WFE.

Portfolio + Monte-Carlo console output:

```
=== Portfolio backtest: ma_cross [fast=12 slow=48], 8 symbols, 1000000 bars ===
symbol           status      netProfit    maxDD%       PF   Sharpe  trades    win%
syn1             ok           17752.83       0.6     9.49     2.09   10458    36.6
...
--- portfolio (aggregated, 8/8 symbols) ---
Net profit                  897060.71
...
=== Monte-Carlo: permute, 1000 iterations, 85600 trades ===
                     P5        P25        P50        P75        P95      worst
netProfit     897060.71  897060.71  897060.71  897060.71  897060.71  897060.71
maxDD             79.75      87.39      94.31     103.44     121.59     166.55
original order: netProfit 897060.71, maxDD 209.61 (trade-level equity)
P(net loss) 0.0%   P(maxDD > original) 0.0%
```

## Tests

`tools/optimizer/tests/` (GoogleTest, ctest label `optimizer`, target
`tacuda_opt_tests`, 115 tests): simulator fills/costs/reversals against
hand-computed cases (including ranged segments used by walk-forward),
look-ahead guards (shifted signals change the outcome; signals are causal
when future data changes), determinism across runs and thread counts,
chunked == unchunked grids, cache dedup counts, CSV/args/fitness parser
edge cases. Genetic: determinism for a fixed seed, no genome is evaluated
twice, monotone best fitness under elitism, mutation clamped to the axes,
and the M2 acceptance test (≥95% of the exhaustive-grid best at ≤10% of
the evaluations). Walk-forward: window arithmetic tiles the series, the
stitched equity matches the window sums, chosen parameters depend only on
in-sample data (no-leak test), WFE formula, skipped windows contribute
flat equity. Portfolio: timestamp-intersection alignment (holes, no
overlap, missing timestamps), aggregated equity/metrics equal the sum of
independent per-symbol runs, merged trades stay chronological,
determinism. Monte-Carlo: permutation keeps net profit constant,
hand-checked drawdown cases, ordered percentiles, seed determinism,
bootstrap variability, empty/invalid inputs. Backend-vs-reference
comparisons (single-series and `ct_*_batch_d` portfolio matrices vs the
C++ reference) run when a GPU / TA-Lib is present and are skipped
otherwise. All engine tests use a plain C++ reference backend, so they
pass on machines with no GPU.

```bash
cd build && ctest -L optimizer --output-on-failure
```

## Current limitations (post-M4 roadmap)

- GPU calls run on the default stream serially outside the portfolio path;
  a stream pool would help short series (< 100K bars), where launch
  overhead dominates.
- Portfolio parameters are shared across symbols (the `ct_*_batch`
  contract) and capital is split equally; per-symbol parameter
  optimization or weighting is out of scope.
- Monte-Carlo operates on closed-trade PnLs (trade-level equity), not on
  per-bar open-position excursions.
- Strategies are compiled in; there is no plugin/scripting layer.
- Walk-forward windows are equal-length rolling (anchored mode is not
  implemented).
