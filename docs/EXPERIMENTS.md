# EXPERIMENTS — 94 classic trading systems on tacuda-opt

Status: shipped. 94 systems from the Omega Research *System Trading &
Development Club* (STAD Club) volumes 1–13 (1997–2000), ported from
TradeStation EasyLanguage to `tacuda-opt` strategies, benchmarked and
published on the site ([experiments.html](https://tacuda.com/experiments.html),
one page per system).

Source: `tools/optimizer/src/strategy/ExpVol01.cpp` … `ExpVol13.cpp`,
shared helpers in `ExpHelpers.h`, registration in `ExpRegistry.h`.
Tests: `tools/optimizer/tests/test_experiments.cpp`.
Benchmark drivers: `tools/optimizer/scripts/bench_experiments.py`
(single-series backtest/GA/WFO) and `bench_experiments_batch.py`
(multi-symbol portfolio, batch API vs device API vs TA-Lib).
Site generator: `tools/optimizer/scripts/gen_experiment_pages.py` +
`experiments_meta.py` (titles, categories, descriptions).

## What was ported

Every volume was cataloged; every system that can run on a single OHLCV
series was implemented. A system was skipped only for one of these
objective reasons:

- needs a second (or third…) data series — intermarket and spread systems;
- needs intraday bars or time-of-day session logic;
- needs data outside OHLCV (open interest, tick distributions, drawing
  objects);
- is not a trading system (educational chapters, screening indicators,
  money-management essays);
- is a pyramiding revision of a system already implemented from an earlier
  volume (Vol. 13 Luxor, Vol. 13 Triple Play).

| Volume | Implemented | Systems | Skipped (reason) |
|--------|-------------|---------|------------------|
| 1 | 7 | MA Crossover, MACD Breakout, Trend Follower, S&R Levels, Pivot Points, Channel Breakout, VolEx | Trend Line Automatic (drawing objects), Spread (second symbol), Pivot Points Intra-Day (intraday) |
| 2 | 8 | ADX Differential, Head & Shoulders, OBV New Highs/Lows, A/D Divergence, Stochastics, CCI Bollinger, Pct Change Channel, Bull Beater | Opening Gap (intraday + Data2) |
| 3 | 9 | Buy Dips/Sell Rallies, Engulfing, Open-Close Histogram, Bollinger Bands, On Balance True Range, Micro-M/W, Variable Zones, StdDev Volatility, Volatility Clusters | Inside Day Breakout Failure (30-min + Data2) |
| 4 | 9 | DMI-ADX, Keltner, LinReg & Momentum, HPI & Channel Breakout, Consecutive Close, Momentum-Retracement, Three EMA, Displaced MA & Volume, Price Distribution | Spread System Beta Up/Down (two symbols) |
| 5 | 7 | Fib .382 Retracement, High/Low Oscillator, Stochastic OB/OS, Pivot Point S&R, Currency Four-Day, Channel Breakout w/ Pullback, Displaced StdDev Bands | Currency/Bond, Intermarket I, Intermarket II (second symbol) |
| 6 | 9 | Accumulation Swing Index, ADX & MA Channel, DMA Channel & Range Leaders, Four MA Pairs, CCI Average, MRTR, First Pullback, Parabolic & Volume, Volume-Weighted Momentum | — |
| 7 | 4 | Derivative MA, Short-Term S/R (Art Putt), MA Support/Resistance, Trading Range Breakout | Dow Theory (Data2-4), Intraday Breakout + Fib (intraday), Intermarket S&P/Bonds/Bills, S&P Implied Volatility (external series) |
| 8 | 4 | Escalator, Jack-in-the-Box, Ryan's Hope, Stampede | ActivityBar Price Distribution (tick data); SystemBuilder/2000i/Fixed-Ratio chapters (no rules) |
| 9 | 8 | Double Your Fun, Luxor, No Hurry, OBV Revisited, Red Rover, Skinny Dipper, Swinger, Traffic Jam | International Index Composite (7 series); RadarScreen chapter (indicator) |
| 10 | 7 | JK Monday We Trade, Serendipity, Eight Ball, Going In Style, Who's Minding the Store, Reference Deviation, Superman | Gemini (weekly Data2); concept chapters (no rules) |
| 11 | 7 | High Five, Hit the Bull's Eye, JK Night Train, Symmetrical Triangle, TrendScore, Triple Play, Win One for the Gapper | MISER (bonds Data2); exit-strategy/report chapters (no system) |
| 12 | 9 | First Prize, In The Zone, Jailbreak, LR Crossing, Not That Way The Other Way, Seems Kinda Shifty, Springs & Upthrusts, Three Musketeers, Tigers in the Mist #2 | Trendiness research note (no rules) |
| 13 | 6 | Open-Close Histogram, DMA & Range Leaders, LinReg & Momentum, Momentum Retracement (MRTS), VolEx, Volatility Clusters — all revisions of earlier systems | Luxor and Triple Play revisions (duplicates), intermarket chapters, ch. 1-3 (educational) |

Strategy names follow `exp<volume>_<system>`: `exp01_ma_crossover` …
`exp13_vol_clusters`. `tacuda-opt list` prints all of them with their
parameters.

## Adaptation conventions (EasyLanguage → tacuda-opt)

The optimizer's engine decides target positions at bar closes and fills
them at the next bar open (see
[STRATEGY_OPTIMIZER.md](STRATEGY_OPTIMIZER.md)). The club systems lean
heavily on intrabar stop/limit orders, pyramiding and dollar stops, so the
pack applies uniform conventions:

- **Stop/limit entries and exits** become *close-confirmed*: the level is
  checked against the finished bar (`buy_stop_hit` / `sell_stop_hit` in
  `ExpHelpers.h` — high/low must reach the level **and** the close must
  confirm the direction), and the fill happens at the next open. This
  trades a bar of latency for zero look-ahead.
- **One unit per side.** Pyramiding (Stampede adds up to 9 units, Luxor up
  to 3) collapses to a single unit; double-size reversal entries become
  plain reversals. Where a filter only gated the add-on units (Luxor's
  ADX), it is dropped with the pyramiding.
- **Dollar stops → ATR multiples.** `$500`-style protective stops,
  breakeven shifts and targets are re-expressed as ATR-multiple
  parameters, so systems stay meaningful on any price scale.
- **Session/day rules** use bar timestamps (`day_of_week`); intraday-only
  session filters are dropped and noted per system.
- **Indicators** come from the shared `Family` request set computed by the
  backend (GPU TaCuda / TA-Lib / test reference). Since 0.4.0 the
  club-specific constructs are native TaCuda families too — Accumulation
  Swing Index, EL-style parabolic, On Balance True Range, weighted OBV,
  Herrick-Payoff proxy, MarketStrength, TrendScore, Reference Deviation,
  rolling kurtosis/skewness, swing pivots (confirmed `strength` bars
  late — never with future data) — and derived lines ("XAverage of
  OBTR") chain inside the backend. Only bespoke pattern logic (head &
  shoulders and friends) stays CPU-side in `signals()`.
- **Common exit suites.** Volumes 8–13 share a toolbox of protective /
  breakeven / trailing / volatility / big-profit ATR stops; the pack
  implements them once per system with the volume's constants as
  parameters.

## Tests

`test_experiments.cpp` (part of `tacuda_opt_tests`, label `optimizer`):

- registry: exactly 94 unique `exp*` systems, every volume 1–13 present;
- declarations: params non-empty, defaults valid and inside ranges,
  indicator requests well-formed;
- fast/slow validation rejects degenerate combinations;
- smoke: every system runs on 3 000 synthetic daily bars — valid result,
  finite metrics, |position| ≤ 1, exits at or after entries, and at least
  2/3 of the pack trades;
- warm-up/range: raw signals are always in {-1, 0, 1};
- portfolio batch path: every system runs on a 3-symbol portfolio through
  the batch-backend interface and aggregates to exactly the sum of the
  per-symbol single-series runs;
- unit tests for `ExpHelpers` (EL-style crossings, day-of-week, rolling
  stats, EMA seeding, stop-hit semantics, `avg_range`,
  `occurred_at_least`).

Backend numerical parity (`test_backends.cpp`) covers all 22 requested
families with per-family tolerances on tacuda, TA-Lib and the plain-C++
reference. Together with the shared engine tests (causality, determinism,
warmup enforcement) the optimizer suite is 129 tests; all pass on the GPU
box.

## Benchmark protocol

`bench_experiments.py` runs three passes per system on identical inputs:

```bash
python tools/optimizer/scripts/bench_experiments.py \
    --bin ./build/tools/optimizer/tacuda-opt \
    --out exp_results --bars 5000 --seed 42 --backend tacuda
```

1. `backtest` at the volume defaults (`--json-trades` for the equity
   curve);
2. `genetic` over the full declared parameter space (population 48, ≤30
   generations, stagnation 10, fitness `netprofit - 2*maxdd`);
3. `wfo` with 6 rolling windows, 75% in-sample, genetic optimizer per
   window (population 32, ≤15 generations).

Shared inputs: synthetic daily series (5 000 bars ≈ 19 years, seed 42,
weekends skipped), commission 2 bps + 1 tick slippage per side, capital
10 000, GPU backend. The full 94-system run takes ~47 s on the RTX 3090 Ti
box (282 optimizer invocations). Results land in `exp_results/*.json`
plus a combined `summary.json`.

These numbers demonstrate the engine, not tradable edges: the series is
synthetic, and genetic results are in-sample by construction. The
walk-forward pass (per-window re-optimization, stitched OOS equity, WFE)
is the honest column.

## Batch-API portfolio path

Every one of the 94 systems also runs multi-symbol through the TaCuda
batch API — no per-strategy work was needed: strategies only declare
`IndicatorRequest`s, and all 22 requested families (SMA, EMA, WMA, TRIMA,
RSI, BBANDS, ATR, MACD, STOCH, CCI, ADX, ±DI, OBV, MOM, ROC, STDDEV,
LINEARREG(+slope), MAX, MIN, TRANGE) are wired to `ct_*_batch_d` in the
portfolio backend. Three interchangeable execution paths:

```bash
# one ct_*_batch_d call per indicator request (numSeries x seriesLen matrix)
tacuda-opt portfolio --synthetic 500000 --symbols 16 \
    --strategy exp01_ma_crossover --backend tacuda
# baseline: same GPU kernels, one single-series ct_*_d call per symbol
tacuda-opt portfolio ... --backend tacuda-device
# per-symbol TA-Lib on the CPU
tacuda-opt portfolio ... --backend talib
```

`bench_experiments_batch.py` drives all 94 systems through four backend
configurations on identical synthetic portfolios and checks net-profit
parity; since 0.4.1 the TA-Lib baseline is measured both single-threaded
and on all CPU cores (`--threads 0` parallelizes the per-symbol CPU
indicator path across symbols). The report's phased timing
(`indicatorSec` vs `signalSec`) isolates the part the backends own.
Results on the RTX 3090 Ti + i3-12100F (4C/8T) box, 94 systems,
commission 2 bps + 1 tick:

| Config | batch | device-loop | TA-Lib 1t | TA-Lib 8t | vs TA-Lib 1t | vs TA-Lib 8t |
|---|---|---|---|---|---|---|
| 0.4.0, 16 sym × 500 k | 22.2 s | 36.3 s | 19.7 s | — | ×0.89 | — |
| 0.4.1, 16 sym × 500 k | 16.8 s | 14.1 s | 19.8 s | 8.1 s | **×1.18** | ×0.48 |
| 0.4.1, 64 sym × 500 k | 48.6 s | 38.1 s | 79.3 s | 31.8 s | **×1.63** | ×0.65 |

(total indicator-phase seconds over all 94 systems; "×N" = speedup of the
batch path)

The 0.4.1 scan rewrite (`ClubPack.cu`) removed the last single-thread GPU
recursions: OBTR/WOBV/ASI became parallel prefix scans of per-bar terms,
HPI an `AffineScan` recurrence, XAverage a seed-aware affine monoid scan
that treats NaN inputs as identities. The club-heavy systems dropped to
the PCIe floor — `exp06_asi` 2.03 s → 0.17 s at 16×500 k (×11.7),
`exp03_obtr` 1.13 s → ~0.2 s — flipping them from the pack's worst GPU
cases (×0.2–0.3 vs one CPU thread in 0.4.0) to ×1.2–1.9 wins. The only
remaining sequential exception is `SAR_EL`, a genuine state machine
(trend flips and acceleration clamps do not associate into a scan).

A scaling sweep over a 12-system representative subset (100 k bars per
symbol, 4 → 1024 symbols; `exp_v3_scale_*` summaries in
`tools/optimizer/bench/`) shows where each backend lives:

- **GPU time is nearly flat until the card fills** (4 → 64 symbols is
  almost free), CPU time grows linearly from the first symbol. The
  crossover against one CPU thread sits around 16–32 symbols; at 1024
  symbols the subset runs ×2.1 faster than one thread.
- **Compute-heavy windowed kernels beat all 8 CPU threads outright**:
  `exp10_ref_dev` ×10.1 vs one thread and ×2.5 vs eight at 1024 sym,
  `exp04_price_dist` ×7.6/×1.7. These ratios keep growing with portfolio
  size while the GPU stays below one second.
- **Trivial request sets stay PCIe-bound** (a couple of SMA lines over a
  big matrix is bytes, not math): there the 8-thread CPU remains
  competitive — that floor is the price of downloading every line for
  the CPU-side simulation, and it disappears entirely for device-resident
  consumers (`ct_*_batch_d`, derived-series chaining).

Findings that still hold from the 0.4.0 round, honestly stated:

- **Parity.** All 94 systems: GPU batch vs GPU device-loop agree to float
  determinism limits, TA-Lib within tolerance. Scan-based kernels (RSI,
  BBANDS, EMA, ATR, club scans…) are only ULP-reproducible run-to-run —
  CUB's decoupled look-back groups float additions differently per
  launch — so a knife-edge threshold can flip a single trade on an
  8M-bar portfolio (`exp02_bull_beater`: 8025 vs 8026 trades across
  *runs of the same backend*). The bench treats GPU paths as matching
  within 0.1% of net profit.
- **End-to-end the backend hardly matters for a single backtest**:
  signals+simulation is CPU work of the same order as the indicator
  phase, so total wall time per system differs by only a few percent.
  The batch API's niche is portfolio-scale indicator computation — and
  it now holds that niche against both GPU-per-symbol loops and the
  multi-core CPU wherever kernels do real work.

The full comparison story (tables, scaling charts, capability matrix) is
published at [tacuda.com/benchmark.html](https://tacuda.com/benchmark.html),
section "Strategy workloads"; the tables are regenerated from the committed
summaries with `python scripts/gen_talib_comparison.py`.

During this work the portfolio batch backend gained two correctness/perf
fixes: the column-matrix upload is now synchronized before the first
`ct_*_batch_d` call (the batch entry points dispatch onto *non-blocking*
streams, which do not order against the legacy-stream upload — a genuine
race), and each output line is downloaded as one matrix-sized D2H
transfer and scattered on the host instead of `numSeries` small copies.

### Batch-API gaps — closed in 0.4.0

The three gap classes identified by the 0.3.x benchmark round were all
closed in 0.4.0:

1. **C-ABI functions without `_batch` variants** — every family with a
   host API now has generated `ct_*_d` and `ct_*_batch[_d]` entry
   points: SAR/SAREXT, MAMA, MACDFIX, STOCHRSI, TRIX, NVI/PVI, the six
   `ct_ht_*`, APO/PPO/PVO, BOP, IMI, AVGDEV, MIDPOINT/MIDPRICE, ±DM,
   MAXINDEX/MININDEX/MINMAX(INDEX), CHANGE, MACD_LINE, MA/MAVP,
   CORREL/BETA and all 63 candlestick patterns (see
   [BATCH_API.md](BATCH_API.md)). Vector math stays intentionally
   unbatched (elementwise kernels are shape-agnostic — the 0.3.0
   decision).
2. **Club indicators became native TaCuda families** (`ClubPack.cu`):
   `ct_asi`, `ct_sar_el` (EL-seeded parabolic), `ct_obtr`, `ct_wobv`,
   `ct_hpi`, `ct_market_strength`, `ct_trendscore`, `ct_refdev`,
   `ct_skew`, `ct_kurtosis`, `ct_pivot_high`/`ct_pivot_low`,
   `ct_xaverage` and the NaN-strict `ct_rolling_{mean,max,min,std}` —
   each with host, device and batch entry points, mirrored in the
   optimizer as `Family::ASI` … `Family::ROLLING_STD` on all backends
   (GPU kernels, shared plain-C++ port for TA-Lib/test backends).
3. **Derived-series chaining.** `IndicatorRequest` now takes an optional
   `chain` (upstream request) — "XAverage *of OBTR*", "rolling mean *of
   weighted OBV*", "rolling MAX/MIN *of HPI*" are single requests; GPU
   backends compute the whole chain device-side (batch backend: on the
   `numSeries × seriesLen` matrices) with no host round-trip, and the
   cache key folds the chain in. Chaining is limited to the NaN-aware
   club families over single-output upstream requests
   (`family_chainable()` / `validate_chain()`).

Seven systems moved their custom math from `signals()` into requests:
`exp03_obtr` (OBTR + XAverage chain), `exp04_hpi_breakout` (HPI +
rolling MAX/MIN chains), `exp04_price_dist` (KURTOSIS + SKEW),
`exp06_asi` (ASI + PIVOT_HIGH/LOW), `exp06_parabolic_vol` (SAR_EL),
`exp09_obv_revisited` (WOBV + rolling-mean chain), `exp10_ref_dev`
(REFDEV), `exp10_superman` (MARKET_STRENGTH), `exp11_trendscore`
(TRENDSCORE + rolling-mean chain). Swing-pivot pattern systems with
bespoke pivot semantics (head & shoulders, micro M/W, symmetrical
triangle) keep their CPU logic.

In 0.4.1 the club recursions were re-formulated as parallel scans (see
the benchmark section above), closing the last performance gap the 0.4.0
round documented.

## Site generation

The per-system pages published at
[tacuda.com/experiments.html](https://tacuda.com/experiments.html) are
generated by a script in this repository (the website tree itself lives
outside the repo — pass the directory that contains `site/`):

```bash
python tools/optimizer/scripts/gen_experiment_pages.py --repo <website-root>
```

Reads `experiments_meta.py` (authored titles/categories/descriptions per
system, volume blurbs, skip lists), the rule summaries straight from the
`ExpVol*.cpp` implementation comments, and the benchmark JSONs from
`tools/optimizer/bench/exp_results/`. Emits the hub page, one static
self-contained page per system (concept, adaptation notes, implemented
rules, parameter table, reproduce commands, backtest metrics, inline SVG
equity curve, GA top-5, walk-forward windows with OOS summary) and a
refreshed `sitemap.xml`.
