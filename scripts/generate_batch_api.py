#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Generate the batched (multi-symbol) API surface from a single family table.

Emits:

* ``include/tacuda.h``                        - batch declarations between markers
* ``include/tacuda_device.h``                 - generated ct_*_d declarations
* ``src/api/api_batch_generated.inc``         - host+device batch implementations
* ``src/api/api_device_generated.inc``        - generated ct_*_d implementations
* ``src/api/api_device_generated_includes.inc`` - indicator headers for the above
* ``bindings/nodejs/batch_generated.js``      - koffi decls + JS wrappers
* ``bindings/csharp/TacudaBatch.Generated.cs``- friendly C# helpers

Python wrappers are built at runtime in ``bindings/python/tacuda/__init__.py``
from ``_generated.py`` specs, so nothing is emitted for Python here. After
running this script, re-run ``bindings/generate_bindings.py`` so the ctypes /
P-Invoke layers pick up the new symbols.

Families are dispatched per-series to ``ct_*_d`` device functions via the
generic runners in ``api_batch.cu``, unless they have a fused kernel
(``fused``) or a row-batched plan (``ROWS``) that computes the whole matrix
in a fixed number of launches. Device functions that did not exist
before 0.4.0 (SAR/SAREXT, MAMA, MACDFIX, STOCHRSI, TRIX, NVI/PVI, HT_*,
APO/PPO/PVO, BOP, IMI, AVGDEV, MIDPOINT/MIDPRICE, +/-DM, MAXINDEX/MININDEX/
MINMAX(INDEX), CHANGE, MACD_LINE, MA/MAVP, CORREL/BETA, the 63 candlestick
patterns and the STAD "club pack") are generated here as thin wrappers over
the existing CUDA indicator classes. Club-pack families additionally get a
plain single-series host entry point (``ct_asi`` etc.).

Vector-math functions (sin, add, ...) are intentionally excluded:
element-wise kernels are shape-agnostic, so calling e.g. ``ct_sin`` on the
flattened matrix is already the optimal batch.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path
from typing import Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "tacuda.h"
HEADER_DEVICE = ROOT / "include" / "tacuda_device.h"
IMPL_INC = ROOT / "src" / "api" / "api_batch_generated.inc"
DEVICE_IMPL_INC = ROOT / "src" / "api" / "api_device_generated.inc"
DEVICE_INCLUDES_INC = ROOT / "src" / "api" / "api_device_generated_includes.inc"
NODE_OUT = ROOT / "bindings" / "nodejs" / "batch_generated.js"
CSHARP_OUT = ROOT / "bindings" / "csharp" / "TacudaBatch.Generated.cs"

HDR_BEGIN = "/* --- BEGIN GENERATED BATCH API (scripts/generate_batch_api.py) --- */"
HDR_END = "/* --- END GENERATED BATCH API --- */"
DEV_HDR_BEGIN = "/* --- BEGIN GENERATED DEVICE API (scripts/generate_batch_api.py) --- */"
DEV_HDR_END = "/* --- END GENERATED DEVICE API --- */"

Scalar = Tuple[str, str]  # (ctype, name)


@dataclasses.dataclass(frozen=True)
class Family:
    name: str                 # ct_{name}_batch / ct_{name}_batch_d
    d_func: str               # underlying single-series device function
    inputs: Sequence[str]     # logical plural column names
    outputs: Sequence[str]    # logical plural output names
    scalars: Sequence[Scalar] = ()
    host_single: bool = False  # also emit a plain ct_{name} host wrapper
    fused: str = ""           # FusedFamily enumerator; empty keeps legacy dispatch
    output_dtypes: Sequence[str] = ()  # empty = legacy float outputs
    reference: str = ""       # pinned TA-Lib 0.8.3 operation, if present


@dataclasses.dataclass(frozen=True)
class DeviceFamily:
    """A ct_*_d wrapper generated over an existing CUDA indicator class."""

    name: str                  # ct_{name}_d
    cls: str                   # tacuda:: class name
    inputs: Sequence[str]      # logical plural names (order = calculate order)
    outputs: Sequence[str]
    scalars: Sequence[Scalar] = ()
    ctor: str = ""             # C++ constructor argument expression
    call: str = "direct"       # direct | stacked | pick0
    stacked_lines: int = 0     # lines the class writes for stacked/pick0
    header: str = ""           # indicator header (defaults to <cls>.h)
    host_single: bool = False  # club pack: also emit plain ct_{name} host entry
    output_dtypes: Sequence[str] = ()
    reference: str = ""


IN_SINGLE = ("inputs",)
IN_HL = ("highs", "lows")
IN_HLC = ("highs", "lows", "closes")
IN_OHLC = ("opens", "highs", "lows", "closes")
IN_OHLCV = ("opens", "highs", "lows", "closes", "volumes")
IN_HLCV = ("highs", "lows", "closes", "volumes")
IN_CV = ("closes", "volumes")
OUT_SINGLE = ("outputs",)

PERIOD = (("int", "period"),)


def _simple(name: str, extra: Sequence[Scalar] = PERIOD, fused: str = "") -> Family:
    return Family(name, f"ct_{name}_d", IN_SINGLE, OUT_SINGLE, extra, fused=fused)


# Families whose ct_*_d existed before 0.4.0 (hand-written in api_device.cu).
BASE_FAMILIES: Sequence[Family] = (
    # --- single input, single output, period ---
    _simple("wma", fused="Wma"), _simple("dema"), _simple("tema"), _simple("trima"),
    _simple("momentum"), _simple("roc"), _simple("rocp"), _simple("rocr"),
    _simple("rocr100"), _simple("cmo"), _simple("stddev", fused="Stddev"), _simple("var"),
    _simple("sum", fused="Sum"), _simple("max", fused="Max"), _simple("min", fused="Min"), _simple("linearreg"),
    _simple("linearreg_slope"), _simple("linearreg_intercept"),
    _simple("linearreg_angle"), _simple("tsf"),
    _simple("kama", (("int", "period"), ("int", "fastPeriod"),
                     ("int", "slowPeriod"))),
    _simple("t3", (("int", "period"), ("float", "vFactor"))),
    # --- high/low/close input, single output ---
    Family("natr", "ct_natr_d", IN_HLC, OUT_SINGLE, PERIOD, fused="Natr"),
    Family("adx", "ct_adx_d", IN_HLC, OUT_SINGLE, PERIOD),
    Family("adxr", "ct_adxr_d", IN_HLC, OUT_SINGLE, PERIOD),
    Family("dx", "ct_dx_d", IN_HLC, OUT_SINGLE, PERIOD),
    Family("plus_di", "ct_plus_di_d", IN_HLC, OUT_SINGLE, PERIOD),
    Family("minus_di", "ct_minus_di_d", IN_HLC, OUT_SINGLE, PERIOD),
    Family("willr", "ct_willr_d", IN_HLC, OUT_SINGLE, PERIOD),
    Family("cci", "ct_cci_d", IN_HLC, OUT_SINGLE, PERIOD),
    Family("trange", "ct_trange_d", IN_HLC, OUT_SINGLE, ()),
    Family("typprice", "ct_typprice_d", IN_HLC, OUT_SINGLE, ()),
    Family("wclprice", "ct_wclprice_d", IN_HLC, OUT_SINGLE, ()),
    Family("ultosc", "ct_ultosc_d", IN_HLC, OUT_SINGLE,
           (("int", "period1"), ("int", "period2"), ("int", "period3"))),
    # --- high/low input ---
    Family("medprice", "ct_medprice_d", IN_HL, OUT_SINGLE, ()),
    Family("aroonosc", "ct_aroonosc_d", IN_HL, OUT_SINGLE, PERIOD),
    Family("aroon", "ct_aroon_d", IN_HL, ("aroon_downs", "aroon_ups"), PERIOD),
    # --- oscillators with multiple outputs ---
    Family("stochf", "ct_stochastic_fast_d", IN_HLC, ("fast_ks", "fast_ds"),
           (("int", "kPeriod"), ("int", "dPeriod"), ("int", "dMaType"))),
    Family("macdext", "ct_macdext_d", IN_SINGLE,
           ("macds", "signals", "hists"),
           (("int", "fastPeriod"), ("int", "fastType"),
            ("int", "slowPeriod"), ("int", "slowType"),
            ("int", "signalPeriod"), ("int", "signalType")), reference="MACDEXT"),
    Family("accbands", "ct_accbands_d", IN_HLC,
           ("uppers", "middles", "lowers"), PERIOD),
    # --- price transforms / volume ---
    Family("avgprice", "ct_avgprice_d", IN_OHLC, OUT_SINGLE, ()),
    Family("obv", "ct_obv_d", IN_CV, OUT_SINGLE, (), fused="Obv"),
    Family("mfi", "ct_mfi_d", IN_HLCV, OUT_SINGLE, PERIOD, fused="Mfi"),
    Family("ad", "ct_ad_d", IN_HLCV, OUT_SINGLE, (), fused="Ad"),
    Family("adosc", "ct_adosc_d", IN_HLCV, OUT_SINGLE,
           (("int", "fastPeriod"), ("int", "slowPeriod")), fused="Adosc"),
)

# 63 candlestick patterns: ct_cdl_<name> -> tacuda class (OHLC in, one out).
CDL_PATTERNS: Sequence[Tuple[str, str]] = (
    ("cdl_doji", "Doji"), ("cdl_hammer", "Hammer"),
    ("cdl_inverted_hammer", "InvertedHammer"),
    ("cdl_bullish_engulfing", "BullishEngulfing"),
    ("cdl_bearish_engulfing", "BearishEngulfing"),
    ("cdl_three_white_soldiers", "ThreeWhiteSoldiers"),
    ("cdl_abandoned_baby", "AbandonedBaby"),
    ("cdl_advance_block", "AdvanceBlock"), ("cdl_belt_hold", "BeltHold"),
    ("cdl_breakaway", "Breakaway"), ("cdl_two_crows", "TwoCrows"),
    ("cdl_three_black_crows", "ThreeBlackCrows"),
    ("cdl_three_inside", "ThreeInside"), ("cdl_three_outside", "CDL3OUTSIDE"),
    ("cdl_three_line_strike", "ThreeLineStrike"),
    ("cdl_three_stars_in_south", "ThreeStarsInSouth"),
    ("cdl_closing_marubozu", "ClosingMarubozu"),
    ("cdl_conceal_baby_swallow", "ConcealBabySwallow"),
    ("cdl_counterattack", "CounterAttack"),
    ("cdl_dark_cloud_cover", "DarkCloudCover"), ("cdl_doji_star", "DojiStar"),
    ("cdl_dragonfly_doji", "DragonflyDoji"), ("cdl_engulfing", "Engulfing"),
    ("cdl_evening_doji_star", "EveningDojiStar"),
    ("cdl_evening_star", "EveningStar"),
    ("cdl_gap_side_side_white", "GapSideSideWhite"),
    ("cdl_gravestone_doji", "GravestoneDoji"),
    ("cdl_hanging_man", "HangingMan"), ("cdl_harami", "Harami"),
    ("cdl_harami_cross", "HaramiCross"), ("cdl_high_wave", "HighWave"),
    ("cdl_hikkake", "Hikkake"), ("cdl_hikkake_mod", "HikkakeMod"),
    ("cdl_homing_pigeon", "HomingPigeon"),
    ("cdl_identical_three_crows", "IdenticalThreeCrows"),
    ("cdl_in_neck", "InNeck"), ("cdl_kicking", "Kicking"),
    ("cdl_kicking_by_length", "KickingByLength"),
    ("cdl_ladder_bottom", "LadderBottom"),
    ("cdl_long_legged_doji", "LongLeggedDoji"), ("cdl_long_line", "LongLine"),
    ("cdl_marubozu", "Marubozu"), ("cdl_matching_low", "MatchingLow"),
    ("cdl_mat_hold", "MatHold"),
    ("cdl_morning_doji_star", "MorningDojiStar"),
    ("cdl_morning_star", "MorningStar"), ("cdl_on_neck", "OnNeck"),
    ("cdl_piercing", "Piercing"), ("cdl_rickshaw_man", "RickshawMan"),
    ("cdl_rise_fall3_methods", "RiseFall3Methods"),
    ("cdl_separating_lines", "SeparatingLines"),
    ("cdl_shooting_star", "ShootingStar"), ("cdl_short_line", "ShortLine"),
    ("cdl_spinning_top", "SpinningTop"),
    ("cdl_stalled_pattern", "StalledPattern"),
    ("cdl_stick_sandwich", "StickSandwich"), ("cdl_takuri", "Takuri"),
    ("cdl_tasuki_gap", "TasukiGap"), ("cdl_thrusting", "Thrusting"),
    ("cdl_tristar", "Tristar"), ("cdl_unique_3_river", "Unique3River"),
    ("cdl_upside_gap_2_crows", "UpsideGap2Crows"),
    ("cdl_xside_gap_3_methods", "XSideGap3Methods"),
)

FAST_SLOW = (("int", "fastPeriod"), ("int", "slowPeriod"))

# ct_*_d wrappers generated over existing indicator classes (0.4.0).
DEVICE_FAMILIES: Sequence[DeviceFamily] = tuple(
    [
        # --- single input, single output ---
        DeviceFamily("trix", "TRIX", IN_SINGLE, OUT_SINGLE, PERIOD, "period"),
        DeviceFamily("change", "Change", IN_SINGLE, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("midpoint", "MIDPOINT", IN_SINGLE, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("avgdev", "AVGDEV", IN_SINGLE, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("maxindex", "MAXINDEX", IN_SINGLE, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("minindex", "MININDEX", IN_SINGLE, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("apo", "APO", IN_SINGLE, OUT_SINGLE, FAST_SLOW,
                     "fastPeriod, slowPeriod"),
        DeviceFamily("ppo", "PPO", IN_SINGLE, OUT_SINGLE, FAST_SLOW,
                     "fastPeriod, slowPeriod"),
        DeviceFamily("pvo", "PVO", ("volumes",), OUT_SINGLE, FAST_SLOW,
                     "fastPeriod, slowPeriod"),
        DeviceFamily("ma", "MA", IN_SINGLE, OUT_SINGLE,
                     (("int", "period"), ("int", "maType")),
                     "period, static_cast<tacuda::MAType>(maType)"),
        DeviceFamily("ht_dcperiod", "HT_DCPERIOD", IN_SINGLE, OUT_SINGLE),
        DeviceFamily("ht_dcphase", "HT_DCPHASE", IN_SINGLE, OUT_SINGLE),
        DeviceFamily("ht_trendline", "HT_TRENDLINE", IN_SINGLE, OUT_SINGLE),
        DeviceFamily("ht_trendmode", "HT_TRENDMODE", IN_SINGLE, OUT_SINGLE),
        # --- single input, stacked multi-output classes ---
        DeviceFamily("stochrsi", "StochRSI", IN_SINGLE,
                     ("fast_ks", "fast_ds"),
                     (("int", "rsiPeriod"), ("int", "kPeriod"),
                      ("int", "dPeriod")),
                     "rsiPeriod, kPeriod, dPeriod", "stacked", 2),
        DeviceFamily("macdfix", "MACDFIX", IN_SINGLE,
                     ("macds", "signals", "hists"),
                     (("int", "signalPeriod"),), "signalPeriod", "stacked", 3),
        DeviceFamily("mama", "MAMA", IN_SINGLE, ("mamas", "famas"),
                     (("float", "fastLimit"), ("float", "slowLimit")),
                     "fastLimit, slowLimit", "stacked", 2),
        DeviceFamily("minmax", "MINMAX", IN_SINGLE, ("mins", "maxs"), PERIOD,
                     "period", "stacked", 2),
        DeviceFamily("minmaxindex", "MINMAXINDEX", IN_SINGLE,
                     ("minidxs", "maxidxs"), PERIOD, "period", "stacked", 2),
        DeviceFamily("ht_phasor", "HT_PHASOR", IN_SINGLE,
                     ("inphases", "quadratures"), (), "", "stacked", 2),
        DeviceFamily("ht_sine", "HT_SINE", IN_SINGLE, ("sines", "leadsines"),
                     (), "", "stacked", 2),
        DeviceFamily("macd_line", "MACDEXT", IN_SINGLE, OUT_SINGLE, FAST_SLOW,
                     "fastPeriod, slowPeriod, 1, tacuda::MAType::EMA",
                     "pick0", 3, header="MACDEXT.h"),
        # --- two inputs ---
        DeviceFamily("midprice", "MIDPRICE", IN_HL, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("plus_dm", "PlusDM", IN_HL, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("minus_dm", "MinusDM", IN_HL, OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("sar", "SAR", IN_HL, OUT_SINGLE,
                     (("float", "step"), ("float", "maxAcceleration")),
                     "step, maxAcceleration"),
        DeviceFamily("sarext", "SAREXT", IN_HL, OUT_SINGLE,
                     (("float", "startValue"), ("float", "offsetOnReverse"),
                      ("float", "accInitLong"), ("float", "accLong"),
                      ("float", "accMaxLong"), ("float", "accInitShort"),
                      ("float", "accShort"), ("float", "accMaxShort")),
                     "startValue, offsetOnReverse, accInitLong, accLong, "
                     "accMaxLong, accInitShort, accShort, accMaxShort"),
        DeviceFamily("imi", "IMI", ("opens", "closes"), OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("nvi", "NVI", IN_CV, OUT_SINGLE),
        DeviceFamily("pvi", "PVI", IN_CV, OUT_SINGLE),
        DeviceFamily("correl", "Correl", ("xs", "ys"), OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("beta", "Beta", ("xs", "ys"), OUT_SINGLE, PERIOD,
                     "period"),
        DeviceFamily("mavp", "MAVP", ("values", "periods"), OUT_SINGLE,
                     (("int", "minPeriod"), ("int", "maxPeriod"),
                      ("int", "maType")),
                     "minPeriod, maxPeriod, static_cast<tacuda::MAType>(maType)",
                     header="MAVP.h", reference="MAVP"),
        # --- OHLC ---
        DeviceFamily("bop", "BOP", IN_OHLC, OUT_SINGLE),
    ]
    + [
        DeviceFamily(name, cls, IN_OHLC, OUT_SINGLE, (), "", header=f"{cls}.h")
        for name, cls in CDL_PATTERNS
    ]
    + [
        # --- STAD club pack (ClubPack.h) ---
        DeviceFamily("asi", "ASI", IN_OHLC, OUT_SINGLE, (), "",
                     header="ClubPack.h", host_single=True),
        DeviceFamily("sar_el", "SAREL", IN_HLC, OUT_SINGLE,
                     (("float", "accFact"), ("float", "maxAccel")),
                     "accFact, maxAccel", header="ClubPack.h",
                     host_single=True),
        DeviceFamily("obtr", "OBTR", IN_HLC, OUT_SINGLE, (), "",
                     header="ClubPack.h", host_single=True),
        DeviceFamily("wobv", "WOBV", IN_OHLCV, OUT_SINGLE, (), "",
                     header="ClubPack.h", host_single=True),
        DeviceFamily("hpi", "HPI", ("highs", "lows", "volumes"), OUT_SINGLE,
                     (("float", "factor"),), "factor", header="ClubPack.h",
                     host_single=True),
        DeviceFamily("market_strength", "MarketStrength", IN_SINGLE,
                     OUT_SINGLE, PERIOD, "period", header="ClubPack.h",
                     host_single=True),
        DeviceFamily("trendscore", "TrendScore", IN_SINGLE, OUT_SINGLE,
                     (("int", "lookback"),), "lookback", header="ClubPack.h",
                     host_single=True),
        DeviceFamily("refdev", "RefDev", IN_SINGLE, OUT_SINGLE,
                     (("int", "smaPeriod"), ("int", "sumPeriod")),
                     "smaPeriod, sumPeriod", header="ClubPack.h",
                     host_single=True),
        DeviceFamily("skew", "RollingSkew", IN_SINGLE, OUT_SINGLE, PERIOD,
                     "period", header="ClubPack.h", host_single=True),
        DeviceFamily("kurtosis", "RollingKurtosis", IN_SINGLE, OUT_SINGLE,
                     PERIOD, "period", header="ClubPack.h", host_single=True),
        DeviceFamily("pivot_high", "PivotHigh", IN_SINGLE, OUT_SINGLE,
                     (("int", "strength"),), "strength", header="ClubPack.h",
                     host_single=True),
        DeviceFamily("pivot_low", "PivotLow", IN_SINGLE, OUT_SINGLE,
                     (("int", "strength"),), "strength", header="ClubPack.h",
                     host_single=True),
        DeviceFamily("xaverage", "XAverage", IN_SINGLE, OUT_SINGLE, PERIOD,
                     "period", header="ClubPack.h", host_single=True),
        DeviceFamily("rolling_mean", "RollingMean", IN_SINGLE, OUT_SINGLE,
                     PERIOD, "period", header="ClubPack.h", host_single=True),
        DeviceFamily("rolling_max", "RollingMax", IN_SINGLE, OUT_SINGLE,
                     PERIOD, "period", header="ClubPack.h", host_single=True),
        DeviceFamily("rolling_min", "RollingMin", IN_SINGLE, OUT_SINGLE,
                     PERIOD, "period", header="ClubPack.h", host_single=True),
        DeviceFamily("rolling_std", "RollingStd", IN_SINGLE, OUT_SINGLE,
                     PERIOD, "period", header="ClubPack.h", host_single=True),
    ]
)

from talib08_catalog import NEW as TA08_NEW
DEVICE_FAMILIES = tuple(DEVICE_FAMILIES) + tuple(
    DeviceFamily(e['api'], '', e['inputs'], e['outputs'], e['scalars'],
                 host_single=e['name'] not in ['APO','PPO'], output_dtypes=e['output_dtypes'], reference=e['name'])
    for e in TA08_NEW
)

# Batch families = base (pre-0.4.0 device functions) + everything generated.
FAMILIES: Sequence[Family] = tuple(BASE_FAMILIES) + tuple(
    Family(df.name, f"ct_{df.name}_d", df.inputs, df.outputs, df.scalars,
           host_single=df.host_single, output_dtypes=df.output_dtypes, reference=df.reference)
    for df in DEVICE_FAMILIES
)


# ---------------------------------------------------------------------------
# Row-batched legacy families (RowsPlan in api_batch.cu, docs/BATCH_API.md)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RowsPlan:
    """Whole-matrix execution of a legacy family with its own arithmetic.

    "flat": the family's ct_*_d over the flattened matrix, in blocks of
    whole rows. The family's kernels read only the window that ends at the
    element they write, so elements at or past the first valid index
    `warmup` of a row see only that row; the runner restores the elements
    before it from the block's first row. "rows": `launcher` runs the sequential single-series state
    machine with one GPU thread per row. `warmup` and `valid` are C++
    expressions over the family's scalars; `valid` false keeps the
    per-series dispatcher (and its error reporting). `long_row` overrides
    the row length from which a flat plan needs 64 rows to pay off.
    """

    kind: str
    warmup: str = "0"
    valid: str = "true"
    launcher: str = ""
    long_row: str = ""


def _flat(warmup: str, valid: str = "true", long_row: str = "") -> RowsPlan:
    return RowsPlan("flat", warmup, valid, long_row=long_row)


def _rows(launcher: str, valid: str = "true") -> RowsPlan:
    return RowsPlan("rows", "0", valid, launcher)


_P = "static_cast<long long>(period)"
_P_OK = "period > 0"
_LAG = ("momentum", "roc", "rocp", "rocr", "rocr100", "change")
_WINDOW = ("willr", "midprice", "var", "linearreg", "linearreg_slope",
           "linearreg_intercept", "linearreg_angle", "tsf", "minmax", "cci",
           "avgdev", "skew", "kurtosis", "rolling_mean", "rolling_max",
           "rolling_min", "rolling_std")
_SMA_DIRECT = "tacuda::detail::sma_direct_window"
_D = "tacuda::detail::"


def _candle_warmup(cls: str) -> str:
    """First valid index of a candlestick kernel: its `lookback` constant."""
    src = (ROOT / "src" / "indicators" / f"{cls}.cu").read_text(encoding="utf-8")
    found = re.findall(r"const int lookback = (\d+);", src)
    if len(found) == 1:
        return found[0]
    if not found and "if (idx > 0)" in src:  # engulfing: the previous candle only
        return "1"
    raise SystemExit(f"cannot derive the warm-up of candlestick kernel {cls}.cu")


ROWS = {
    **{n: _flat("0") for n in ("typprice", "wclprice", "medprice", "avgprice", "bop")},
    "trange": _flat("1"),
    **{n: _flat(_P, _P_OK) for n in _LAG},
    **{n: _flat(_P + " - 1", _P_OK) for n in _WINDOW},
    "midpoint": _rows(_D + "midpoint_rows(in[0], out[0], seriesLen, numSeries, period, s)", _P_OK),
    "aroon": _flat(_P, _P_OK),
    "aroonosc": _flat(_P, _P_OK),
    "ultosc": _flat("std::max({period1, period2, period3})",
                    "period1 > 0 && period2 > 0 && period3 > 0"),
    "stochf": _flat("static_cast<long long>(kPeriod) + dPeriod - 2",
                    "kPeriod > 0 && dPeriod > 0"),
    # SMA composites: their flattened call runs the SMA passes one after
    # another, without the per-series stream overlap.
    "trima": _flat(_P + " - 1",
                   f"period > 0 && {_SMA_DIRECT}((period + 1) / 2) && "
                   f"{_SMA_DIRECT}(period % 2 == 0 ? (period + 1) / 2 + 1 : (period + 1) / 2)",
                   long_row="1LL << 14"),
    "accbands": _flat(_P + " - 1", f"period > 0 && {_SMA_DIRECT}(period)",
                      long_row="1LL << 14"),
    "market_strength": _flat(_P, _P_OK),
    "trendscore": _flat("static_cast<long long>(lookback) + 10", "lookback >= 0"),
    "refdev": _flat("static_cast<long long>(smaPeriod) + sumPeriod - 2",
                    "smaPeriod > 0 && sumPeriod > 0"),
    "pivot_high": _flat("2LL * strength", "strength > 0"),
    "pivot_low": _flat("2LL * strength", "strength > 0"),
    **{name: _flat(_candle_warmup(cls)) for name, cls in CDL_PATTERNS},
    "sar": _rows(_D + "sar_rows(in[0], in[1], out[0], seriesLen, numSeries, "
                 "step, maxAcceleration, s)"),
    "sarext": _rows(_D + "sarext_rows(in[0], in[1], out[0], seriesLen, numSeries, "
                    "startValue, offsetOnReverse, accInitLong, accLong, accMaxLong, "
                    "accInitShort, accShort, accMaxShort, s)"),
    "sar_el": _rows(_D + "sar_el_rows(in[0], in[1], in[2], out[0], seriesLen, "
                    "numSeries, accFact, maxAccel, s)"),
    "maxindex": _rows(_D + "max_index_rows(in[0], out[0], seriesLen, numSeries, period, s)", _P_OK),
    "minindex": _rows(_D + "min_index_rows(in[0], out[0], seriesLen, numSeries, period, s)", _P_OK),
    "minmaxindex": _rows(_D + "min_max_index_rows(in[0], out[0], out[1], seriesLen, "
                         "numSeries, period, s)", _P_OK),
    **{name: _rows(_D + f"{name}_rows(in[0], out[0], seriesLen, numSeries, s)")
       for name in ("ht_dcperiod", "ht_dcphase", "ht_trendline", "ht_trendmode")},
    "ht_phasor": _rows(_D + "ht_phasor_rows(in[0], out[0], out[1], seriesLen, numSeries, s)"),
    "ht_sine": _rows(_D + "ht_sine_rows(in[0], out[0], out[1], seriesLen, numSeries, s)"),
    "mama": _rows(_D + "mama_rows(in[0], out[0], out[1], seriesLen, numSeries, "
                  "fastLimit, slowLimit, s)"),
    "macdfix": _rows(_D + "macdfix_rows(in[0], out[0], out[1], out[2], seriesLen, "
                     "numSeries, signalPeriod, s)", "signalPeriod > 0"),
}

SHORT_ROWS = {
    **{n.lower(): (n, 'period > 0', 'period')
       for n in ('Dema', 'Tema', 'Trix', 'Adx', 'Dx')},
    **{n: (f, 'period > 0', 'period') for n, f in
       [('plus_di', 'PlusDi'), ('minus_di', 'MinusDi')]},
    **{n: (f, f'{_D}short_float_wilder(seriesLen, period)', 'period')
       for n, f in [('cmo', 'Cmo'), ('plus_dm', 'PlusDm'), ('minus_dm', 'MinusDm')]},
    **{n.lower(): (n, f'period > 0 && period <= {_D}kShortWindow', 'period')
       for n in ('Adxr', 'Beta', 'Correl')},
    't3': ('T3', 'period >= 2', 'period, 0, 0, vFactor'),
    'kama': ('Kama', f'period > 0 && period <= {_D}kShortWindow && fastPeriod > 0 && slowPeriod > 0',
             'period, fastPeriod, slowPeriod'),
    **{n: (f, 'fastPeriod > 0 && slowPeriod > 0', 'fastPeriod, slowPeriod')
       for n, f in [('apo', 'Apo'), ('ppo', 'Ppo'), ('macd_line', 'MacdLine')]},
    'pvo': ('Pvo', 'fastPeriod > 0 && fastPeriod < slowPeriod', 'fastPeriod, slowPeriod'),
    'ma': ('Ma', f'period > 0 && period <= {_D}kShortWindow && maType >= 0 && maType <= 8 && '
                 'maType != 7 && (maType != 8 || period >= 2)', 'period, maType'),
    'stochrsi': ('StochRsi', f'rsiPeriod > 0 && kPeriod > 0 && kPeriod <= {_D}kShortWindow && '
                            f'dPeriod > 0 && dPeriod <= {_D}kShortWindow', 'rsiPeriod, kPeriod, dPeriod'),
    'imi': ('Imi', f'period >= 2 && period <= {_D}kShortWindow', 'period'),
    'hpi': ('Hpi', 'factor > 0.0f && factor <= 1.0f', '0, 0, 0, factor'),
    'xaverage': ('Xaverage', 'period > 0 && period <= seriesLen', 'period'),
    **{n.lower(): (n, 'true', '') for n in ('Nvi', 'Pvi', 'Asi', 'Obtr', 'Wobv')},
}

_unknown_rows = sorted((set(ROWS) | set(SHORT_ROWS)) - {f.name for f in FAMILIES})
if _unknown_rows:
    raise SystemExit(f"row plans for unknown batch families: {_unknown_rows}")
_double_plans = sorted(f.name for f in FAMILIES if f.name in ROWS and (f.fused or f.reference))
if _double_plans:
    raise SystemExit(f"families with both a row plan and a fused/reference path: {_double_plans}")

# JS object keys for named (multi) outputs.
JS_OUT_KEYS = {
    "aroon_downs": "down", "aroon_ups": "up",
    "fast_ks": "k", "fast_ds": "d",
    "macds": "macd", "signals": "signal", "hists": "hist",
    "uppers": "upper", "middles": "middle", "lowers": "lower",
    "mamas": "mama", "famas": "fama",
    "mins": "min", "maxs": "max",
    "minidxs": "minIdx", "maxidxs": "maxIdx",
    "inphases": "inPhase", "quadratures": "quadrature",
    "sines": "sine", "leadsines": "leadSine",
}

CS_OUT_KEYS = {
    "aroon_downs": "Down", "aroon_ups": "Up",
    "fast_ks": "K", "fast_ds": "D",
    "macds": "Macd", "signals": "Signal", "hists": "Hist",
    "uppers": "Upper", "middles": "Middle", "lowers": "Lower",
    "mamas": "Mama", "famas": "Fama",
    "mins": "Min", "maxs": "Max",
    "minidxs": "MinIdx", "maxidxs": "MaxIdx",
    "inphases": "InPhase", "quadratures": "Quadrature",
    "sines": "Sine", "leadsines": "LeadSine",
}

# Singular parameter names for generated ct_*_d declarations.
SINGULAR = {
    "inputs": "input", "outputs": "output", "opens": "open", "highs": "high",
    "lows": "low", "closes": "close", "volumes": "volume", "xs": "x",
    "ys": "y", "values": "values", "periods": "periods",
    "aroon_downs": "aroon_down", "aroon_ups": "aroon_up",
    "fast_ks": "fast_k", "fast_ds": "fast_d",
    "macds": "macd", "signals": "signal", "hists": "hist",
    "mamas": "mama", "famas": "fama", "mins": "min", "maxs": "max",
    "minidxs": "minidx", "maxidxs": "maxidx",
    "inphases": "inphase", "quadratures": "quadrature",
    "sines": "sine", "leadsines": "leadsine",
}


def singular(name: str) -> str:
    return SINGULAR.get(name, name[:-1] if name.endswith("s") else name)


def camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def pascal(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in name.split("_"))


def cs_input_name(logical: str) -> str:
    return camel(logical)


for _family in FAMILIES:
    for _output in _family.outputs:
        JS_OUT_KEYS.setdefault(_output, camel(_output))
        CS_OUT_KEYS.setdefault(_output, pascal(_output))


def output_dtypes(family):
    return family.output_dtypes or ('float',) * len(family.outputs)


def render_reference(family, variant):
    device = variant.startswith('device')
    batch = variant.endswith('batch')
    prefix = 'd_' if device else 'host_'
    suffix = '_batch_d' if variant=='device_batch' else '_batch' if batch else '_d' if device else ''
    fn = 'ct_' + family.name + suffix
    params = decl_params(family, device) if batch else device_decl_params(family) if device else host_single_params(family)
    params = params.replace(' CTAPI_STREAM_DEFAULT','')
    head = f'ctStatus_t {fn}('
    ins = ', '.join(prefix + (n if batch else singular(n)) for n in family.inputs)
    outs = ', '.join(prefix + (n if batch else singular(n)) for n in family.outputs)
    packed = ', '.join(f'static_cast<double>({n})' for _,n in family.scalars)
    shape = 'seriesLen, numSeries' if batch else 'size, 1'
    runner = 'run_device' if device else 'run_host'
    return [f'{head}{wrap_decl(params,len(head))}) {{',
            f'    const float* ins[] = {{{ins}}};',
            f'    void* outs[] = {{{outs}}};',
            f'    return tacuda::talib08::{runner}(tacuda::talib08::Function::{family.reference},',
            f'        ins, outs, {shape}, tacuda::talib08::Params{{{{{packed}}}}}, stream);',
            '}', '']


# ---------------------------------------------------------------------------
# C header declarations
# ---------------------------------------------------------------------------

def decl_params(fam: Family, device: bool) -> str:
    prefix = "d_" if device else "host_"
    parts = [f"const float *{prefix}{n}" for n in fam.inputs]
    parts += [f"{t} *{prefix}{n}" for t,n in zip(output_dtypes(fam), fam.outputs)]
    parts += ["int seriesLen", "int numSeries"]
    parts += [f"{t} {n}" for t, n in fam.scalars]
    parts += ["int maxConcurrency", "cudaStream_t stream CTAPI_STREAM_DEFAULT"]
    return ", ".join(parts)


def host_single_params(fam: Family) -> str:
    parts = [f"const float *host_{singular(n)}" for n in fam.inputs]
    parts += [f"{t} *host_{singular(n)}" for t,n in zip(output_dtypes(fam), fam.outputs)]
    parts += ["int size"]
    parts += [f"{t} {n}" for t, n in fam.scalars]
    parts += ["cudaStream_t stream CTAPI_STREAM_DEFAULT"]
    return ", ".join(parts)


def device_decl_params(df: DeviceFamily) -> str:
    parts = [f"const float* d_{singular(n)}" for n in df.inputs]
    parts += [f"{t}* d_{singular(n)}" for t,n in zip(output_dtypes(df), df.outputs)]
    parts += ["int size"]
    parts += [f"{t} {n}" for t, n in df.scalars]
    parts += ["cudaStream_t stream CTAPI_STREAM_DEFAULT"]
    return ", ".join(parts)


def wrap_decl(text: str, indent: int) -> str:
    """Wrap a long declaration at commas to stay near 80 columns."""
    words = text.split(", ")
    lines, cur = [], ""
    for w in words:
        cand = w if not cur else cur + ", " + w
        if len(cand) + indent > 78 and cur:
            lines.append(cur + ",")
            cur = w
        else:
            cur = cand
    lines.append(cur)
    pad = " " * indent
    return ("\n" + pad).join(lines)


def render_header() -> str:
    out = [HDR_BEGIN,
           "/* Generated batch entry points: same matrix layout and scheduling as the",
           " * hand-written ct_*_batch functions above. Club-pack families also get a",
           " * plain single-series host entry point. Do not edit by hand. */"]
    for fam in FAMILIES:
        if fam.host_single:
            fn = f"ct_{fam.name}"
            head = f"CTAPI_EXPORT ctStatus_t {fn}("
            out.append(f"{head}{wrap_decl(host_single_params(fam), len(head))});")
        for device in (False, True):
            fn = f"ct_{fam.name}_batch" + ("_d" if device else "")
            head = f"CTAPI_EXPORT ctStatus_t {fn}("
            body = wrap_decl(decl_params(fam, device), len(head))
            out.append(f"{head}{body});")
    out.append(HDR_END)
    return "\n".join(out)


def render_device_header() -> str:
    out = [DEV_HDR_BEGIN,
           "/* Generated device entry points (thin wrappers over the CUDA indicator",
           " * classes; see scripts/generate_batch_api.py). Do not edit by hand. */"]
    for df in DEVICE_FAMILIES:
        fn = f"ct_{df.name}_d"
        head = f"CTAPI_EXPORT ctStatus_t {fn}("
        out.append(f"{head}{wrap_decl(device_decl_params(df), len(head))});")
    out.append(DEV_HDR_END)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CUDA implementations (.inc files)
# ---------------------------------------------------------------------------

def render_device_includes() -> str:
    headers = sorted({df.header or f"{df.cls}.h" for df in DEVICE_FAMILIES if not df.reference})
    out = [
        "// Auto-generated by scripts/generate_batch_api.py. Do not edit.",
        "// Indicator class headers used by api_device_generated.inc.",
        "#include <utils/DeviceBufferPool.h>",
        "#include <utils/Talib08.h>",
    ]
    out += [f"#include <indicators/{h}>" for h in headers]
    return "\n".join(out) + "\n"


def render_device_impl() -> str:
    out = [
        "// Auto-generated by scripts/generate_batch_api.py. Do not edit.",
        "// Included from api_device.cu inside extern \"C\"; uses",
        "// classify_exception from that translation unit.",
        "",
    ]
    for df in DEVICE_FAMILIES:
        if df.reference:
            out += render_reference(df, 'device')
            continue
        fn = f"ct_{df.name}_d"
        params = device_decl_params(df).replace(" CTAPI_STREAM_DEFAULT", "")
        head = f"ctStatus_t {fn}("
        in_names = [f"d_{singular(n)}" for n in df.inputs]
        out_names = [f"d_{singular(n)}" for n in df.outputs]
        null_check = " || !".join(in_names + out_names)
        out.append(f"{head}{wrap_decl(params, len(head))}) {{")
        out.append("    tacuda::detail::clear_last_error();")
        out.append(f"    if (!{null_check} || size <= 0) {{")
        out.append(f"        tacuda::detail::set_last_error(\"{fn}: null pointer or invalid size\");")
        out.append("        return CT_STATUS_INVALID_ARGUMENT;")
        out.append("    }")
        out.append("    try {")
        ctor = f"({df.ctor})" if df.ctor else ""
        out.append(f"        tacuda::{df.cls} ind{ctor};")
        if df.call == "direct":
            args = ", ".join(in_names + out_names)
            out.append(f"        ind.calculate({args}, size, stream);")
        else:
            lines = df.stacked_lines
            out.append("        auto tmp = acquireDeviceBufferAsync<float>(")
            out.append(f"            static_cast<size_t>(size) * {lines}, stream);")
            out.append(f"        ind.calculate({in_names[0]}, tmp.get(), size, stream);")
            copies = out_names if df.call == "stacked" else out_names[:1]
            for k, dst in enumerate(copies):
                src = "tmp.get()" if k == 0 else f"tmp.get() + {k} * static_cast<size_t>(size)"
                out.append(f"        if (cudaMemcpyAsync({dst}, {src},")
                out.append("                static_cast<size_t>(size) * sizeof(float),")
                out.append("                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {")
                out.append(f"            tacuda::detail::set_last_error(\"{fn}: device copy failed\");")
                out.append("            return CT_STATUS_COPY_FAILED;")
                out.append("        }")
        out.append("        return CT_STATUS_SUCCESS;")
        out.append("    } catch (...) {")
        out.append("        return classify_exception();")
        out.append("    }")
        out.append("}")
        out.append("")
    return "\n".join(out)


def render_impl() -> str:
    out = [
        "// Auto-generated by scripts/generate_batch_api.py. Do not edit.",
        "// Included from api_batch.cu inside extern \"C\"; uses the generic",
        "// run_host_batch / run_device_batch runners from that translation unit.",
        "",
    ]
    for fam in FAMILIES:
        if fam.reference:
            if fam.host_single:
                out += render_reference(fam, 'host')
            out += render_reference(fam, 'host_batch')
            out += render_reference(fam, 'device_batch')
            continue
        n_in, n_out = len(fam.inputs), len(fam.outputs)
        scalar_args = "".join(f", {n}" for _, n in fam.scalars)
        for device in (False, True):
            prefix = "d_" if device else "host_"
            fn = f"ct_{fam.name}_batch" + ("_d" if device else "")
            runner = "run_device_batch" if device else "run_host_batch"
            params = decl_params(fam, device).replace(" CTAPI_STREAM_DEFAULT", "")
            head = f"ctStatus_t {fn}("
            in_list = ", ".join(prefix + n for n in fam.inputs)
            out_list = ", ".join(prefix + n for n in fam.outputs)
            call_ins = ", ".join(f"in[{i}]" for i in range(n_in))
            call_outs = ", ".join(f"out[{i}]" for i in range(n_out))
            out.append(f"{head}{wrap_decl(params, len(head))}) {{")
            extended = {'ma':('MA','maType >= 9'),
                        'mavp':('MAVP','maType >= 9 || maType == 3 || maType == 4 || maType == 7 || maType == 8'),
                        'macdext':('MACDEXT','!((fastType == 1 && slowType == 1 && signalType == 1) || ((fastType == 0 || fastType == 2 || fastType == 5) && (slowType == 0 || slowType == 2 || slowType == 5) && (signalType == 0 || signalType == 2 || signalType == 5)))')}
            if fam.name in extended:
                operation, condition = extended[fam.name]
                out.append(f'    if ({condition}) {{')
                body = render_reference(dataclasses.replace(fam,reference=operation), 'device_batch' if device else 'host_batch')
                out.extend('    '+line for line in body[1:-2])
                out.append('    }')
            out.append(f"    const float* ins[{n_in}] = {{{in_list}}};")
            out.append(f"    float* outs[{n_out}] = {{{out_list}}};")
            out.append(f"    return {runner}(")
            out.append(f"        ins, {n_in}, outs, {n_out}, seriesLen, numSeries, maxConcurrency, stream,")
            out.append("        [&](const float* const* in, float* const* out, cudaStream_t s) {")
            out.append(f"            return {fam.d_func}({call_ins}, {call_outs}, seriesLen{scalar_args}, s);")
            plan = ROWS.get(fam.name)
            if fam.fused:
                out.append(f"        }}, fused_call(FusedFamily::{fam.fused}{scalar_args}));")
            elif plan and plan.kind == "flat":
                long_row = f", {plan.long_row}" if plan.long_row else ""
                out.append(f"        }}, FusedCall{{}}, rows_flat({plan.valid}, {plan.warmup}{long_row}),")
                out.append("        [&](const float* const* in, float* const* out, int n, cudaStream_t s) {")
                out.append(f"            return {fam.d_func}({call_ins}, {call_outs}, n{scalar_args}, s);")
                out.append("        });")
            elif plan:
                out.append(f"        }}, FusedCall{{}}, rows_kernel({plan.valid}),")
                out.append("        [&](const float* const* in, float* const* out, int, cudaStream_t s) {")
                out.append(f"            {plan.launcher};")
                out.append("            return CT_STATUS_SUCCESS;")
                out.append("        });")
            elif fam.name in SHORT_ROWS:
                family, valid, options = SHORT_ROWS[fam.name]
                out.append(f"        }}, FusedCall{{}}, rows_kernel(({valid}) && {_D}short_batch_shape(seriesLen, numSeries)),")
                out.append("        [&](const float* const* in, float* const* out, int, cudaStream_t s) {")
                out.append(f"            {_D}launch_short_rows({_D}ShortFamily::{family}, in, {n_in}, out, {n_out},"
                           f" seriesLen, numSeries, {{{options}}}, s);")
                out.append("            return CT_STATUS_SUCCESS;")
                out.append("        });")
            else:
                out.append("        });")
            out.append("}")
            out.append("")
        if fam.host_single:
            # Plain single-series host entry point: the host batch runner with
            # numSeries = 1 (uploads, kernel, download, sync).
            fn = f"ct_{fam.name}"
            params = host_single_params(fam).replace(" CTAPI_STREAM_DEFAULT", "")
            head = f"ctStatus_t {fn}("
            in_list = ", ".join(f"host_{singular(n)}" for n in fam.inputs)
            out_list = ", ".join(f"host_{singular(n)}" for n in fam.outputs)
            call_ins = ", ".join(f"in[{i}]" for i in range(n_in))
            call_outs = ", ".join(f"out[{i}]" for i in range(n_out))
            out.append(f"{head}{wrap_decl(params, len(head))}) {{")
            out.append(f"    const float* ins[{n_in}] = {{{in_list}}};")
            out.append(f"    float* outs[{n_out}] = {{{out_list}}};")
            out.append("    return run_host_batch(")
            out.append(f"        ins, {n_in}, outs, {n_out}, size, 1, 1, stream,")
            out.append("        [&](const float* const* in, float* const* out, cudaStream_t s) {")
            out.append(f"            return {fam.d_func}({call_ins}, {call_outs}, size{scalar_args}, s);")
            out.append("        });")
            out.append("}")
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Node.js bindings
# ---------------------------------------------------------------------------

def node_sig(fam: Family, device: bool) -> str:
    fn = f"ct_{fam.name}_batch" + ("_d" if device else "")
    if device:
        parts = [f"const void *d_{n}" for n in fam.inputs]
        parts += [f"void *d_{n}" for n in fam.outputs]
    else:
        parts = [f"const float *{n}" for n in fam.inputs]
        parts += [f"{t} *{n}" for t,n in zip(output_dtypes(fam), fam.outputs)]
    parts += ["int seriesLen", "int numSeries"]
    parts += [f"{t} {n}" for t, n in fam.scalars]
    parts += ["int maxConcurrency", "void *stream"]
    return f"int {fn}({', '.join(parts)})"


def render_node() -> str:
    out = [
        "// Auto-generated by scripts/generate_batch_api.py. Do not edit.",
        "// Registers generated ct_*_batch[_d] families; wired up from index.js.",
        "'use strict';",
        "",
        "module.exports = function registerGeneratedBatch(ctx) {",
        "    const { lib, call, deviceCall, toBatchMatrix, splitBatchRows } = ctx;",
        "    const host = {};",
        "    const device = {};",
        "    const ohlcvColumns = {};",
        "",
    ]
    for fam in FAMILIES:
        for device in (False, True):
            fn = f"ct_{fam.name}_batch" + ("_d" if device else "")
            out.append(f"    const {fn} = lib.func('{node_sig(fam, device)}');")
    out.append("")

    for fam in FAMILIES:
        js = camel(fam.name) + "Batch"
        fn = f"ct_{fam.name}_batch"
        scalar_names = [n for _, n in fam.scalars]
        args = [camel(n) for n in fam.inputs] + scalar_names + ["opts = {}"]
        out.append(f"    host.{js} = function {js}({', '.join(args)}) {{")
        first = camel(fam.inputs[0])
        if len(fam.inputs) == 1:
            out.append(f"        const {{ matrix, seriesLen, numSeries }} = toBatchMatrix({first}, opts.seriesLen);")
            mats = ["matrix"]
        else:
            mats = []
            for n in fam.inputs:
                v = f"m_{camel(n)}"
                out.append(f"        const {v} = toBatchMatrix({camel(n)}, opts.seriesLen);")
                mats.append(f"{v}.matrix")
            base = f"m_{first}"
            # Both dimensions must match: a 2x10 and a 4x5 matrix have the
            # same element count but are different shapes.
            for n in fam.inputs[1:]:
                out.append(f"        if (m_{camel(n)}.seriesLen !== {base}.seriesLen || m_{camel(n)}.numSeries !== {base}.numSeries) throw new RangeError('all input matrices must have the same shape (seriesLen x numSeries)');")
            out.append(f"        const {{ seriesLen, numSeries }} = {base};")
        outs = []
        for t,n in zip(output_dtypes(fam), fam.outputs):
            v = "out" if len(fam.outputs) == 1 else f"o_{camel(n)}"
            src = "matrix.length" if len(fam.inputs) == 1 else f"m_{first}.matrix.length"
            out.append(f"        const {v} = new {'Int32Array' if t=='int' else 'Float32Array'}({src});")
            outs.append(v)
        call_args = mats + outs + ["seriesLen", "numSeries"] + scalar_names + ["opts.maxConcurrency ?? 0"]
        out.append(f"        call({fn}, '{fn}', {', '.join(call_args)});")
        if len(fam.outputs) == 1:
            out.append("        return splitBatchRows(out, seriesLen, numSeries);")
        else:
            fields = ", ".join(
                f"{JS_OUT_KEYS[n]}: splitBatchRows(o_{camel(n)}, seriesLen, numSeries)"
                for n in fam.outputs)
            out.append(f"        return {{ {fields} }};")
        out.append("    };")

        # device wrapper: raw pointers, C argument order
        fn_d = fn + "_d"
        d_args = ([f"d{pascal(n)}" for n in fam.inputs]
                  + [f"d{pascal(n)}" for n in fam.outputs]
                  + ["seriesLen", "numSeries"] + scalar_names
                  + ["maxConcurrency = 0"])
        pass_args = [a.split(" = ")[0] for a in d_args]
        out.append(f"    device.{js} = function ({', '.join(d_args)}) {{")
        out.append(f"        deviceCall({fn_d}, '{fn_d}', {', '.join(pass_args)});")
        out.append("    };")

        # Families whose inputs are all OHLCV columns can also accept an
        # array of Ohlcv containers; index.js wraps them using this map.
        col_map = {"opens": "open", "highs": "high", "lows": "low",
                   "closes": "close", "volumes": "volume"}
        if all(n in col_map for n in fam.inputs):
            cols = ", ".join(f"'{col_map[n]}'" for n in fam.inputs)
            out.append(f"    ohlcvColumns.{js} = [{cols}];")
        out.append("")

    # Typed single-series host wrappers for the new reference-backed families.
    for fam in FAMILIES:
        if not fam.reference or not fam.host_single:
            continue
        fn = f'ct_{fam.name}'
        parts = [f'const float *{singular(n)}' for n in fam.inputs]
        parts += [f'{t} *{singular(n)}' for t,n in zip(output_dtypes(fam),fam.outputs)]
        parts += ['int size'] + [f'{t} {n}' for t,n in fam.scalars] + ['void *stream']
        out.append(f"    const {fn} = lib.func('int {fn}({', '.join(parts)})');")
        js = camel(fam.name)
        args = [camel(singular(n)) for n in fam.inputs] + [n for _,n in fam.scalars]
        out.append(f"    host.{js} = function ({', '.join(args)}) {{")
        mats = []
        for n in fam.inputs:
            var = 'm_' + camel(n)
            out.append(f'        const {var} = toBatchMatrix([{camel(singular(n))}]);')
            mats.append(var)
        for var in mats[1:]:
            out.append(f"        if ({var}.seriesLen !== {mats[0]}.seriesLen) throw new RangeError('all inputs must have the same length');")
        outs = []
        for i,t in enumerate(output_dtypes(fam)):
            var = f'output{i}'
            out.append(f'        const {var} = new {"Int32Array" if t=="int" else "Float32Array"}({mats[0]}.seriesLen);')
            outs.append(var)
        native = [v+'.matrix' for v in mats] + outs + [mats[0]+'.seriesLen'] + [n for _,n in fam.scalars]
        out.append(f"        call({fn}, '{fn}', {', '.join(native)});")
        if len(outs)==1:
            out.append('        return output0;')
        else:
            out.append('        return {'+', '.join(JS_OUT_KEYS[n]+': '+v for n,v in zip(fam.outputs,outs))+'};')
        out.append('    };')

    # Single-series ct_*_d wrappers over the generated device surface (raw
    # GPU pointers). Completes Node's coverage of the C ABI; index.js merges
    # these into `device` without overriding its hand-written wrappers.
    out.append("    // Generated single-series ct_*_d device wrappers (raw GPU pointers).")
    for df in DEVICE_FAMILIES:
        fn = f"ct_{df.name}_d"
        parts = [f"const void *d_{singular(n)}" for n in df.inputs]
        parts += [f"void *d_{singular(n)}" for n in df.outputs]
        parts += ["int size"]
        parts += [f"{t} {n}" for t, n in df.scalars]
        parts += ["void *stream"]
        out.append(f"    const {fn} = lib.func('int {fn}({', '.join(parts)})');")
        js = camel(df.name)
        d_args = ([f"d{pascal(singular(n))}" for n in df.inputs]
                  + [f"d{pascal(singular(n))}" for n in df.outputs]
                  + ["size"] + [n for _, n in df.scalars])
        out.append(f"    device.{js} = function ({', '.join(d_args)}) {{")
        out.append(f"        deviceCall({fn}, '{fn}', {', '.join(d_args)});")
        out.append("    };")
    out.append("")

    out.append("    return { host, device, ohlcvColumns };")
    out.append("};")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# C# friendly wrappers (partial class TacudaBatch)
# ---------------------------------------------------------------------------

def render_csharp() -> str:
    out = [
        "// Auto-generated by scripts/generate_batch_api.py. Do not edit.",
        "using System;",
        "",
        "namespace Tacuda.Bindings;",
        "",
        "public static partial class TacudaBatch",
        "{",
    ]
    for fam in FAMILIES:
        method = pascal(fam.name)
        c_fn = f"ct_{fam.name}_batch"
        scalar_params = ", ".join(f"{t} {n}" for t, n in fam.scalars)
        scalar_args = "".join(f", {n}" for _, n in fam.scalars)
        in_params = ", ".join(f"float[][] {cs_input_name(n)}" for n in fam.inputs)
        sig_scalars = f", {scalar_params}" if scalar_params else ""

        if len(fam.outputs) == 1:
            ret = output_dtypes(fam)[0] + "[][]"
        else:
            ret = "(" + ", ".join(
                f"{t}[][] {CS_OUT_KEYS[n]}" for t,n in zip(output_dtypes(fam), fam.outputs)) + ")"

        out.append(f"    public static {ret} {method}({in_params}{sig_scalars}, int maxConcurrency = 0)")
        out.append("    {")
        first = cs_input_name(fam.inputs[0])
        mat_vars = []
        for n in fam.inputs:
            v = f"m{pascal(n)}"
            out.append(f"        var (len{'' if n == fam.inputs[0] else pascal(n)}, count{'' if n == fam.inputs[0] else pascal(n)}, {v}) = FlattenRows({cs_input_name(n)});")
            mat_vars.append(v)
        for n in fam.inputs[1:]:
            out.append(f"        if (len{pascal(n)} != len || count{pascal(n)} != count)")
            out.append("        {")
            out.append("            throw new ArgumentException(\"all input matrices must have the same shape\");")
            out.append("        }")
        out_vars = []
        for t,n in zip(output_dtypes(fam), fam.outputs):
            v = "output" if len(fam.outputs) == 1 else f"o{pascal(n)}"
            out.append(f"        var {v} = new {t}[len * count];")
            out_vars.append(v)
        call_args = ", ".join(mat_vars + out_vars) + ", len, count" + scalar_args + ", maxConcurrency"
        out.append("        NativeMethods.ThrowOnError(")
        out.append(f"            NativeMethods.{c_fn}({call_args}),")
        out.append(f"            nameof(NativeMethods.{c_fn}));")
        if len(fam.outputs) == 1:
            out.append("        return Split(output, len, count);")
        else:
            tup = ", ".join(f"Split(o{pascal(n)}, len, count)" for n in fam.outputs)
            out.append(f"        return ({tup});")
        out.append("    }")
        out.append("")
    out.append("}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# File plumbing
# ---------------------------------------------------------------------------

# Project signature stamped on every generated file (keep in sync with
# scripts/add_source_headers.py, which stamps the hand-written sources).
_SIGN_LINES = (
    "TaCuda | GPU-accelerated technical analysis | https://tacuda.com",
    "Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>",
    "SPDX-License-Identifier: Apache-2.0",
)


def signature(style: str) -> str:
    if style == "c":
        bar = "/* " + "=" * 72 + " */"
        body = [f"/*  {ln:<71} */" for ln in _SIGN_LINES]
    else:  # "//" line comments (C#, JS)
        bar = "// " + "=" * 74
        body = [f"//  {ln}" for ln in _SIGN_LINES]
    return "\n".join([bar, *body, bar]) + "\n"


def render_region(path: Path, begin: str, end: str, payload: str,
                  anchor: str | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    if begin in text:
        pattern = re.compile(re.escape(begin) + ".*?" + re.escape(end), re.DOTALL)
        text = pattern.sub(payload, text)
    else:
        if anchor is None:
            anchor = "#ifdef __cplusplus\n} // extern \"C\"\n#endif"
        if anchor not in text:
            raise SystemExit(f"anchor not found in {path}")
        text = text.replace(anchor, payload + "\n\n" + anchor)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check generated files without writing')
    args = parser.parse_args()
    n_batch = 2 * len(FAMILIES)
    n_dev = len(DEVICE_FAMILIES)
    n_host = sum(1 for f in FAMILIES if f.host_single)
    payloads = {
        HEADER: render_region(HEADER, HDR_BEGIN, HDR_END, render_header()),
        HEADER_DEVICE: render_region(HEADER_DEVICE, DEV_HDR_BEGIN, DEV_HDR_END,
                                     render_device_header(), anchor="#ifdef __cplusplus\n}\n#endif"),
        IMPL_INC: signature('c') + render_impl(), DEVICE_IMPL_INC: signature('c') + render_device_impl(),
        DEVICE_INCLUDES_INC: signature('c') + render_device_includes(), NODE_OUT: signature('//') + render_node(),
        CSHARP_OUT: signature('//') + render_csharp(),
    }
    stale = [p for p, text in payloads.items() if not p.exists() or p.read_text(encoding='utf-8') != text]
    if args.check:
        if stale:
            raise SystemExit('Generated batch files are stale: ' + ', '.join(str(p) for p in stale))
        print('Batch API generated files: current')
        return
    for path in stale:
        path.write_text(payloads[path], encoding='utf-8', newline='\n')
    print(f"generated {len(FAMILIES)} batch families ({n_batch} symbols), "
          f"{n_dev} device wrappers, {n_host} host single entries")
    print(f"  header       : {HEADER}")
    print(f"  device header: {HEADER_DEVICE}")
    print(f"  batch impl   : {IMPL_INC}")
    print(f"  device impl  : {DEVICE_IMPL_INC}")
    print(f"  device incs  : {DEVICE_INCLUDES_INC}")
    print(f"  nodejs       : {NODE_OUT}")
    print(f"  csharp       : {CSHARP_OUT}")
    print("next: python bindings/generate_bindings.py")


if __name__ == "__main__":
    main()
