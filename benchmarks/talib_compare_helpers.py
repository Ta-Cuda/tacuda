# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    group: str
    input_kind: str
    tacuda_fn: Callable[[Dict[str, np.ndarray]], Any]
    talib_fn: Callable[[Dict[str, np.ndarray]], Any]
    tolerance: float = 1e-4
    # Datasets excluded from the differential corpus for documented reasons
    # (each exclusion must carry a comment at the spec definition).
    skip_datasets: Tuple[str, ...] = ()
    # Discrete outputs (candlestick +-100/0 signals): `tolerance` is then the
    # maximum allowed FRACTION of flipped bars — float32 vs float64 rounding
    # can flip individual borderline candles, but a logic difference flips
    # orders of magnitude more.
    discrete: bool = False
    # Exact TA-Lib function name this spec covers; filled by _maybe_add so
    # the corpus coverage check needs no fuzzy name matching.
    talib_name: str = ""


# Only legitimate use left: aligning the ct_macd_line EXTENSION (which
# exposes earlier bars than TA-Lib's signal-aligned MACD outputs). Never use
# it to hide early finite values of a TA-Lib-equivalent function.
def slice_valid(arr: np.ndarray, start: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float32)
    if start < arr.size:
        out[start:] = arr[start:]
    return out


def to_talib_f64(arr: np.ndarray, baseline_dtype: str = "float32") -> np.ndarray:
    if baseline_dtype == "float16":
        return np.asarray(arr, dtype=np.float16).astype(np.float64)
    if baseline_dtype == "float64":
        return np.asarray(arr, dtype=np.float64)
    # Default legacy behavior: float32-quantized baseline.
    return np.asarray(arr, dtype=np.float32).astype(np.float64)


def generate_inputs(n: int, seed: int = 42) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    close = np.random.rand(n) * 100 + 100
    high = close + np.random.rand(n) * 5
    low = close - np.random.rand(n) * 5
    open_ = close + (np.random.rand(n) - 0.5) * 2
    volume = np.random.rand(n) * 1_000_000
    close2 = high
    periods = np.random.randint(2, 31, size=n).astype(np.float32)
    return {
        "close": close,
        "high": high,
        "low": low,
        "open": open_,
        "volume": volume,
        "close2": close2,
        "periods": periods,
    }


def _maybe_add(specs: List[IndicatorSpec], talib_obj: Any, talib_name: str, spec: IndicatorSpec) -> None:
    if hasattr(talib_obj, talib_name):
        specs.append(dataclasses.replace(spec, talib_name=talib_name))


def talib_index_output(arr: np.ndarray, lookback: int) -> np.ndarray:
    """TA-Lib integer outputs (MAXINDEX/MININDEX/...) come back from the
    Python wrapper as int arrays with 0 in the warm-up region; convert to
    float with NaN warm-up so masks compare like every other function."""
    out = np.asarray(arr, dtype=np.float64).copy()
    if lookback > 0:
        out[:lookback] = np.nan
    return out


def compare_discrete(lhs: np.ndarray, rhs: np.ndarray, max_flip_fraction: float) -> Tuple[bool, float, float, int]:
    """Oracle for discrete signal outputs (+-100/0): NaN masks must match
    exactly and (with the default max_flip_fraction of 0) EVERY signal must
    match. Reports the flip fraction over non-zero signals of either side —
    a far more honest denominator than all bars, most of which are zero.
    Returns (ok, signalFlipFraction, rawFlipFraction, nan_mismatch), shaped
    like compare_arrays for the corpus runner."""
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.shape != rhs.shape:
        return False, float("inf"), float("inf"), -1
    lhs_nan = np.isnan(lhs)
    rhs_nan = np.isnan(rhs)
    nan_mismatch = int(np.sum(lhs_nan != rhs_nan))
    valid = ~lhs_nan & ~rhs_nan
    if not np.any(valid):
        ok = nan_mismatch == 0
        return ok, 0.0, 0.0, nan_mismatch
    a = lhs[valid]
    b = rhs[valid]
    diff = a != b
    raw_flips = float(np.mean(diff))
    signal = (a != 0.0) | (b != 0.0)
    n_signal = int(np.sum(signal))
    signal_flips = (float(np.sum(diff & signal)) / n_signal) if n_signal else 0.0
    if max_flip_fraction == 0.0:
        ok = (not bool(diff.any())) and nan_mismatch == 0
    else:
        ok = (signal_flips <= max_flip_fraction) and nan_mismatch == 0
    return ok, signal_flips, raw_flips, nan_mismatch


def build_specs(tacuda: Any, talib: Any, dtype: str = "float32") -> List[IndicatorSpec]:
    specs: List[IndicatorSpec] = []
    close32 = lambda d: d["close"].astype(np.float32)
    high32 = lambda d: d["high"].astype(np.float32)
    low32 = lambda d: d["low"].astype(np.float32)
    is_fp16 = dtype == "float16"

    _maybe_add(specs, talib, "SMA", IndicatorSpec("SMA(14)", "overlap", "close",
        lambda d: tacuda.sma(close32(d), 14, dtype=dtype),
        lambda d: talib.SMA(to_talib_f64(d["close"], dtype), 14), tolerance=(8e-2 if is_fp16 else 1e-4)))
    _maybe_add(specs, talib, "EMA", IndicatorSpec("EMA(12)", "overlap", "close",
        lambda d: tacuda.ema(close32(d), 12, dtype=dtype),
        lambda d: talib.EMA(to_talib_f64(d["close"], dtype), 12), tolerance=(8e-2 if is_fp16 else 1e-4)))
    _maybe_add(specs, talib, "WMA", IndicatorSpec("WMA(10)", "overlap", "close",
        lambda d: tacuda.wma(d["close"].astype(np.float32), 10),
        lambda d: talib.WMA(to_talib_f64(d["close"]), 10)))
    _maybe_add(specs, talib, "DEMA", IndicatorSpec("DEMA(14)", "overlap", "close",
        lambda d: tacuda.dema(d["close"].astype(np.float32), 14),
        lambda d: talib.DEMA(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "TEMA", IndicatorSpec("TEMA(14)", "overlap", "close",
        lambda d: tacuda.tema(d["close"].astype(np.float32), 14),
        lambda d: talib.TEMA(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "TRIMA", IndicatorSpec("TRIMA(14)", "overlap", "close",
        lambda d: tacuda.trima(d["close"].astype(np.float32), 14),
        lambda d: talib.TRIMA(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "KAMA", IndicatorSpec("KAMA(14)", "overlap", "close",
        lambda d: tacuda.kama(d["close"].astype(np.float32), 14, 2, 30),
        lambda d: talib.KAMA(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "T3", IndicatorSpec("T3(14)", "overlap", "close",
        lambda d: tacuda.t3(d["close"].astype(np.float32), 14, 0.7),
        lambda d: talib.T3(to_talib_f64(d["close"]), 14, 0.7), tolerance=2e-4))
    _maybe_add(specs, talib, "MAVP", IndicatorSpec("MAVP(2..30)", "overlap", "close_periods",
        # Warm-up is natively TA-Lib aligned (maxPeriod-1); per-bar EMA in
        # float32 needs a relaxed scaled tolerance.
        lambda d: tacuda.mavp(d["close"].astype(np.float32), d["periods"], 2, 30, 1),
        lambda d: talib.MAVP(to_talib_f64(d["close"]), to_talib_f64(d["periods"]), 2, 30, matype=1), tolerance=5e-3))
    _maybe_add(specs, talib, "MAMA", IndicatorSpec("MAMA(0.5,0.05)", "overlap", "close",
        lambda d: tacuda.mama(d["close"].astype(np.float32), 0.5, 0.05),
        # KNOWN DEVIATION: the float32 Hilbert phase accumulator can lock
        # onto a different adaptive period than TA-Lib's double math, which
        # occasionally produces large point divergences. Tracked in
        # docs/IMPLEMENTATION_NOTES.md; scaled tolerance is deliberately lax.
        lambda d: talib.MAMA(to_talib_f64(d["close"]), 0.5, 0.05), tolerance=0.25))
    _maybe_add(specs, talib, "MIDPOINT", IndicatorSpec("MIDPOINT(14)", "overlap", "close",
        lambda d: tacuda.midpoint(d["close"].astype(np.float32), 14),
        lambda d: talib.MIDPOINT(to_talib_f64(d["close"]), 14)))
    # TA_MA dispatcher with two representative MA types.
    _maybe_add(specs, talib, "MA", IndicatorSpec("MA(14,SMA)", "overlap", "close",
        lambda d: tacuda.ma(d["close"].astype(np.float32), 14, 0),
        lambda d: talib.MA(to_talib_f64(d["close"]), 14, matype=0)))
    _maybe_add(specs, talib, "MA", IndicatorSpec("MA(14,EMA)", "overlap", "close",
        lambda d: tacuda.ma(d["close"].astype(np.float32), 14, 1),
        lambda d: talib.MA(to_talib_f64(d["close"]), 14, matype=1)))
    # Linear regression family + TSF.
    _maybe_add(specs, talib, "LINEARREG", IndicatorSpec("LINEARREG(14)", "stat", "close",
        lambda d: tacuda.linearreg(d["close"].astype(np.float32), 14),
        lambda d: talib.LINEARREG(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "LINEARREG_SLOPE", IndicatorSpec("LINEARREG_SLOPE(14)", "stat", "close",
        lambda d: tacuda.linearreg_slope(d["close"].astype(np.float32), 14),
        lambda d: talib.LINEARREG_SLOPE(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "LINEARREG_INTERCEPT", IndicatorSpec("LINEARREG_INTERCEPT(14)", "stat", "close",
        lambda d: tacuda.linearreg_intercept(d["close"].astype(np.float32), 14),
        lambda d: talib.LINEARREG_INTERCEPT(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "LINEARREG_ANGLE", IndicatorSpec("LINEARREG_ANGLE(14)", "stat", "close",
        # atan of the slope in degrees; slope is a difference of large
        # f32 sums, so allow a slightly relaxed scaled tolerance.
        lambda d: tacuda.linearreg_angle(d["close"].astype(np.float32), 14),
        lambda d: talib.LINEARREG_ANGLE(to_talib_f64(d["close"]), 14), tolerance=1e-3))
    _maybe_add(specs, talib, "TSF", IndicatorSpec("TSF(14)", "stat", "close",
        lambda d: tacuda.tsf(d["close"].astype(np.float32), 14),
        lambda d: talib.TSF(to_talib_f64(d["close"]), 14)))
    # IMI exists in TA-Lib 0.6.x; align end-of-window with leading warm-up.
    _maybe_add(specs, talib, "IMI", IndicatorSpec("IMI(14)", "momentum", "oc",
        lambda d: tacuda.imi(d["open"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.IMI(to_talib_f64(d["open"]), to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "MIDPRICE", IndicatorSpec("MIDPRICE(14)", "overlap", "hl",
        lambda d: tacuda.midprice(d["high"].astype(np.float32), d["low"].astype(np.float32), 14),
        lambda d: talib.MIDPRICE(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 14)))

    _maybe_add(specs, talib, "RSI", IndicatorSpec("RSI(14)", "momentum", "close",
        lambda d: tacuda.rsi(close32(d), 14, dtype=dtype),
        lambda d: talib.RSI(to_talib_f64(d["close"], dtype), 14), tolerance=(3e-2 if is_fp16 else 1e-4)))
    _maybe_add(specs, talib, "CMO", IndicatorSpec("CMO(14)", "momentum", "close",
        lambda d: tacuda.cmo(d["close"].astype(np.float32), 14),
        lambda d: talib.CMO(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "TRIX", IndicatorSpec("TRIX(14)", "momentum", "close",
        # Percentage change of a triple EMA: float32 error is amplified as
        # the smoothed line crosses zero, hence the relaxed scaled tolerance.
        lambda d: tacuda.trix(d["close"].astype(np.float32), 14),
        lambda d: talib.TRIX(to_talib_f64(d["close"]), 14), tolerance=5e-2))
    _maybe_add(specs, talib, "MOM", IndicatorSpec("Momentum(10)", "momentum", "close",
        lambda d: tacuda.momentum(d["close"].astype(np.float32), 10),
        lambda d: talib.MOM(to_talib_f64(d["close"]), 10)))
    _maybe_add(specs, talib, "ROC", IndicatorSpec("ROC(10)", "momentum", "close",
        lambda d: tacuda.roc(d["close"].astype(np.float32), 10),
        lambda d: talib.ROC(to_talib_f64(d["close"]), 10)))
    _maybe_add(specs, talib, "ROCP", IndicatorSpec("ROCP(10)", "momentum", "close",
        lambda d: tacuda.rocp(d["close"].astype(np.float32), 10),
        lambda d: talib.ROCP(to_talib_f64(d["close"]), 10)))
    _maybe_add(specs, talib, "ROCR", IndicatorSpec("ROCR(10)", "momentum", "close",
        lambda d: tacuda.rocr(d["close"].astype(np.float32), 10),
        lambda d: talib.ROCR(to_talib_f64(d["close"]), 10)))
    _maybe_add(specs, talib, "ROCR100", IndicatorSpec("ROCR100(10)", "momentum", "close",
        lambda d: tacuda.rocr100(d["close"].astype(np.float32), 10),
        lambda d: talib.ROCR100(to_talib_f64(d["close"]), 10)))
    _maybe_add(specs, talib, "APO", IndicatorSpec("APO(12,26)", "momentum", "close",
        lambda d: tacuda.apo(d["close"].astype(np.float32), 12, 26),
        lambda d: talib.APO(to_talib_f64(d["close"]), 12, 26, matype=1)))
    _maybe_add(specs, talib, "PPO", IndicatorSpec("PPO(12,26)", "momentum", "close",
        # Percentage of a slow EMA: float32 error is amplified without bound
        # as the denominator crosses zero, hence the relaxed scaled tolerance.
        lambda d: tacuda.ppo(d["close"].astype(np.float32), 12, 26),
        lambda d: talib.PPO(to_talib_f64(d["close"]), 12, 26, matype=1), tolerance=5e-3))
    _maybe_add(specs, talib, "MACD", IndicatorSpec("MACD(12,26,9)", "macd", "close",
        lambda d: tacuda.macd(close32(d), 12, 26, 9, 1, dtype=dtype),
        lambda d: talib.MACD(to_talib_f64(d["close"], dtype), 12, 26, 9), tolerance=(2e-2 if is_fp16 else 1e-4)))
    _maybe_add(specs, talib, "MACDFIX", IndicatorSpec("MACDFIX(9)", "macd", "close",
        lambda d: tacuda.macdfix(d["close"].astype(np.float32), 9),
        lambda d: talib.MACDFIX(to_talib_f64(d["close"]), 9)))
    _maybe_add(specs, talib, "MACD", IndicatorSpec("MACD_LINE(12,26)", "macd", "close",
        # ct_macd_line is a tacuda extension (macd line only): its first
        # value appears at slow-1, while TA-Lib's MACD outputs are aligned
        # to slow-1 + signal-1 (signal default 9). Mask ours to TA-Lib's
        # alignment for the comparison.
        lambda d: slice_valid(tacuda.macd_line(d["close"].astype(np.float32), 12, 26), (26 - 1) + (9 - 1)),
        lambda d: talib.MACD(to_talib_f64(d["close"]), 12, 26, 9)[0]))

    _maybe_add(specs, talib, "ATR", IndicatorSpec("ATR(14)", "volatility", "hlc",
        lambda d: tacuda.atr(high32(d), low32(d), close32(d), 14, 0.0, dtype=dtype),
        lambda d: talib.ATR(to_talib_f64(d["high"], dtype), to_talib_f64(d["low"], dtype), to_talib_f64(d["close"], dtype), 14), tolerance=(3e-2 if is_fp16 else 1e-4)))
    _maybe_add(specs, talib, "NATR", IndicatorSpec("NATR(14)", "volatility", "hlc",
        lambda d: tacuda.natr(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.NATR(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14),
        # TA-Lib bug (ta_NATR.c): when a later close is zero it writes the
        # guard's 0.0 into outReal[0] instead of outReal[outIdx], corrupting
        # the FIRST output bar. TaCuda emits 0 at the zero-close bar itself
        # (the obviously intended behavior), so the zero-crossing dataset is
        # excluded rather than reproducing the out-of-place write.
        skip_datasets=("zeros",)))
    _maybe_add(specs, talib, "TRANGE", IndicatorSpec("TRANGE", "volatility", "hlc",
        lambda d: tacuda.trange(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32)),
        lambda d: talib.TRANGE(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "BBANDS", IndicatorSpec("BBANDS(20,2)", "volatility", "close",
        lambda d: tacuda.bbands(close32(d), 20, 2.0, 2.0, dtype=dtype),
        lambda d: talib.BBANDS(to_talib_f64(d["close"], dtype), 20, 2, 2), tolerance=(8e-2 if is_fp16 else 1e-4)))
    _maybe_add(specs, talib, "ACCBANDS", IndicatorSpec("ACCBANDS(20)", "volatility", "hlc",
        lambda d: tacuda.accbands(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 20),
        lambda d: talib.ACCBANDS(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 20)))
    _maybe_add(specs, talib, "STDDEV", IndicatorSpec("StdDev(14)", "stat", "close",
        lambda d: tacuda.stddev(d["close"].astype(np.float32), 14),
        lambda d: talib.STDDEV(to_talib_f64(d["close"]), 14, 1)))
    _maybe_add(specs, talib, "VAR", IndicatorSpec("Variance(14)", "stat", "close",
        lambda d: tacuda.var(d["close"].astype(np.float32), 14),
        lambda d: talib.VAR(to_talib_f64(d["close"]), 14, 1)))
    _maybe_add(specs, talib, "SUM", IndicatorSpec("SUM(14)", "math", "close",
        # Since 0.5.0 ct_sum is natively end-aligned (TA-Lib warm-up).
        lambda d: tacuda.sum(d["close"].astype(np.float32), 14),
        lambda d: talib.SUM(to_talib_f64(d["close"]), 14), tolerance=1.5e-4))

    _maybe_add(specs, talib, "OBV", IndicatorSpec("OBV", "volume", "cv",
        lambda d: tacuda.obv(d["close"].astype(np.float32), d["volume"].astype(np.float32)),
        lambda d: talib.OBV(to_talib_f64(d["close"]), to_talib_f64(d["volume"])).astype(np.float32)))
    _maybe_add(specs, talib, "AD", IndicatorSpec("AD", "volume", "hlcv",
        lambda d: tacuda.ad(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), d["volume"].astype(np.float32)),
        # Cumulative line in the 1e8 range: the scaled criterion makes the
        # default tolerance meaningful again (3e-8 relative observed).
        lambda d: talib.AD(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), to_talib_f64(d["volume"])).astype(np.float32),
        tolerance=1e-4))
    _maybe_add(specs, talib, "ADOSC", IndicatorSpec("ADOSC(3,10)", "volume", "hlcv",
        lambda d: tacuda.adosc(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), d["volume"].astype(np.float32), 3, 10),
        # Difference of two EMAs of a ~1e8-scale cumulative line: float32
        # cancellation near zero crossings needs a relaxed scaled tolerance.
        lambda d: talib.ADOSC(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), to_talib_f64(d["volume"]), 3, 10).astype(np.float32),
        tolerance=5e-2))
    _maybe_add(specs, talib, "MFI", IndicatorSpec("MFI(14)", "volume", "hlcv",
        lambda d: tacuda.mfi(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), d["volume"].astype(np.float32), 14),
        lambda d: talib.MFI(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), to_talib_f64(d["volume"]), 14)))

    _maybe_add(specs, talib, "ADX", IndicatorSpec("ADX(14)", "trend", "hlc",
        lambda d: tacuda.adx(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.ADX(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "ADXR", IndicatorSpec("ADXR(14)", "trend", "hlc",
        lambda d: tacuda.adxr(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.ADXR(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "DX", IndicatorSpec("DX(14)", "trend", "hlc",
        lambda d: tacuda.dx(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.DX(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "PLUS_DI", IndicatorSpec("PlusDI(14)", "trend", "hlc",
        lambda d: tacuda.plus_di(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.PLUS_DI(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "MINUS_DI", IndicatorSpec("MinusDI(14)", "trend", "hlc",
        lambda d: tacuda.minus_di(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.MINUS_DI(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "PLUS_DM", IndicatorSpec("PlusDM(14)", "trend", "hl",
        lambda d: tacuda.plus_dm(d["high"].astype(np.float32), d["low"].astype(np.float32), 14),
        lambda d: talib.PLUS_DM(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 14), tolerance=2.1e-4))
    _maybe_add(specs, talib, "MINUS_DM", IndicatorSpec("MinusDM(14)", "trend", "hl",
        lambda d: tacuda.minus_dm(d["high"].astype(np.float32), d["low"].astype(np.float32), 14),
        lambda d: talib.MINUS_DM(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 14), tolerance=2e-4))
    _maybe_add(specs, talib, "AROON", IndicatorSpec("AROON(14)", "trend", "hl",
        lambda d: tacuda.aroon(d["high"].astype(np.float32), d["low"].astype(np.float32), 14, 14),
        lambda d: (lambda r: (r[1], r[0], talib.AROONOSC(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 14)))(
            talib.AROON(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 14))))
    _maybe_add(specs, talib, "AROONOSC", IndicatorSpec("AROONOSC(14)", "trend", "hl",
        lambda d: tacuda.aroonosc(d["high"].astype(np.float32), d["low"].astype(np.float32), 14),
        lambda d: talib.AROONOSC(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 14)))
    _maybe_add(specs, talib, "CCI", IndicatorSpec("CCI(14)", "trend", "hlc",
        lambda d: tacuda.cci(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.CCI(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14), tolerance=3e-4))
    _maybe_add(specs, talib, "ULTOSC", IndicatorSpec("ULTOSC(7,14,28)", "trend", "hlc",
        lambda d: tacuda.ultosc(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 7, 14, 28),
        lambda d: talib.ULTOSC(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 7, 14, 28)))
    _maybe_add(specs, talib, "WILLR", IndicatorSpec("WILLR(14)", "trend", "hlc",
        lambda d: tacuda.willr(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 14),
        lambda d: talib.WILLR(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "SAR", IndicatorSpec("SAR(0.02,0.2)", "trend", "hl",
        lambda d: tacuda.sar(d["high"].astype(np.float32), d["low"].astype(np.float32), 0.02, 0.2),
        lambda d: talib.SAR(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 0.02, 0.2)))
    _maybe_add(specs, talib, "SAREXT", IndicatorSpec("SAREXT", "trend", "hl",
        lambda d: tacuda.sarext(d["high"].astype(np.float32), d["low"].astype(np.float32), 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2),
        lambda d: talib.SAREXT(to_talib_f64(d["high"]), to_talib_f64(d["low"]), 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2), tolerance=1e-3))

    _maybe_add(specs, talib, "STOCH", IndicatorSpec("STOCH(5,3)", "oscillators", "hlc",
        lambda d: tacuda.stochastic(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 5, 3),
        lambda d: talib.STOCH(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)))
    _maybe_add(specs, talib, "STOCHF", IndicatorSpec("STOCHF(5,3)", "oscillators", "hlc",
        lambda d: tacuda.stochf(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32), 5, 3),
        lambda d: talib.STOCHF(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]), fastk_period=5, fastd_period=3, fastd_matype=0)))
    _maybe_add(specs, talib, "STOCHRSI", IndicatorSpec("STOCHRSI(14,5,3)", "oscillators", "close",
        lambda d: tacuda.stochrsi(d["close"].astype(np.float32), 14, 5, 3),
        lambda d: talib.STOCHRSI(to_talib_f64(d["close"]), 14, 5, 3, 0), tolerance=1e-3))

    _maybe_add(specs, talib, "MIN", IndicatorSpec("MIN(14)", "math", "close",
        lambda d: tacuda.min(d["close"].astype(np.float32), 14),
        lambda d: talib.MIN(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "MAX", IndicatorSpec("MAX(14)", "math", "close",
        lambda d: tacuda.max(d["close"].astype(np.float32), 14),
        lambda d: talib.MAX(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "MAXINDEX", IndicatorSpec("MAXINDEX(14)", "math", "close",
        lambda d: tacuda.maxindex(d["close"].astype(np.float32), 14),
        lambda d: talib_index_output(talib.MAXINDEX(to_talib_f64(d["close"]), 14), 13)))
    _maybe_add(specs, talib, "MININDEX", IndicatorSpec("MININDEX(14)", "math", "close",
        lambda d: tacuda.minindex(d["close"].astype(np.float32), 14),
        lambda d: talib_index_output(talib.MININDEX(to_talib_f64(d["close"]), 14), 13)))
    _maybe_add(specs, talib, "MINMAX", IndicatorSpec("MINMAX(14)", "math", "close",
        # Since 0.5.0 ct_minmax is natively end-aligned (TA-Lib warm-up).
        lambda d: tacuda.minmax(d["close"].astype(np.float32), 14),
        lambda d: talib.MINMAX(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "MINMAXINDEX", IndicatorSpec("MINMAXINDEX(14)", "math", "close",
        lambda d: (lambda r: (r[0], r[1]))(
            tacuda.minmaxindex(d["close"].astype(np.float32), 14)),
        lambda d: (lambda r: (talib_index_output(r[0], 13), talib_index_output(r[1], 13)))(
            talib.MINMAXINDEX(to_talib_f64(d["close"]), 14))))
    _maybe_add(specs, talib, "AVGDEV", IndicatorSpec("AVGDEV(14)", "math", "close",
        lambda d: tacuda.avgdev(d["close"].astype(np.float32), 14),
        lambda d: talib.AVGDEV(to_talib_f64(d["close"]), 14)))
    _maybe_add(specs, talib, "BOP", IndicatorSpec("BOP", "math", "ohlc",
        # (close-open)/(high-low) in float32: tiny ranges amplify rounding.
        lambda d: tacuda.bop(d["open"].astype(np.float32), d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32)),
        lambda d: talib.BOP(to_talib_f64(d["open"]), to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"])), tolerance=2e-3))

    _maybe_add(specs, talib, "AVGPRICE", IndicatorSpec("AvgPrice", "price", "ohlc",
        lambda d: tacuda.avgprice(d["open"].astype(np.float32), d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32)),
        lambda d: talib.AVGPRICE(to_talib_f64(d["open"]), to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "MEDPRICE", IndicatorSpec("MedPrice", "price", "hl",
        lambda d: tacuda.medprice(d["high"].astype(np.float32), d["low"].astype(np.float32)),
        lambda d: talib.MEDPRICE(to_talib_f64(d["high"]), to_talib_f64(d["low"]))))
    _maybe_add(specs, talib, "TYPPRICE", IndicatorSpec("TypPrice", "price", "hlc",
        lambda d: tacuda.typprice(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32)),
        lambda d: talib.TYPPRICE(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "WCLPRICE", IndicatorSpec("WclPrice", "price", "hlc",
        lambda d: tacuda.wclprice(d["high"].astype(np.float32), d["low"].astype(np.float32), d["close"].astype(np.float32)),
        lambda d: talib.WCLPRICE(to_talib_f64(d["high"]), to_talib_f64(d["low"]), to_talib_f64(d["close"]))))

    _maybe_add(specs, talib, "BETA", IndicatorSpec("Beta(14)", "stat", "xy",
        lambda d: tacuda.beta(d["close"].astype(np.float32), d["close2"].astype(np.float32), 14),
        lambda d: talib.BETA(to_talib_f64(d["close"]), to_talib_f64(d["close2"]), 14)))
    _maybe_add(specs, talib, "CORREL", IndicatorSpec("Correl(14)", "stat", "xy",
        lambda d: tacuda.correl(d["close"].astype(np.float32), d["close2"].astype(np.float32), 14),
        lambda d: talib.CORREL(to_talib_f64(d["close"]), to_talib_f64(d["close2"]), 14)))

    # MACDEXT with independent MA types (SMA fast/slow/signal is bit-exact vs TA-Lib).
    _maybe_add(specs, talib, "MACDEXT", IndicatorSpec("MACDEXT(12,26,9,SMA)", "macd", "close",
        lambda d: tacuda.macdext(close32(d), 12, 0, 26, 0, 9, 0),
        lambda d: talib.MACDEXT(to_talib_f64(d["close"]), fastperiod=12, fastmatype=0,
                                slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0),
        tolerance=5e-3))

    # APO/PPO with TA-Lib's default MA type (SMA) via the _ma variants.
    if hasattr(tacuda, "apo_ma"):
        _maybe_add(specs, talib, "APO", IndicatorSpec("APO(12,26,SMA)", "momentum", "close",
            lambda d: tacuda.apo_ma(d["close"].astype(np.float32), 12, 26, 0),
            lambda d: talib.APO(to_talib_f64(d["close"]), 12, 26, matype=0)))
    if hasattr(tacuda, "ppo_ma"):
        _maybe_add(specs, talib, "PPO", IndicatorSpec("PPO(12,26,SMA)", "momentum", "close",
            lambda d: tacuda.ppo_ma(d["close"].astype(np.float32), 12, 26, 0),
            lambda d: talib.PPO(to_talib_f64(d["close"]), 12, 26, matype=0), tolerance=5e-3))

    # Hilbert Transform family (sequential double ports of ta_HT_*.c).
    _maybe_add(specs, talib, "HT_DCPERIOD", IndicatorSpec("HT_DCPERIOD", "hilbert", "close",
        lambda d: tacuda.ht_dcperiod(close32(d)),
        lambda d: talib.HT_DCPERIOD(to_talib_f64(d["close"])), tolerance=1e-3))
    _maybe_add(specs, talib, "HT_DCPHASE", IndicatorSpec("HT_DCPHASE", "hilbert", "close",
        # Phase in degrees: values wrap around +-315, tolerance is scaled.
        lambda d: tacuda.ht_dcphase(close32(d)),
        lambda d: talib.HT_DCPHASE(to_talib_f64(d["close"])), tolerance=5e-3))
    _maybe_add(specs, talib, "HT_PHASOR", IndicatorSpec("HT_PHASOR", "hilbert", "close",
        lambda d: tacuda.ht_phasor(close32(d)),
        lambda d: talib.HT_PHASOR(to_talib_f64(d["close"])), tolerance=5e-3))
    _maybe_add(specs, talib, "HT_SINE", IndicatorSpec("HT_SINE", "hilbert", "close",
        lambda d: tacuda.ht_sine(close32(d)),
        lambda d: talib.HT_SINE(to_talib_f64(d["close"])), tolerance=5e-3))
    _maybe_add(specs, talib, "HT_TRENDLINE", IndicatorSpec("HT_TRENDLINE", "hilbert", "close",
        lambda d: tacuda.ht_trendline(close32(d)),
        lambda d: talib.HT_TRENDLINE(to_talib_f64(d["close"])), tolerance=1e-3))
    _maybe_add(specs, talib, "HT_TRENDMODE", IndicatorSpec("HT_TRENDMODE", "hilbert", "close",
        # Binary flag; BOTH sides 0-fill the warm-up (tacuda mirrors TA-Lib's
        # int-output convention here), so raw arrays compare directly.
        lambda d: tacuda.ht_trendmode(close32(d)),
        lambda d: np.asarray(talib.HT_TRENDMODE(to_talib_f64(d["close"])), dtype=np.float64)))

    # Candlestick patterns: EXACT signal equality vs TA-Lib over the whole
    # array — no head mask, no flip budget. Both sides 0-fill their warm-up
    # (identical TA-Lib lookbacks), and the predicates run in double like
    # TA-Lib, so borderline candles resolve identically.
    cdl_map = [
        ("cdl_abandoned_baby", "CDLABANDONEDBABY"),
        ("cdl_advance_block", "CDLADVANCEBLOCK"),
        ("cdl_belt_hold", "CDLBELTHOLD"),
        ("cdl_breakaway", "CDLBREAKAWAY"),
        ("cdl_closing_marubozu", "CDLCLOSINGMARUBOZU"),
        ("cdl_conceal_baby_swallow", "CDLCONCEALBABYSWALL"),
        ("cdl_counterattack", "CDLCOUNTERATTACK"),
        ("cdl_dark_cloud_cover", "CDLDARKCLOUDCOVER"),
        ("cdl_doji", "CDLDOJI"),
        ("cdl_doji_star", "CDLDOJISTAR"),
        ("cdl_dragonfly_doji", "CDLDRAGONFLYDOJI"),
        ("cdl_engulfing", "CDLENGULFING"),
        ("cdl_evening_doji_star", "CDLEVENINGDOJISTAR"),
        ("cdl_evening_star", "CDLEVENINGSTAR"),
        ("cdl_gap_side_side_white", "CDLGAPSIDESIDEWHITE"),
        ("cdl_gravestone_doji", "CDLGRAVESTONEDOJI"),
        ("cdl_hammer", "CDLHAMMER"),
        ("cdl_hanging_man", "CDLHANGINGMAN"),
        ("cdl_harami", "CDLHARAMI"),
        ("cdl_harami_cross", "CDLHARAMICROSS"),
        ("cdl_high_wave", "CDLHIGHWAVE"),
        ("cdl_hikkake", "CDLHIKKAKE"),
        ("cdl_hikkake_mod", "CDLHIKKAKEMOD"),
        ("cdl_homing_pigeon", "CDLHOMINGPIGEON"),
        ("cdl_identical_three_crows", "CDLIDENTICAL3CROWS"),
        ("cdl_in_neck", "CDLINNECK"),
        ("cdl_inverted_hammer", "CDLINVERTEDHAMMER"),
        ("cdl_kicking", "CDLKICKING"),
        ("cdl_kicking_by_length", "CDLKICKINGBYLENGTH"),
        ("cdl_ladder_bottom", "CDLLADDERBOTTOM"),
        ("cdl_long_legged_doji", "CDLLONGLEGGEDDOJI"),
        ("cdl_long_line", "CDLLONGLINE"),
        ("cdl_marubozu", "CDLMARUBOZU"),
        ("cdl_mat_hold", "CDLMATHOLD"),
        ("cdl_matching_low", "CDLMATCHINGLOW"),
        ("cdl_morning_doji_star", "CDLMORNINGDOJISTAR"),
        ("cdl_morning_star", "CDLMORNINGSTAR"),
        ("cdl_on_neck", "CDLONNECK"),
        ("cdl_piercing", "CDLPIERCING"),
        ("cdl_rickshaw_man", "CDLRICKSHAWMAN"),
        ("cdl_rise_fall3_methods", "CDLRISEFALL3METHODS"),
        ("cdl_separating_lines", "CDLSEPARATINGLINES"),
        ("cdl_shooting_star", "CDLSHOOTINGSTAR"),
        ("cdl_short_line", "CDLSHORTLINE"),
        ("cdl_spinning_top", "CDLSPINNINGTOP"),
        ("cdl_stalled_pattern", "CDLSTALLEDPATTERN"),
        ("cdl_stick_sandwich", "CDLSTICKSANDWICH"),
        ("cdl_takuri", "CDLTAKURI"),
        ("cdl_tasuki_gap", "CDLTASUKIGAP"),
        ("cdl_three_black_crows", "CDL3BLACKCROWS"),
        ("cdl_three_inside", "CDL3INSIDE"),
        ("cdl_three_line_strike", "CDL3LINESTRIKE"),
        ("cdl_three_outside", "CDL3OUTSIDE"),
        ("cdl_three_stars_in_south", "CDL3STARSINSOUTH"),
        ("cdl_three_white_soldiers", "CDL3WHITESOLDIERS"),
        ("cdl_thrusting", "CDLTHRUSTING"),
        ("cdl_tristar", "CDLTRISTAR"),
        ("cdl_two_crows", "CDL2CROWS"),
        ("cdl_unique_3_river", "CDLUNIQUE3RIVER"),
        ("cdl_upside_gap_2_crows", "CDLUPSIDEGAP2CROWS"),
        ("cdl_xside_gap_3_methods", "CDLXSIDEGAP3METHODS"),
    ]
    for tac_name, talib_name in cdl_map:
        tac_fn = getattr(tacuda, tac_name, None)
        if tac_fn is None or not hasattr(talib, talib_name):
            continue
        talib_fn = getattr(talib, talib_name)
        specs.append(IndicatorSpec(
            talib_name, "candle", "ohlc",
            (lambda f: lambda d: np.asarray(f(
                d["open"].astype(np.float32), d["high"].astype(np.float32),
                d["low"].astype(np.float32), d["close"].astype(np.float32)),
                dtype=np.float64))(tac_fn),
            (lambda f: lambda d: np.asarray(f(
                to_talib_f64(d["open"]), to_talib_f64(d["high"]),
                to_talib_f64(d["low"]), to_talib_f64(d["close"])),
                dtype=np.float64))(talib_fn),
            tolerance=0.0, discrete=True, talib_name=talib_name))

    # Element-wise math transforms / operators.
    # unit-scale input keeps ASIN/ACOS in domain and EXP/TAN well conditioned.
    unit32 = lambda d: ((d["close"] - 150.0) / 50.0).astype(np.float32)
    unit64 = lambda d: to_talib_f64((d["close"] - 150.0) / 50.0)
    _maybe_add(specs, talib, "SIN", IndicatorSpec("SIN", "vector_math", "close",
        lambda d: tacuda.sin(close32(d)), lambda d: talib.SIN(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "COS", IndicatorSpec("COS", "vector_math", "close",
        lambda d: tacuda.cos(close32(d)), lambda d: talib.COS(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "TAN", IndicatorSpec("TAN", "vector_math", "close",
        lambda d: tacuda.tan(unit32(d)), lambda d: talib.TAN(unit64(d)), tolerance=1e-3))
    _maybe_add(specs, talib, "ASIN", IndicatorSpec("ASIN", "vector_math", "close",
        lambda d: tacuda.asin(unit32(d)), lambda d: talib.ASIN(unit64(d)), tolerance=1e-3))
    _maybe_add(specs, talib, "ACOS", IndicatorSpec("ACOS", "vector_math", "close",
        lambda d: tacuda.acos(unit32(d)), lambda d: talib.ACOS(unit64(d)), tolerance=1e-3))
    _maybe_add(specs, talib, "ATAN", IndicatorSpec("ATAN", "vector_math", "close",
        lambda d: tacuda.atan(close32(d)), lambda d: talib.ATAN(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "SINH", IndicatorSpec("SINH", "vector_math", "close",
        lambda d: tacuda.sinh(unit32(d)), lambda d: talib.SINH(unit64(d))))
    _maybe_add(specs, talib, "COSH", IndicatorSpec("COSH", "vector_math", "close",
        lambda d: tacuda.cosh(unit32(d)), lambda d: talib.COSH(unit64(d))))
    _maybe_add(specs, talib, "TANH", IndicatorSpec("TANH", "vector_math", "close",
        lambda d: tacuda.tanh(unit32(d)), lambda d: talib.TANH(unit64(d))))
    _maybe_add(specs, talib, "EXP", IndicatorSpec("EXP", "vector_math", "close",
        lambda d: tacuda.exp(unit32(d)), lambda d: talib.EXP(unit64(d))))
    _maybe_add(specs, talib, "LN", IndicatorSpec("LN", "vector_math", "close",
        lambda d: tacuda.ln(close32(d)), lambda d: talib.LN(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "LOG10", IndicatorSpec("LOG10", "vector_math", "close",
        lambda d: tacuda.log10(close32(d)), lambda d: talib.LOG10(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "SQRT", IndicatorSpec("SQRT", "vector_math", "close",
        lambda d: tacuda.sqrt(close32(d)), lambda d: talib.SQRT(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "CEIL", IndicatorSpec("CEIL", "vector_math", "close",
        lambda d: tacuda.ceil(close32(d)), lambda d: talib.CEIL(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "FLOOR", IndicatorSpec("FLOOR", "vector_math", "close",
        lambda d: tacuda.floor(close32(d)), lambda d: talib.FLOOR(to_talib_f64(d["close"]))))
    _maybe_add(specs, talib, "ADD", IndicatorSpec("ADD", "vector_math", "hl",
        lambda d: tacuda.add(high32(d), low32(d)),
        lambda d: talib.ADD(to_talib_f64(d["high"]), to_talib_f64(d["low"]))))
    _maybe_add(specs, talib, "SUB", IndicatorSpec("SUB", "vector_math", "hl",
        lambda d: tacuda.sub(high32(d), low32(d)),
        lambda d: talib.SUB(to_talib_f64(d["high"]), to_talib_f64(d["low"]))))
    _maybe_add(specs, talib, "MULT", IndicatorSpec("MULT", "vector_math", "hl",
        lambda d: tacuda.mult(high32(d), low32(d)),
        lambda d: talib.MULT(to_talib_f64(d["high"]), to_talib_f64(d["low"])), tolerance=1e-2))
    _maybe_add(specs, talib, "DIV", IndicatorSpec("DIV", "vector_math", "hl",
        lambda d: tacuda.div(high32(d), low32(d)),
        lambda d: talib.DIV(to_talib_f64(d["high"]), to_talib_f64(d["low"]))))

    return specs


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray, tolerance: float) -> Tuple[bool, float, float, int]:
    """Strict oracle: a PASS requires
      * identical shapes,
      * identical NaN masks (lookback / warm-up / propagation),
      * identical +-Inf positions and signs,
      * numeric agreement within `tolerance` on the mutually finite part.
    A NaN-mask mismatch is a hard failure — TaCuda=[2,3] vs TA-Lib=[NaN,3]
    must never PASS, and all-NaN-vs-values even less so.

    The numeric criterion is mixed absolute/relative: each element is
    checked as |a-b| / max(1, |a|, |b|) <= tolerance. Small magnitudes are
    compared absolutely, large magnitudes relatively — a float32 engine
    cannot hold an absolute epsilon on outputs in the 1e4..1e8 range
    (cumulative AD, percentage ratios near zero denominators, ...), and a
    purely absolute bound would either mask real errors on small values or
    flag pure rounding on large ones."""
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        return False, float("inf"), float("inf"), -1

    lhs_nan = np.isnan(lhs)
    rhs_nan = np.isnan(rhs)
    nan_mismatch = int(np.sum(lhs_nan != rhs_nan))

    lhs_inf = np.isinf(lhs)
    rhs_inf = np.isinf(rhs)
    inf_mismatch = int(np.sum(lhs_inf != rhs_inf))
    both_inf = lhs_inf & rhs_inf
    if np.any(both_inf):
        inf_mismatch += int(np.sum(np.sign(lhs[both_inf]) != np.sign(rhs[both_inf])))

    valid_mask = ~lhs_nan & ~rhs_nan & ~lhs_inf & ~rhs_inf
    if not np.any(valid_mask):
        ok = nan_mismatch == 0 and inf_mismatch == 0
        return ok, 0.0, 0.0, nan_mismatch
    a = lhs[valid_mask].astype(np.float64)
    b = rhs[valid_mask].astype(np.float64)
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    diff = np.abs(a - b) / scale
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    ok = (max_diff <= tolerance) and nan_mismatch == 0 and inf_mismatch == 0
    return ok, max_diff, mean_diff, nan_mismatch
