#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Pipeline test: one data source, multiple indicators.
Similar to C++ test_context_pipeline.cpp
"""

import os
import sys
import math

import numpy as np

# Add the tacuda module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set library path for Windows
tacuda_dll = os.path.join(os.path.dirname(__file__), "..", "..", "build", "Release", "tacuda.dll")
if os.path.exists(tacuda_dll):
    os.environ["TACUDA_LIBRARY"] = tacuda_dll

import tacuda


def generate_ohlcv(size: int):
    """Generate deterministic synthetic OHLCV data (matches C++ test)."""
    open_ = np.zeros(size, dtype=np.float32)
    high = np.zeros(size, dtype=np.float32)
    low = np.zeros(size, dtype=np.float32)
    close = np.zeros(size, dtype=np.float32)
    volume = np.zeros(size, dtype=np.float32)
    
    for i in range(size):
        t = float(i)
        
        # Deterministic synthetic price series
        base = 100.0 + 2.0 * math.sin(0.01 * t) + 0.5 * math.cos(0.005 * t) + 0.0001 * t
        
        o = base + 0.1 * math.sin(0.03 * t)
        c = base + 0.1 * math.cos(0.02 * t)
        
        h = max(o, c) + 0.2 + 0.3 * abs(math.sin(0.01 * t))
        l = min(o, c) - 0.2 - 0.3 * abs(math.cos(0.01 * t))
        
        open_[i] = o
        high[i] = h
        low[i] = l
        close[i] = c
        
        # Positive, varying volume
        volume[i] = 100000.0 + ((i * 37) % 1000) * 10.0
    
    return open_, high, low, close, volume


def validate_range(arr, min_val, max_val, name, eps=1e-6):
    """Check values are in range (ignoring NaN, with tolerance for float errors)."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return True
    # Allow small epsilon for floating point errors
    if np.any(valid < min_val - eps) or np.any(valid > max_val + eps):
        bad = valid[(valid < min_val - eps) | (valid > max_val + eps)]
        print(f"    {name} out of range [{min_val},{max_val}]: {bad[:5]}")
        return False
    return True


def validate_non_negative(arr, name, eps=1e-6):
    """Check values are non-negative (ignoring NaN, with tolerance)."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return True
    if np.any(valid < -eps):
        bad = valid[valid < -eps]
        print(f"    {name} has negative values: {bad[:5]}")
        return False
    return True


def validate_bbands(upper, middle, lower):
    """Check BBands order: upper >= middle >= lower."""
    mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
    u, m, l_ = upper[mask], middle[mask], lower[mask]
    if len(u) == 0:
        return True
    if np.any(u < m) or np.any(m < l_):
        idx = np.where((u < m) | (m < l_))[0][0]
        print(f"    BBands order violated at {idx}: U={u[idx]}, M={m[idx]}, L={l_[idx]}")
        return False
    return True


def test_result(name, success, results):
    """Record test result."""
    if success:
        print(f"  [PASS] {name}")
        results["passed"] += 1
    else:
        print(f"  [FAIL] {name}")
        results["failed"] += 1


def run_pipeline_test():
    """Run multi-indicator pipeline test."""
    print("=" * 70)
    print("TACUDA Python Pipeline Test (Multi-Indicator)")
    print("=" * 70)
    
    N = 4096
    
    # Generate synthetic OHLCV data
    open_, high, low, close, volume = generate_ohlcv(N)
    
    print(f"Data size: {N} samples")
    print()
    
    results = {"passed": 0, "failed": 0}
    
    # ====================================================================
    # Calculate multiple indicators on the SAME data source
    # ====================================================================
    
    # RSI
    try:
        rsi = tacuda.rsi(close, period=14)
        test_result("RSI(14)", validate_range(rsi, 0, 100, "RSI"), results)
    except Exception as e:
        print(f"  [ERROR] RSI: {e}")
        results["failed"] += 1
    
    # ATR
    try:
        atr = tacuda.atr(high, low, close, period=14, initial=0.0)
        test_result("ATR(14)", validate_non_negative(atr, "ATR"), results)
    except Exception as e:
        print(f"  [ERROR] ATR: {e}")
        results["failed"] += 1
    
    # ADX
    try:
        adx = tacuda.adx(high, low, close, period=14)
        test_result("ADX(14)", validate_range(adx, 0, 100, "ADX"), results)
    except Exception as e:
        print(f"  [ERROR] ADX: {e}")
        results["failed"] += 1
    
    # OBV
    try:
        obv = tacuda.obv(close, volume)
        test_result("OBV", True, results)
    except Exception as e:
        print(f"  [ERROR] OBV: {e}")
        results["failed"] += 1
    
    # MFI
    try:
        mfi = tacuda.mfi(high, low, close, volume, period=14)
        test_result("MFI(14)", validate_range(mfi, 0, 100, "MFI"), results)
    except Exception as e:
        print(f"  [ERROR] MFI: {e}")
        results["failed"] += 1
    
    # EMA (fast)
    try:
        ema_fast = tacuda.ema(close, period=12)
        test_result("EMA(12)", True, results)
    except Exception as e:
        print(f"  [ERROR] EMA(12): {e}")
        results["failed"] += 1
    
    # EMA (slow)
    try:
        ema_slow = tacuda.ema(close, period=26)
        test_result("EMA(26)", True, results)
    except Exception as e:
        print(f"  [ERROR] EMA(26): {e}")
        results["failed"] += 1
    
    # Trend = EMA(12) - EMA(26)
    try:
        trend = np.where(np.isnan(ema_fast) | np.isnan(ema_slow), np.nan, ema_fast - ema_slow)
        test_result("Trend (EMA12-EMA26)", True, results)
    except Exception as e:
        print(f"  [ERROR] Trend: {e}")
        results["failed"] += 1
    
    # MACD
    try:
        macd, signal, hist = tacuda.macd(close, fastPeriod=12, slowPeriod=26, signalPeriod=9, type=0)
        test_result("MACD(12,26,9)", True, results)
    except Exception as e:
        print(f"  [ERROR] MACD: {e}")
        results["failed"] += 1
    
    # Stochastic
    try:
        stoch_k, stoch_d = tacuda.stochastic(high, low, close, kPeriod=14, dPeriod=3)
        valid_k = validate_range(stoch_k, 0, 100, "StochK")
        valid_d = validate_range(stoch_d, 0, 100, "StochD")
        test_result("Stochastic(14,3)", valid_k and valid_d, results)
    except Exception as e:
        print(f"  [ERROR] Stochastic: {e}")
        results["failed"] += 1
    
    # Bollinger Bands
    try:
        bb_upper, bb_middle, bb_lower = tacuda.bbands(close, period=20, upperMul=2.0, lowerMul=2.0)
        test_result("BBANDS(20,2,2)", validate_bbands(bb_upper, bb_middle, bb_lower), results)
    except Exception as e:
        print(f"  [ERROR] BBANDS: {e}")
        results["failed"] += 1
    
    # SMA
    try:
        sma = tacuda.sma(close, period=20)
        test_result("SMA(20)", True, results)
    except Exception as e:
        print(f"  [ERROR] SMA: {e}")
        results["failed"] += 1
    
    # WMA
    try:
        wma = tacuda.wma(close, period=10)
        test_result("WMA(10)", True, results)
    except Exception as e:
        print(f"  [ERROR] WMA: {e}")
        results["failed"] += 1
    
    # StdDev
    try:
        stddev = tacuda.stddev(close, period=20)
        test_result("StdDev(20)", validate_non_negative(stddev, "StdDev"), results)
    except Exception as e:
        print(f"  [ERROR] StdDev: {e}")
        results["failed"] += 1
    
    # Sum
    try:
        sum_result = tacuda.sum(close, period=10)
        test_result("Sum(10)", True, results)
    except Exception as e:
        print(f"  [ERROR] Sum: {e}")
        results["failed"] += 1
    
    # ====================================================================
    # MACDEXT: EMA/EMA/EMA must match classic MACD; mixed types must run
    # ====================================================================
    try:
        macd_ref, _, _ = tacuda.macd(close, fastPeriod=12, slowPeriod=26, signalPeriod=9, type=1)
        ext_macd, ext_signal, ext_hist = tacuda.macdext(
            close, fastPeriod=12, fastType=1, slowPeriod=26, slowType=1,
            signalPeriod=9, signalType=1)
        mask = ~(np.isnan(macd_ref) | np.isnan(ext_macd))
        ok = mask.sum() > 0 and np.max(np.abs(macd_ref[mask] - ext_macd[mask])) < 1e-3
        test_result("MACDEXT(EMA,EMA,EMA) == MACD(EMA)", ok, results)
        
        ext2 = tacuda.macdext(close, fastPeriod=12, fastType=0, slowPeriod=26, slowType=1,
                              signalPeriod=9, signalType=2)
        test_result("MACDEXT(SMA,EMA,WMA) runs", np.any(~np.isnan(ext2[0])), results)
    except Exception as e:
        print(f"  [ERROR] MACDEXT: {e}")
        results["failed"] += 2
    
    # ====================================================================
    # Vector math: element-wise kernels vs numpy references
    # ====================================================================
    unary_cases = {
        "acos": (np.linspace(-0.95, 0.95, N, dtype=np.float32), np.arccos),
        "asin": (np.linspace(-0.95, 0.95, N, dtype=np.float32), np.arcsin),
        "atan": (np.linspace(-3, 3, N, dtype=np.float32), np.arctan),
        "ceil": (np.linspace(-5, 5, N, dtype=np.float32), np.ceil),
        "cos": (np.linspace(-3, 3, N, dtype=np.float32), np.cos),
        "cosh": (np.linspace(-2, 2, N, dtype=np.float32), np.cosh),
        "exp": (np.linspace(-2, 2, N, dtype=np.float32), np.exp),
        "floor": (np.linspace(-5, 5, N, dtype=np.float32), np.floor),
        "ln": (np.linspace(0.1, 10, N, dtype=np.float32), np.log),
        "log10": (np.linspace(0.1, 10, N, dtype=np.float32), np.log10),
        "sin": (np.linspace(-3, 3, N, dtype=np.float32), np.sin),
        "sinh": (np.linspace(-2, 2, N, dtype=np.float32), np.sinh),
        "sqrt": (np.linspace(0.1, 10, N, dtype=np.float32), np.sqrt),
        "tan": (np.linspace(-1.2, 1.2, N, dtype=np.float32), np.tan),
        "tanh": (np.linspace(-2, 2, N, dtype=np.float32), np.tanh),
    }
    vm_ok = True
    for name, (src, ref_fn) in unary_cases.items():
        try:
            out = getattr(tacuda, name)(src)
            expected = ref_fn(src.astype(np.float64)).astype(np.float32)
            diff = np.max(np.abs(out - expected) / np.maximum(1.0, np.abs(expected)))
            if diff > 1e-4:
                print(f"    {name}: max rel diff {diff:.2e}")
                vm_ok = False
        except Exception as e:
            print(f"    {name}: {e}")
            vm_ok = False
    test_result("Vector math unary (15 kernels)", vm_ok, results)
    
    try:
        a = np.linspace(0.1, 10, N, dtype=np.float32)
        b = np.linspace(-3, 3, N, dtype=np.float32)
        ok = (np.allclose(tacuda.add(a, b), a + b, atol=1e-4)
              and np.allclose(tacuda.sub(a, b), a - b, atol=1e-4)
              and np.allclose(tacuda.mult(a, b), a * b, atol=1e-3)
              and np.allclose(tacuda.div(a, b)[np.abs(b) > 0.01],
                              (a / b)[np.abs(b) > 0.01], rtol=1e-3))
        test_result("Vector math binary (add/sub/mult/div)", ok, results)
    except Exception as e:
        print(f"  [ERROR] Vector math binary: {e}")
        results["failed"] += 1
    
    # ====================================================================
    # Device API: upload -> sma_d + sin_d -> download
    # ====================================================================
    try:
        d_in = tacuda.device_alloc(N)
        d_out = tacuda.device_alloc(N)
        try:
            tacuda.device_upload(close, d_in, N)
            tacuda.sma_d(d_in, d_out, N, 20)
            tacuda.device_sync()
            gpu_out = np.asarray(tacuda.device_download(d_out, N), dtype=np.float32)
            mask = ~(np.isnan(sma) | np.isnan(gpu_out))
            ok = mask.sum() > 0 and np.max(np.abs(sma[mask] - gpu_out[mask])) < 1e-4
            test_result("Device API: sma_d == sma", ok, results)
            
            tacuda.sin_d(d_in, d_out, N)
            tacuda.device_sync()
            gpu_out = np.asarray(tacuda.device_download(d_out, N), dtype=np.float32)
            expected = np.sin(close)
            ok = np.max(np.abs(gpu_out - expected)) < 1e-4
            test_result("Device API: sin_d == np.sin", ok, results)
        finally:
            tacuda.device_free(d_in)
            tacuda.device_free(d_out)
    except Exception as e:
        print(f"  [ERROR] Device API: {e}")
        results["failed"] += 2
    
    # ====================================================================
    # OHLCV container: structured inputs must match column-array calls
    # ====================================================================
    try:
        series = tacuda.OHLCV.from_columns(open_, high, low, close, volume)
        ohlcv_ok = True

        atr_series = tacuda.atr(series, period=14)
        mask = ~(np.isnan(atr) | np.isnan(atr_series))
        if not (mask.sum() > 0 and np.max(np.abs(atr[mask] - atr_series[mask])) < 1e-5):
            print("    atr(OHLCV) != atr(high, low, close)")
            ohlcv_ok = False

        adx_series = tacuda.adx(series, period=14)
        mask = ~(np.isnan(adx) | np.isnan(adx_series))
        if not (mask.sum() > 0 and np.max(np.abs(adx[mask] - adx_series[mask])) < 1e-5):
            print("    adx(OHLCV) != adx(high, low, close)")
            ohlcv_ok = False

        mfi_series = tacuda.mfi(series, period=14)
        mask = ~(np.isnan(mfi) | np.isnan(mfi_series))
        if not (mask.sum() > 0 and np.max(np.abs(mfi[mask] - mfi_series[mask])) < 1e-5):
            print("    mfi(OHLCV) != mfi(high, low, close, volume)")
            ohlcv_ok = False

        doji_cols = tacuda.cdl_doji(open_, high, low, close)
        doji_series = tacuda.cdl_doji(series)
        if not np.array_equal(doji_cols, doji_series):
            print("    cdl_doji(OHLCV) != cdl_doji(o, h, l, c)")
            ohlcv_ok = False

        test_result("OHLCV structured inputs (atr/adx/mfi/cdl_doji)", ohlcv_ok, results)
    except Exception as e:
        print(f"  [ERROR] OHLCV structured inputs: {e}")
        results["failed"] += 1

    # ====================================================================
    # Batched (multi-symbol) API vs per-series host API
    # ====================================================================
    try:
        num_series, series_len = 5, 1024
        rng = np.random.default_rng(42)
        base = 100.0 + np.cumsum(rng.normal(0, 0.5, (num_series, series_len)), axis=1)
        matrix = base.astype(np.float32)

        batch_ok = True
        batch_rsi = tacuda.rsi_batch(matrix, period=14, max_concurrency=3)
        for s in range(num_series):
            single = tacuda.rsi(matrix[s], period=14)
            mask = ~(np.isnan(single) | np.isnan(batch_rsi[s]))
            if not (mask.sum() > 0 and np.max(np.abs(single[mask] - batch_rsi[s][mask])) < 1e-4):
                print(f"    rsi_batch row {s} mismatch")
                batch_ok = False

        b_macd, b_signal, b_hist = tacuda.macd_batch(matrix)
        m0, s0, h0 = tacuda.macd(matrix[0], fastPeriod=12, slowPeriod=26, signalPeriod=9, type=1)
        mask = ~(np.isnan(m0) | np.isnan(b_macd[0]))
        if not (mask.sum() > 0 and np.max(np.abs(m0[mask] - b_macd[0][mask])) < 1e-4):
            print("    macd_batch row 0 mismatch")
            batch_ok = False

        highs = matrix + 1.0
        lows = matrix - 1.0
        batch_k, batch_d = tacuda.stochastic_batch(highs, lows, matrix, kPeriod=14, dPeriod=3)
        k0, d0 = tacuda.stochastic(highs[0], lows[0], matrix[0], kPeriod=14, dPeriod=3)
        mask = ~(np.isnan(k0) | np.isnan(batch_k[0]))
        if not (mask.sum() > 0 and np.max(np.abs(k0[mask] - batch_k[0][mask])) < 1e-4):
            print("    stochastic_batch row 0 mismatch")
            batch_ok = False

        test_result(f"Batch API ({num_series} series x {series_len})", batch_ok, results)
    except Exception as e:
        print(f"  [ERROR] Batch API: {e}")
        results["failed"] += 1

    # ====================================================================
    # Generated batch families (0.3.0): wma/adx/mfi/aroon/macdext vs single
    # ====================================================================
    try:
        num_series, series_len = 4, 768
        rng = np.random.default_rng(7)
        matrix = (100.0 + np.cumsum(rng.normal(0, 0.5, (num_series, series_len)),
                                    axis=1)).astype(np.float32)
        highs = matrix + 1.0
        lows = matrix - 1.0
        volumes = (1000.0 + 500.0 * np.abs(np.sin(
            np.arange(num_series * series_len, dtype=np.float32)
        ))).reshape(num_series, series_len)

        def rows_match(batch_rows, single_fn, tol=1e-4):
            for s in range(num_series):
                single = single_fn(s)
                mask = ~(np.isnan(single) | np.isnan(batch_rows[s]))
                if not (mask.sum() > 0 and
                        np.max(np.abs(single[mask] - batch_rows[s][mask])) < tol):
                    return False
            return True

        gen_ok = True
        b_wma = tacuda.wma_batch(matrix, 10, max_concurrency=2)
        gen_ok &= rows_match(b_wma, lambda s: tacuda.wma(matrix[s], period=10))

        b_adx = tacuda.adx_batch(highs, lows, matrix, 14)
        gen_ok &= rows_match(b_adx, lambda s: tacuda.adx(highs[s], lows[s], matrix[s], period=14))

        b_mfi = tacuda.mfi_batch(highs, lows, matrix, volumes, 14)
        gen_ok &= rows_match(b_mfi, lambda s: tacuda.mfi(highs[s], lows[s], matrix[s], volumes[s], period=14))

        b_macd, b_signal, b_hist = tacuda.macdext_batch(matrix, 12, 1, 26, 1, 9, 1)
        m0, s0, h0 = tacuda.macd(matrix[0], fastPeriod=12, slowPeriod=26, signalPeriod=9, type=1)
        mask = ~(np.isnan(m0) | np.isnan(b_macd[0]))
        gen_ok &= bool(mask.sum() > 0 and np.max(np.abs(m0[mask] - b_macd[0][mask])) < 1e-4)

        b_down, b_up = tacuda.aroon_batch(highs, lows, 14)
        gen_ok &= b_down.shape == matrix.shape and b_up.shape == matrix.shape
        for arr in (b_down, b_up):  # aroon values are bounded to [0, 100]
            vals = arr[~np.isnan(arr)]
            gen_ok &= bool(vals.size > 0 and vals.min() >= 0.0 and vals.max() <= 100.0)

        test_result("Generated batch families (wma/adx/mfi/macdext/aroon)", gen_ok, results)
    except Exception as e:
        print(f"  [ERROR] Generated batch families: {e}")
        results["failed"] += 1

    # ====================================================================
    # Library version metadata
    # ====================================================================
    try:
        ver = tacuda.version()
        test_result(f"version() == '{ver}'", bool(ver) and ver.count(".") == 2, results)
    except Exception as e:
        print(f"  [ERROR] version(): {e}")
        results["failed"] += 1

    # ====================================================================
    # Cross-validation: SMA(20) should match BBands middle band
    # ====================================================================
    try:
        mask = ~(np.isnan(sma) | np.isnan(bb_middle))
        sma_valid = sma[mask]
        bb_mid_valid = bb_middle[mask]
        
        if len(sma_valid) > 0:
            max_diff = np.max(np.abs(sma_valid - bb_mid_valid))
            matches = max_diff < 1e-4
            test_result(f"Cross-check: SMA(20) == BBands.Middle ({len(sma_valid)} samples, maxdiff={max_diff:.2e})", 
                       matches, results)
        else:
            test_result("Cross-check: SMA(20) == BBands.Middle (no valid samples)", False, results)
    except Exception as e:
        print(f"  [ERROR] Cross-check: {e}")
        results["failed"] += 1
    

    # ====================================================================
    # Streaming API: appending must reproduce the full-series result
    # ====================================================================
    try:
        n = len(close)
        with tacuda.stream.Rsi(14) as st:
            test_result(f"stream.Rsi lookback == 14", st.lookback == 14, results)
            warm = st.append(close[: n // 2])
            tail = np.concatenate(
                [st.append(close[i : i + 1]) for i in range(n // 2, n)]
            )
            streamed = np.concatenate([warm, tail])
            test_result(f"stream.Rsi bars == {n}", st.bars == n, results)

        full = tacuda.rsi(close, 14)
        mask_ok = np.array_equal(np.isnan(streamed), np.isnan(full))
        test_result("stream.Rsi NaN mask == full-series", mask_ok, results)
        both = ~np.isnan(streamed) & ~np.isnan(full)
        diff = np.max(np.abs(streamed[both] - full[both])) if both.any() else 0.0
        test_result(f"stream.Rsi values == full-series (maxdiff={diff:.2e})",
                    diff < 1e-3, results)

        # Chunking must not change the result.
        with tacuda.stream.Rsi(14) as st:
            bulk = st.append(close)
        test_result("stream.Rsi chunked == bulk",
                    np.array_equal(np.nan_to_num(bulk, nan=-1.0),
                                   np.nan_to_num(streamed, nan=-1.0)),
                    results)

        # Multi-output family returns one array per line.
        with tacuda.stream.Macd(12, 26, 9) as st:
            lines = st.append(close)
        test_result("stream.Macd returns 3 lines", len(lines) == 3, results)

        # Batch: every row equals the single-series stream.
        series = 4
        with tacuda.stream.RsiBatch(14, num_series=series) as bt:
            bt.set_exec(tacuda.stream.EXEC_GPU)
            mat = np.repeat(close[None, :], series, axis=0)
            out = bt.append(mat)
        rows_ok = all(
            np.array_equal(np.nan_to_num(out[r], nan=-1.0),
                           np.nan_to_num(streamed, nan=-1.0))
            for r in range(series)
        )
        test_result(f"stream.RsiBatch({series}, GPU) rows == single stream",
                    rows_ok, results)

        # A destroyed handle must be rejected, not dereferenced.
        dead = tacuda.stream.Sma(10)
        dead.close()
        try:
            dead.append(close[:1])
            test_result("stream: use-after-close rejected", False, results)
        except Exception:
            test_result("stream: use-after-close rejected", True, results)
    except Exception as e:
        print(f"  [ERROR] streaming: {e}")
        results["failed"] += 1

    # ====================================================================
    # Results
    # ====================================================================
    print()
    print("=" * 70)
    print(f"PIPELINE RESULTS: {results['passed']} passed, {results['failed']} failed")
    print("=" * 70)
    
    return 1 if results["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(run_pipeline_test())

