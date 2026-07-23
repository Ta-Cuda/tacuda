#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Volatility Indicators: tacuda vs TA-Lib
Комплексное тестирование всех индикаторов волатильности:
- ATR (Average True Range)
- NATR (Normalized ATR)
- TRANGE (True Range)
- BBANDS (Bollinger Bands)
- StdDev (Standard Deviation)
"""

import time
import numpy as np
import talib
import ctypes
import os
import sys

# ==================== Загрузка tacuda ====================
tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release', 'tacuda.dll')
if not os.path.exists(tacuda_lib_path):
    tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libtacuda.so')

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
if os.path.exists(cuda_path):
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(cuda_path)
    except (AttributeError, FileNotFoundError):
        pass

tacuda_dir = os.path.dirname(tacuda_lib_path)
if os.path.isdir(tacuda_dir):
    try:
        os.add_dll_directory(os.path.abspath(tacuda_dir))
    except (AttributeError, FileNotFoundError):
        pass

try:
    tacuda = ctypes.CDLL(tacuda_lib_path)
except Exception as e:
    print(f"Failed to load tacuda library: {e}")
    print(f"Library path: {tacuda_lib_path}")
    sys.exit(1)

# ==================== Настройка функций ====================

# ATR
try:
    tacuda_atr = tacuda.ct_atr
    tacuda_atr.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_int,                    # period
        ctypes.c_float,                  # initial
        ctypes.c_void_p                  # stream
    ]
    tacuda_atr.restype = ctypes.c_int
    HAS_ATR = True
except AttributeError:
    HAS_ATR = False
    print("Warning: ct_atr not found")

# NATR
try:
    tacuda_natr = tacuda.ct_natr
    tacuda_natr.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_int,                    # period
        ctypes.c_void_p                  # stream
    ]
    tacuda_natr.restype = ctypes.c_int
    HAS_NATR = True
except AttributeError:
    HAS_NATR = False
    print("Warning: ct_natr not found")

# TRANGE
try:
    tacuda_trange = tacuda.ct_trange
    tacuda_trange.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_void_p                  # stream
    ]
    tacuda_trange.restype = ctypes.c_int
    HAS_TRANGE = True
except AttributeError:
    HAS_TRANGE = False
    print("Warning: ct_trange not found")

# BBANDS
try:
    tacuda_bbands = tacuda.ct_bbands
    tacuda_bbands.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # upper
        ctypes.POINTER(ctypes.c_float),  # middle
        ctypes.POINTER(ctypes.c_float),  # lower
        ctypes.c_int,                    # size
        ctypes.c_int,                    # period
        ctypes.c_float,                  # upperMul
        ctypes.c_float,                  # lowerMul
        ctypes.c_void_p                  # stream
    ]
    tacuda_bbands.restype = ctypes.c_int
    HAS_BBANDS = True
except AttributeError:
    HAS_BBANDS = False
    print("Warning: ct_bbands not found")

# StdDev
try:
    tacuda_stddev = tacuda.ct_stddev
    tacuda_stddev.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_int,                    # period
        ctypes.c_void_p                  # stream
    ]
    tacuda_stddev.restype = ctypes.c_int
    HAS_STDDEV = True
except AttributeError:
    HAS_STDDEV = False
    print("Warning: ct_stddev not found")

# ==================== Генерация данных ====================

def generate_ohlc_data(size, seed=42):
    """Генерация реалистичных OHLC данных"""
    np.random.seed(seed)
    
    # Базовый тренд
    trend = np.linspace(100, 120, size)
    
    # Волатильность (цикличная)
    x = np.linspace(0, 20, size)
    volatility = 2 + np.sin(x) * 1.5
    
    # Close цена
    close = trend + np.cumsum(np.random.randn(size) * 0.3)
    
    # High и Low на основе volatility
    high = close + np.abs(np.random.randn(size)) * volatility
    low = close - np.abs(np.random.randn(size)) * volatility
    
    # Open между high и low
    open_price = low + (high - low) * np.random.rand(size)
    
    return {
        'high': high.astype(np.float32),
        'low': low.astype(np.float32),
        'close': close.astype(np.float32),
        'open': open_price.astype(np.float32)
    }

# ==================== Benchmark Functions ====================

def benchmark_atr(ohlc, period, iterations=50):
    """Benchmark ATR"""
    if not HAS_ATR:
        return None, None, 0, 0
    
    size = len(ohlc['high'])
    
    # TA-Lib
    high_f64 = ohlc['high'].astype(np.float64)
    low_f64 = ohlc['low'].astype(np.float64)
    close_f64 = ohlc['close'].astype(np.float64)
    
    for _ in range(5):
        _ = talib.ATR(high_f64, low_f64, close_f64, timeperiod=period)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_result = talib.ATR(high_f64, low_f64, close_f64, timeperiod=period)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda
    output = np.full(size, np.nan, dtype=np.float32)
    high_ptr = ohlc['high'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = ohlc['low'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = ohlc['close'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_atr(high_ptr, low_ptr, close_ptr, out_ptr, size, period, ctypes.c_float(0.0), None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_atr(high_ptr, low_ptr, close_ptr, out_ptr, size, period, ctypes.c_float(0.0), None)
    t1 = time.perf_counter()
    cuda_time = (t1 - t0) / iterations * 1000
    
    return ta_result.astype(np.float32), output, ta_time, cuda_time

def benchmark_natr(ohlc, period, iterations=50):
    """Benchmark NATR"""
    if not HAS_NATR:
        return None, None, 0, 0
    
    size = len(ohlc['high'])
    
    # TA-Lib
    high_f64 = ohlc['high'].astype(np.float64)
    low_f64 = ohlc['low'].astype(np.float64)
    close_f64 = ohlc['close'].astype(np.float64)
    
    for _ in range(5):
        _ = talib.NATR(high_f64, low_f64, close_f64, timeperiod=period)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_result = talib.NATR(high_f64, low_f64, close_f64, timeperiod=period)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda
    output = np.full(size, np.nan, dtype=np.float32)
    high_ptr = ohlc['high'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = ohlc['low'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = ohlc['close'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_natr(high_ptr, low_ptr, close_ptr, out_ptr, size, period, None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_natr(high_ptr, low_ptr, close_ptr, out_ptr, size, period, None)
    t1 = time.perf_counter()
    cuda_time = (t1 - t0) / iterations * 1000
    
    return ta_result.astype(np.float32), output, ta_time, cuda_time

def benchmark_trange(ohlc, iterations=50):
    """Benchmark TRANGE"""
    if not HAS_TRANGE:
        return None, None, 0, 0
    
    size = len(ohlc['high'])
    
    # TA-Lib
    high_f64 = ohlc['high'].astype(np.float64)
    low_f64 = ohlc['low'].astype(np.float64)
    close_f64 = ohlc['close'].astype(np.float64)
    
    for _ in range(5):
        _ = talib.TRANGE(high_f64, low_f64, close_f64)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_result = talib.TRANGE(high_f64, low_f64, close_f64)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda
    output = np.full(size, np.nan, dtype=np.float32)
    high_ptr = ohlc['high'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = ohlc['low'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = ohlc['close'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_trange(high_ptr, low_ptr, close_ptr, out_ptr, size, None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_trange(high_ptr, low_ptr, close_ptr, out_ptr, size, None)
    t1 = time.perf_counter()
    cuda_time = (t1 - t0) / iterations * 1000
    
    return ta_result.astype(np.float32), output, ta_time, cuda_time

def benchmark_bbands(close, period, nbdevup=2.0, nbdevdn=2.0, iterations=50):
    """Benchmark BBANDS"""
    if not HAS_BBANDS:
        return None, None, 0, 0
    
    size = len(close)
    
    # TA-Lib
    close_f64 = close.astype(np.float64)
    
    for _ in range(5):
        _ = talib.BBANDS(close_f64, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        upper, middle, lower = talib.BBANDS(close_f64, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    ta_result = np.stack([upper, middle, lower], axis=0).astype(np.float32)
    
    # tacuda - три отдельных массива
    upper_out = np.full(size, np.nan, dtype=np.float32)
    middle_out = np.full(size, np.nan, dtype=np.float32)
    lower_out = np.full(size, np.nan, dtype=np.float32)
    
    in_ptr = close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    upper_ptr = upper_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    middle_ptr = middle_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lower_ptr = lower_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_bbands(in_ptr, upper_ptr, middle_ptr, lower_ptr, size, period, 
                     ctypes.c_float(nbdevup), ctypes.c_float(nbdevdn), None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_bbands(in_ptr, upper_ptr, middle_ptr, lower_ptr, size, period, 
                     ctypes.c_float(nbdevup), ctypes.c_float(nbdevdn), None)
    t1 = time.perf_counter()
    cuda_time = (t1 - t0) / iterations * 1000
    
    cuda_result = np.stack([upper_out, middle_out, lower_out], axis=0)
    
    return ta_result, cuda_result, ta_time, cuda_time

def benchmark_stddev(close, period, nbdev=1.0, iterations=50):
    """Benchmark StdDev"""
    if not HAS_STDDEV:
        return None, None, 0, 0
    
    size = len(close)
    
    # TA-Lib - nbdev применяется вручную
    close_f64 = close.astype(np.float64)
    
    for _ in range(5):
        _ = talib.STDDEV(close_f64, timeperiod=period, nbdev=nbdev)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_result = talib.STDDEV(close_f64, timeperiod=period, nbdev=nbdev)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda - без nbdev, возвращает чистый StdDev
    output = np.full(size, np.nan, dtype=np.float32)
    in_ptr = close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_stddev(in_ptr, out_ptr, size, period, None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_stddev(in_ptr, out_ptr, size, period, None)
    t1 = time.perf_counter()
    cuda_time = (t1 - t0) / iterations * 1000
    
    # Применяем nbdev к результату tacuda для fair сравнения
    output_scaled = output * nbdev
    
    return ta_result.astype(np.float32), output_scaled, ta_time, cuda_time

# ==================== Comparison ====================

def compare_results(ta_result, cuda_result, tolerance=1e-4):
    """Сравнение результатов с учетом NaN"""
    if ta_result is None or cuda_result is None:
        return False, 0.0, 0.0
    
    # Для многомерных результатов (BBANDS)
    if ta_result.ndim > 1:
        all_match = True
        max_diffs = []
        mean_diffs = []
        
        for i in range(ta_result.shape[0]):
            mask = ~(np.isnan(ta_result[i]) | np.isnan(cuda_result[i]))
            if not np.any(mask):
                continue
            
            ta_valid = ta_result[i][mask]
            cuda_valid = cuda_result[i][mask]
            
            diff = np.abs(ta_valid - cuda_valid)
            max_diffs.append(np.max(diff))
            mean_diffs.append(np.mean(diff))
            
            match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
            all_match = all_match and match
        
        return all_match, max(max_diffs) if max_diffs else 0.0, np.mean(mean_diffs) if mean_diffs else 0.0
    
    # Для одномерных результатов
    mask = ~(np.isnan(ta_result) | np.isnan(cuda_result))
    if not np.any(mask):
        return False, 0.0, 0.0
    
    ta_valid = ta_result[mask]
    cuda_valid = cuda_result[mask]
    
    diff = np.abs(ta_valid - cuda_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
    return match, max_diff, mean_diff

# ==================== Main ====================

def print_separator():
    print("=" * 110)

def main():
    print_separator()
    print("Volatility Indicators Benchmark: tacuda vs TA-Lib".center(110))
    print("Testing: ATR, NATR, TRANGE, BBANDS, StdDev".center(110))
    print_separator()
    print()
    
    data_size = 500000
    print(f"Data size: {data_size:,} elements (OHLC)")
    print()
    
    # Генерация OHLC данных
    print("Generating realistic OHLC data...")
    ohlc = generate_ohlc_data(data_size)
    print(f"Price range: High={np.max(ohlc['high']):.2f}, Low={np.min(ohlc['low']):.2f}, Close avg={np.mean(ohlc['close']):.2f}")
    print()
    
    # Периоды для тестирования
    test_periods = [5, 10, 14, 20, 30, 50, 100]
    
    # ==================== ATR ====================
    if HAS_ATR:
        print_separator()
        print("ATR (Average True Range) - Volatility Measure".center(110))
        print_separator()
        print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        for period in test_periods:
            ta_result, cuda_result, ta_time, cuda_time = benchmark_atr(ohlc, period)
            match, max_diff, mean_diff = compare_results(ta_result, cuda_result)
            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"
            
            print(f"{period:<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        
        print_separator()
        print()
    
    # ==================== NATR ====================
    if HAS_NATR:
        print_separator()
        print("NATR (Normalized ATR) - ATR as % of Close Price".center(110))
        print_separator()
        print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        for period in test_periods:
            ta_result, cuda_result, ta_time, cuda_time = benchmark_natr(ohlc, period)
            match, max_diff, mean_diff = compare_results(ta_result, cuda_result)
            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"
            
            print(f"{period:<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        
        print_separator()
        print()
    
    # ==================== TRANGE ====================
    if HAS_TRANGE:
        print_separator()
        print("TRANGE (True Range) - Instantaneous Volatility".center(110))
        print_separator()
        
        ta_result, cuda_result, ta_time, cuda_time = benchmark_trange(ohlc)
        match, max_diff, mean_diff = compare_results(ta_result, cuda_result)
        speedup = ta_time / cuda_time if cuda_time > 0 else 0
        match_str = "PASS" if match else "FAIL"
        
        print(f"{'N/A':<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        print_separator()
        print()
    
    # ==================== BBANDS ====================
    if HAS_BBANDS:
        print_separator()
        print("BBANDS (Bollinger Bands) - Upper/Middle/Lower Bands".center(110))
        print_separator()
        print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        for period in test_periods:
            ta_result, cuda_result, ta_time, cuda_time = benchmark_bbands(ohlc['close'], period)
            match, max_diff, mean_diff = compare_results(ta_result, cuda_result)
            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"
            
            print(f"{period:<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        
        print_separator()
        print()
    
    # ==================== StdDev ====================
    if HAS_STDDEV:
        print_separator()
        print("StdDev (Standard Deviation) - Price Dispersion".center(110))
        print_separator()
        print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        for period in test_periods:
            ta_result, cuda_result, ta_time, cuda_time = benchmark_stddev(ohlc['close'], period)
            match, max_diff, mean_diff = compare_results(ta_result, cuda_result)
            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"
            
            print(f"{period:<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        
        print_separator()
        print()
    
    print()
    print("Notes:")
    print("  - ATR: Average True Range over period (smoothed with Wilder's method)")
    print("  - NATR: Normalized ATR = 100 * ATR / Close")
    print("  - TRANGE: max(H-L, |H-C_prev|, |L-C_prev|)")
    print("  - BBANDS: Middle ± StdDev * nbdev")
    print("  - StdDev: Population standard deviation over period")
    print()

if __name__ == '__main__':
    main()

