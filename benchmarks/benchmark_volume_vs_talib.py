#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Volume Indicators: tacuda vs TA-Lib
Тестируем: OBV, AD, ADOSC, MFI, IMI, NVI, PVI
"""

import time
import numpy as np
import talib
import ctypes
import os
import sys

# Загрузка tacuda
tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release', 'tacuda.dll')
if not os.path.exists(tacuda_lib_path):
    tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libtacuda.so')

# Добавляем путь к CUDA DLLs
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
if os.path.exists(cuda_path):
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(cuda_path)
    except (AttributeError, FileNotFoundError):
        pass

# Добавляем каталог tacuda.dll
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
    print(f"Make sure CUDA runtime DLLs are in PATH")
    sys.exit(1)

# Загружаем функции Volume индикаторов
try:
    tacuda_obv = tacuda.ct_obv
    tacuda_obv.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # volume
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_void_p                  # stream
    ]
    tacuda_obv.restype = ctypes.c_int
except AttributeError:
    tacuda_obv = None
    print("Warning: ct_obv not found")

try:
    tacuda_ad = tacuda.ct_ad
    tacuda_ad.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # volume
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_void_p                  # stream
    ]
    tacuda_ad.restype = ctypes.c_int
except AttributeError:
    tacuda_ad = None
    print("Warning: ct_ad not found")

try:
    tacuda_adosc = tacuda.ct_adosc
    tacuda_adosc.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # volume
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_int,                    # fast period
        ctypes.c_int,                    # slow period
        ctypes.c_void_p                  # stream
    ]
    tacuda_adosc.restype = ctypes.c_int
except AttributeError:
    tacuda_adosc = None
    print("Warning: ct_adosc not found")

try:
    tacuda_mfi = tacuda.ct_mfi
    tacuda_mfi.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # volume
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                    # size
        ctypes.c_int,                    # period
        ctypes.c_void_p                  # stream
    ]
    tacuda_mfi.restype = ctypes.c_int
except AttributeError:
    tacuda_mfi = None
    print("Warning: ct_mfi not found")

# ====================== BENCHMARK FUNCTIONS ======================

def generate_ohlcv_data(size, seed=42):
    """Генерация реалистичных OHLCV данных"""
    np.random.seed(seed)
    
    # Базовая цена с трендом и шумом
    t = np.linspace(0, 100, size)
    trend = 100 + 0.01 * t
    noise = np.random.randn(size) * 2
    close = trend + noise + 10 * np.sin(t / 10)
    
    # High и Low относительно Close
    high = close + np.abs(np.random.randn(size) * 0.5)
    low = close - np.abs(np.random.randn(size) * 0.5)
    
    # Volume с вариацией
    base_volume = 1000000
    volume = base_volume + np.random.randn(size) * base_volume * 0.2
    volume = np.abs(volume)  # Volume всегда положительный
    
    # Open между High и Low
    open_price = low + (high - low) * np.random.rand(size)
    
    return {
        'high': high.astype(np.float32),
        'low': low.astype(np.float32),
        'close': close.astype(np.float32),
        'open': open_price.astype(np.float32),
        'volume': volume.astype(np.float32)
    }

# ====================== OBV BENCHMARKS ======================

def benchmark_obv_talib(close, volume, iterations=50):
    """Бенчмарк TA-Lib OBV"""
    close_f64 = close.astype(np.float64)
    volume_f64 = volume.astype(np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib.OBV(close_f64, volume_f64)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.OBV(close_f64, volume_f64)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_obv_tacuda(close, volume, iterations=50):
    """Бенчмарк tacuda OBV"""
    if tacuda_obv is None:
        return None, 0.0
    
    size = len(close)
    output = np.full(size, np.nan, dtype=np.float32)
    
    close_ptr = close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    volume_ptr = volume.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Прогрев GPU
    for _ in range(5):
        tacuda_obv(close_ptr, volume_ptr, output_ptr, size, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_obv(close_ptr, volume_ptr, output_ptr, size, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return output, avg_time_ms

# ====================== AD BENCHMARKS ======================

def benchmark_ad_talib(high, low, close, volume, iterations=50):
    """Бенчмарк TA-Lib AD"""
    high_f64 = high.astype(np.float64)
    low_f64 = low.astype(np.float64)
    close_f64 = close.astype(np.float64)
    volume_f64 = volume.astype(np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib.AD(high_f64, low_f64, close_f64, volume_f64)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.AD(high_f64, low_f64, close_f64, volume_f64)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_ad_tacuda(high, low, close, volume, iterations=50):
    """Бенчмарк tacuda AD"""
    if tacuda_ad is None:
        return None, 0.0
    
    size = len(close)
    output = np.full(size, np.nan, dtype=np.float32)
    
    high_ptr = high.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = low.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    volume_ptr = volume.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Прогрев GPU
    for _ in range(5):
        tacuda_ad(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr, size, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_ad(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr, size, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return output, avg_time_ms

# ====================== MFI BENCHMARKS ======================

def benchmark_mfi_talib(high, low, close, volume, period, iterations=50):
    """Бенчмарк TA-Lib MFI"""
    high_f64 = high.astype(np.float64)
    low_f64 = low.astype(np.float64)
    close_f64 = close.astype(np.float64)
    volume_f64 = volume.astype(np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib.MFI(high_f64, low_f64, close_f64, volume_f64, timeperiod=period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.MFI(high_f64, low_f64, close_f64, volume_f64, timeperiod=period)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_mfi_tacuda(high, low, close, volume, period, iterations=50):
    """Бенчмарк tacuda MFI"""
    if tacuda_mfi is None:
        return None, 0.0
    
    size = len(close)
    output = np.full(size, np.nan, dtype=np.float32)
    
    high_ptr = high.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = low.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    volume_ptr = volume.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Прогрев GPU
    for _ in range(5):
        tacuda_mfi(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr, size, period, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_mfi(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr, size, period, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return output, avg_time_ms

# ====================== ADOSC BENCHMARKS ======================

def benchmark_adosc_talib(high, low, close, volume, fast, slow, iterations=50):
    """Бенчмарк TA-Lib ADOSC"""
    high_f64 = high.astype(np.float64)
    low_f64 = low.astype(np.float64)
    close_f64 = close.astype(np.float64)
    volume_f64 = volume.astype(np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib.ADOSC(high_f64, low_f64, close_f64, volume_f64, fastperiod=fast, slowperiod=slow)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.ADOSC(high_f64, low_f64, close_f64, volume_f64, fastperiod=fast, slowperiod=slow)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_adosc_tacuda(high, low, close, volume, fast, slow, iterations=50):
    """Бенчмарк tacuda ADOSC"""
    if tacuda_adosc is None:
        return None, 0.0
    
    size = len(close)
    output = np.full(size, np.nan, dtype=np.float32)
    
    high_ptr = high.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = low.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    volume_ptr = volume.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Прогрев GPU
    for _ in range(5):
        tacuda_adosc(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr, size, fast, slow, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_adosc(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr, size, fast, slow, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return output, avg_time_ms

# ====================== COMPARISON & REPORTING ======================

def compare_results(ta_result, cuda_result, tolerance=1e-4):
    """Сравнение результатов с учетом NaN"""
    if cuda_result is None:
        return False, 0.0, 0.0
    
    # Маска валидных значений
    mask = ~(np.isnan(ta_result) | np.isnan(cuda_result))
    
    if not np.any(mask):
        return True, 0.0, 0.0
    
    ta_valid = ta_result[mask]
    cuda_valid = cuda_result[mask]
    
    diff = np.abs(ta_valid - cuda_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
    return match, max_diff, mean_diff

def print_separator():
    print("=" * 110)

def print_header(title):
    print_separator()
    print(title.center(110))
    print_separator()

def main():
    print_header("Volume Indicators Benchmark: tacuda vs TA-Lib")
    print()
    
    data_size = 500000
    print(f"Data size: {data_size:,} elements (OHLCV)")
    print()
    
    # Генерация OHLCV данных
    print("Generating realistic OHLCV data...")
    data = generate_ohlcv_data(data_size)
    print(f"Price range: Low={np.min(data['low']):.2f}, High={np.max(data['high']):.2f}, Close avg={np.mean(data['close']):.2f}")
    print(f"Volume range: Min={np.min(data['volume']):.0f}, Max={np.max(data['volume']):.0f}, Avg={np.mean(data['volume']):.0f}")
    print()
    
    # ==================== OBV ====================
    print_header("1. OBV (On Balance Volume)")
    if tacuda_obv is not None:
        print(f"{'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        ta_result, ta_time = benchmark_obv_talib(data['close'], data['volume'])
        cu_result, cu_time = benchmark_obv_tacuda(data['close'], data['volume'])
        match, max_diff, mean_diff = compare_results(ta_result, cu_result)
        speedup = ta_time / cu_time if cu_time > 0 else 0
        
        print(f"{ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {'PASS' if match else 'FAIL':<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        print()
        
        # Sample values
        print("Sample OBV values:")
        print(f"{'Index':<10} {'TA-Lib':<15} {'tacuda':<15} {'Diff':<15}")
        print("-" * 60)
        for idx in [0, 100, 1000, 10000, 100000]:
            if idx < len(ta_result):
                diff = abs(ta_result[idx] - cu_result[idx]) if not (np.isnan(ta_result[idx]) or np.isnan(cu_result[idx])) else np.nan
                print(f"{idx:<10} {ta_result[idx]:<15.2f} {cu_result[idx]:<15.2f} {diff:<15.6e}")
        print()
    else:
        print("OBV not available")
        print()
    
    # ==================== AD ====================
    print_header("2. AD (Accumulation/Distribution)")
    if tacuda_ad is not None:
        print(f"{'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        ta_result, ta_time = benchmark_ad_talib(data['high'], data['low'], data['close'], data['volume'])
        cu_result, cu_time = benchmark_ad_tacuda(data['high'], data['low'], data['close'], data['volume'])
        match, max_diff, mean_diff = compare_results(ta_result, cu_result)
        speedup = ta_time / cu_time if cu_time > 0 else 0
        
        print(f"{ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {'PASS' if match else 'FAIL':<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        print()
    else:
        print("AD not available")
        print()
    
    # ==================== MFI ====================
    print_header("3. MFI (Money Flow Index)")
    if tacuda_mfi is not None:
        periods = [7, 14, 21, 28]
        print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        for period in periods:
            ta_result, ta_time = benchmark_mfi_talib(data['high'], data['low'], data['close'], data['volume'], period)
            cu_result, cu_time = benchmark_mfi_tacuda(data['high'], data['low'], data['close'], data['volume'], period)
            match, max_diff, mean_diff = compare_results(ta_result, cu_result)
            speedup = ta_time / cu_time if cu_time > 0 else 0
            
            print(f"{period:<10} {ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {'PASS' if match else 'FAIL':<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        print()
    else:
        print("MFI not available")
        print()
    
    # ==================== ADOSC ====================
    print_header("4. ADOSC (Accumulation/Distribution Oscillator)")
    if tacuda_adosc is not None:
        configs = [(3, 10), (5, 15), (7, 21)]
        print(f"{'Fast/Slow':<12} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
        print_separator()
        
        for fast, slow in configs:
            ta_result, ta_time = benchmark_adosc_talib(data['high'], data['low'], data['close'], data['volume'], fast, slow)
            cu_result, cu_time = benchmark_adosc_tacuda(data['high'], data['low'], data['close'], data['volume'], fast, slow)
            match, max_diff, mean_diff = compare_results(ta_result, cu_result)
            speedup = ta_time / cu_time if cu_time > 0 else 0
            
            print(f"{fast}/{slow:<10} {ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {'PASS' if match else 'FAIL':<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        print()
    else:
        print("ADOSC not available")
        print()
    
    print_separator()
    print()
    print("Notes:")
    print("  - OBV: On Balance Volume - cumulative volume indicator")
    print("  - AD: Accumulation/Distribution - volume flow indicator")
    print("  - MFI: Money Flow Index - volume-weighted RSI")
    print("  - ADOSC: Chaikin Oscillator - EMA of AD")
    print()

if __name__ == '__main__':
    main()

