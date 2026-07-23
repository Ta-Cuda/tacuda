#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Candlestick Patterns: tacuda vs TA-Lib
Сравниваем производительность распознавания свечных паттернов
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

# Candlestick паттерны для тестирования
PATTERNS = [
    # Название паттерна, функция C API, функция TA-Lib
    ("Doji", "ct_cdl_doji", talib.CDLDOJI),
    ("Hammer", "ct_cdl_hammer", talib.CDLHAMMER),
    ("InvertedHammer", "ct_cdl_inverted_hammer", talib.CDLINVERTEDHAMMER),
    ("HangingMan", "ct_cdl_hanging_man", talib.CDLHANGINGMAN),
    ("ShootingStar", "ct_cdl_shooting_star", talib.CDLSHOOTINGSTAR),
    ("Engulfing", "ct_cdl_engulfing", talib.CDLENGULFING),
    ("Harami", "ct_cdl_harami", talib.CDLHARAMI),
    ("MorningStar", "ct_cdl_morning_star", talib.CDLMORNINGSTAR),
    ("EveningStar", "ct_cdl_evening_star", talib.CDLEVENINGSTAR),
    ("ThreeWhiteSoldiers", "ct_cdl_three_white_soldiers", talib.CDL3WHITESOLDIERS),
    ("ThreeBlackCrows", "ct_cdl_three_black_crows", talib.CDL3BLACKCROWS),
    ("DarkCloudCover", "ct_cdl_dark_cloud_cover", talib.CDLDARKCLOUDCOVER),
    ("Piercing", "ct_cdl_piercing", talib.CDLPIERCING),
    ("BullishEngulfing", "ct_cdl_bullish_engulfing", talib.CDLENGULFING),
    ("BearishEngulfing", "ct_cdl_bearish_engulfing", talib.CDLENGULFING),
    ("DojiStar", "ct_cdl_doji_star", talib.CDLDOJISTAR),
    ("DragonflyDoji", "ct_cdl_dragonfly_doji", talib.CDLDRAGONFLYDOJI),
    ("GravestoneDoji", "ct_cdl_gravestone_doji", talib.CDLGRAVESTONEDOJI),
    ("MorningDojiStar", "ct_cdl_morning_doji_star", talib.CDLMORNINGDOJISTAR),
    ("EveningDojiStar", "ct_cdl_evening_doji_star", talib.CDLEVENINGDOJISTAR),
]

def get_tacuda_function(func_name):
    """Получение функции из библиотеки tacuda"""
    try:
        func = getattr(tacuda, func_name)
        # OHLC candlestick patterns: 4 inputs + 1 output + size + stream
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # open
            ctypes.POINTER(ctypes.c_float),  # high
            ctypes.POINTER(ctypes.c_float),  # low
            ctypes.POINTER(ctypes.c_float),  # close
            ctypes.POINTER(ctypes.c_float),  # output (float values: -100/0/100)
            ctypes.c_int,                    # size
            ctypes.c_void_p                  # stream
        ]
        func.restype = ctypes.c_int
        return func
    except AttributeError:
        return None

def generate_ohlc_data(size, seed=42):
    """Генерация реалистичных OHLC данных"""
    np.random.seed(seed)
    
    # Базовая цена с трендом и волатильностью
    base_price = 100.0
    trend = np.linspace(0, 20, size)
    noise = np.cumsum(np.random.randn(size) * 0.5)
    close = base_price + trend + noise
    
    # Генерируем OHLC с реалистичными отношениями
    high_offset = np.abs(np.random.randn(size) * 2)
    low_offset = np.abs(np.random.randn(size) * 2)
    open_offset = np.random.randn(size) * 1
    
    high = close + high_offset
    low = close - low_offset
    open_price = close + open_offset
    
    # Корректируем, чтобы high >= max(open, close) и low <= min(open, close)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    return open_price.astype(np.float32), high.astype(np.float32), low.astype(np.float32), close.astype(np.float32)

def benchmark_talib_pattern(open_arr, high, low, close, talib_func, iterations=50):
    """Бенчмарк TA-Lib паттерна"""
    # Конвертируем в float64 для TA-Lib
    o = np.asarray(open_arr, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib_func(o, h, l, c)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib_func(o, h, l, c)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.int32), avg_time_ms

def benchmark_tacuda_pattern(open_arr, high, low, close, tacuda_func, iterations=50):
    """Бенчмарк tacuda паттерна"""
    size = len(open_arr)
    
    # Входные данные (float32)
    o = np.asarray(open_arr, dtype=np.float32)
    h = np.asarray(high, dtype=np.float32)
    l = np.asarray(low, dtype=np.float32)
    c = np.asarray(close, dtype=np.float32)
    
    # Выходной массив (float values -100/0/100)
    output_arr = np.zeros(size, dtype=np.float32)
    
    o_ptr = o.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_ptr = h.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    l_ptr = l.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Прогрев GPU
    for _ in range(5):
        tacuda_func(o_ptr, h_ptr, l_ptr, c_ptr, out_ptr, size, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_func(o_ptr, h_ptr, l_ptr, c_ptr, out_ptr, size, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms

def compare_results(ta_result, cuda_result, tolerance=0):
    """Сравнение результатов (точное совпадение для целых чисел)"""
    ta_int = np.asarray(ta_result, dtype=np.int32)
    cuda_int = np.asarray(np.rint(cuda_result), dtype=np.int32)
    match = np.array_equal(ta_int, cuda_int)
    
    if not match:
        diff_mask = ta_int != cuda_int
        num_diff = np.sum(diff_mask)
        total = len(ta_result)
        
        # Находим несовпадающие значения
        diff_indices = np.where(diff_mask)[0]
        if len(diff_indices) > 0:
            sample_idx = diff_indices[0]
            ta_val = ta_result[sample_idx]
            cuda_val = cuda_result[sample_idx]
        else:
            ta_val = cuda_val = 0
            
        return match, num_diff, total, ta_val, cuda_val
    
    return match, 0, len(ta_result), 0, 0

def print_separator():
    print("=" * 120)

def main():
    print_separator()
    print("Candlestick Patterns Benchmark: tacuda vs TA-Lib".center(120))
    print("Pattern Recognition Performance Comparison".center(120))
    print_separator()
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} candlesticks (OHLC bars)")
    print()
    
    # Генерация реалистичных OHLC данных
    print("Generating realistic OHLC data...")
    open_data, high_data, low_data, close_data = generate_ohlc_data(data_size)
    
    print(f"Price range: Open=[{open_data.min():.2f}, {open_data.max():.2f}], "
          f"High=[{high_data.min():.2f}, {high_data.max():.2f}], "
          f"Low=[{low_data.min():.2f}, {low_data.max():.2f}], "
          f"Close=[{close_data.min():.2f}, {close_data.max():.2f}]")
    print()

    print_separator()
    print(f"{'Pattern':<25} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<10} {'Signals':<15}")
    print_separator()

    results = []
    available_patterns = []
    missing_patterns = []

    for pattern_name, c_func_name, talib_func in PATTERNS:
        # Проверяем наличие функции в tacuda
        tacuda_func = get_tacuda_function(c_func_name)
        
        if tacuda_func is None:
            missing_patterns.append(pattern_name)
            continue
        
        available_patterns.append(pattern_name)
        
        try:
            # TA-Lib benchmark
            ta_result, ta_time = benchmark_talib_pattern(
                open_data, high_data, low_data, close_data, talib_func, iterations=50
            )
            
            # tacuda benchmark
            cuda_result, cuda_time = benchmark_tacuda_pattern(
                open_data, high_data, low_data, close_data, tacuda_func, iterations=50
            )
            
            # Сравнение результатов
            match, num_diff, total, ta_sample, cuda_sample = compare_results(ta_result, cuda_result)
            
            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else f"FAIL ({num_diff}/{total})"
            
            # Подсчет сигналов
            num_signals_ta = np.sum(ta_result != 0)
            num_signals_cuda = np.sum(cuda_result != 0)
            signals_str = f"TA:{num_signals_ta} / CU:{num_signals_cuda}"
            
            print(f"{pattern_name:<25} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<10} {signals_str:<15}")
            
            results.append({
                'pattern': pattern_name,
                'ta_time': ta_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'match': match,
                'num_signals_ta': num_signals_ta,
                'num_signals_cuda': num_signals_cuda
            })
            
        except Exception as e:
            print(f"{pattern_name:<25} ERROR: {str(e)}")

    print_separator()
    print()

    if not results:
        print("No patterns were successfully tested!")
        return

    # Статистика
    speedups = [r['speedup'] for r in results]
    avg_speedup = np.mean(speedups) if speedups else 0.0
    max_speedup = np.max(speedups) if speedups else 0.0
    min_speedup = np.min(speedups) if speedups else 0.0

    all_match = all(r['match'] for r in results)
    match_count = sum(1 for r in results if r['match'])

    print("Summary:")
    print(f"  Patterns tested:  {len(results)}/{len(PATTERNS)}")
    print(f"  Average speedup:  {avg_speedup:.2f}x")
    print(f"  Max speedup:      {max_speedup:.2f}x")
    print(f"  Min speedup:      {min_speedup:.2f}x")
    print(f"  Patterns matched: {match_count}/{len(results)} {'[OK]' if all_match else '[FAIL]'}")
    print()

    # Throughput анализ
    print("Throughput Analysis (tacuda):")
    for r in results:
        # Размер данных: 4 массива float32 (OHLC) + 1 массив int32 (output) = 5 * size * 4 bytes
        data_bytes = data_size * 4 * 5
        time_sec = r['cuda_time'] / 1000
        throughput_gbps = (data_bytes / time_sec) / (1024**3)
        print(f"  {r['pattern']:<25}: {throughput_gbps:6.2f} GB/s")
    print()

    if missing_patterns:
        print(f"Missing patterns ({len(missing_patterns)}):")
        for p in missing_patterns:
            print(f"  - {p}")
        print()

    # Дополнительная статистика по сигналам
    print("Pattern Detection Statistics:")
    print(f"{'Pattern':<25} {'TA-Lib Signals':<20} {'tacuda Signals':<20} {'Match':<10}")
    print("-" * 80)
    for r in results:
        match_sym = "OK" if r['num_signals_ta'] == r['num_signals_cuda'] else "DIFF"
        print(f"{r['pattern']:<25} {r['num_signals_ta']:<20} {r['num_signals_cuda']:<20} {match_sym:<10}")
    print()

    print("Notes:")
    print("  - Candlestick patterns return integer values: -100 (bearish), 0 (no pattern), 100 (bullish)")
    print("  - Pattern recognition is deterministic and should match exactly")
    print("  - Performance gain comes from GPU parallel processing of OHLC data")
    print()

if __name__ == '__main__':
    main()

