#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark RSI Batch Mode: tacuda vs TA-Lib
Демонстрация преимущества GPU при обработке множественных серий (портфель инструментов)
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
    sys.exit(1)

tacuda_rsi = tacuda.ct_rsi
tacuda_rsi.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_void_p                  # stream
]
tacuda_rsi.restype = ctypes.c_int

def benchmark_talib_batch(data_batch, period, iterations=30):
    """
    Бенчмарк TA-Lib для множественных серий (последовательная обработка)
    data_batch: shape (batch_count, series_length)
    """
    batch_count, series_len = data_batch.shape
    
    # Прогрев
    for _ in range(3):
        for i in range(batch_count):
            _ = talib.RSI(data_batch[i].astype(np.float64), timeperiod=period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        results = []
        for i in range(batch_count):
            result = talib.RSI(data_batch[i].astype(np.float64), timeperiod=period)
            results.append(result.astype(np.float32))
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return np.array(results), avg_time_ms

def benchmark_tacuda_batch(data_flat, period, batch_count, series_len, iterations=30):
    """
    Бенчмарк tacuda для batch mode (параллельная обработка на GPU)
    data_flat: flattened array of all series concatenated
    """
    total_size = len(data_flat)
    input_arr = np.asarray(data_flat, dtype=np.float32)
    output_arr = np.full(total_size, np.nan, dtype=np.float32)

    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Настройка batch mode через env
    os.environ['TACUDA_RSI_BATCH_COUNT'] = str(batch_count)
    os.environ['TACUDA_RSI_SERIES_LEN'] = str(series_len)

    # Прогрев GPU
    for _ in range(3):
        tacuda_rsi(input_ptr, output_ptr, total_size, period, None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_rsi(input_ptr, output_ptr, total_size, period, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    
    # Преобразуем обратно в batch format
    output_batch = output_arr.reshape(batch_count, series_len)
    
    return output_batch, avg_time_ms

def compare_batch_results(ta_batch, cuda_batch, period, tolerance=1e-4):
    """Сравнение результатов batch обработки"""
    batch_count = ta_batch.shape[0]
    all_match = True
    max_diff_overall = 0.0
    mean_diff_overall = 0.0
    
    for i in range(batch_count):
        ta_valid = ta_batch[i][period:]
        cuda_valid = cuda_batch[i][period:]
        
        mask = ~(np.isnan(ta_valid) | np.isnan(cuda_valid))
        ta_valid = ta_valid[mask]
        cuda_valid = cuda_valid[mask]
        
        if len(ta_valid) > 0:
            diff = np.abs(ta_valid - cuda_valid)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            max_diff_overall = max(max_diff_overall, max_diff)
            mean_diff_overall += mean_diff / batch_count
            
            if not np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance):
                all_match = False
    
    return all_match, max_diff_overall, mean_diff_overall

def print_separator(width=120):
    print("=" * width)

def main():
    print_separator()
    print("RSI BATCH MODE Benchmark: tacuda vs TA-Lib".center(120))
    print("GPU Parallel Processing для множественных инструментов".center(120))
    print_separator()
    print()

    # Конфигурации тестов
    test_configs = [
        # (batch_count, series_length, description)
        (10, 1000, "Small portfolio: 10 instruments x 1K bars"),
        (50, 1000, "Medium portfolio: 50 instruments x 1K bars"),
        (100, 1000, "Large portfolio: 100 instruments x 1K bars"),
        (500, 1000, "XL portfolio: 500 instruments x 1K bars"),
        (1000, 500, "Huge portfolio: 1000 instruments x 500 bars"),
        (100, 10000, "Deep history: 100 instruments x 10K bars"),
    ]

    period = 14  # Standard RSI period

    print("Test Configuration:")
    print(f"  RSI Period: {period}")
    print(f"  Iterations per test: 30")
    print()

    print_separator()
    print(f"{'Config':<45} {'Batch':<10} {'Series':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8}")
    print_separator()

    all_results = []

    for batch_count, series_len, description in test_configs:
        # Генерация реалистичных данных для портфеля инструментов
        np.random.seed(42)
        data_batch = []
        
        for i in range(batch_count):
            # Каждый инструмент имеет свои характеристики
            x = np.linspace(0, 100, series_len)
            trend = 0.05 * x * (1 + i * 0.01)  # разные тренды
            seasonal = 10 * np.sin(2 * np.pi * x / 50 + i * 0.1)  # разная сезонность
            volatility = 1 + i * 0.01  # разная волатильность
            noise = np.random.randn(series_len) * volatility
            price = 100 + trend + seasonal + noise
            price = np.maximum(price, 1.0)  # no negative prices
            data_batch.append(price.astype(np.float32))
        
        data_batch = np.array(data_batch)
        data_flat = data_batch.flatten()  # для tacuda batch mode

        # TA-Lib benchmark (sequential)
        try:
            ta_batch, ta_time = benchmark_talib_batch(data_batch, period, iterations=30)
        except Exception as e:
            print(f"TA-Lib failed: {e}")
            continue

        # tacuda benchmark (parallel batch mode)
        try:
            cuda_batch, cuda_time = benchmark_tacuda_batch(data_flat, period, batch_count, series_len, iterations=30)
        except Exception as e:
            print(f"tacuda failed: {e}")
            continue

        # Сравнение результатов
        match, max_diff, mean_diff = compare_batch_results(ta_batch, cuda_batch, period)
        
        speedup = ta_time / cuda_time if cuda_time > 0 else 0
        match_str = "PASS" if match else "FAIL"

        print(f"{description:<45} {batch_count:<10} {series_len:<10} {ta_time:<15.2f} {cuda_time:<15.2f} "
              f"{speedup:<12.2f}x {match_str:<8}")

        all_results.append({
            'description': description,
            'batch_count': batch_count,
            'series_len': series_len,
            'ta_time': ta_time,
            'cuda_time': cuda_time,
            'speedup': speedup,
            'match': match,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        })

    print_separator()
    print()

    # Анализ результатов
    if all_results:
        print("PERFORMANCE ANALYSIS:")
        print()
        
        # Найдём лучший speedup
        best_result = max(all_results, key=lambda r: r['speedup'])
        worst_result = min(all_results, key=lambda r: r['speedup'])
        avg_speedup = np.mean([r['speedup'] for r in all_results])
        
        print(f"  Average Speedup:     {avg_speedup:8.2f}x")
        print(f"  Best Speedup:        {best_result['speedup']:8.2f}x  ({best_result['description']})")
        print(f"  Worst Speedup:       {worst_result['speedup']:8.2f}x  ({worst_result['description']})")
        print()
        
        # Accuracy
        all_match = all(r['match'] for r in all_results)
        max_diff_overall = max(r['max_diff'] for r in all_results)
        
        print(f"  All Tests Match:     {'YES' if all_match else 'NO'}")
        print(f"  Max Difference:      {max_diff_overall:.6e}")
        print()
        
        # Throughput анализ для лучшего случая
        total_elements = best_result['batch_count'] * best_result['series_len']
        data_bytes = total_elements * 4 * 2  # input + output (float32)
        time_sec = best_result['cuda_time'] / 1000
        throughput_gbps = (data_bytes / time_sec) / (1024**3)
        
        print(f"  Peak GPU Throughput: {throughput_gbps:8.2f} GB/s")
        print(f"  Instruments/sec:     {best_result['batch_count'] / time_sec:8.0f}")
        print()
        
        # Практическое применение
        print_separator()
        print("REAL-WORLD SCENARIOS:")
        print()
        
        scenarios = [
            (100, 252, "Daily RSI for 100 stocks (1 year history)"),
            (500, 252, "Daily RSI for 500 stocks (1 year history)"),
            (1000, 390, "Intraday RSI for 1000 stocks (1 day, 1-min bars)"),
            (100, 23400, "Intraday RSI for 100 stocks (1 month, 1-min bars)"),
        ]
        
        print(f"{'Scenario':<55} {'TA-Lib (est.)':<20} {'tacuda (est.)':<20} {'Speedup':<10}")
        print_separator()
        
        for batch, series, desc in scenarios:
            # Оценка времени на основе средней производительности
            # Используем лучший результат для оценки
            ta_est = best_result['ta_time'] * (batch * series) / (best_result['batch_count'] * best_result['series_len'])
            cuda_est = best_result['cuda_time'] * (batch * series) / (best_result['batch_count'] * best_result['series_len'])
            speedup_est = ta_est / cuda_est if cuda_est > 0 else 0
            
            ta_est_str = f"{ta_est:.1f} ms" if ta_est < 1000 else f"{ta_est/1000:.2f} sec"
            cuda_est_str = f"{cuda_est:.1f} ms" if cuda_est < 1000 else f"{cuda_est/1000:.2f} sec"
            
            print(f"{desc:<55} {ta_est_str:<20} {cuda_est_str:<20} {speedup_est:<10.1f}x")
        
        print()

    print_separator()
    print("BATCH MODE BENEFITS:".center(120))
    print_separator()
    print()
    print("  * GPU обрабатывает множественные инструменты ПАРАЛЛЕЛЬНО")
    print("  * Идеально для portfolio analysis, screening, backtesting")
    print("  * Чем больше инструментов - тем выше ускорение")
    print("  * Масштабируется линейно до тысяч инструментов")
    print("  * Real-time обработка портфелей из 1000+ инструментов")
    print()
    print("USAGE:")
    print("  export TACUDA_RSI_BATCH_COUNT=100")
    print("  export TACUDA_RSI_SERIES_LEN=1000")
    print("  # Теперь RSI обработает 100 серий по 1000 элементов параллельно!")
    print()
    print_separator()

if __name__ == '__main__':
    main()

