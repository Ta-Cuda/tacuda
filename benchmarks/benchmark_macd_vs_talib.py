#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark MACD: tacuda vs TA-Lib
Сравниваем производительность Moving Average Convergence Divergence на разных параметрах
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

# Добавляем каталог tacuda.dll для корректной загрузки зависимостей (Python 3.8+)
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

try:
    tacuda_macd = tacuda.ct_macd
except AttributeError as exc:
    raise SystemExit("ct_macd symbol not found in tacuda library") from exc

# ct_macd(const float* host_input, float* host_macd, float* host_signal, float* host_hist,
#         int size, int fastPeriod, int slowPeriod, int signalPeriod, ctMaType_t type, cudaStream_t stream)
tacuda_macd.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output macd
    ctypes.POINTER(ctypes.c_float),  # output signal
    ctypes.POINTER(ctypes.c_float),  # output histogram
    ctypes.c_int,                    # size
    ctypes.c_int,                    # fastPeriod
    ctypes.c_int,                    # slowPeriod
    ctypes.c_int,                    # signalPeriod
    ctypes.c_int,                    # maType (0=SMA, 1=EMA)
    ctypes.c_void_p                  # stream
]
tacuda_macd.restype = ctypes.c_int

def benchmark_talib(data, fast=12, slow=26, signal=9, iterations=100):
    """Бенчмарк TA-Lib MACD"""
    input_arr = np.asarray(data, dtype=np.float64)
    # Прогрев
    for _ in range(5):
        _ = talib.MACD(input_arr, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    start = time.perf_counter()
    for _ in range(iterations):
        macd, signal_line, hist = talib.MACD(input_arr, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    end = time.perf_counter()
    avg_time_ms = (end - start) / iterations * 1000
    return (macd.astype(np.float32), signal_line.astype(np.float32), hist.astype(np.float32)), avg_time_ms

def benchmark_tacuda(data, fast=12, slow=26, signal=9, iterations=100, use_device_signal=False):
    """Бенчмарк tacuda MACD"""
    # Enable device-side signal calculation if requested
    if use_device_signal:
        os.environ['TACUDA_USE_DEVICE_SIGNAL'] = '1'
    else:
        os.environ.pop('TACUDA_USE_DEVICE_SIGNAL', None)
    
    size = len(data)
    input_arr = np.asarray(data, dtype=np.float32)
    output_macd = np.full(size, np.nan, dtype=np.float32)
    output_signal = np.full(size, np.nan, dtype=np.float32)
    output_hist = np.full(size, np.nan, dtype=np.float32)

    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    macd_ptr = output_macd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    signal_ptr = output_signal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hist_ptr = output_hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        tacuda_macd(input_ptr, macd_ptr, signal_ptr, hist_ptr, size, fast, slow, signal, 1, None)  # 1=EMA

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_macd(input_ptr, macd_ptr, signal_ptr, hist_ptr, size, fast, slow, signal, 1, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return (output_macd, output_signal, output_hist), avg_time_ms

def compare_results(ta_results, cuda_results, slow, signal, tolerance=5e-4):
    """
    Сравнение результатов MACD
    
    TA-Lib не возвращает MACD/Signal/Hist пока Signal не готов (индекс slow+signal-2).
    tacuda возвращает MACD раньше, но для честного сравнения сравниваем начиная 
    с первого валидного индекса TA-Lib.
    """
    ta_macd, ta_signal, ta_hist = ta_results
    cuda_macd, cuda_signal, cuda_hist = cuda_results
    
    # TA-Lib первый валидный индекс (для всех трёх компонентов)
    signal_start = slow + signal - 2
    
    results = {}
    
    # Сравнение MACD line (начиная с signal_start где TA-Lib имеет первое значение)
    ta_macd_valid = ta_macd[signal_start:]
    cuda_macd_valid = cuda_macd[signal_start:]
    mask = ~(np.isnan(ta_macd_valid) | np.isnan(cuda_macd_valid))
    if mask.any():
        diff = np.abs(ta_macd_valid[mask] - cuda_macd_valid[mask])
        results['macd'] = {
            'match': np.allclose(ta_macd_valid[mask], cuda_macd_valid[mask], rtol=tolerance, atol=tolerance),
            'max_diff': np.max(diff),
            'mean_diff': np.mean(diff)
        }
    else:
        results['macd'] = {'match': False, 'max_diff': 0.0, 'mean_diff': 0.0}
    
    # Сравнение Signal line
    ta_signal_valid = ta_signal[signal_start:]
    cuda_signal_valid = cuda_signal[signal_start:]
    mask = ~(np.isnan(ta_signal_valid) | np.isnan(cuda_signal_valid))
    if mask.any():
        diff = np.abs(ta_signal_valid[mask] - cuda_signal_valid[mask])
        results['signal'] = {
            'match': np.allclose(ta_signal_valid[mask], cuda_signal_valid[mask], rtol=tolerance, atol=tolerance),
            'max_diff': np.max(diff),
            'mean_diff': np.mean(diff)
        }
    else:
        results['signal'] = {'match': False, 'max_diff': 0.0, 'mean_diff': 0.0}
    
    # Сравнение Histogram
    ta_hist_valid = ta_hist[signal_start:]
    cuda_hist_valid = cuda_hist[signal_start:]
    mask = ~(np.isnan(ta_hist_valid) | np.isnan(cuda_hist_valid))
    if mask.any():
        diff = np.abs(ta_hist_valid[mask] - cuda_hist_valid[mask])
        results['hist'] = {
            'match': np.allclose(ta_hist_valid[mask], cuda_hist_valid[mask], rtol=tolerance, atol=tolerance),
            'max_diff': np.max(diff),
            'mean_diff': np.mean(diff)
        }
    else:
        results['hist'] = {'match': False, 'max_diff': 0.0, 'mean_diff': 0.0}
    
    # Общий результат
    all_match = all(r['match'] for r in results.values())
    max_diff = max(r['max_diff'] for r in results.values())
    mean_diff = np.mean([r['mean_diff'] for r in results.values()])
    
    return all_match, max_diff, mean_diff, results

def print_separator():
    print("=" * 130)

def main():
    print("=" * 130)
    print("MACD Benchmark: tacuda vs TA-Lib".center(130))
    print("=" * 130)
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements")
    print()

    # Генерация реалистичных тестовых данных (цены с трендом и волатильностью)
    np.random.seed(42)
    x = np.linspace(0, 100, data_size)
    trend = 0.1 * x  # восходящий тренд
    seasonal = 10 * np.sin(2 * np.pi * x / 50)  # сезонные колебания
    noise = np.random.randn(data_size) * 2  # случайный шум
    data = 100 + trend + seasonal + noise  # базовая цена 100
    data = np.maximum(data, 1.0)  # no negative prices

    # Разные конфигурации MACD
    test_configs = [
        # (fast, slow, signal, description)
        (12, 26, 9, "Standard MACD (12,26,9)"),
        (5, 13, 5, "Fast MACD (5,13,5)"),
        (19, 39, 9, "Slow MACD (19,39,9)"),
        (8, 17, 9, "Custom MACD (8,17,9)"),
        (6, 12, 6, "Short-term MACD (6,12,6)"),
        (24, 52, 18, "Long-term MACD (24,52,18)"),
    ]

    # Тест обоих режимов: Host-path (default) и Device-path (optimized)
    for mode_name, use_device in [("Host-path (default)", False), ("Device-path (optimized)", True)]:
        print_separator()
        print(f"{mode_name} - {'TACUDA_USE_DEVICE_SIGNAL=1' if use_device else 'Standard mode'}".center(130))
        print_separator()
        print(f"{'Config':<30} {'Fast':<6} {'Slow':<6} {'Signal':<8} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} "
              f"{'Speedup':<12} {'Match':<8} {'Max Diff':<15}")
        print_separator()

        results = []

        for fast, slow, signal, description in test_configs:
            if slow >= data_size or fast >= slow:
                continue

            # TA-Lib benchmark
            ta_result, ta_time = benchmark_talib(data.astype(np.float32), fast, slow, signal, iterations=50)

            # tacuda benchmark
            cuda_result, cuda_time = benchmark_tacuda(data, fast, slow, signal, iterations=50, use_device_signal=use_device)

            # Сравнение результатов
            match, max_diff, mean_diff, detailed = compare_results(ta_result, cuda_result, slow, signal)

            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"

            print(f"{description:<30} {fast:<6} {slow:<6} {signal:<8} {ta_time:<15.3f} {cuda_time:<15.3f} "
                  f"{speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e}")

            results.append({
                'config': description,
                'fast': fast,
                'slow': slow,
                'signal': signal,
                'ta_time': ta_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'match': match,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'detailed': detailed,
                'mode': mode_name
            })

        print_separator()
        print()

        # Статистика для текущего режима
        if results:
            speedups = [r['speedup'] for r in results]
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            min_speedup = np.min(speedups)

            all_match = all(r['match'] for r in results)

            print(f"Summary for {mode_name}:")
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Max speedup:     {max_speedup:.2f}x")
            print(f"  Min speedup:     {min_speedup:.2f}x")
            print(f"  All tests match: {'YES' if all_match else 'NO'}")
            print()

            # Детальная точность для каждого компонента
            print("Accuracy by Component:")
            for component in ['macd', 'signal', 'hist']:
                component_matches = sum(1 for r in results if r['detailed'][component]['match'])
                max_diff_component = max(r['detailed'][component]['max_diff'] for r in results)
                print(f"  {component.upper():<10}: {component_matches}/{len(results)} matches, "
                      f"max diff = {max_diff_component:.6e}")
            print()

            # Throughput анализ для лучшего случая
            best_result = max(results, key=lambda r: r['speedup'])
            # MACD обрабатывает 1 вход и 3 выхода
            data_bytes = data_size * 4 * 4  # input + 3 outputs (float32)
            time_sec = best_result['cuda_time'] / 1000
            throughput_gbps = (data_bytes / time_sec) / (1024**3)
            
            print(f"Throughput Analysis (tacuda - {mode_name}):")
            print(f"  Config: {best_result['config']}")
            print(f"  Throughput: {throughput_gbps:.2f} GB/s")
            print(f"  Speedup: {best_result['speedup']:.2f}x")
            print()

    print("MACD Components Explained:".center(130))
    print_separator()
    print()
    print("  * MACD Line    = Fast EMA - Slow EMA    (основная линия индикатора)")
    print("  * Signal Line  = EMA of MACD Line       (сигнальная линия)")
    print("  * Histogram    = MACD Line - Signal     (гистограмма расхождения)")
    print()
    print("Trading Signals:")
    print("  * MACD > Signal (Histogram > 0): Бычий сигнал (покупка)")
    print("  * MACD < Signal (Histogram < 0): Медвежий сигнал (продажа)")
    print("  * Histogram растет: Усиление тренда")
    print("  * Histogram уменьшается: Ослабление тренда")
    print()
    print_separator()

if __name__ == '__main__':
    main()

