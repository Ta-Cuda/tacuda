#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark MAVP (Moving Average with Variable Period): tacuda vs TA-Lib
Тестируем адаптивные скользящие средние с переменным периодом
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
    tacuda_mavp = tacuda.ct_mavp
except AttributeError as exc:
    raise SystemExit("ct_mavp symbol not found in tacuda library") from exc

# ct_mavp(const float* values, const float* periods, float* output,
#         int size, int minPeriod, int maxPeriod, int maType, cudaStream_t stream)
tacuda_mavp.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # values
    ctypes.POINTER(ctypes.c_float),  # periods
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # minPeriod
    ctypes.c_int,                    # maxPeriod
    ctypes.c_int,                    # maType (0=SMA, 1=EMA, etc.)
    ctypes.c_void_p                  # stream
]
tacuda_mavp.restype = ctypes.c_int

def generate_periods(size, min_period, max_period, pattern='sine'):
    """
    Генерация массива периодов для MAVP
    
    Паттерны:
    - 'sine': синусоидальное изменение периода
    - 'trend': линейное изменение от min к max
    - 'random': случайные периоды
    - 'step': ступенчатое изменение
    - 'constant': константный период (для тестирования)
    """
    if pattern == 'sine':
        # Синусоидальное изменение: медленно меняющийся период
        x = np.linspace(0, 4 * np.pi, size)
        normalized = (np.sin(x) + 1) / 2  # [0, 1]
        periods = min_period + normalized * (max_period - min_period)
    elif pattern == 'trend':
        # Линейный тренд от min к max
        periods = np.linspace(min_period, max_period, size)
    elif pattern == 'random':
        # Случайные периоды
        periods = np.random.uniform(min_period, max_period, size)
    elif pattern == 'step':
        # Ступенчатое изменение каждые size/10 точек
        steps = 10
        step_size = size // steps
        periods = np.repeat(np.linspace(min_period, max_period, steps), step_size)[:size]
    elif pattern == 'constant':
        # Константный период (средний)
        avg_period = (min_period + max_period) // 2
        periods = np.full(size, avg_period, dtype=np.float32)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return periods.astype(np.float32)

def benchmark_talib(data, periods, min_period, max_period, matype=0, iterations=50):
    """
    Бенчмарк TA-Lib MAVP
    matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
    """
    data_arr = np.asarray(data, dtype=np.float64)
    periods_arr = np.asarray(periods, dtype=np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib.MAVP(data_arr, periods_arr, 
                       minperiod=min_period, 
                       maxperiod=max_period,
                       matype=matype)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.MAVP(data_arr, periods_arr,
                           minperiod=min_period,
                           maxperiod=max_period, 
                           matype=matype)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_tacuda(data, periods, min_period, max_period, matype=0, iterations=50):
    """Бенчмарк tacuda MAVP"""
    size = len(data)
    data_arr = np.asarray(data, dtype=np.float32)
    periods_arr = np.asarray(periods, dtype=np.float32)
    output_arr = np.full(size, np.nan, dtype=np.float32)

    data_ptr = data_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    periods_ptr = periods_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        tacuda_mavp(data_ptr, periods_ptr, output_ptr, 
                    size, min_period, max_period, matype, None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_mavp(data_ptr, periods_ptr, output_ptr,
                    size, min_period, max_period, matype, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms

def compare_results(ta_result, cuda_result, tolerance=1e-4):
    """Сравнение результатов (MAVP может иметь больше NaN из-за переменных периодов)"""
    # Находим валидные данные (не NaN в обоих массивах)
    mask = ~(np.isnan(ta_result) | np.isnan(cuda_result))
    
    if np.sum(mask) == 0:
        return False, 0.0, 0.0, 0
    
    ta_valid = ta_result[mask]
    cuda_valid = cuda_result[mask]
    
    diff = np.abs(ta_valid - cuda_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    valid_count = len(ta_valid)
    
    match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
    return match, max_diff, mean_diff, valid_count

def print_separator():
    print("=" * 120)

def get_matype_name(matype):
    """Получить имя типа MA по номеру"""
    names = {
        0: 'SMA',
        1: 'EMA', 
        2: 'WMA',
        3: 'DEMA',
        4: 'TEMA',
        5: 'TRIMA',
        6: 'KAMA',
        7: 'MAMA',
        8: 'T3'
    }
    return names.get(matype, f'Unknown({matype})')

def main():
    print_separator()
    print("MAVP (Moving Average with Variable Period) Benchmark: tacuda vs TA-Lib".center(120))
    print_separator()
    print()

    data_size = 100000
    print(f"Data size: {data_size:,} elements")
    print()

    # Генерация тестовых данных
    np.random.seed(42)
    x = np.linspace(0, 100, data_size)
    data = np.sin(x) + np.random.randn(data_size) * 0.05

    # Тестовые конфигурации
    test_configs = [
        # (min_period, max_period, pattern, matype, name)
        (5, 50, 'sine', 0, 'SMA-Sine'),
        (10, 100, 'sine', 1, 'EMA-Sine'),
        (5, 50, 'trend', 0, 'SMA-Trend'),
        (10, 100, 'trend', 1, 'EMA-Trend'),
        (10, 50, 'random', 0, 'SMA-Random'),
        (10, 50, 'step', 0, 'SMA-Step'),
        (20, 20, 'constant', 0, 'SMA-Const(20)'),  # Эквивалентно обычной SMA
        (5, 50, 'sine', 2, 'WMA-Sine'),
        (10, 100, 'sine', 3, 'DEMA-Sine'),
    ]

    print_separator()
    print(f"{'Config':<20} {'MAType':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} "
          f"{'Speedup':<12} {'Match':<8} {'Valid pts':<12} {'Max Diff':<15}")
    print_separator()

    results = []

    for min_p, max_p, pattern, matype, name in test_configs:
        # Генерируем периоды
        periods = generate_periods(data_size, min_p, max_p, pattern)
        
        try:
            # TA-Lib benchmark
            ta_result, ta_time = benchmark_talib(data, periods, min_p, max_p, matype, iterations=30)

            # tacuda benchmark
            cuda_result, cuda_time = benchmark_tacuda(data, periods, min_p, max_p, matype, iterations=30)

            # Сравнение результатов
            match, max_diff, mean_diff, valid_count = compare_results(ta_result, cuda_result)

            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"

            print(f"{name:<20} {get_matype_name(matype):<10} {ta_time:<15.3f} {cuda_time:<15.3f} "
                  f"{speedup:<12.2f}x {match_str:<8} {valid_count:<12} {max_diff:<15.6e}")

            results.append({
                'name': name,
                'matype': matype,
                'min_period': min_p,
                'max_period': max_p,
                'pattern': pattern,
                'ta_time': ta_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'match': match,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'valid_count': valid_count
            })
        except Exception as e:
            print(f"{name:<20} {get_matype_name(matype):<10} ERROR: {str(e)}")

    print_separator()
    print()

    # Статистика
    if results:
        speedups = [r['speedup'] for r in results]
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)

        all_match = all(r['match'] for r in results)

        print("Summary:")
        print(f"  Tests run:       {len(results)}")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Max speedup:     {max_speedup:.2f}x")
        print(f"  Min speedup:     {min_speedup:.2f}x")
        print(f"  All tests match: {'YES' if all_match else 'NO'}")
        print()

        # Анализ по типам MA
        print("Performance by MA Type:")
        matype_groups = {}
        for r in results:
            mt = r['matype']
            if mt not in matype_groups:
                matype_groups[mt] = []
            matype_groups[mt].append(r['speedup'])
        
        for mt in sorted(matype_groups.keys()):
            avg_sp = np.mean(matype_groups[mt])
            print(f"  {get_matype_name(mt):<10}: {avg_sp:6.2f}x average speedup")
        print()

        # Throughput анализ для самого быстрого теста
        best = max(results, key=lambda r: r['speedup'])
        print(f"Best Performance (tacuda {best['name']}):")
        # Размер данных: values + periods + output (3x для float32)
        data_bytes = data_size * 4 * 3
        time_sec = best['cuda_time'] / 1000
        throughput_gbps = (data_bytes / time_sec) / (1024**3)
        print(f"  Throughput: {throughput_gbps:6.2f} GB/s")
        print(f"  Time:       {best['cuda_time']:.3f} ms")
        print(f"  Speedup:    {best['speedup']:.2f}x")
        print()

if __name__ == '__main__':
    main()

