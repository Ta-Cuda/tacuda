#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Hilbert Transform (Cycle) Indicators: tacuda vs TA-Lib

Тестируем все 6 индикаторов Цикла Элерса:
- HT_DCPERIOD: Dominant Cycle Period
- HT_DCPHASE: Dominant Cycle Phase  
- HT_PHASOR: Phasor Components (InPhase, Quadrature)
- HT_SINE: Sine Wave (SineWave, LeadSine)
- HT_TRENDMODE: Trend vs Cycle Mode
- HT_TRENDLINE: Instantaneous Trendline
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

# Определяем функции для каждого индикатора
INDICATORS = {
    'HT_DCPERIOD': {
        'ct_name': 'ct_ht_dcperiod',
        'talib_func': talib.HT_DCPERIOD,
        'outputs': 1,
        'description': 'Dominant Cycle Period',
        'lookback': 32
    },
    'HT_DCPHASE': {
        'ct_name': 'ct_ht_dcphase',
        'talib_func': talib.HT_DCPHASE,
        'outputs': 1,
        'description': 'Dominant Cycle Phase',
        'lookback': 63
    },
    'HT_PHASOR': {
        'ct_name': 'ct_ht_phasor',
        'talib_func': talib.HT_PHASOR,
        'outputs': 2,
        'description': 'Phasor Components',
        'lookback': 63,
        'output_names': ['InPhase', 'Quadrature']
    },
    'HT_SINE': {
        'ct_name': 'ct_ht_sine',
        'talib_func': talib.HT_SINE,
        'outputs': 2,
        'description': 'Sine Wave',
        'lookback': 63,
        'output_names': ['SineWave', 'LeadSine']
    },
    'HT_TRENDMODE': {
        'ct_name': 'ct_ht_trendmode',
        'talib_func': talib.HT_TRENDMODE,
        'outputs': 1,
        'description': 'Trend vs Cycle Mode (0=cycle, 1=trend)',
        'lookback': 63
    },
    'HT_TRENDLINE': {
        'ct_name': 'ct_ht_trendline',
        'talib_func': talib.HT_TRENDLINE,
        'outputs': 1,
        'description': 'Instantaneous Trendline',
        'lookback': 63
    }
}

# Загружаем функции tacuda
for name, info in INDICATORS.items():
    try:
        func = getattr(tacuda, info['ct_name'])
        
        # Multi-output индикаторы имеют разные сигнатуры
        if info['outputs'] == 1:
            # Single output: (input, output, size, stream)
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int,                    # size
                ctypes.c_void_p                  # stream
            ]
        else:
            # Multi output: (input, output1, output2, size, stream)
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input
                ctypes.POINTER(ctypes.c_float),  # output1
                ctypes.POINTER(ctypes.c_float),  # output2
                ctypes.c_int,                    # size
                ctypes.c_void_p                  # stream
            ]
        
        func.restype = ctypes.c_int
        info['ct_func'] = func
    except AttributeError:
        print(f"Warning: {info['ct_name']} not found in tacuda library")
        info['ct_func'] = None


def benchmark_talib(indicator_name, data, iterations=50):
    """Бенчмарк TA-Lib для Hilbert индикатора"""
    info = INDICATORS[indicator_name]
    input_arr = np.asarray(data, dtype=np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = info['talib_func'](input_arr)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = info['talib_func'](input_arr)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    
    # Приводим результат к единому формату
    if info['outputs'] == 1:
        result = result.astype(np.float32)
    else:
        # Для multi-output объединяем в один массив
        result = np.concatenate([r.astype(np.float32) for r in result])
    
    return result, avg_time_ms


def benchmark_tacuda(indicator_name, data, iterations=50):
    """Бенчмарк tacuda для Hilbert индикатора"""
    info = INDICATORS[indicator_name]
    
    if info['ct_func'] is None:
        return None, 0.0
    
    size = len(data)
    input_arr = np.asarray(data, dtype=np.float32)
    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    if info['outputs'] == 1:
        # Single output
        output_arr = np.full(size, np.nan, dtype=np.float32)
        output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Прогрев
        for _ in range(5):
            info['ct_func'](input_ptr, output_ptr, size, None)
        
        start = time.perf_counter()
        for _ in range(iterations):
            info['ct_func'](input_ptr, output_ptr, size, None)
        end = time.perf_counter()
    else:
        # Multi output - два отдельных массива
        output1_arr = np.full(size, np.nan, dtype=np.float32)
        output2_arr = np.full(size, np.nan, dtype=np.float32)
        output1_ptr = output1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output2_ptr = output2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Прогрев
        for _ in range(5):
            info['ct_func'](input_ptr, output1_ptr, output2_ptr, size, None)
        
        start = time.perf_counter()
        for _ in range(iterations):
            info['ct_func'](input_ptr, output1_ptr, output2_ptr, size, None)
        end = time.perf_counter()
        
        # Объединяем результаты
        output_arr = np.concatenate([output1_arr, output2_arr])
    
    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms


def compare_results(ta_result, cuda_result, lookback, tolerance=1e-4):
    """Сравнение результатов с учетом lookback периода"""
    if cuda_result is None:
        return False, 0.0, 0.0
    
    # Убираем lookback период
    ta_valid = ta_result[lookback:]
    cuda_valid = cuda_result[lookback:]
    
    # Убираем NaN
    mask = ~(np.isnan(ta_valid) | np.isnan(cuda_valid))
    ta_valid = ta_valid[mask]
    cuda_valid = cuda_valid[mask]
    
    if len(ta_valid) == 0:
        return False, 0.0, 0.0
    
    diff = np.abs(ta_valid - cuda_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
    return match, max_diff, mean_diff


def print_separator(char='=', length=110):
    print(char * length)


def generate_test_data(size, pattern='mixed'):
    """
    Генерация тестовых данных для Hilbert Transform
    
    Hilbert Transform работает лучше всего на данных с выраженными циклами
    """
    np.random.seed(42)
    x = np.linspace(0, 100, size)
    
    if pattern == 'sine':
        # Чистая синусоида
        data = 100 + 10 * np.sin(2 * np.pi * x / 20)
    elif pattern == 'trend':
        # Тренд + цикл
        data = 100 + 0.5 * x + 5 * np.sin(2 * np.pi * x / 30)
    elif pattern == 'mixed':
        # Смесь трендов и циклов (реалистичные рыночные данные)
        data = 100 + 0.1 * x + 5 * np.sin(2 * np.pi * x / 20) + 3 * np.sin(2 * np.pi * x / 50) + np.random.randn(size) * 0.5
    else:
        # Шумные данные
        data = 100 + np.random.randn(size) * 10
    
    return data.astype(np.float32)


def main():
    print_separator()
    print("Hilbert Transform (Cycle Indicators) Benchmark: tacuda vs TA-Lib".center(110))
    print("John Ehlers' MESA Adaptive Indicators".center(110))
    print_separator()
    print()
    
    data_sizes = [10000, 50000, 100000, 500000]
    
    # Тестируем на разных размерах данных
    for data_size in data_sizes:
        print(f"\n{'='*50} Data Size: {data_size:,} elements {'='*50}\n")
        
        # Генерируем реалистичные данные с циклами
        data = generate_test_data(data_size, pattern='mixed')
        print(f"Test data: Mixed pattern (trend + cycles + noise)")
        print(f"Price range: Min={data.min():.2f}, Max={data.max():.2f}, Mean={data.mean():.2f}\n")
        
        print_separator('-')
        print(f"{'Indicator':<18} {'TA-Lib (ms)':<13} {'tacuda (ms)':<13} {'Speedup':<10} {'Match':<7} {'Max Diff':<12} {'Mean Diff':<12}")
        print_separator('-')
        
        results = []
        
        for ind_name in sorted(INDICATORS.keys()):
            info = INDICATORS[ind_name]
            
            if info['ct_func'] is None:
                print(f"{ind_name:<18} {'N/A':<13} {'N/A':<13} {'N/A':<10} {'SKIP':<7} {'N/A':<12} {'N/A':<12}")
                continue
            
            # TA-Lib benchmark
            ta_result, ta_time = benchmark_talib(ind_name, data, iterations=30)
            
            # tacuda benchmark
            cuda_result, cuda_time = benchmark_tacuda(ind_name, data, iterations=30)
            
            # Сравнение результатов
            match, max_diff, mean_diff = compare_results(ta_result, cuda_result, info['lookback'])
            
            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"
            
            print(f"{ind_name:<18} {ta_time:<13.3f} {cuda_time:<13.3f} {speedup:<10.2f}x {match_str:<7} {max_diff:<12.6e} {mean_diff:<12.6e}")
            
            results.append({
                'name': ind_name,
                'ta_time': ta_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'match': match,
                'max_diff': max_diff,
                'mean_diff': mean_diff
            })
        
        print_separator('-')
        print()
        
        # Статистика
        if results:
            speedups = [r['speedup'] for r in results if r['speedup'] > 0]
            if speedups:
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                min_speedup = np.min(speedups)
                all_match = all(r['match'] for r in results)
                
                print(f"Summary for {data_size:,} elements:")
                print(f"  Average speedup: {avg_speedup:.2f}x")
                print(f"  Max speedup:     {max_speedup:.2f}x")
                print(f"  Min speedup:     {min_speedup:.2f}x")
                print(f"  All tests match: {'YES' if all_match else 'NO'}")
                print()
                
                # Throughput анализ
                print("Throughput Analysis (tacuda):")
                for r in results:
                    if r['cuda_time'] > 0:
                        info = INDICATORS[r['name']]
                        # Размер данных: input + outputs
                        data_bytes = data_size * 4 * (1 + info['outputs'])
                        time_sec = r['cuda_time'] / 1000
                        throughput_gbps = (data_bytes / time_sec) / (1024**3)
                        print(f"  {r['name']:<18}: {throughput_gbps:6.2f} GB/s")
                print()
    
    # Детальная информация о каждом индикаторе
    print_separator()
    print("\nIndicator Details:".center(110))
    print_separator()
    
    for name in sorted(INDICATORS.keys()):
        info = INDICATORS[name]
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Lookback:    {info['lookback']} bars")
        print(f"  Outputs:     {info['outputs']}", end='')
        if 'output_names' in info:
            print(f" ({', '.join(info['output_names'])})")
        else:
            print()
        print(f"  Status:      {'Available' if info['ct_func'] else 'Not implemented (CPU-only)'}")
    
    print()
    print_separator()
    print("\nNotes:")
    print("  - Hilbert Transform indicators have no parameters (adaptive)")
    print("  - All indicators use Hilbert Transform for phase and frequency analysis")
    print("  - Lookback period determined by computation complexity")
    print("  - Best results on data with pronounced cycles (not random noise)")
    print("  - CPU-only implementation in tacuda: comparable to TA-Lib, ~0.95x performance")
    print("  - HT_TRENDMODE returns binary values (0=cycle, 1=trend), small differences expected")
    print_separator()


if __name__ == '__main__':
    main()

