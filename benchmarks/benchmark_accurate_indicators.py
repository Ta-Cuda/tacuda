#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark: tacuda vs TA-Lib - Only Verified Accurate Indicators
Compares performance for indicators that have been verified to match TA-Lib exactly.
"""

import numpy as np
import time
import ctypes
import os
import sys
import talib

# Load tacuda library
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
    sys.exit(1)

def load_func(name, argtypes):
    try:
        func = getattr(tacuda, name)
        func.argtypes = argtypes
        func.restype = ctypes.c_int
        return func
    except AttributeError:
        return None

# Load verified accurate indicators
ct_rsi = load_func('ct_rsi', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_ema = load_func('ct_ema', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_sma = load_func('ct_sma', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_atr = load_func('ct_atr', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
])

ct_stochastic = load_func('ct_stochastic', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_bbands = load_func('ct_bbands', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p
])

ct_willr = load_func('ct_willr', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_roc = load_func('ct_roc', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_momentum = load_func('ct_momentum', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_wma = load_func('ct_wma', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])


def print_header(title, width=110):
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def generate_ohlcv(size):
    """Generate synthetic OHLCV data"""
    np.random.seed(42)
    
    price = 100.0
    opens = np.zeros(size, dtype=np.float64)
    highs = np.zeros(size, dtype=np.float64)
    lows = np.zeros(size, dtype=np.float64)
    closes = np.zeros(size, dtype=np.float64)
    volumes = np.random.uniform(100000, 2000000, size).astype(np.float64)
    
    for i in range(size):
        change = np.random.randn() * 0.5
        price = max(1.0, price + change)
        
        o = price
        h = price + abs(np.random.randn()) * 2
        l = price - abs(np.random.randn()) * 2
        c = price + np.random.randn()
        
        h = max(o, h, c)
        l = min(o, l, c)
        
        opens[i] = o
        highs[i] = h
        lows[i] = l
        closes[i] = c
        price = c
    
    return opens, highs, lows, closes, volumes


def benchmark_talib(closes64, highs64, lows64, iterations=10):
    """TA-Lib benchmark for verified indicators"""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # 10 verified indicators
        rsi = talib.RSI(closes64, timeperiod=14)
        ema12 = talib.EMA(closes64, timeperiod=12)
        ema26 = talib.EMA(closes64, timeperiod=26)
        sma20 = talib.SMA(closes64, timeperiod=20)
        wma14 = talib.WMA(closes64, timeperiod=14)
        atr = talib.ATR(highs64, lows64, closes64, timeperiod=14)
        slowk, slowd = talib.STOCH(highs64, lows64, closes64, fastk_period=14, slowk_period=3, slowd_period=3)
        upper, middle, lower = talib.BBANDS(closes64, timeperiod=20, nbdevup=2, nbdevdn=2)
        willr = talib.WILLR(highs64, lows64, closes64, timeperiod=14)
        roc = talib.ROC(closes64, timeperiod=10)
        mom = talib.MOM(closes64, timeperiod=10)
        
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times)


def benchmark_tacuda(closes32, highs32, lows32, iterations=10):
    """tacuda benchmark for verified indicators"""
    size = len(closes32)
    
    # Output buffers
    rsi_out = np.zeros(size, dtype=np.float32)
    ema12_out = np.zeros(size, dtype=np.float32)
    ema26_out = np.zeros(size, dtype=np.float32)
    sma20_out = np.zeros(size, dtype=np.float32)
    wma14_out = np.zeros(size, dtype=np.float32)
    atr_out = np.zeros(size, dtype=np.float32)
    stoch_k = np.zeros(size, dtype=np.float32)
    stoch_d = np.zeros(size, dtype=np.float32)
    bb_upper = np.zeros(size, dtype=np.float32)
    bb_middle = np.zeros(size, dtype=np.float32)
    bb_lower = np.zeros(size, dtype=np.float32)
    willr_out = np.zeros(size, dtype=np.float32)
    roc_out = np.zeros(size, dtype=np.float32)
    mom_out = np.zeros(size, dtype=np.float32)
    
    close_ptr = closes32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    high_ptr = highs32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = lows32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # 10 verified indicators (same as TA-Lib)
        ct_rsi(close_ptr, rsi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 14, None)
        ct_ema(close_ptr, ema12_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 12, None)
        ct_ema(close_ptr, ema26_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 26, None)
        ct_sma(close_ptr, sma20_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 20, None)
        ct_wma(close_ptr, wma14_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 14, None)
        ct_atr(high_ptr, low_ptr, close_ptr, atr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 14, 0.0, None)
        ct_stochastic(high_ptr, low_ptr, close_ptr,
                      stoch_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      stoch_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 14, 3, None)
        ct_bbands(close_ptr, 
                  bb_upper.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  bb_middle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  bb_lower.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 20, 2.0, 2.0, None)
        ct_willr(high_ptr, low_ptr, close_ptr, willr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 14, None)
        ct_roc(close_ptr, roc_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 10, None)
        ct_momentum(close_ptr, mom_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 10, None)
        
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times)


def main():
    print_header("Verified Indicators Benchmark: tacuda vs TA-Lib")
    print()
    print("Testing 11 verified accurate indicators:")
    print("  RSI(14), EMA(12), EMA(26), SMA(20), WMA(14), ATR(14),")
    print("  Stochastic(14,3,3), Bollinger(20,2,2), Williams %R(14), ROC(10), Momentum(10)")
    print()
    
    sizes = [50000, 100000, 250000, 500000, 1000000, 2000000]
    
    print(f"{'Size':>12} | {'TA-Lib (ms)':>14} | {'tacuda (ms)':>14} | {'Speedup':>10} | {'Winner':>10}")
    print("-" * 75)
    
    results = []
    
    for size in sizes:
        opens, highs, lows, closes, volumes = generate_ohlcv(size)
        
        closes32 = closes.astype(np.float32)
        highs32 = highs.astype(np.float32)
        lows32 = lows.astype(np.float32)
        
        talib_time = benchmark_talib(closes, highs, lows)
        tacuda_time = benchmark_tacuda(closes32, highs32, lows32)
        
        speedup = talib_time / tacuda_time if tacuda_time > 0 else 0
        winner = "tacuda" if speedup > 1 else "TA-Lib"
        
        print(f"{size:>12,} | {talib_time:>14.2f} | {tacuda_time:>14.2f} | {speedup:>10.2f}x | {winner:>10}")
        
        results.append({
            'size': size,
            'talib': talib_time,
            'tacuda': tacuda_time,
            'speedup': speedup
        })
    
    print()
    print("=" * 75)
    print()
    
    # Analysis
    crossover_size = None
    for r in results:
        if r['speedup'] >= 1.0:
            crossover_size = r['size']
            break
    
    if crossover_size:
        print(f"Crossover point: tacuda becomes faster at ~{crossover_size:,} bars")
    else:
        print("tacuda is slower than TA-Lib for all tested sizes")
        print("This is due to host<->device transfer overhead for each indicator call")
    
    print()
    print("Key insight:")
    print("  - Each ct_* call does: allocate GPU mem -> copy to GPU -> compute -> copy back -> free")
    print("  - This overhead dominates for smaller datasets")
    print("  - tacuda::Context (Pipeline) eliminates intermediate transfers")
    print("  - Expected Pipeline speedup: 2-5x over current results")
    print()
    
    # Calculate per-indicator times
    print("Per-indicator timing (1M bars):")
    r = results[-2] if len(results) >= 2 else results[-1]  # Use 1M result
    print(f"  TA-Lib:  {r['talib']/11:.2f} ms per indicator")
    print(f"  tacuda:  {r['tacuda']/11:.2f} ms per indicator")
    print(f"  Overhead per call: ~{(r['tacuda']/11 - r['talib']/11):.2f} ms (transfer + alloc)")


if __name__ == "__main__":
    main()

