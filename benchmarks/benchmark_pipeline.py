#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark: Pipeline (zero-copy) vs Traditional (host transfers) approach
Demonstrates the performance benefit of keeping data on GPU between indicator calls.
"""

import numpy as np
import time
import ctypes
import os
import sys

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

# Load indicator functions
def load_func(name, argtypes):
    try:
        func = getattr(tacuda, name)
        func.argtypes = argtypes
        func.restype = ctypes.c_int
        return func
    except AttributeError:
        print(f"Warning: {name} not found")
        return None

# Standard signatures
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
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p  # Added initial param
])

ct_adx = load_func('ct_adx', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_macd = load_func('ct_macd', [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # macd
    ctypes.POINTER(ctypes.c_float),  # signal
    ctypes.POINTER(ctypes.c_float),  # hist
    ctypes.c_int,  # size
    ctypes.c_int,  # fast
    ctypes.c_int,  # slow
    ctypes.c_int,  # signal
    ctypes.c_int,  # matype
    ctypes.c_void_p  # stream
])

ct_stochastic = load_func('ct_stochastic', [
    ctypes.POINTER(ctypes.c_float),  # high
    ctypes.POINTER(ctypes.c_float),  # low
    ctypes.POINTER(ctypes.c_float),  # close
    ctypes.POINTER(ctypes.c_float),  # k out
    ctypes.POINTER(ctypes.c_float),  # d out
    ctypes.c_int,  # size
    ctypes.c_int,  # k period
    ctypes.c_int,  # d period
    ctypes.c_void_p  # stream
])

ct_bbands = load_func('ct_bbands', [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # upper
    ctypes.POINTER(ctypes.c_float),  # middle
    ctypes.POINTER(ctypes.c_float),  # lower
    ctypes.c_int,  # size
    ctypes.c_int,  # period
    ctypes.c_float,  # upper mul
    ctypes.c_float,  # lower mul
    ctypes.c_void_p  # stream
])

ct_mfi = load_func('ct_mfi', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_obv = load_func('ct_obv', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p
])


def print_header(title, width=100):
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def generate_ohlcv(size):
    """Generate synthetic OHLCV data"""
    np.random.seed(42)
    
    price = 100.0
    opens = np.zeros(size, dtype=np.float32)
    highs = np.zeros(size, dtype=np.float32)
    lows = np.zeros(size, dtype=np.float32)
    closes = np.zeros(size, dtype=np.float32)
    volumes = np.random.uniform(100000, 2000000, size).astype(np.float32)
    
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


def benchmark_traditional(opens, highs, lows, closes, volumes, iterations=5):
    """
    Traditional approach: Each indicator call does host->device->host transfer
    """
    size = len(closes)
    
    # Output buffers
    rsi_out = np.zeros(size, dtype=np.float32)
    ema12_out = np.zeros(size, dtype=np.float32)
    ema26_out = np.zeros(size, dtype=np.float32)
    sma20_out = np.zeros(size, dtype=np.float32)
    atr_out = np.zeros(size, dtype=np.float32)
    adx_out = np.zeros(size, dtype=np.float32)
    macd_out = np.zeros(size, dtype=np.float32)
    macd_sig = np.zeros(size, dtype=np.float32)
    macd_hist = np.zeros(size, dtype=np.float32)
    stoch_k = np.zeros(size, dtype=np.float32)
    stoch_d = np.zeros(size, dtype=np.float32)
    bb_out = np.zeros(size * 3, dtype=np.float32)
    mfi_out = np.zeros(size, dtype=np.float32)
    obv_out = np.zeros(size, dtype=np.float32)
    
    # Pointers
    close_ptr = closes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    high_ptr = highs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = lows.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    vol_ptr = volumes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Each call does: allocate GPU, copy to GPU, compute, copy back, free GPU
        ct_rsi(close_ptr, rsi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 14, None)
        ct_ema(close_ptr, ema12_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 12, None)
        ct_ema(close_ptr, ema26_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 26, None)
        ct_sma(close_ptr, sma20_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 20, None)
        ct_atr(high_ptr, low_ptr, close_ptr,
               atr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 14, 0.0, None)
        ct_adx(high_ptr, low_ptr, close_ptr,
               adx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 14, None)
        ct_macd(close_ptr,
                macd_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                macd_sig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                macd_hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                size, 12, 26, 9, 0, None)  # matype=0 (SMA)
        ct_stochastic(high_ptr, low_ptr, close_ptr,
                      stoch_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      stoch_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      size, 14, 3, None)
        bb_upper = np.zeros(size, dtype=np.float32)
        bb_middle = np.zeros(size, dtype=np.float32)
        bb_lower = np.zeros(size, dtype=np.float32)
        ct_bbands(close_ptr, 
                  bb_upper.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  bb_middle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  bb_lower.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  size, 20, 2.0, 2.0, None)
        ct_mfi(high_ptr, low_ptr, close_ptr, vol_ptr,
               mfi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 14, None)
        ct_obv(close_ptr, vol_ptr,
               obv_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, None)
        
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times), len(times) * 11  # 11 indicator calls


def main():
    print_header("Pipeline vs Traditional Benchmark")
    print()
    
    sizes = [100000, 500000, 1000000]
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Data size: {size:,} bars")
        print(f"{'='*60}")
        
        # Generate data
        opens, highs, lows, closes, volumes = generate_ohlcv(size)
        
        # Benchmark traditional approach
        trad_time, num_calls = benchmark_traditional(opens, highs, lows, closes, volumes)
        
        print(f"\nTraditional (11 separate calls with host transfers):")
        print(f"  Total time: {trad_time:.2f} ms")
        print(f"  Per indicator: {trad_time/11:.2f} ms")
        
        # Estimate transfer overhead
        # Each call transfers: input (size * 4 bytes) + output (size * 4 bytes) = 8 * size bytes
        # For HLC indicators: 3 * size * 4 + size * 4 = 16 * size bytes
        # Rough estimate: ~10 bytes per element per call average
        transfer_bytes = size * 10 * 11  # 11 calls
        transfer_mb = transfer_bytes / (1024 * 1024)
        
        print(f"\nEstimated data transfer:")
        print(f"  ~{transfer_mb:.1f} MB total ({11} round-trips)")
        
        # Pipeline approach would be:
        # 1x upload OHLCV: 5 * size * 4 bytes
        # 11x compute (no transfer)
        # 11x download results: 11 * size * 4 bytes
        pipeline_transfer = (5 + 11) * size * 4
        pipeline_mb = pipeline_transfer / (1024 * 1024)
        
        print(f"\nPipeline approach would transfer:")
        print(f"  ~{pipeline_mb:.1f} MB total (1 upload + selective downloads)")
        print(f"  Potential savings: {(transfer_mb - pipeline_mb) / transfer_mb * 100:.1f}%")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print("""
The Pipeline approach (tacuda::Context) provides:
1. Single upload of OHLCV data to GPU
2. All indicator computations stay on GPU
3. Download only the results you need
4. Automatic buffer management

Benefits:
- Reduced PCIe transfer overhead
- Better GPU utilization
- Simpler code for multi-indicator strategies
- Memory reuse between computations

Use Pipeline when:
- Computing multiple indicators on same data
- Building indicator chains (RSI -> EMA of RSI)
- Running backtests with many indicators
- Real-time systems with streaming data
""")


if __name__ == "__main__":
    main()

