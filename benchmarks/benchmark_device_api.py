#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark: tacuda Device API vs Host API vs TA-Lib

Compares three approaches:
1. TA-Lib: CPU-based, highly optimized
2. tacuda Host API (ct_*): Each call does host->device->host transfer
3. tacuda Device API (ct_*_device): Data stays on GPU, no intermediate transfers

Expected results:
- Device API should be 5-10x faster than Host API for multiple indicators
- Device API should be faster than TA-Lib for large datasets
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

def load_func(name, argtypes, restype=ctypes.c_int):
    try:
        func = getattr(tacuda, name)
        func.argtypes = argtypes
        func.restype = restype
        return func
    except AttributeError:
        print(f"Warning: {name} not found")
        return None

# Host API functions (existing)
ct_sma = load_func('ct_sma', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_ema = load_func('ct_ema', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_rsi = load_func('ct_rsi', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_bbands = load_func('ct_bbands', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p
])
ct_momentum = load_func('ct_momentum', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_roc = load_func('ct_roc', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_stddev = load_func('ct_stddev', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

# Device API functions (using existing _device suffix)
ct_sma_device = load_func('ct_sma_device', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_ema_device = load_func('ct_ema_device', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_rsi_device = load_func('ct_rsi_device', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_momentum_device = load_func('ct_momentum_device', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_roc_device = load_func('ct_roc_device', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])
ct_stddev_device = load_func('ct_stddev_device', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])


def print_header(title, width=110):
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def generate_data(size):
    """Generate synthetic price data"""
    np.random.seed(42)
    price = 100.0
    closes = np.zeros(size, dtype=np.float64)
    
    for i in range(size):
        change = np.random.randn() * 0.5
        price = max(1.0, price + change)
        closes[i] = price
    
    return closes


def benchmark_talib(closes64, iterations=10):
    """TA-Lib benchmark: 6 indicators"""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        sma20 = talib.SMA(closes64, timeperiod=20)
        ema12 = talib.EMA(closes64, timeperiod=12)
        ema26 = talib.EMA(closes64, timeperiod=26)
        rsi14 = talib.RSI(closes64, timeperiod=14)
        mom10 = talib.MOM(closes64, timeperiod=10)
        roc10 = talib.ROC(closes64, timeperiod=10)
        std20 = talib.STDDEV(closes64, timeperiod=20)
        
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times)


def benchmark_host_api(closes32, iterations=10):
    """tacuda Host API: Each call transfers data"""
    size = len(closes32)
    
    # Output buffers
    sma_out = np.zeros(size, dtype=np.float32)
    ema12_out = np.zeros(size, dtype=np.float32)
    ema26_out = np.zeros(size, dtype=np.float32)
    rsi_out = np.zeros(size, dtype=np.float32)
    mom_out = np.zeros(size, dtype=np.float32)
    roc_out = np.zeros(size, dtype=np.float32)
    std_out = np.zeros(size, dtype=np.float32)
    
    close_ptr = closes32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        ct_sma(close_ptr, sma_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 20, None)
        ct_ema(close_ptr, ema12_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 12, None)
        ct_ema(close_ptr, ema26_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 26, None)
        ct_rsi(close_ptr, rsi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 14, None)
        ct_momentum(close_ptr, mom_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 10, None)
        ct_roc(close_ptr, roc_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 10, None)
        ct_stddev(close_ptr, std_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 20, None)
        
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times)


def benchmark_device_api(closes32, iterations=10):
    """tacuda Device API: Data stays on GPU using cudaMalloc/cudaMemcpy directly"""
    size = len(closes32)
    
    # Check if device functions are available
    if not all([ct_sma_device, ct_ema_device, ct_rsi_device, ct_momentum_device, ct_roc_device, ct_stddev_device]):
        print("Device API functions not available, skipping...")
        return -1
    
    # Use cudaMalloc/cudaMemcpy via ctypes
    try:
        cudart = ctypes.CDLL('cudart64_129.dll')
    except:
        try:
            cudart = ctypes.CDLL('cudart64_12.dll')
        except:
            print("CUDA runtime not found, skipping device API test")
            return -1
    
    cudaMalloc = cudart.cudaMalloc
    cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudaMalloc.restype = ctypes.c_int
    
    cudaFree = cudart.cudaFree
    cudaFree.argtypes = [ctypes.c_void_p]
    cudaFree.restype = ctypes.c_int
    
    cudaMemcpy = cudart.cudaMemcpy
    cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudaMemcpy.restype = ctypes.c_int
    
    cudaDeviceSynchronize = cudart.cudaDeviceSynchronize
    cudaDeviceSynchronize.restype = ctypes.c_int
    
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    
    # Allocate device buffers
    d_input = ctypes.c_void_p()
    d_sma = ctypes.c_void_p()
    d_ema12 = ctypes.c_void_p()
    d_ema26 = ctypes.c_void_p()
    d_rsi = ctypes.c_void_p()
    d_mom = ctypes.c_void_p()
    d_roc = ctypes.c_void_p()
    d_std = ctypes.c_void_p()
    
    buf_size = size * ctypes.sizeof(ctypes.c_float)
    
    err = cudaMalloc(ctypes.byref(d_input), buf_size)
    if err != 0:
        print(f"cudaMalloc failed: {err}")
        return -1
    cudaMalloc(ctypes.byref(d_sma), buf_size)
    cudaMalloc(ctypes.byref(d_ema12), buf_size)
    cudaMalloc(ctypes.byref(d_ema26), buf_size)
    cudaMalloc(ctypes.byref(d_rsi), buf_size)
    cudaMalloc(ctypes.byref(d_mom), buf_size)
    cudaMalloc(ctypes.byref(d_roc), buf_size)
    cudaMalloc(ctypes.byref(d_std), buf_size)
    
    close_ptr = closes32.ctypes.data_as(ctypes.c_void_p)
    host_out = np.zeros(size, dtype=np.float32)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Upload data once
        cudaMemcpy(d_input, close_ptr, buf_size, cudaMemcpyHostToDevice)
        
        # All computations on GPU - no intermediate transfers!
        ct_sma_device(ctypes.cast(d_input, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(d_sma, ctypes.POINTER(ctypes.c_float)), size, 20, None)
        ct_ema_device(ctypes.cast(d_input, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(d_ema12, ctypes.POINTER(ctypes.c_float)), size, 12, None)
        ct_ema_device(ctypes.cast(d_input, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(d_ema26, ctypes.POINTER(ctypes.c_float)), size, 26, None)
        ct_rsi_device(ctypes.cast(d_input, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(d_rsi, ctypes.POINTER(ctypes.c_float)), size, 14, None)
        ct_momentum_device(ctypes.cast(d_input, ctypes.POINTER(ctypes.c_float)),
                           ctypes.cast(d_mom, ctypes.POINTER(ctypes.c_float)), size, 10, None)
        ct_roc_device(ctypes.cast(d_input, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(d_roc, ctypes.POINTER(ctypes.c_float)), size, 10, None)
        ct_stddev_device(ctypes.cast(d_input, ctypes.POINTER(ctypes.c_float)),
                         ctypes.cast(d_std, ctypes.POINTER(ctypes.c_float)), size, 20, None)
        
        # Download only what we need (just RSI for verification)
        cudaMemcpy(host_out.ctypes.data_as(ctypes.c_void_p), d_rsi, buf_size, cudaMemcpyDeviceToHost)
        
        # Sync
        cudaDeviceSynchronize()
        
        times.append((time.perf_counter() - start) * 1000)
    
    # Free device memory
    cudaFree(d_input)
    cudaFree(d_sma)
    cudaFree(d_ema12)
    cudaFree(d_ema26)
    cudaFree(d_rsi)
    cudaFree(d_mom)
    cudaFree(d_roc)
    cudaFree(d_std)
    
    return np.median(times)


def main():
    print_header("Device API Benchmark: tacuda Device vs Host vs TA-Lib")
    print()
    print("Testing 7 indicators: SMA(20), EMA(12), EMA(26), RSI(14), MOM(10), ROC(10), STDDEV(20)")
    print()
    print("Approaches:")
    print("  1. TA-Lib:        CPU-based, highly optimized C code")
    print("  2. tacuda Host:   Each ct_*() call does host->device->host transfer")
    print("  3. tacuda Device: Upload once -> compute all -> download results")
    print()
    
    sizes = [100000, 250000, 500000, 1000000, 2000000]
    
    print(f"{'Size':>12} | {'TA-Lib':>12} | {'Host API':>12} | {'Device API':>12} | {'Device/Host':>12} | {'Device/TA-Lib':>14}")
    print("-" * 95)
    
    results = []
    
    for size in sizes:
        closes64 = generate_data(size)
        closes32 = closes64.astype(np.float32)
        
        talib_time = benchmark_talib(closes64)
        host_time = benchmark_host_api(closes32)
        device_time = benchmark_device_api(closes32)
        
        if device_time > 0:
            speedup_vs_host = host_time / device_time
            speedup_vs_talib = talib_time / device_time
            print(f"{size:>12,} | {talib_time:>10.2f}ms | {host_time:>10.2f}ms | {device_time:>10.2f}ms | {speedup_vs_host:>11.2f}x | {speedup_vs_talib:>13.2f}x")
        else:
            speedup_vs_host = 0
            speedup_vs_talib = 0
            print(f"{size:>12,} | {talib_time:>10.2f}ms | {host_time:>10.2f}ms | {'N/A':>12} | {'N/A':>12} | {'N/A':>14}")
        
        results.append({
            'size': size,
            'talib': talib_time,
            'host': host_time,
            'device': device_time,
            'speedup_host': speedup_vs_host,
            'speedup_talib': speedup_vs_talib
        })
    
    print()
    print("=" * 95)
    print()
    
    # Summary
    valid_results = [r for r in results if r['device'] > 0]
    if valid_results:
        avg_speedup_host = np.mean([r['speedup_host'] for r in valid_results])
        avg_speedup_talib = np.mean([r['speedup_talib'] for r in valid_results])
        
        print(f"Average Device API speedup vs Host API:  {avg_speedup_host:.2f}x")
        print(f"Average Device API speedup vs TA-Lib:    {avg_speedup_talib:.2f}x")
    else:
        print("Device API test was not run (CUDA runtime not available or functions missing)")
    
    print()
    
    # Analysis
    print("=" * 95)
    print(f"{'ANALYSIS':^95}")
    print("=" * 95)
    print("""
Key findings:

1. Device API eliminates 7 host->device + 7 device->host transfers per iteration
   - Host API: 14 transfers per call batch
   - Device API: 1 upload + 1 download = 2 transfers

2. Memory bandwidth savings:
   - For 1M floats: 14 * 4MB = 56MB vs 2 * 4MB = 8MB per iteration
   - That's 7x less PCIe bandwidth usage!

3. GPU kernel launch overhead is amortized:
   - Device pointers are already valid, no allocation needed
   - Kernels can be queued back-to-back

Recommendations:
- Use Device API (ct_*_device) when computing multiple indicators on same data
- Use Host API (ct_*) for single indicator calls or simple scripts
- For real-time streaming, keep buffers on GPU and update incrementally
""")


if __name__ == "__main__":
    import io
    
    # Capture output to both console and file
    output_file = __file__.replace('.py', '_results.txt')
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    import sys
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = Tee(sys.__stdout__, f)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
    
    print(f"\nResults saved to: {output_file}")
