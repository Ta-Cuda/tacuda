#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark: tacuda Pipeline vs TA-Lib
Compares performance of computing multiple indicators on the same dataset.

tacuda Pipeline: Upload once → Compute all → Download results
TA-Lib: Each indicator processes data independently
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

# tacuda function signatures
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

ct_adx = load_func('ct_adx', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_macd = load_func('ct_macd', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
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

ct_mfi = load_func('ct_mfi', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_obv = load_func('ct_obv', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p
])

ct_cci = load_func('ct_cci', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

ct_willr = load_func('ct_willr', [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
])

# Device API helpers (upload once -> compute batch -> download once)
ct_device_alloc = getattr(tacuda, 'ct_device_alloc', None)
if ct_device_alloc is not None:
    ct_device_alloc.argtypes = [ctypes.c_int]
    ct_device_alloc.restype = ctypes.c_void_p

ct_device_free = getattr(tacuda, 'ct_device_free', None)
if ct_device_free is not None:
    ct_device_free.argtypes = [ctypes.c_void_p]
    ct_device_free.restype = None

ct_device_upload = load_func('ct_device_upload', [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p
])
ct_device_download = load_func('ct_device_download', [
    ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
])
ct_device_sync = load_func('ct_device_sync', [ctypes.c_void_p])

ct_rsi_d = load_func('ct_rsi_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_ema_d = load_func('ct_ema_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_sma_d = load_func('ct_sma_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_atr_d = load_func('ct_atr_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p])
ct_adx_d = load_func('ct_adx_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_macd_d = load_func('ct_macd_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_stochastic_d = load_func('ct_stochastic_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_bbands_d = load_func('ct_bbands_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p])
ct_mfi_d = load_func('ct_mfi_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_obv_d = load_func('ct_obv_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p])
ct_cci_d = load_func('ct_cci_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
ct_willr_d = load_func('ct_willr_d', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])


def print_header(title, width=100):
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def generate_ohlcv(size):
    """Generate synthetic OHLCV data"""
    np.random.seed(42)
    
    price = 100.0
    opens = np.zeros(size, dtype=np.float64)  # TA-Lib needs float64
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


def benchmark_talib(opens, highs, lows, closes, volumes, iterations=5):
    """
    TA-Lib: Each indicator call processes numpy arrays
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # All indicators computed by TA-Lib
        rsi = talib.RSI(closes, timeperiod=14)
        ema12 = talib.EMA(closes, timeperiod=12)
        ema26 = talib.EMA(closes, timeperiod=26)
        sma20 = talib.SMA(closes, timeperiod=20)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        adx = talib.ADX(highs, lows, closes, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
        upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
        mfi = talib.MFI(highs, lows, closes, volumes, timeperiod=14)
        obv = talib.OBV(closes, volumes)
        cci = talib.CCI(highs, lows, closes, timeperiod=14)
        willr = talib.WILLR(highs, lows, closes, timeperiod=14)
        
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times), {
        'rsi': rsi,
        'ema12': ema12,
        'ema26': ema26,
        'sma20': sma20,
        'atr': atr,
        'adx': adx,
        'macd': macd,
        'macd_signal': macd_signal,
        'stoch_k': slowk,
        'stoch_d': slowd,
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'mfi': mfi,
        'obv': obv,
        'cci': cci,
        'willr': willr
    }


def benchmark_tacuda(opens, highs, lows, closes, volumes, iterations=5):
    """
    tacuda: Each indicator call does host->device->host transfer
    """
    size = len(closes)
    
    # Convert to float32 for tacuda
    opens32 = opens.astype(np.float32)
    highs32 = highs.astype(np.float32)
    lows32 = lows.astype(np.float32)
    closes32 = closes.astype(np.float32)
    volumes32 = volumes.astype(np.float32)
    
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
    bb_upper = np.zeros(size, dtype=np.float32)
    bb_middle = np.zeros(size, dtype=np.float32)
    bb_lower = np.zeros(size, dtype=np.float32)
    mfi_out = np.zeros(size, dtype=np.float32)
    obv_out = np.zeros(size, dtype=np.float32)
    cci_out = np.zeros(size, dtype=np.float32)
    willr_out = np.zeros(size, dtype=np.float32)
    
    # Pointers
    close_ptr = closes32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    high_ptr = highs32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = lows32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    vol_ptr = volumes32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
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
                size, 12, 26, 9, 0, None)
        ct_stochastic(high_ptr, low_ptr, close_ptr,
                      stoch_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      stoch_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      size, 14, 3, None)
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
        ct_cci(high_ptr, low_ptr, close_ptr,
               cci_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               size, 14, None)
        ct_willr(high_ptr, low_ptr, close_ptr,
                 willr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 size, 14, None)
        
        times.append((time.perf_counter() - start) * 1000)
    
    return np.median(times), {
        'rsi': rsi_out,
        'ema12': ema12_out,
        'ema26': ema26_out,
        'sma20': sma20_out,
        'atr': atr_out,
        'adx': adx_out,
        'macd': macd_out,
        'macd_signal': macd_sig,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'mfi': mfi_out,
        'obv': obv_out,
        'cci': cci_out,
        'willr': willr_out
    }


def benchmark_tacuda_device_pipeline(opens, highs, lows, closes, volumes, iterations=5):
    """
    tacuda Device API pipeline:
    upload once per iteration -> compute all indicators on device -> download outputs once.
    No host round-trip between indicators.
    """
    required = [ct_device_alloc, ct_device_free, ct_device_upload, ct_device_download, ct_device_sync,
                ct_rsi_d, ct_ema_d, ct_sma_d, ct_atr_d, ct_adx_d, ct_macd_d, ct_stochastic_d,
                ct_bbands_d, ct_mfi_d, ct_obv_d, ct_cci_d, ct_willr_d]
    if any(f is None for f in required):
        raise RuntimeError("Device API functions are unavailable in current build")

    size = len(closes)
    opens32 = opens.astype(np.float32)
    highs32 = highs.astype(np.float32)
    lows32 = lows.astype(np.float32)
    closes32 = closes.astype(np.float32)
    volumes32 = volumes.astype(np.float32)

    # Host output buffers
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
    bb_upper = np.zeros(size, dtype=np.float32)
    bb_middle = np.zeros(size, dtype=np.float32)
    bb_lower = np.zeros(size, dtype=np.float32)
    mfi_out = np.zeros(size, dtype=np.float32)
    obv_out = np.zeros(size, dtype=np.float32)
    cci_out = np.zeros(size, dtype=np.float32)
    willr_out = np.zeros(size, dtype=np.float32)

    d_close = ct_device_alloc(size)
    d_high = ct_device_alloc(size)
    d_low = ct_device_alloc(size)
    d_volume = ct_device_alloc(size)
    d_rsi = ct_device_alloc(size)
    d_ema12 = ct_device_alloc(size)
    d_ema26 = ct_device_alloc(size)
    d_sma20 = ct_device_alloc(size)
    d_atr = ct_device_alloc(size)
    d_adx = ct_device_alloc(size)
    d_macd = ct_device_alloc(size)
    d_macd_sig = ct_device_alloc(size)
    d_macd_hist = ct_device_alloc(size)
    d_stoch_k = ct_device_alloc(size)
    d_stoch_d = ct_device_alloc(size)
    d_bb_upper = ct_device_alloc(size)
    d_bb_middle = ct_device_alloc(size)
    d_bb_lower = ct_device_alloc(size)
    d_mfi = ct_device_alloc(size)
    d_obv = ct_device_alloc(size)
    d_cci = ct_device_alloc(size)
    d_willr = ct_device_alloc(size)

    dev_ptrs = [d_close, d_high, d_low, d_volume, d_rsi, d_ema12, d_ema26, d_sma20, d_atr, d_adx,
                d_macd, d_macd_sig, d_macd_hist, d_stoch_k, d_stoch_d, d_bb_upper, d_bb_middle,
                d_bb_lower, d_mfi, d_obv, d_cci, d_willr]
    if any(p is None or p == 0 for p in dev_ptrs):
        for p in dev_ptrs:
            if p:
                ct_device_free(p)
        raise RuntimeError("Failed to allocate one or more device buffers")

    times = []
    try:
        close_ptr = closes32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        high_ptr = highs32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        low_ptr = lows32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vol_ptr = volumes32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        for _ in range(iterations):
            start = time.perf_counter()

            # Upload once
            ct_device_upload(d_close, close_ptr, size, None)
            ct_device_upload(d_high, high_ptr, size, None)
            ct_device_upload(d_low, low_ptr, size, None)
            ct_device_upload(d_volume, vol_ptr, size, None)

            # Compute batch on device
            ct_rsi_d(d_close, d_rsi, size, 14, None)
            ct_ema_d(d_close, d_ema12, size, 12, None)
            ct_ema_d(d_close, d_ema26, size, 26, None)
            ct_sma_d(d_close, d_sma20, size, 20, None)
            ct_atr_d(d_high, d_low, d_close, d_atr, size, 14, 0.0, None)
            ct_adx_d(d_high, d_low, d_close, d_adx, size, 14, None)
            ct_macd_d(d_close, d_macd, d_macd_sig, d_macd_hist, size, 12, 26, 9, 1, None)
            ct_stochastic_d(d_high, d_low, d_close, d_stoch_k, d_stoch_d, size, 14, 3, None)
            ct_bbands_d(d_close, d_bb_upper, d_bb_middle, d_bb_lower, size, 20, 2.0, 2.0, None)
            ct_mfi_d(d_high, d_low, d_close, d_volume, d_mfi, size, 14, None)
            ct_obv_d(d_close, d_volume, d_obv, size, None)
            ct_cci_d(d_high, d_low, d_close, d_cci, size, 14, None)
            ct_willr_d(d_high, d_low, d_close, d_willr, size, 14, None)

            # Download once
            ct_device_download(rsi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_rsi, size, None)
            ct_device_download(ema12_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_ema12, size, None)
            ct_device_download(ema26_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_ema26, size, None)
            ct_device_download(sma20_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_sma20, size, None)
            ct_device_download(atr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_atr, size, None)
            ct_device_download(adx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_adx, size, None)
            ct_device_download(macd_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_macd, size, None)
            ct_device_download(macd_sig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_macd_sig, size, None)
            ct_device_download(macd_hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_macd_hist, size, None)
            ct_device_download(stoch_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_stoch_k, size, None)
            ct_device_download(stoch_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_stoch_d, size, None)
            ct_device_download(bb_upper.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_bb_upper, size, None)
            ct_device_download(bb_middle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_bb_middle, size, None)
            ct_device_download(bb_lower.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_bb_lower, size, None)
            ct_device_download(mfi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_mfi, size, None)
            ct_device_download(obv_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_obv, size, None)
            ct_device_download(cci_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_cci, size, None)
            ct_device_download(willr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d_willr, size, None)
            ct_device_sync(None)

            times.append((time.perf_counter() - start) * 1000)
    finally:
        for p in dev_ptrs:
            ct_device_free(p)

    return np.median(times), {
        'rsi': rsi_out, 'ema12': ema12_out, 'ema26': ema26_out, 'sma20': sma20_out,
        'atr': atr_out, 'adx': adx_out, 'macd': macd_out, 'macd_signal': macd_sig,
        'stoch_k': stoch_k, 'stoch_d': stoch_d, 'bb_upper': bb_upper,
        'bb_middle': bb_middle, 'bb_lower': bb_lower, 'mfi': mfi_out,
        'obv': obv_out, 'cci': cci_out, 'willr': willr_out
    }


def compare_results(ta_results, cuda_results, name, tolerance=1e-2):
    """Compare results between TA-Lib and tacuda"""
    ta_arr = ta_results[name]
    cuda_arr = cuda_results[name]
    
    # Skip NaN values
    mask = ~np.isnan(ta_arr) & ~np.isnan(cuda_arr)
    if np.sum(mask) == 0:
        return True, 0.0, 0.0
    
    ta_valid = ta_arr[mask]
    cuda_valid = cuda_arr[mask].astype(np.float64)
    
    diff = np.abs(ta_valid - cuda_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Relative tolerance for large values
    rel_diff = diff / (np.abs(ta_valid) + 1e-10)
    max_rel = np.max(rel_diff)
    
    match = max_rel < tolerance or max_diff < tolerance
    
    return match, max_diff, mean_diff


def main():
    print_header("Multi-Indicator Benchmark: tacuda vs TA-Lib")
    print()
    print("Computing 13 indicators on the same OHLCV dataset:")
    print("  RSI(14), EMA(12), EMA(26), SMA(20), ATR(14), ADX(14),")
    print("  MACD(12,26,9), Stochastic(14,3,3), Bollinger(20,2,2),")
    print("  MFI(14), OBV, CCI(14), Williams %R(14)")
    print()
    
    sizes = [100000, 500000, 1000000]
    
    all_results = []
    
    for size in sizes:
        print(f"\n{'='*100}")
        print(f"Data size: {size:,} bars")
        print(f"{'='*100}")
        
        # Generate data
        opens, highs, lows, closes, volumes = generate_ohlcv(size)
        
        # Benchmark TA-Lib
        talib_time, talib_results = benchmark_talib(opens, highs, lows, closes, volumes)
        
        # Benchmark tacuda (host API)
        tacuda_time, tacuda_results = benchmark_tacuda(opens, highs, lows, closes, volumes)
        # Benchmark tacuda (device pipeline API)
        pipeline_time, pipeline_results = benchmark_tacuda_device_pipeline(opens, highs, lows, closes, volumes)
        
        speedup = talib_time / tacuda_time if tacuda_time > 0 else 0
        
        pipeline_speedup = talib_time / pipeline_time if pipeline_time > 0 else 0
        print(f"\n{'Metric':<25} {'TA-Lib':<16} {'tacuda Host':<16} {'tacuda Device':<16}")
        print("-" * 80)
        print(f"{'Total time (ms)':<25} {talib_time:<16.2f} {tacuda_time:<16.2f} {pipeline_time:<16.2f}")
        print(f"{'Per indicator (ms)':<25} {talib_time/13:<16.2f} {tacuda_time/13:<16.2f} {pipeline_time/13:<16.2f}")
        print(f"{'Throughput (bars/sec)':<25} {size*13/talib_time*1000:<16,.0f} {size*13/tacuda_time*1000:<16,.0f} {size*13/pipeline_time*1000:<16,.0f}")
        print(f"{'Speedup vs TA-Lib':<25} {'-':<16} {speedup:<16.2f}x {pipeline_speedup:<16.2f}x")
        
        # Verify accuracy
        print(f"\n{'Indicator':<20} {'Match':<10} {'Max Diff':<20} {'Mean Diff':<20}")
        print("-" * 70)
        
        indicators = ['rsi', 'ema12', 'ema26', 'sma20', 'atr', 'adx', 'macd', 
                      'stoch_k', 'bb_upper', 'mfi', 'obv', 'cci', 'willr']
        
        all_match = True
        for ind in indicators:
            host_match, host_max_diff, host_mean_diff = compare_results(talib_results, tacuda_results, ind)
            dev_match, dev_max_diff, dev_mean_diff = compare_results(talib_results, pipeline_results, ind)
            status = "PASS" if (host_match and dev_match) else "FAIL"
            all_match = all_match and host_match and dev_match
            print(f"{ind:<20} {status:<10} host:{host_max_diff:<8.2e} dev:{dev_max_diff:<8.2e} "
                  f"host_mean:{host_mean_diff:<8.2e} dev_mean:{dev_mean_diff:<8.2e}")
        
        all_results.append({
            'size': size,
            'talib_time': talib_time,
            'tacuda_time': tacuda_time,
            'pipeline_time': pipeline_time,
            'speedup': speedup,
            'pipeline_speedup': pipeline_speedup,
            'all_match': all_match
        })
    
    # Summary table
    print("\n" + "=" * 100)
    print(f"{'SUMMARY':^100}")
    print("=" * 100)
    print()
    print(f"{'Data Size':<12} {'TA-Lib':<12} {'Host API':<12} {'Device API':<12} {'Host x':<10} {'Device x':<10} {'Accuracy':<12}")
    print("-" * 90)
    
    for r in all_results:
        status = "ALL PASS" if r['all_match'] else "SOME FAIL"
        print(f"{r['size']:>10,}   {r['talib_time']:<12.2f} {r['tacuda_time']:<12.2f} {r['pipeline_time']:<12.2f} "
              f"{r['speedup']:<10.2f}x {r['pipeline_speedup']:<10.2f}x {status:<12}")
    
    # Calculate averages
    avg_speedup = np.mean([r['speedup'] for r in all_results])
    
    print()
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print()
    
    # Performance analysis
    print("=" * 100)
    print(f"{'PERFORMANCE ANALYSIS':^100}")
    print("=" * 100)
    print("""
tacuda advantages:
  + GPU parallel computation for each indicator
  + Optimized CUDA kernels with shared memory
  + Efficient memory coalescing
  + Can leverage GPU's high memory bandwidth

TA-Lib advantages:
  + No GPU transfer overhead
  + Highly optimized C code
  + No GPU memory allocation cost
  + Better for small datasets

Recommendations:
  - For < 50,000 bars: TA-Lib may be faster (transfer overhead dominates)
  - For > 100,000 bars: tacuda provides significant speedup
  - For real-time streaming: tacuda Pipeline keeps data on GPU
  - For backtesting millions of bars: tacuda is the clear winner

Note: tacuda Pipeline (Context class) can provide additional speedup by:
  - Uploading data to GPU once
  - Computing all indicators without intermediate transfers
  - Downloading only final results
  - Estimated additional 30-40% improvement over separate calls
""")


if __name__ == "__main__":
    main()

