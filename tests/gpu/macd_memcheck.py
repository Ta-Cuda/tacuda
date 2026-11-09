# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Run MACD precision bounds checks under Compute Sanitizer via late loading.

Set TACUDA_LIBRARY to the exact candidate native. --negative checks one known
old-code reproducer; a nonzero sanitizer exit is expected only for that control.
"""
import ctypes as c
import os
import sys

import numpy as np


def main():
    # Loading the native after the Python process starts avoids the sanitizer's
    # early-CUDA-initialization failure in the directly linked C++ test binary.
    lib = c.CDLL(os.environ['TACUDA_LIBRARY'])
    cuda = c.CDLL('cudart64_12.dll' if os.name == 'nt' else 'libcudart.so.12')
    signature = [c.c_void_p] * 4 + [c.c_int] * 7 + [c.c_void_p]
    for name in ('ct_macd_ex', 'ct_macd_ex_d'):
        getattr(lib, name).argtypes = signature
        getattr(lib, name).restype = c.c_int
    cuda.cudaMalloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t]
    cuda.cudaMemcpy.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int]
    cuda.cudaFree.argtypes = [c.c_void_p]
    cuda.cudaDeviceSynchronize.argtypes = []
    types = {0: np.float16, 1: np.float32, 2: np.float64}
    pairs = [(0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]
    cases = [(n, fast, slow, signal) for n in (1, 3, 16, 32)
             for fast, slow, signal in ((32, 4, 3), (4, 32, 3), (12, 26, 9))]
    cases += [(128, 12, 26, 9)]
    if '--negative' in sys.argv:
        pairs, cases = [(2, 2)], [(16, 32, 4, 3)]
    count = 0
    for storage, compute in pairs:
        for size, fast, slow, signal in cases:
            x = np.arange(1, size + 1, dtype=types[storage])
            outputs = [np.full(size, -123, dtype=x.dtype) for _ in range(3)]
            parameters = [size, fast, slow, signal, 1, storage, compute, None]
            assert lib.ct_macd_ex(x.ctypes.data, *(a.ctypes.data for a in outputs), *parameters) == 0
            lookback = max(fast, slow) - 1 + signal - 1
            for array in outputs:
                assert np.isnan(array[:min(size, lookback)]).all(), (storage, compute, size)
                if size > lookback:
                    assert np.isfinite(array[lookback:]).all()
            count += 1
            pointers = [c.c_void_p() for _ in range(4)]
            try:
                for pointer in pointers:
                    assert cuda.cudaMalloc(c.byref(pointer), x.nbytes) == 0
                assert cuda.cudaMemcpy(pointers[0], x.ctypes.data, x.nbytes, 1) == 0
                assert lib.ct_macd_ex_d(*pointers, *parameters) == 0
                assert cuda.cudaDeviceSynchronize() == 0
                for pointer, reference in zip(pointers[1:], outputs):
                    result = np.empty_like(reference)
                    assert cuda.cudaMemcpy(result.ctypes.data, pointer, result.nbytes, 2) == 0
                    np.testing.assert_array_equal(result, reference)
                count += 1
            finally:
                for pointer in pointers:
                    if pointer:
                        cuda.cudaFree(pointer)
    print(f'PASS: {count} host/device MACD calls across {len(pairs)} storage/compute pairs, '
          'including fastPeriod > size and finite-output controls')


if __name__ == '__main__':
    main()
