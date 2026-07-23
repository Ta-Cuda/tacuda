#!/usr/bin/env python
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Comprehensive benchmark and correctness comparison between TACUDA and TA-Lib.

This script compares:
1. Performance (speed)
2. Numerical accuracy (comparing output values)
3. Memory usage

Usage:
    python compare_with_talib.py
"""

import os
import sys

# Add tacuda Python bindings to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
tacuda_bindings_path = os.path.join(project_root, 'bindings', 'python')
sys.path.insert(0, project_root)
sys.path.insert(0, tacuda_bindings_path)

# Add CUDA to DLL search path for Windows (needed for tacuda.dll dependencies)
if sys.platform.startswith('win'):
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    ]
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            # Python 3.8+ supports os.add_dll_directory
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(cuda_path)
            os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
            print(f"Added CUDA to DLL search path: {cuda_path}")
            break

# Set DLL path for Windows
dll_path = os.path.join(project_root, 'build', 'Release', 'tacuda.dll')
if not os.path.exists(dll_path):
    dll_path = os.path.join(project_root, 'build', 'Debug', 'tacuda.dll')
    
if os.path.exists(dll_path):
    os.environ['TACUDA_LIBRARY'] = dll_path
    print(f"Using TACUDA library: {dll_path}")
else:
    print(f"ERROR: Cannot find TACUDA library at {dll_path}")
    sys.exit(1)

import time
import numpy as np
import tacuda  # MUST import tacuda BEFORE talib to avoid DLL conflicts
import talib
from typing import Callable, Tuple, Dict, Any
from benchmarks.talib_compare_helpers import build_specs, compare_arrays, generate_inputs

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def bench(f: Callable, *args, repeats: int = 5) -> float:
    """Measure median execution time over multiple runs."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = f(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def compare_indicator(name: str, tacuda_func: Callable, talib_func: Callable,
                     data: Dict[str, np.ndarray], params: Dict[str, Any],
                     tolerance: float = 1e-5):
    """
    Compare a single indicator between TACUDA and TA-Lib.
    
    Args:
        name: Indicator name
        tacuda_func: TACUDA function (lambda with all parameters baked in)
        talib_func: TA-Lib function (lambda with all parameters baked in)
        data: Input data dictionary (e.g., {'close': array, 'high': array})
        params: Parameters dictionary (unused, for compatibility)
        tolerance: Numerical tolerance for correctness check
    """
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Run benchmarks
    try:
        t_tacuda = bench(tacuda_func, data)
        t_talib = bench(talib_func, data)
        
        # Get results for correctness check
        tacuda_result = tacuda_func(data)
        talib_result = talib_func(data)
        
        # Handle multiple outputs (like MACD, Stochastic)
        if isinstance(tacuda_result, tuple):
            # Compare each output separately
            all_correct = True
            max_diffs = []
            for i, (tr, tl) in enumerate(zip(tacuda_result, talib_result)):
                is_correct_i, max_diff_i, mean_diff_i, nan_mismatch = compare_arrays(np.asarray(tr), np.asarray(tl), tolerance)
                if nan_mismatch:
                    print(f"  {YELLOW}Warning: NaN positions differ ({nan_mismatch} mismatches){RESET}")
                if not is_correct_i:
                    print(f"  {RED}Mismatch detected: max_diff={max_diff_i:.6e}, mean_diff={mean_diff_i:.6e}{RESET}")
                all_correct = all_correct and is_correct_i
                max_diffs.append(max_diff_i)
            is_correct = all_correct
            max_diff = max(max_diffs) if max_diffs else 0.0  # Maximum across all outputs
        else:
            is_correct, max_diff, mean_diff, nan_mismatch = compare_arrays(np.asarray(tacuda_result), np.asarray(talib_result), tolerance)
            if nan_mismatch:
                print(f"  {YELLOW}Warning: NaN positions differ ({nan_mismatch} mismatches){RESET}")
            if not is_correct:
                print(f"  {RED}Mismatch detected: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}{RESET}")
        
        # Calculate speedup
        speedup = t_talib / max(t_tacuda, 1e-9)
        
        # Print results
        correctness_str = f"{GREEN}✓ PASS{RESET}" if is_correct else f"{RED}✗ FAIL{RESET}"
        speedup_color = GREEN if speedup > 1 else RED
        
        print(f"Correctness: {correctness_str} (max_diff: {max_diff:.2e})")
        print(f"TA-Lib:      {t_talib*1000:.3f} ms")
        print(f"TACUDA:      {t_tacuda*1000:.3f} ms")
        print(f"Speedup:     {speedup_color}{speedup:.1f}x{RESET}")
        
        return {
            'name': name,
            'correct': is_correct,
            'max_diff': max_diff,
            't_talib': t_talib,
            't_tacuda': t_tacuda,
            'speedup': speedup
        }
        
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        return {
            'name': name,
            'correct': False,
            'error': str(e)
        }


def run_groups(selected_groups=None):
    # Windows console encoding fix
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    print(f"{BLUE}{'='*59}{RESET}")
    print(f"{BLUE}   TACUDA vs TA-Lib: Performance & Correctness Test   {RESET}")
    print(f"{BLUE}{'='*59}{RESET}")
    
    dtypes = [d.strip() for d in os.getenv("TACUDA_BENCH_DTYPES", "float32").split(",") if d.strip()]
    final_rc = 0
    for dtype in dtypes:
        print(f"\n{BLUE}Running dtype profile: {dtype}{RESET}")
        data = generate_inputs(10_000, seed=42)
        all_specs = build_specs(tacuda, talib, dtype=dtype)
        if selected_groups is None:
            test_specs = all_specs
        else:
            test_specs = [s for s in all_specs if s.group in selected_groups]

        # Run comparisons
        results = []
        for spec in test_specs:
            result = compare_indicator(spec.name, spec.tacuda_fn, spec.talib_fn, data, {}, tolerance=spec.tolerance)
            results.append(result)
    
        # Summary
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}SUMMARY ({dtype}){RESET}")
        print(f"{BLUE}{'='*60}{RESET}")

        total = len(results)
        passed = sum(1 for r in results if r.get('correct', False))
        failed = total - passed
    
        print(f"\nTotal indicators tested: {total}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        if failed > 0:
            print(f"{RED}Failed: {failed}{RESET}")
    
        # Performance summary
        print(f"\n{BLUE}Performance Summary:{RESET}")
        avg_speedup = np.mean([r['speedup'] for r in results if 'speedup' in r])
        print(f"Average speedup: {avg_speedup:.1f}x")

        # Best performers
        sorted_results = sorted([r for r in results if 'speedup' in r],
                            key=lambda x: x['speedup'], reverse=True)
        print(f"\n{GREEN}Top 3 speedups:{RESET}")
        for i, r in enumerate(sorted_results[:3], 1):
            print(f"  {i}. {r['name']}: {r['speedup']:.1f}x")

        # Failed tests
        if failed > 0:
            print(f"\n{RED}Failed tests:{RESET}")
            for r in results:
                if not r.get('correct', False):
                    if 'error' in r:
                        print(f"  - {r['name']}: {r['error']}")
                    else:
                        print(f"  - {r['name']}: max_diff={r.get('max_diff', 'N/A'):.2e}")
            final_rc = 1
    
        print(f"\n{BLUE}{'='*60}{RESET}")

    return final_rc


def main():
    return run_groups()


if __name__ == '__main__':
    sys.exit(main())

