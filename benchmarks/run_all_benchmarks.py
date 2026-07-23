# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Запускает все бенчмарки и собирает результаты для анализа производительности
"""
import subprocess
import re
import sys
from pathlib import Path

# Список основных бенчмарков
BENCHMARKS = [
    "benchmark_overlap_studies_vs_talib.py",
    "benchmark_momentum_vs_talib.py",
    "benchmark_volatility_vs_talib.py",
    "benchmark_volume_vs_talib.py",
    "benchmark_Statistics_vs_talib.py",
    "benchmark_Cycle_vs_talib.py",
    "benchmark_candlestick_vs_talib.py",
]

def run_benchmark(script_name):
    """Запускает один бенчмарк и возвращает результаты"""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print('='*80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 минут таймаут
            encoding='utf-8',
            errors='replace'
        )
        
        return {
            'name': script_name,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {script_name}")
        return {
            'name': script_name,
            'stdout': '',
            'stderr': 'TIMEOUT',
            'returncode': -1
        }
    except Exception as e:
        print(f"ERROR running {script_name}: {e}")
        return {
            'name': script_name,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def parse_results(results):
    """Парсит результаты и извлекает информацию о производительности"""
    analysis = {
        'slow_indicators': [],  # speedup < 0.5x
        'moderate_indicators': [],  # 0.5x <= speedup < 1.0x
        'good_indicators': [],  # 1.0x <= speedup < 2.0x
        'excellent_indicators': [],  # speedup >= 2.0x
        'errors': []
    }
    
    for result in results:
        if result['returncode'] != 0:
            analysis['errors'].append({
                'benchmark': result['name'],
                'error': result['stderr']
            })
            continue
        
        # Парсим строки вида: "Pattern/Indicator   TA-Lib (ms)   tacuda (ms)   Speedup   Match"
        lines = result['stdout'].split('\n')
        for line in lines:
            # Ищем строки с данными (содержат speedup)
            match = re.search(r'(\w+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)x?\s', line)
            if match:
                name = match.group(1)
                try:
                    speedup = float(match.group(2))
                    
                    indicator_info = {
                        'name': name,
                        'speedup': speedup,
                        'benchmark': result['name']
                    }
                    
                    if speedup < 0.5:
                        analysis['slow_indicators'].append(indicator_info)
                    elif speedup < 1.0:
                        analysis['moderate_indicators'].append(indicator_info)
                    elif speedup < 2.0:
                        analysis['good_indicators'].append(indicator_info)
                    else:
                        analysis['excellent_indicators'].append(indicator_info)
                except ValueError:
                    continue
    
    return analysis

def print_analysis(analysis):
    """Выводит анализ производительности"""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    if analysis['errors']:
        print(f"\n[ERROR] ({len(analysis['errors'])} benchmarks failed):")
        for error in analysis['errors']:
            print(f"  - {error['benchmark']}: {error['error'][:100]}")
    
    print(f"\n[SLOW] INDICATORS (speedup < 0.5x) - {len(analysis['slow_indicators'])} found:")
    if analysis['slow_indicators']:
        for ind in sorted(analysis['slow_indicators'], key=lambda x: x['speedup']):
            print(f"  - {ind['name']:<25} {ind['speedup']:.2f}x  [{ind['benchmark']}]")
    else:
        print("  None!")
    
    print(f"\n[MODERATE] INDICATORS (0.5x <= speedup < 1.0x) - {len(analysis['moderate_indicators'])} found:")
    if analysis['moderate_indicators']:
        for ind in sorted(analysis['moderate_indicators'], key=lambda x: x['speedup']):
            print(f"  - {ind['name']:<25} {ind['speedup']:.2f}x  [{ind['benchmark']}]")
    else:
        print("  None!")
    
    print(f"\n[GOOD] INDICATORS (1.0x <= speedup < 2.0x) - {len(analysis['good_indicators'])} found:")
    for ind in sorted(analysis['good_indicators'], key=lambda x: x['speedup'])[:10]:
        print(f"  - {ind['name']:<25} {ind['speedup']:.2f}x  [{ind['benchmark']}]")
    if len(analysis['good_indicators']) > 10:
        print(f"  ... and {len(analysis['good_indicators']) - 10} more")
    
    print(f"\n[EXCELLENT] INDICATORS (speedup >= 2.0x) - {len(analysis['excellent_indicators'])} found:")
    for ind in sorted(analysis['excellent_indicators'], key=lambda x: -x['speedup'])[:15]:
        print(f"  - {ind['name']:<25} {ind['speedup']:.2f}x  [{ind['benchmark']}]")
    if len(analysis['excellent_indicators']) > 15:
        print(f"  ... and {len(analysis['excellent_indicators']) - 15} more")
    
    # Статистика
    total = (len(analysis['slow_indicators']) + len(analysis['moderate_indicators']) + 
             len(analysis['good_indicators']) + len(analysis['excellent_indicators']))
    
    print(f"\n[STATISTICS]:")
    print(f"  Total indicators tested: {total}")
    if total > 0:
        print(f"  Slower than TA-Lib: {len(analysis['slow_indicators']) + len(analysis['moderate_indicators'])} ({(len(analysis['slow_indicators']) + len(analysis['moderate_indicators']))/total*100:.1f}%)")
        print(f"  Faster than TA-Lib: {len(analysis['good_indicators']) + len(analysis['excellent_indicators'])} ({(len(analysis['good_indicators']) + len(analysis['excellent_indicators']))/total*100:.1f}%)")
    print("="*80)

def main():
    print("Starting comprehensive benchmark suite...")
    print(f"Running {len(BENCHMARKS)} benchmark scripts...")
    
    results = []
    for benchmark in BENCHMARKS:
        result = run_benchmark(benchmark)
        results.append(result)
        # Выводим краткий результат
        if result['returncode'] == 0:
            print(f"[OK] {benchmark} completed")
        else:
            print(f"[FAIL] {benchmark} failed")
    
    print("\n\nParsing results...")
    analysis = parse_results(results)
    
    print_analysis(analysis)
    
    # Сохраняем подробные результаты
    output_file = Path(__file__).parent / "ALL_BENCHMARKS_RESULTS.txt"
    with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE BENCHMARK RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"\n{'='*80}\n")
            f.write(f"Benchmark: {result['name']}\n")
            f.write(f"Return code: {result['returncode']}\n")
            f.write('='*80 + "\n\n")
            f.write(result['stdout'])
            if result['stderr']:
                f.write("\n\nSTDERR:\n")
                f.write(result['stderr'])
            f.write("\n\n")
    
    print(f"\nFull results saved to: {output_file}")
    
    return 0 if not analysis['errors'] else 1

if __name__ == "__main__":
    sys.exit(main())

