#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Simple wrapper to run device API benchmark and save output"""
import sys
import os

# Redirect stdout to file
output_file = os.path.join(os.path.dirname(__file__), 'device_api_results.txt')
sys.stdout = open(output_file, 'w', encoding='utf-8')
sys.stderr = sys.stdout

try:
    # Import and run benchmark
    import benchmark_device_api
    benchmark_device_api.main()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    sys.stdout.close()
    
# Print confirmation to original stdout
sys.__stdout__.write(f"Results saved to: {output_file}\n")

