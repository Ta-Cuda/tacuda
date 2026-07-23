# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
import time
import numpy as np


def bench(f, *args, repeats=5):
    """Return median runtime over `repeats` executions."""
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f(*args)
        xs.append(time.perf_counter() - t0)
    return float(np.median(xs))
