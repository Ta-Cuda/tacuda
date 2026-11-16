# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""The released compatibility target must never shrink with the local install."""
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / 'tests/reference/talib-0.8.3.json'

def require_reference(talib):
    target = json.loads(MANIFEST.read_text())
    wrapper = talib.__version__
    native = talib.__ta_version__
    if isinstance(native, bytes): native = native.decode()
    native_version = native.split()[0]
    actual = set(talib.get_functions())
    wanted = set(target['functions'])
    if wrapper != target['python_version'] or native_version != target['native_version'] or actual != wanted:
        raise RuntimeError(f"Expected pinned TA-Lib Python/native 0.8.3 and 237 functions; got {wrapper}/{native_version}, {len(actual)} functions; missing={sorted(wanted-actual)}, extra={sorted(actual-wanted)}")
    return target

if __name__ == '__main__':
    import talib
    require_reference(talib)
    print('PASS pinned TA-Lib Python/native 0.8.3; 237 functions')
