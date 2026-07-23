# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Audit binding coverage (Python / Node.js / C#) vs public C headers.

Scans index.js AND batch_generated.js for Node, the generated specs for
Python and C#. Exits non-zero when any binding misses a header symbol, so
CI can gate on it.

Run from anywhere: python bindings/audit_bindings.py
"""
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

h = (ROOT / "include/tacuda.h").read_text(encoding="utf-8")
hd = (ROOT / "include/tacuda_device.h").read_text(encoding="utf-8")
pat = re.compile(r"CTAPI_EXPORT\s+(?:ctStatus_t|float\s*\*|void\s*\*|void|size_t)\s*(ct_[a-zA-Z0-9_]+)\s*\(")
host = set(pat.findall(h))
dev = set(pat.findall(hd))
print("HOST header:", len(host))
print("DEVICE header:", len(dev))

failures = 0

# --- Node.js (hand-written index.js + generated batch_generated.js) ---
node = (ROOT / "bindings/nodejs/index.js").read_text(encoding="utf-8")
node += (ROOT / "bindings/nodejs/batch_generated.js").read_text(encoding="utf-8")
node_funcs = set(re.findall(r"(?:int|void\*?|uint64_t) (ct_[a-zA-Z0-9_]+)\(", node))
cdl_block = node[node.find("CDL_NAMES"):node.find("const cdlFuncs")]
for m in re.findall(r"'([a-z0-9_]+)'", cdl_block):
    node_funcs.add("ct_cdl_" + m)


def js_string_array(var_name: str) -> list:
    m = re.search(var_name + r"\s*=\s*\[(.*?)\]", node, re.DOTALL)
    return re.findall(r"'([a-z0-9_]+)'", m.group(1)) if m else []


unary = js_string_array("UNARY_MATH_NAMES")
binary = js_string_array("BINARY_MATH_NAMES")
dev_period = js_string_array("DEVICE_PERIOD_FUNCS")
dev_hlc = js_string_array("DEVICE_HLC_PERIOD_FUNCS")
dev_legacy = js_string_array("DEVICE_LEGACY_PERIOD_FUNCS")
for n in unary + binary:
    node_funcs.add(f"ct_{n}")
    node_funcs.add(f"ct_{n}_d")
for n in dev_period + dev_hlc:
    node_funcs.add(f"ct_{n}_d")
for n in dev_legacy:
    node_funcs.add(f"ct_{n}_device")
missing_node_host = sorted(host - node_funcs)
missing_node_dev = sorted(dev - node_funcs)
print()
print("NODE declares:", len(node_funcs))
print("NODE missing host funcs:", len(missing_node_host))
for f in missing_node_host:
    print("   ", f)
print("NODE missing device funcs:", len(missing_node_dev))
for f in missing_node_dev:
    print("   ", f)
failures += len(missing_node_host) + len(missing_node_dev)

# --- Python generated ---
gen = (ROOT / "bindings/python/tacuda/_generated.py").read_text(encoding="utf-8")
py_funcs = set(re.findall(r'c_name="(ct_[a-zA-Z0-9_]+)"', gen))
missing_py_host = sorted(host - py_funcs)
missing_py_dev = sorted(dev - py_funcs)
print()
print("PYTHON generated:", len(py_funcs))
print("PYTHON missing host:", missing_py_host)
print("PYTHON missing device:", missing_py_dev)
failures += len(missing_py_host) + len(missing_py_dev)

# --- C# generated ---
cs = (ROOT / "bindings/csharp/Tacuda.Generated.cs").read_text(encoding="utf-8")
cs_funcs = set(re.findall(r"(?:public|internal) static extern \S+ (ct_[a-zA-Z0-9_]+)\(", cs))
missing_cs_host = sorted(host - cs_funcs)
missing_cs_dev = sorted(dev - cs_funcs)
print()
print("CSHARP generated:", len(cs_funcs))
print("CSHARP missing host:", missing_cs_host)
print("CSHARP missing device:", missing_cs_dev)
failures += len(missing_cs_host) + len(missing_cs_dev)

# What node has that header doesn't (stale decls)
stale_node = sorted(f for f in node_funcs if f not in host and f not in dev)
print()
print("NODE stale decls (not in headers):", stale_node)
failures += len(stale_node)

if failures:
    print(f"\nAUDIT FAIL: {failures} coverage gaps / stale declarations")
    sys.exit(1)
print("\naudit OK: full C ABI coverage in Python, Node.js and C#")
