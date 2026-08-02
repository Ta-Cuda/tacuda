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
hs = (ROOT / "include/tacuda_stream.h").read_text(encoding="utf-8")
pat = re.compile(r"CTAPI_EXPORT\s+(?:ctStatus_t|float\s*\*|void\s*\*|void|size_t)\s*(ct_[a-zA-Z0-9_]+)\s*\(")
host = set(pat.findall(h))
dev = set(pat.findall(hd))
# Streaming is handle-based, so it is wrapped by hand in every language
# instead of coming out of the array-in/array-out generator. It still has to
# be covered everywhere, hence its own bucket rather than an exemption.
stream = set(pat.findall(hs))
print("HOST header:", len(host))
print("DEVICE header:", len(dev))
print("STREAM header:", len(stream))


def stream_names(stems):
    """Expand family stems into the four entry points each one gets."""
    out = set()
    for stem in stems:
        out.add(f"ct_{stem}_stream_create")
        out.add(f"ct_{stem}_stream_append")
        out.add(f"ct_{stem}_stream_batch_create")
        out.add(f"ct_{stem}_stream_batch_append")
    return out

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
# Streaming classes are built from a table of family stems.
node_funcs |= stream_names(re.findall(r"stem:\s*'([a-z0-9_]+)'", node))
node_funcs |= set(re.findall(r"lib\.func\('int (ct_stream[a-z0-9_]*)\(", node))

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
missing_node_stream = sorted(stream - node_funcs)
print("NODE missing stream funcs:", len(missing_node_stream))
for f in missing_node_stream:
    print("   ", f)
failures += len(missing_node_host) + len(missing_node_dev) + len(missing_node_stream)

# --- Python generated ---
gen = (ROOT / "bindings/python/tacuda/_generated.py").read_text(encoding="utf-8")
py_funcs = set(re.findall(r'c_name="(ct_[a-zA-Z0-9_]+)"', gen))
py_stream_src = (ROOT / "bindings/python/tacuda/stream.py").read_text(encoding="utf-8")
py_funcs |= set(re.findall(r'_prototype\("(ct_[a-zA-Z0-9_]+)"', py_stream_src))
py_funcs |= stream_names(re.findall(r'\("[A-Za-z]+",\s*"([a-z0-9_]+)"', py_stream_src))
missing_py_host = sorted(host - py_funcs)
missing_py_dev = sorted(dev - py_funcs)
print()
print("PYTHON generated:", len(py_funcs))
print("PYTHON missing host:", missing_py_host)
print("PYTHON missing device:", missing_py_dev)
missing_py_stream = sorted(stream - py_funcs)
print("PYTHON missing stream:", missing_py_stream)
failures += len(missing_py_host) + len(missing_py_dev) + len(missing_py_stream)

# --- C# generated ---
cs = (ROOT / "bindings/csharp/Tacuda.Generated.cs").read_text(encoding="utf-8")
cs += (ROOT / "bindings/csharp/Stream.cs").read_text(encoding="utf-8")
cs_funcs = set(re.findall(r"(?:public|internal) static extern \S+ (ct_[a-zA-Z0-9_]+)\(", cs))
missing_cs_host = sorted(host - cs_funcs)
missing_cs_dev = sorted(dev - cs_funcs)
print()
print("CSHARP generated:", len(cs_funcs))
print("CSHARP missing host:", missing_cs_host)
print("CSHARP missing device:", missing_cs_dev)
missing_cs_stream = sorted(stream - cs_funcs)
print("CSHARP missing stream:", missing_cs_stream)
failures += len(missing_cs_host) + len(missing_cs_dev) + len(missing_cs_stream)

# What node has that header doesn't (stale decls)
stale_node = sorted(f for f in node_funcs
                    if f not in host and f not in dev and f not in stream)
print()
print("NODE stale decls (not in headers):", stale_node)
failures += len(stale_node)

if failures:
    print(f"\nAUDIT FAIL: {failures} coverage gaps / stale declarations")
    sys.exit(1)
print("\naudit OK: full C ABI coverage (host + device + streaming) "
      "in Python, Node.js and C#")
