# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Stamp the TaCuda signature header on top of every source file.

Idempotent: files already carrying the signature (tacuda.com in the first
few lines) are skipped. Generated files are skipped too — their emitters
(scripts/generate_batch_api.py, bindings/generate_bindings.py) stamp the
header themselves, so regeneration never loses it.

Run from anywhere: python scripts/add_source_headers.py [--check]
`--check` exits 1 listing files missing the header instead of writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WIDTH = 78
LINES = [
    "TaCuda | GPU-accelerated technical analysis | https://tacuda.com",
    "Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>",
    "SPDX-License-Identifier: Apache-2.0",
]


def c_style() -> str:
    bar = "/* " + "=" * (WIDTH - 6) + " */"
    body = [f"/*  {ln:<{WIDTH - 7}} */" for ln in LINES]
    return "\n".join([bar, *body, bar]) + "\n"


def line_style(prefix: str) -> str:
    bar = prefix + " " + "=" * (WIDTH - len(prefix) - 1)
    body = [f"{prefix}  {ln}" for ln in LINES]
    return "\n".join([bar, *body, bar]) + "\n"


HEADERS = {
    "c": c_style(),          # .h .hpp .cpp .cc .cu .cuh .inc
    "slash": line_style("//"),   # .cs .js
    "hash": line_style("#"),     # .py .cmake CMakeLists.txt
    "bat": line_style("rem"),    # .bat
}

STYLE_BY_EXT = {
    ".h": "c", ".hpp": "c", ".cpp": "c", ".cc": "c", ".cu": "c",
    ".cuh": "c", ".inc": "c",
    ".cs": "slash", ".js": "slash",
    ".py": "hash", ".cmake": "hash",
    ".bat": "bat",
}

# Emitted by generators which stamp the header themselves.
GENERATED = {
    "src/api/api_batch_generated.inc",
    "src/api/api_device_generated.inc",
    "src/api/api_device_generated_includes.inc",
    "bindings/nodejs/batch_generated.js",
    "bindings/csharp/Tacuda.Generated.cs",
    "bindings/csharp/TacudaBatch.Generated.cs",
    "bindings/python/tacuda/_generated.py",
}

SKIP_DIRS = {".git", "node_modules", "build", "build-win", "dist", "bin",
             "obj", "__pycache__", ".vs", "runtimes"}


def wants_header(path: Path) -> str | None:
    if path.name == "CMakeLists.txt":
        return "hash"
    return STYLE_BY_EXT.get(path.suffix.lower())


def has_header(text: str) -> bool:
    head = "\n".join(text.splitlines()[:6])
    return "tacuda.com" in head and "Deecalov" in head


def insert(text: str, header: str, style: str) -> str:
    if "\r\n" in text:
        header = header.replace("\n", "\r\n")
    lines = text.splitlines(keepends=True)
    pos = 0
    if style in ("hash", "slash") and lines and lines[0].startswith("#!"):
        pos = 1                     # keep the shebang first (python, node)
    if style == "bat" and lines and lines[0].lower().startswith("@echo off"):
        pos = 1                     # a header before @echo off would be echoed
    prefix = "".join(lines[:pos])
    rest = "".join(lines[pos:])
    return prefix + header + rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    missing: list[str] = []
    stamped = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if rel in GENERATED:
            continue
        style = wants_header(path)
        if style is None:
            continue
        raw = path.read_bytes()
        bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
        text = raw[len(bom):].decode("utf-8")
        if has_header(text):
            continue
        if args.check:
            missing.append(rel)
            continue
        new_text = insert(text, HEADERS[style], style)
        path.write_bytes(bom + new_text.encode("utf-8"))
        stamped += 1

    if args.check:
        if missing:
            print(f"{len(missing)} files missing the signature header:")
            for m in missing:
                print("  ", m)
            return 1
        print("all source files carry the signature header")
        return 0
    print(f"stamped {stamped} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
