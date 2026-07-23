#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Generate language bindings for the tacuda C API.

This script parses ``include/tacuda.h`` and ``include/tacuda_device.h`` and produces:

* ``bindings/python/_generated.py`` - static description of the API.
* ``bindings/csharp/Tacuda.Generated.cs`` - P/Invoke declarations.

It supports both Host API (managed arrays) and Device API (raw pointers/IntPtr).
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "tacuda.h"
HEADER_DEVICE = ROOT / "include" / "tacuda_device.h"
PYTHON_OUT = ROOT / "bindings" / "python" / "tacuda" / "_generated.py"
CSHARP_OUT = ROOT / "bindings" / "csharp" / "Tacuda.Generated.cs"


@dataclasses.dataclass(frozen=True)
class Parameter:
    """Representation of a parameter in the ``tacuda`` C API."""

    name: str
    type_text: str
    base_type: str
    pointer: bool
    const: bool
    default: Optional[str] = None

    @property
    def is_stream(self) -> bool:
        return self.base_type == "cudaStream_t"

    @property
    def is_device_ptr(self) -> bool:
        """Check if this parameter is a device pointer."""
        # Heuristic: names starting with d_ or device_ are device pointers
        return self.pointer and (self.name.startswith("d_") or self.name.startswith("device_"))


@dataclasses.dataclass(frozen=True)
class FunctionSpec:
    """Representation of a single exported API function."""

    c_name: str
    return_type: str
    parameters: Sequence[Parameter]

    @property
    def py_name(self) -> str:
        if self.c_name.startswith("ct_"):
            return self.c_name[3:]
        return self.c_name

    @property
    def inputs(self) -> List[Parameter]:
        return [p for p in self.parameters if p.pointer and p.const]

    @property
    def outputs(self) -> List[Parameter]:
        return [p for p in self.parameters if p.pointer and not p.const]

    @property
    def size_param(self) -> Optional[Parameter]:
        for param in self.parameters:
            if not param.pointer and param.name == "size":
                return param
        return None

    @property
    def scalar_params(self) -> List[Parameter]:
        scalars: List[Parameter] = []
        for param in self.parameters:
            if param.pointer:
                continue
            if param.name == "size":
                continue
            scalars.append(param)
        return scalars


# Regex to capture return type, function name, and parameter list
FUNC_PATTERN = re.compile(
    r"CTAPI_EXPORT\s+(ctStatus_t|float\s*\*|void\s*\*|void|size_t)\s*(ct_[a-zA-Z0-9_]+)\s*\((.*?)\)\s*;",
    re.DOTALL,
)


def _normalise_param(param: str) -> Parameter:
    original = param
    param = param.strip()
    if not param:
        raise ValueError("Empty parameter component in declaration")

    default: Optional[str] = None
    if "CTAPI_STREAM_DEFAULT" in param:
        param = param.replace("CTAPI_STREAM_DEFAULT", "").strip()
        default = "0"

    if "=" in param:
        param, default = [part.strip() for part in param.split("=", 1)]

    param = param.replace("*", " * ")
    param = re.sub(r"\s+", " ", param).strip()
    if " " not in param:
        raise ValueError(f"Unable to split parameter '{original}'")

    type_part, name = param.rsplit(" ", 1)
    pointer = "*" in type_part
    tokens = [t for t in type_part.replace("*", " ").split(" ") if t]
    const = "const" in tokens
    base_tokens = [t for t in tokens if t != "const"]
    if not base_tokens:
        raise ValueError(f"Could not determine base type for '{original}'")
    base_type = base_tokens[-1]

    return Parameter(
        name=name,
        type_text=type_part,
        base_type=base_type,
        pointer=pointer,
        const=const,
        default=default,
    )


def parse_header(text: str) -> List[FunctionSpec]:
    functions: List[FunctionSpec] = []
    for match in FUNC_PATTERN.finditer(text):
        ret_type = match.group(1)
        if "*" in ret_type:
            ret_type = re.sub(r"\s+", "", ret_type)
        name = match.group(2)
        params_blob = match.group(3)
        
        if not params_blob or params_blob.strip() == "void":
            parameters = []
        else:
            params_raw = [p.strip() for p in params_blob.split(",") if p.strip()]
            parameters = [_normalise_param(p) for p in params_raw]
            
        functions.append(FunctionSpec(c_name=name, return_type=ret_type, parameters=parameters))
    return functions


def _format_python_tuple(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return "()"
    if len(items) == 1:
        return f"({items[0]},)"
    return "(" + ", ".join(items) + ")"


def render_python(functions: Sequence[FunctionSpec]) -> str:
    lines: List[str] = []
    lines.append("# This file is auto-generated by bindings/generate_bindings.py.")
    lines.append("# Do not edit by hand.")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass")
    lines.append("from typing import Tuple")
    lines.append("")
    lines.append("")
    lines.append("@dataclass(frozen=True)")
    lines.append("class ScalarParam:")
    lines.append("    name: str")
    lines.append("    ctype: str")
    lines.append("    default: str | None = None")
    lines.append("")
    lines.append("")
    lines.append("@dataclass(frozen=True)")
    lines.append("class FunctionSpec:")
    lines.append("    c_name: str")
    lines.append("    return_type: str")
    lines.append("    inputs: Tuple[str, ...]")
    lines.append("    outputs: Tuple[str, ...]")
    lines.append("    input_types: Tuple[str, ...]")  # New: tracking pointer types
    lines.append("    output_types: Tuple[str, ...]") # New: tracking pointer types
    lines.append("    size_param: str | None")
    lines.append("    scalars: Tuple[ScalarParam, ...]")
    lines.append("")
    lines.append("")
    lines.append("FUNCTIONS = {")
    
    for func in functions:
        # Skip memory management functions for Python dict if not strictly standard
        # But we want all of them. We just need to handle them in __init__.py
        
        inputs = [f"\"{p.name}\"" for p in func.inputs]
        outputs = [f"\"{p.name}\"" for p in func.outputs]
        
        # Determine pointer types for inputs/outputs
        input_types = []
        for p in func.inputs:
            if p.is_device_ptr:
                input_types.append("\"device\"")
            else:
                input_types.append("\"host\"")
                
        output_types = []
        for p in func.outputs:
            if p.is_device_ptr:
                output_types.append("\"device\"")
            else:
                output_types.append("\"host\"")
        
        scalars: List[str] = []
        for param in func.scalar_params:
            default = param.default
            if default is None:
                scalars.append(
                    "            ScalarParam(name=\"%s\", ctype=\"%s\")," % (param.name, param.base_type)
                )
            else:
                scalars.append(
                    "            ScalarParam(name=\"%s\", ctype=\"%s\", default=\"%s\"),"
                    % (param.name, param.base_type, default)
                )

        lines.append(f"    \"{func.py_name}\": FunctionSpec(")
        lines.append(f"        c_name=\"{func.c_name}\",")
        lines.append(f"        return_type=\"{func.return_type}\",")
        lines.append(f"        inputs={_format_python_tuple(inputs)},")
        lines.append(f"        outputs={_format_python_tuple(outputs)},")
        lines.append(f"        input_types={_format_python_tuple(input_types)},")
        lines.append(f"        output_types={_format_python_tuple(output_types)},")
        
        if func.size_param:
            lines.append(f"        size_param=\"{func.size_param.name}\",")
        else:
            lines.append(f"        size_param=None,")
            
        if scalars:
            lines.append("        scalars=(")
            lines.extend(scalars)
            lines.append("        ),")
        else:
            lines.append("        scalars=(),")
        lines.append("    ),")
        
    lines.append("}")
    lines.append("")
    lines.append("__all__ = tuple(sorted(FUNCTIONS))")
    lines.append("")
    return "\n".join(lines) + "\n"


def _csharp_library_name_block() -> List[str]:
    # A bare library name lets the .NET runtime probe platform-specific
    # variants (tacuda.dll / libtacuda.so / libtacuda.dylib) on every OS.
    # Preprocessor constants like WINDOWS are only defined for OS-specific
    # TFMs (net8.0-windows), so an #if-based selection breaks plain net8.0.
    return [
        'private const string LibraryName = "tacuda";',
    ]


def _parse_ctdatatype_enum(header_text: str) -> List[tuple[str, str]]:
    match = re.search(
        r"typedef\s+enum\s+ctDataType\s*\{(?P<body>.*?)\}\s*ctDataType_t\s*;",
        header_text,
        re.DOTALL,
    )
    if not match:
        return [
            ("CT_DTYPE_F16", "0"),
            ("CT_DTYPE_F32", "1"),
            ("CT_DTYPE_F64", "2"),
        ]

    values: List[tuple[str, str]] = []
    for raw in match.group("body").split(","):
        line = raw.strip()
        if not line or "=" not in line:
            continue
        name, value = [part.strip() for part in line.split("=", 1)]
        if name.startswith("CT_DTYPE_"):
            values.append((name, value))

    return values or [
        ("CT_DTYPE_F16", "0"),
        ("CT_DTYPE_F32", "1"),
        ("CT_DTYPE_F64", "2"),
    ]


def _csharp_type(param: Parameter) -> str:
    if param.pointer:
        if param.is_device_ptr:
            return "IntPtr"
        # Precision (_ex) entry points take ``void *`` whose element type is
        # selected at runtime by ctDataType_t — marshalling them as float[]
        # would corrupt F16/F64 data. Expose them as IntPtr.
        if param.base_type == "void":
            return "IntPtr"
        # The remaining pointer parameters in the public API are ``float *``.
        return "float[]"

    mapping = {
        "int": "int",
        "float": "float",
        "size_t": "ulong", # or UIntPtr/IntPtr/long depending on arch, ulong covers 64bit
        "ctMaType_t": "int",
        "cudaStream_t": "IntPtr",
    }
    try:
        return mapping[param.base_type]
    except KeyError as exc:
        # Fallback for unknown types
        return param.base_type


def _csharp_return_type(ret_type: str) -> str:
    mapping = {
        "ctStatus_t": "int",
        "float*": "IntPtr",
        "void*": "IntPtr",
        "void": "void",
        "size_t": "ulong"
    }
    return mapping.get(ret_type, ret_type)


def render_csharp(functions: Sequence[FunctionSpec], header_text: str) -> str:
    lines: List[str] = []
    lines.append("// <auto-generated>")
    lines.append("// This file was generated by bindings/generate_bindings.py.")
    lines.append("// Do not edit manually.")
    lines.append("// </auto-generated>")
    lines.append("using System;")
    lines.append("using System.Runtime.InteropServices;")
    lines.append("")
    lines.append("namespace Tacuda.Bindings;")
    lines.append("")
    lines.append("public enum ctDataType_t : int")
    lines.append("{")
    for name, value in _parse_ctdatatype_enum(header_text):
        lines.append(f"    {name} = {value},")
    lines.append("}")
    lines.append("")
    lines.append("public static partial class NativeMethods")
    lines.append("{")
    for line in _csharp_library_name_block():
        if line.startswith("#"):
            lines.append(line)
        else:
            lines.append("    " + line)
    lines.append("")

    for func in functions:
        parameters = []
        for param in func.parameters:
            csharp_type = _csharp_type(param)
            if param.is_stream:
                parameters.append(f"{csharp_type} {param.name} = default")
            else:
                parameters.append(f"{csharp_type} {param.name}")
        params_joined = ", ".join(parameters)
        
        ret_type = _csharp_return_type(func.return_type)
        
        lines.append("    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]")
        lines.append(
            f"    public static extern {ret_type} {func.c_name}({params_joined});"
        )
        lines.append("")

    lines.append("}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_if_different(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


# Project signature stamped on every generated file (keep in sync with
# scripts/add_source_headers.py, which stamps the hand-written sources).
_SIGN_LINES = (
    "TaCuda | GPU-accelerated technical analysis | https://tacuda.com",
    "Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>",
    "SPDX-License-Identifier: Apache-2.0",
)


def signature(prefix: str) -> str:
    bar = prefix + " " + "=" * (77 - len(prefix))
    body = [f"{prefix}  {ln}" for ln in _SIGN_LINES]
    return "\n".join([bar, *body, bar]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bindings from tacuda.h")
    parser.add_argument("--check", action="store_true", help="verify files are up to date")
    args = parser.parse_args()

    header_text = HEADER.read_text(encoding="utf-8")
    functions = parse_header(header_text)
    
    # Add device API functions
    if HEADER_DEVICE.exists():
        print(f"Reading device header from {HEADER_DEVICE}")
        header_device_text = HEADER_DEVICE.read_text(encoding="utf-8")
        functions_device = parse_header(header_device_text)
        print(f"Found {len(functions_device)} functions in device header")
        
        existing_names = {f.c_name for f in functions}
        for f in functions_device:
            if f.c_name not in existing_names:
                functions.append(f)
    else:
        print(f"Device header not found at {HEADER_DEVICE}")

    python_content = signature("#") + render_python(functions)
    csharp_content = signature("//") + render_csharp(functions, header_text)

    if args.check:
        stale = []
        try:
            existing_python = PYTHON_OUT.read_text(encoding="utf-8")
        except FileNotFoundError:
            stale.append(str(PYTHON_OUT.relative_to(ROOT)))
        else:
            if existing_python != python_content:
                stale.append(str(PYTHON_OUT.relative_to(ROOT)))

        try:
            existing_csharp = CSHARP_OUT.read_text(encoding="utf-8")
        except FileNotFoundError:
            stale.append(str(CSHARP_OUT.relative_to(ROOT)))
        else:
            if existing_csharp != csharp_content:
                stale.append(str(CSHARP_OUT.relative_to(ROOT)))
        if stale:
            print("The following generated files are out of date:")
            for path in stale:
                print(f"  {path}")
            print("Run bindings/generate_bindings.py to update them.")
            return 1
        return 0

    changed = write_if_different(PYTHON_OUT, python_content)
    changed |= write_if_different(CSHARP_OUT, csharp_content)
    if changed:
        print("Bindings updated.")
    else:
        print("Bindings already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
