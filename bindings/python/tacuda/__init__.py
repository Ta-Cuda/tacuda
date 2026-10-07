# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""ctypes based Python bindings for the :mod:`tacuda` library.

The shared library must be built prior to importing this module.  The
bindings are generated from the public ``tacuda.h`` header in order to keep
the Python API in sync with the C++ implementation and to avoid large amounts
of manually written boilerplate.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import inspect
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from . import _generated


def _load_lib() -> ctypes.CDLL:
    """Locate and load the ``tacuda`` shared library."""

    if sys.platform.startswith("win"):
        names = ["tacuda.dll"]
    elif sys.platform == "darwin":
        names = ["libtacuda.dylib", "tacuda.dylib"]
    else:
        names = ["libtacuda.so", "tacuda.so"]

    # Environment variables for explicit locations
    env_candidates: List[str] = []
    env = os.environ.get("TACUDA_LIBRARY")
    if env:
        env_candidates.append(env)

    env_dir = os.environ.get("TACUDA_LIBRARY_PATH") or os.environ.get(
        "TACUDA_LIBRARY_DIR"
    )
    if env_dir:
        for n in names:
            env_candidates.append(os.path.join(env_dir, n))

    for candidate in env_candidates:
        if candidate and os.path.exists(candidate):
            try:
                # Python 3.8+: use winmode to search for dependencies in PATH
                if sys.version_info >= (3, 8) and sys.platform.startswith("win"):
                    lib = ctypes.CDLL(candidate, winmode=0)  # 0 = default search
                else:
                    lib = ctypes.CDLL(candidate)
                return lib
            except OSError:
                pass

    # Try system paths via ctypes.util.find_library
    libname = ctypes.util.find_library("tacuda")
    if libname:
        try:
            return ctypes.CDLL(libname)
        except OSError:
            pass

    # Fall back to package-relative locations
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), "build"),
        here,
        os.path.join(here, "lib"),
        root,
        os.path.join(root, "lib"),
        os.path.join(root, "build"),
    ]
    for base in search_paths:
        for n in names:
            path = os.path.join(base, n)
            if os.path.exists(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue

    raise OSError(
        "tacuda library not found. Build the project first or set "
        "TACUDA_LIBRARY[_PATH]."
    )


_lib = _load_lib()

# String-returning introspection is outside the array binding generator.
_lib.ct_status_string.argtypes = [ctypes.c_int]
_lib.ct_status_string.restype = ctypes.c_char_p
_lib.ct_get_last_error.argtypes = []
_lib.ct_get_last_error.restype = ctypes.c_char_p


def status_string(status: int) -> str:
    return _lib.ct_status_string(status).decode("utf-8", "replace")


def get_last_error() -> str:
    return _lib.ct_get_last_error().decode("utf-8", "replace")


CT_DTYPE_F16 = 0
CT_DTYPE_F32 = 1
CT_DTYPE_F64 = 2

_DTYPE_NAME_TO_ENUM = {
    "float16": CT_DTYPE_F16,
    "f16": CT_DTYPE_F16,
    "float32": CT_DTYPE_F32,
    "f32": CT_DTYPE_F32,
    "float64": CT_DTYPE_F64,
    "f64": CT_DTYPE_F64,
}

_ENUM_TO_NP_DTYPE = {
    CT_DTYPE_F16: np.float16,
    CT_DTYPE_F32: np.float32,
    CT_DTYPE_F64: np.float64,
}


def _normalize_dtype(dtype: str | np.dtype | type) -> int:
    if isinstance(dtype, np.dtype):
        dtype = str(dtype)
    elif isinstance(dtype, type):
        dtype = np.dtype(dtype).name
    key = str(dtype).lower()
    if key not in _DTYPE_NAME_TO_ENUM:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use float16/float32/float64.")
    return _DTYPE_NAME_TO_ENUM[key]


def _compute_dtype_for_storage(storage_enum: int) -> int:
    return CT_DTYPE_F32 if storage_enum == CT_DTYPE_F16 else storage_enum


def _as_typed_array(arr: Any, dtype_enum: int) -> np.ndarray:
    np_dtype = _ENUM_TO_NP_DTYPE[dtype_enum]
    out = np.asarray(arr, dtype=np_dtype)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out, dtype=np_dtype)
    return out


def _as_float_array(arr: Any) -> Tuple[np.ndarray, ctypes.POINTER(ctypes.c_float)]:
    """Return ``arr`` as a contiguous ``float32`` array and pointer."""

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
    if arr.dtype != np.float32 or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


_C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_SPEC = _generated.FUNCTIONS

# Column pulled from an OHLCV container for each recognised C parameter name.
# Functions whose host inputs all appear here accept an OHLCV instance as the
# first positional argument instead of individual column arrays.
_OHLCV_INPUT_COLUMNS = {
    "host_open": "open",
    "host_high": "high",
    "host_low": "low",
    "host_close": "close",
    "host_volume": "volume",
    "host_price": "close",
}


def _scalar_ctype(name: str) -> Any:
    mapping: Dict[str, Any] = {
        "int": ctypes.c_int,
        "float": ctypes.c_float,
        "ctMaType_t": ctypes.c_int,
        "ctDataType_t": ctypes.c_int,
        "cudaStream_t": ctypes.c_void_p,
        "size_t": ctypes.c_size_t,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise KeyError(f"Unsupported scalar type '{name}' in tacuda bindings") from exc


def _prepare_stream(value: Any) -> ctypes.c_void_p:
    if value is None:
        return ctypes.c_void_p(0)
    if isinstance(value, ctypes.c_void_p):
        return value
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    raise TypeError("Stream parameters must be integers or ctypes.c_void_p instances")


def _prepare_device_ptr(value: Any) -> ctypes.c_void_p:
    if value is None:
        return ctypes.c_void_p(0)
    if isinstance(value, ctypes.c_void_p):
        return value
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    # Also accept ctypes pointers
    if hasattr(value, "contents"):
        return ctypes.cast(value, ctypes.c_void_p)
    raise TypeError(f"Device pointer must be int or c_void_p, got {type(value)}")


def _bind_function(name: str, spec: _generated.FunctionSpec) -> None:
    cfunc = getattr(_lib, spec.c_name)

    # Precision (_ex) host functions take void* buffers whose element type is
    # selected by the storageType scalar; the generic float32 path must NOT
    # be used for them (casting F16/F64 data to float32 would make the native
    # side read the wrong element size).
    storage_aware = any(s.name == "storageType" for s in spec.scalars)

    # Determine argument types
    argtypes: List[Any] = []
    
    outputs_first = spec.c_name in ("ct_device_upload", "ct_device_upload_ex", "ct_device_download", "ct_device_download_ex")

    def append_inputs():
        if hasattr(spec, 'input_types'):
            for i, type_name in enumerate(spec.input_types):
                if type_name == "device" or storage_aware:
                    argtypes.append(ctypes.c_void_p) # Device or dtype-selected pointer
                else:
                    argtypes.append(_C_FLOAT_PTR) # Host pointer (float*)
        else:
            # Fallback for old spec format (all float*)
            argtypes.extend([_C_FLOAT_PTR] * len(spec.inputs))

    def append_outputs():
        if hasattr(spec, 'output_types'):
            for i, type_name in enumerate(spec.output_types):
                if type_name == "device" or storage_aware:
                    argtypes.append(ctypes.c_void_p) # Device or dtype-selected pointer
                else:
                    argtypes.append(_C_FLOAT_PTR) # Host pointer (float*)
        else:
            # Fallback for old spec format (all float*)
            argtypes.extend([_C_FLOAT_PTR] * len(spec.outputs))

    if outputs_first:
        append_outputs()
        append_inputs()
    else:
        append_inputs()
        append_outputs()

    # Size param (if exists)
    if spec.size_param:
        argtypes.append(ctypes.c_int)

    # Scalar params
    for scalar in spec.scalars:
        argtypes.append(_scalar_ctype(scalar.ctype))
        
    cfunc.argtypes = argtypes

    # Determine return type
    if hasattr(spec, 'return_type'):
        if spec.return_type == "ctStatus_t":
            cfunc.restype = ctypes.c_int
        elif spec.return_type in ("float*", "void*"):
            cfunc.restype = ctypes.c_void_p # Return as opaque pointer
        elif spec.return_type == "void":
            cfunc.restype = None
        elif spec.return_type == "size_t":
            cfunc.restype = ctypes.c_size_t
        else:
            # Fallback/Unknown
            cfunc.restype = ctypes.c_int
    else:
        cfunc.restype = ctypes.c_int

    # Create Python wrapper
    parameters: List[inspect.Parameter] = []
    for input_name in spec.inputs:
        parameters.append(
            inspect.Parameter(input_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )

    # For device API outputs, they might be passed as arguments!
    # But FunctionSpec splits pointers into inputs (const) and outputs (non-const).
    # In generated bindings, outputs are usually hidden from python signature and created internally.
    # BUT for Device API, the user MUST provide the output buffer pointer.
    # So for Device API functions (where outputs are "device"), we should add them to parameters.
    
    device_outputs_indices = []
    if hasattr(spec, 'output_types'):
        for i, type_name in enumerate(spec.output_types):
            if type_name == "device":
                device_outputs_indices.append(i)
                # Add to python signature
                out_name = spec.outputs[i]
                parameters.append(
                    inspect.Parameter(out_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                )

    has_host_inputs = False
    if hasattr(spec, "input_types"):
        has_host_inputs = any(t == "host" for t in spec.input_types)
    elif spec.inputs:
        has_host_inputs = True

    if spec.size_param and not has_host_inputs:
        parameters.append(
            inspect.Parameter("size", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )

    for scalar in spec.scalars:
        default = inspect._empty
        if scalar.ctype == "cudaStream_t" and scalar.default is not None:
            default = None
        parameters.append(
            inspect.Parameter(
                scalar.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
            )
        )

    signature = inspect.Signature(parameters)

    ohlcv_columns: List[str] | None = None
    host_input_names = [
        n for i, n in enumerate(spec.inputs)
        if not (hasattr(spec, "input_types") and spec.input_types[i] == "device")
    ]
    if host_input_names and all(n in _OHLCV_INPUT_COLUMNS for n in host_input_names):
        ohlcv_columns = [_OHLCV_INPUT_COLUMNS[n] for n in host_input_names]

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], OHLCV):
            if ohlcv_columns is None:
                raise TypeError(
                    f"{name} does not accept an OHLCV container; "
                    "pass the required arrays explicitly"
                )
            series = args[0]
            columns = tuple(getattr(series, col) for col in ohlcv_columns)
            args = columns + args[1:]
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        # Resolve the storage dtype for precision (_ex) functions: host
        # buffers are converted to and returned in exactly that dtype.
        storage_enum = None
        if storage_aware:
            sval = bound.arguments.get("storageType", CT_DTYPE_F32)
            storage_enum = int(sval) if isinstance(sval, (int, np.integer)) else _dtype_to_enum(sval)
            bound.arguments["storageType"] = storage_enum

        c_args: List[Any] = []
        c_args_inputs: List[Any] = []
        c_args_outputs: List[Any] = []
        
        # Process Inputs
        np_inputs: List[np.ndarray] = []
        
        for i, name_ in enumerate(spec.inputs):
            val = bound.arguments[name_]
            is_device = False
            if hasattr(spec, 'input_types') and spec.input_types[i] == "device":
                is_device = True
            
            if is_device:
                c_args_inputs.append(_prepare_device_ptr(val))
            elif storage_enum is not None:
                arr_np = _as_typed_array(val, storage_enum)
                np_inputs.append(arr_np)
                c_args_inputs.append(arr_np.ctypes.data_as(ctypes.c_void_p))
            else:
                arr_np, ptr = _as_float_array(val)
                np_inputs.append(arr_np)
                c_args_inputs.append(ptr)

        # Check sizes if we have numpy inputs
        size = 0
        if np_inputs:
            size = np_inputs[0].size
            for arr in np_inputs[1:]:
                if arr.size != size:
                    raise ValueError("Input arrays must have the same size")
        elif "size" in bound.arguments:
             size = bound.arguments["size"]
        elif "count" in bound.arguments:
             size = bound.arguments["count"]

        # Process Outputs
        outputs_to_return: List[Any] = [] # Only host outputs are returned
        
        for i, name_ in enumerate(spec.outputs):
            is_device = False
            if hasattr(spec, 'output_types') and spec.output_types[i] == "device":
                is_device = True
            
            if is_device:
                # Device outputs are passed as arguments
                val = bound.arguments[name_]
                c_args_outputs.append(_prepare_device_ptr(val))
                # We don't return device pointers, user already has them
            elif storage_enum is not None:
                # Host output in the storage dtype selected by storageType.
                out_arr = np.zeros(size, dtype=_ENUM_TO_NP_DTYPE[storage_enum])
                c_args_outputs.append(out_arr.ctypes.data_as(ctypes.c_void_p))
                outputs_to_return.append(out_arr)
            else:
                # Host output - create numpy array
                if size == 0 and spec.size_param:
                     # We need size. If we have no input arrays, we might need to look at size scalar if it exists?
                     # But usually 'size' is not in bound.arguments if it's the special size param.
                     # Wait, if inputs are device ptrs, we don't know size!
                     # The C function expects 'size'. Where does it come from?
                     pass
                
                # In mixed mode (host input, host output), size comes from input.
                out_arr = np.zeros(size, dtype=np.float32)
                _, ptr = _as_float_array(out_arr)
                c_args_outputs.append(ptr)
                outputs_to_return.append(out_arr)

        if outputs_first:
            c_args.extend(c_args_outputs)
            c_args.extend(c_args_inputs)
        else:
            c_args.extend(c_args_inputs)
            c_args.extend(c_args_outputs)

        # Size parameter
        if spec.size_param:
             # If we have size from numpy arrays, use it.
             # If we rely on manual size passing (e.g. for device-only calls), we need to find where 'size' comes from.
             # But wait, 'size' is EXCLUDED from python signature in 'render_python' logic for standard functions.
             # "FunctionSpec.scalar_params" excludes size.
             
             # Issue: For device-only functions, we MUST pass size manually.
             # My generate_bindings script excluded 'size' from scalars.
             # This is a problem for Device API functions where size cannot be inferred from numpy arrays.
             
             # Quick fix: If size is 0 (no numpy inputs), try to find 'size' in kwargs? No, it's not in signature.
             
             # We will add a hack: if we have no host inputs, we implicitly assume 'size' is passed as argument? 
             # But wrapper signature doesn't have it.
             
             # Real fix: generate_bindings should include 'size' in scalars if it cannot be inferred?
             # Or we assume for now Device API is mixed with Host API or manual calls use ctypes directly.
             
             # Let's assume if np_inputs is empty, we check if 'size' is in bound args (it won't be).
             # This is a limitation. I will assume size is available or 0.
             # For pure device functions, the user currently CANNOT pass size via this wrapper if it was stripped.
             
             # However, 'ct_device_alloc' takes 'count' (renamed to size?). No, 'count'.
             # 'ct_sma_d' takes 'size'.
             
             c_args.append(ctypes.c_int(size))

        # Scalars
        for scalar in spec.scalars:
            val = bound.arguments[scalar.name]
            if scalar.ctype == "cudaStream_t":
                c_args.append(_prepare_stream(val))
            else:
                ctype = _scalar_ctype(scalar.ctype)
                c_args.append(ctype(val))

        rc = cfunc(*c_args)
        
        # Handle return value
        if hasattr(spec, 'return_type') and spec.return_type == "ctStatus_t":
            if rc != 0:
                _call_status(rc, spec.c_name)
            
            if outputs_to_return:
                if len(outputs_to_return) == 1:
                    return outputs_to_return[0]
                return tuple(outputs_to_return)
            return None # Success, no host outputs
        
        return rc # Return whatever it returned (float*, size_t, etc)

    wrapper.__name__ = name
    wrapper.__doc__ = f"Python wrapper for {spec.c_name}"
    wrapper.__signature__ = signature
    globals()[name] = wrapper


for _py_name, _spec in _SPEC.items():
    _bind_function(_py_name, _spec)


def _bind_precision_functions() -> None:
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_float = ctypes.c_float

    _lib.ct_sma_ex.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
    _lib.ct_sma_ex.restype = c_int
    _lib.ct_ema_ex.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
    _lib.ct_ema_ex.restype = c_int
    _lib.ct_rsi_ex.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_void_p]
    _lib.ct_rsi_ex.restype = c_int
    _lib.ct_macd_ex.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_void_p
    ]
    _lib.ct_macd_ex.restype = c_int
    _lib.ct_atr_ex.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_int, c_int, c_void_p
    ]
    _lib.ct_atr_ex.restype = c_int
    _lib.ct_bbands_ex.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float, c_float, c_int, c_int, c_void_p
    ]
    _lib.ct_bbands_ex.restype = c_int


_bind_precision_functions()


_sma_f32 = sma
_ema_f32 = ema
_rsi_f32 = rsi
_macd_f32 = macd
_atr_f32 = atr
_bbands_f32 = bbands


def _call_status(rc: int, func_name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{func_name} failed with code {rc} ({status_string(rc)}): {get_last_error()}")


def sma(input: Any, period: int, stream: Any = None, dtype: str = "float32") -> np.ndarray:
    storage = _normalize_dtype(dtype)
    if storage == CT_DTYPE_F32:
        return _sma_f32(input, period, stream)
    x = _as_typed_array(input, storage)
    out = np.zeros_like(x)
    compute = _compute_dtype_for_storage(storage)
    rc = _lib.ct_sma_ex(
        x.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(x.size), ctypes.c_int(int(period)),
        ctypes.c_int(storage), ctypes.c_int(compute), _prepare_stream(stream)
    )
    _call_status(rc, "ct_sma_ex")
    return out


def ema(input: Any, period: int, stream: Any = None, dtype: str = "float32") -> np.ndarray:
    storage = _normalize_dtype(dtype)
    if storage == CT_DTYPE_F32:
        return _ema_f32(input, period, stream)
    x = _as_typed_array(input, storage)
    out = np.zeros_like(x)
    compute = _compute_dtype_for_storage(storage)
    rc = _lib.ct_ema_ex(
        x.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(x.size), ctypes.c_int(int(period)),
        ctypes.c_int(storage), ctypes.c_int(compute), _prepare_stream(stream)
    )
    _call_status(rc, "ct_ema_ex")
    return out


def rsi(input: Any, period: int, stream: Any = None, dtype: str = "float32") -> np.ndarray:
    storage = _normalize_dtype(dtype)
    if storage == CT_DTYPE_F32:
        return _rsi_f32(input, period, stream)
    x = _as_typed_array(input, storage)
    out = np.zeros_like(x)
    compute = _compute_dtype_for_storage(storage)
    rc = _lib.ct_rsi_ex(
        x.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(x.size), ctypes.c_int(int(period)),
        ctypes.c_int(storage), ctypes.c_int(compute), _prepare_stream(stream)
    )
    _call_status(rc, "ct_rsi_ex")
    return out


def macd(
    input: Any,
    fastPeriod: int,
    slowPeriod: int,
    signalPeriod: int,
    type: int = 1,
    stream: Any = None,
    dtype: str = "float32",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    storage = _normalize_dtype(dtype)
    if storage == CT_DTYPE_F32:
        return _macd_f32(input, fastPeriod, slowPeriod, signalPeriod, type, stream)
    x = _as_typed_array(input, storage)
    m = np.zeros_like(x)
    s = np.zeros_like(x)
    h = np.zeros_like(x)
    compute = _compute_dtype_for_storage(storage)
    rc = _lib.ct_macd_ex(
        x.ctypes.data_as(ctypes.c_void_p), m.ctypes.data_as(ctypes.c_void_p),
        s.ctypes.data_as(ctypes.c_void_p), h.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(x.size), ctypes.c_int(int(fastPeriod)), ctypes.c_int(int(slowPeriod)),
        ctypes.c_int(int(signalPeriod)), ctypes.c_int(int(type)), ctypes.c_int(storage),
        ctypes.c_int(compute), _prepare_stream(stream)
    )
    _call_status(rc, "ct_macd_ex")
    return m, s, h


def atr(
    high: Any, low: Any = None, close: Any = None, period: int = 14,
    initial: float = 0.0, stream: Any = None, dtype: str = "float32",
) -> np.ndarray:
    if isinstance(high, OHLCV):
        # atr(ohlcv, 14) / atr(ohlcv, period=14)
        if low is not None:
            if not np.isscalar(low) or close is not None:
                raise TypeError(
                    "atr(OHLCV, ...) accepts only the period after the container"
                )
            period = int(low)
        series = high
        high, low, close = series.high, series.low, series.close
    elif low is None or close is None:
        raise TypeError("atr() requires high, low and close arrays")
    storage = _normalize_dtype(dtype)
    if storage == CT_DTYPE_F32:
        return _atr_f32(high, low, close, period, initial, stream)
    h = _as_typed_array(high, storage)
    l = _as_typed_array(low, storage)
    c = _as_typed_array(close, storage)
    if h.size != l.size or h.size != c.size:
        raise ValueError("high/low/close arrays must have same size")
    out = np.zeros_like(h)
    compute = _compute_dtype_for_storage(storage)
    rc = _lib.ct_atr_ex(
        h.ctypes.data_as(ctypes.c_void_p), l.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(h.size), ctypes.c_int(int(period)), ctypes.c_float(float(initial)),
        ctypes.c_int(storage), ctypes.c_int(compute), _prepare_stream(stream)
    )
    _call_status(rc, "ct_atr_ex")
    return out


def bbands(
    input: Any, period: int, upperMul: float = 2.0, lowerMul: float = 2.0,
    stream: Any = None, dtype: str = "float32",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    storage = _normalize_dtype(dtype)
    if storage == CT_DTYPE_F32:
        return _bbands_f32(input, period, upperMul, lowerMul, stream)
    x = _as_typed_array(input, storage)
    up = np.zeros_like(x)
    mid = np.zeros_like(x)
    lo = np.zeros_like(x)
    compute = _compute_dtype_for_storage(storage)
    rc = _lib.ct_bbands_ex(
        x.ctypes.data_as(ctypes.c_void_p), up.ctypes.data_as(ctypes.c_void_p),
        mid.ctypes.data_as(ctypes.c_void_p), lo.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(x.size), ctypes.c_int(int(period)), ctypes.c_float(float(upperMul)),
        ctypes.c_float(float(lowerMul)), ctypes.c_int(storage), ctypes.c_int(compute),
        _prepare_stream(stream)
    )
    _call_status(rc, "ct_bbands_ex")
    return up, mid, lo


_lib.ct_version.argtypes = []
_lib.ct_version.restype = ctypes.c_char_p


def version() -> str:
    """Return the native library version string (MAJOR.MINOR.PATCH)."""
    return _lib.ct_version().decode("ascii")


def _as_batch_matrix(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("batch input must be a 2D array (numSeries, seriesLen)")
    return np.ascontiguousarray(a)


def _batch_ptr(a: np.ndarray) -> Any:
    return a.ctypes.data_as(_C_FLOAT_PTR)


def sma_batch(inputs: Any, period: int, max_concurrency: int = 0,
              stream: Any = None) -> np.ndarray:
    """SMA over a (numSeries, seriesLen) matrix; one row per symbol."""
    a = _as_batch_matrix(inputs)
    out = np.zeros_like(a)
    rc = _lib.ct_sma_batch(_batch_ptr(a), _batch_ptr(out),
                           ctypes.c_int(a.shape[1]), ctypes.c_int(a.shape[0]),
                           ctypes.c_int(int(period)),
                           ctypes.c_int(int(max_concurrency)),
                           _prepare_stream(stream))
    _call_status(rc, "ct_sma_batch")
    return out


def ema_batch(inputs: Any, period: int, max_concurrency: int = 0,
              stream: Any = None) -> np.ndarray:
    """EMA over a (numSeries, seriesLen) matrix; one row per symbol."""
    a = _as_batch_matrix(inputs)
    out = np.zeros_like(a)
    rc = _lib.ct_ema_batch(_batch_ptr(a), _batch_ptr(out),
                           ctypes.c_int(a.shape[1]), ctypes.c_int(a.shape[0]),
                           ctypes.c_int(int(period)),
                           ctypes.c_int(int(max_concurrency)),
                           _prepare_stream(stream))
    _call_status(rc, "ct_ema_batch")
    return out


def rsi_batch(inputs: Any, period: int, max_concurrency: int = 0,
              stream: Any = None) -> np.ndarray:
    """RSI over a (numSeries, seriesLen) matrix; one row per symbol."""
    a = _as_batch_matrix(inputs)
    out = np.zeros_like(a)
    rc = _lib.ct_rsi_batch(_batch_ptr(a), _batch_ptr(out),
                           ctypes.c_int(a.shape[1]), ctypes.c_int(a.shape[0]),
                           ctypes.c_int(int(period)),
                           ctypes.c_int(int(max_concurrency)),
                           _prepare_stream(stream))
    _call_status(rc, "ct_rsi_batch")
    return out


def atr_batch(high: Any, low: Any, close: Any, period: int,
              initial: float = 0.0, max_concurrency: int = 0,
              stream: Any = None) -> np.ndarray:
    """ATR over (numSeries, seriesLen) high/low/close matrices."""
    h = _as_batch_matrix(high)
    l = _as_batch_matrix(low)
    c = _as_batch_matrix(close)
    if h.shape != l.shape or h.shape != c.shape:
        raise ValueError("high/low/close matrices must have the same shape")
    out = np.zeros_like(h)
    rc = _lib.ct_atr_batch(_batch_ptr(h), _batch_ptr(l), _batch_ptr(c),
                           _batch_ptr(out),
                           ctypes.c_int(h.shape[1]), ctypes.c_int(h.shape[0]),
                           ctypes.c_int(int(period)),
                           ctypes.c_float(float(initial)),
                           ctypes.c_int(int(max_concurrency)),
                           _prepare_stream(stream))
    _call_status(rc, "ct_atr_batch")
    return out


def macd_batch(inputs: Any, fastPeriod: int = 12, slowPeriod: int = 26,
               signalPeriod: int = 9, type: int = 1, max_concurrency: int = 0,
               stream: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD over a (numSeries, seriesLen) matrix; returns (macd, signal, hist)."""
    a = _as_batch_matrix(inputs)
    m = np.zeros_like(a)
    s = np.zeros_like(a)
    h = np.zeros_like(a)
    rc = _lib.ct_macd_batch(_batch_ptr(a), _batch_ptr(m), _batch_ptr(s),
                            _batch_ptr(h),
                            ctypes.c_int(a.shape[1]), ctypes.c_int(a.shape[0]),
                            ctypes.c_int(int(fastPeriod)),
                            ctypes.c_int(int(slowPeriod)),
                            ctypes.c_int(int(signalPeriod)),
                            ctypes.c_int(int(type)),
                            ctypes.c_int(int(max_concurrency)),
                            _prepare_stream(stream))
    _call_status(rc, "ct_macd_batch")
    return m, s, h


def bbands_batch(inputs: Any, period: int, upperMul: float = 2.0,
                 lowerMul: float = 2.0, max_concurrency: int = 0,
                 stream: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands over a (numSeries, seriesLen) matrix."""
    a = _as_batch_matrix(inputs)
    up = np.zeros_like(a)
    mid = np.zeros_like(a)
    lo = np.zeros_like(a)
    rc = _lib.ct_bbands_batch(_batch_ptr(a), _batch_ptr(up), _batch_ptr(mid),
                              _batch_ptr(lo),
                              ctypes.c_int(a.shape[1]), ctypes.c_int(a.shape[0]),
                              ctypes.c_int(int(period)),
                              ctypes.c_float(float(upperMul)),
                              ctypes.c_float(float(lowerMul)),
                              ctypes.c_int(int(max_concurrency)),
                              _prepare_stream(stream))
    _call_status(rc, "ct_bbands_batch")
    return up, mid, lo


def stochastic_batch(high: Any, low: Any, close: Any, kPeriod: int = 14,
                     dPeriod: int = 3, max_concurrency: int = 0,
                     stream: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic %K/%D over (numSeries, seriesLen) high/low/close matrices."""
    h = _as_batch_matrix(high)
    l = _as_batch_matrix(low)
    c = _as_batch_matrix(close)
    if h.shape != l.shape or h.shape != c.shape:
        raise ValueError("high/low/close matrices must have the same shape")
    k = np.zeros_like(h)
    d = np.zeros_like(h)
    rc = _lib.ct_stochastic_batch(_batch_ptr(h), _batch_ptr(l), _batch_ptr(c),
                                  _batch_ptr(k), _batch_ptr(d),
                                  ctypes.c_int(h.shape[1]),
                                  ctypes.c_int(h.shape[0]),
                                  ctypes.c_int(int(kPeriod)),
                                  ctypes.c_int(int(dPeriod)),
                                  ctypes.c_int(int(max_concurrency)),
                                  _prepare_stream(stream))
    _call_status(rc, "ct_stochastic_batch")
    return k, d


# ---------------------------------------------------------------------------
# Generated batch families: friendly (numSeries, seriesLen)-matrix wrappers
# built from the _generated specs for every remaining ct_*_batch symbol.
# ---------------------------------------------------------------------------

_MANUAL_BATCH = {
    "sma_batch", "ema_batch", "rsi_batch", "atr_batch",
    "macd_batch", "bbands_batch", "stochastic_batch",
}


def _make_batch_wrapper(py_name: str, spec: _generated.FunctionSpec):
    cfunc = getattr(_lib, spec.c_name)
    n_in = len(spec.inputs)
    n_out = len(spec.outputs)
    family_scalars = [
        s for s in spec.scalars
        if s.name not in ("seriesLen", "numSeries", "maxConcurrency")
        and s.ctype != "cudaStream_t"
    ]

    def wrapper(*args: Any, max_concurrency: int = 0, stream: Any = None,
                **kwargs: Any) -> Any:
        if len(args) < n_in or len(args) > n_in + len(family_scalars):
            expected = list(spec.inputs) + [s.name for s in family_scalars]
            raise TypeError(
                f"{py_name} expects arguments {expected} "
                "(plus keyword-only max_concurrency/stream)")
        mats = [_as_batch_matrix(a) for a in args[:n_in]]
        for m in mats[1:]:
            if m.shape != mats[0].shape:
                raise ValueError("input matrices must have the same shape")
        scalar_values = list(args[n_in:])
        for s in family_scalars[len(scalar_values):]:
            if s.name not in kwargs:
                raise TypeError(f"{py_name} missing argument '{s.name}'")
            scalar_values.append(kwargs.pop(s.name))
        if kwargs:
            raise TypeError(
                f"{py_name} got unexpected arguments {sorted(kwargs)}")
        outs = [np.zeros_like(mats[0]) for _ in range(n_out)]

        c_args: list[Any] = [_batch_ptr(m) for m in mats]
        c_args += [_batch_ptr(o) for o in outs]
        c_args += [ctypes.c_int(mats[0].shape[1]),
                   ctypes.c_int(mats[0].shape[0])]
        for s, v in zip(family_scalars, scalar_values):
            c_args.append(ctypes.c_float(float(v)) if s.ctype == "float"
                          else ctypes.c_int(int(v)))
        c_args.append(ctypes.c_int(int(max_concurrency)))
        c_args.append(_prepare_stream(stream))

        _call_status(cfunc(*c_args), spec.c_name)
        return outs[0] if n_out == 1 else tuple(outs)

    wrapper.__name__ = py_name
    param_names = list(spec.inputs) + [s.name for s in family_scalars]
    wrapper.__doc__ = (
        f"Batched wrapper for {spec.c_name}: arguments {param_names}; "
        "inputs are (numSeries, seriesLen) matrices, one row per symbol.")
    return wrapper


for _py_name, _spec in _SPEC.items():
    if (_py_name.endswith("_batch") and _py_name not in _MANUAL_BATCH):
        globals()[_py_name] = _make_batch_wrapper(_py_name, _spec)


class OHLCV:
    """Columnar OHLCV container backed by NumPy arrays."""

    __slots__ = ("_data",)

    def __init__(self, size: int = 0) -> None:
        size = int(size)
        if size < 0:
            raise ValueError("OHLCV size must be non-negative")
        self._data = np.zeros((5, size), dtype=np.float32)

    @property
    def size(self) -> int:
        """Number of rows contained in the series."""

        return int(self._data.shape[1])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.size

    @classmethod
    def from_columns(
        cls,
        open: Any,
        high: Any,
        low: Any,
        close: Any,
        volume: Any | None = None,
    ) -> "OHLCV":
        """Create an :class:`OHLCV` series from column vectors."""

        arrays = []
        size: int | None = None
        for name, source in zip(
            ("open", "high", "low", "close"), (open, high, low, close)
        ):
            arr = np.asarray(source, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            arr = np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
            if size is None:
                size = int(arr.size)
            elif int(arr.size) != size:
                raise ValueError(f"{name} column must have length {size}")
            arrays.append(arr)

        if size is None:
            size = 0

        instance = cls(size)
        for idx, arr in enumerate(arrays):
            instance._data[idx, : size] = arr

        if volume is None:
            instance._data[4, : size].fill(0.0)
        else:
            vol = np.asarray(volume, dtype=np.float32)
            if vol.ndim == 0:
                vol = vol.reshape(1)
            vol = np.ascontiguousarray(vol.reshape(-1), dtype=np.float32)
            if int(vol.size) != size:
                raise ValueError("volume column must match other columns")
            instance._data[4, : size] = vol

        return instance

    @classmethod
    def from_rows(cls, rows: Any) -> "OHLCV":
        """Create an :class:`OHLCV` series from row-wise data."""

        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("rows input must be a 2D array")
        if arr.shape[1] not in (4, 5):
            raise ValueError("rows must have 4 (OHLC) or 5 (OHLCV) columns")

        arr = np.ascontiguousarray(arr, dtype=np.float32)
        columns = [arr[:, i] for i in range(4)]
        volume = arr[:, 4] if arr.shape[1] == 5 else np.zeros(arr.shape[0], dtype=np.float32)
        return cls.from_columns(columns[0], columns[1], columns[2], columns[3], volume)

    def _column_view(self, idx: int) -> np.ndarray:
        return self._data[idx]

    @property
    def open(self) -> np.ndarray:
        return self._column_view(0)

    @property
    def high(self) -> np.ndarray:
        return self._column_view(1)

    @property
    def low(self) -> np.ndarray:
        return self._column_view(2)

    @property
    def close(self) -> np.ndarray:
        return self._column_view(3)

    @property
    def volume(self) -> np.ndarray:
        return self._column_view(4)

    def column_major(self, include_volume: bool = True) -> np.ndarray:
        """Return a view of the data in column-major (SoA) layout."""

        if include_volume:
            return self._data.reshape(-1)
        return self._data[:4].reshape(-1)

    def copy_column_major(self, include_volume: bool = True) -> np.ndarray:
        """Return a copy of the data in column-major layout."""

        return np.array(self.column_major(include_volume), copy=True)

    def to_numpy(self, include_volume: bool = True, copy: bool = True) -> np.ndarray:
        """Return a 2D NumPy array of shape (rows, columns)."""

        data = self._data.T if include_volume else self._data[:4].T
        return np.array(data, copy=copy)

    def set(
        self,
        index: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> None:
        if index < 0 or index >= self.size:
            raise IndexError("OHLCV row index out of range")
        self.open[index] = float(open)
        self.high[index] = float(high)
        self.low[index] = float(low)
        self.close[index] = float(close)
        self.volume[index] = float(volume)

    def __array__(self, dtype: Any | None = None) -> np.ndarray:  # pragma: no cover - numpy protocol
        if dtype is None or dtype == self._data.dtype:
            return self._data
        return self._data.astype(dtype, copy=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"OHLCV(size={self.size})"

# Imported last: stream.py needs the loaded `_lib` from this module.
from . import stream  # noqa: E402

__all_list = sorted(_SPEC)
__all_list.extend(("OHLCV", "version", "status_string", "get_last_error", "stream"))
__all__ = tuple(__all_list)
