# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Streaming (incremental update) API — see docs/STREAMING_API.md.

A stream carries indicator state across calls, so appending a newly closed
bar costs a state update instead of a full-series recompute::

    import numpy as np, tacuda

    with tacuda.stream.Rsi(14) as rsi:
        warm = rsi.append(history)        # replay: ordinary append
        value = rsi.append(new_close)[-1] # then one call per closed bar

Batch streams advance a whole portfolio in one call; matrices are
series-major, shaped ``(num_series, n)``::

    with tacuda.stream.RsiBatch(14, num_series=256) as rsi:
        out = rsi.append(closes)          # (256, n) in, (256, n) out

These wrappers are hand-written rather than generated: the flat generator
models array-in/array-out calls, while a stream is a handle with a lifetime.
"""

from __future__ import annotations

import ctypes
from typing import Any, Sequence, Tuple

import numpy as np

from . import _lib

__all__ = [
    "TacudaStreamError",
    "EXEC_AUTO",
    "EXEC_CPU",
    "EXEC_GPU",
    "Sma", "Ema", "Wma", "Rsi", "Atr", "Natr", "Obv", "Ad", "Macd", "Bbands",
    "SmaBatch", "EmaBatch", "WmaBatch", "RsiBatch", "AtrBatch", "NatrBatch",
    "ObvBatch", "AdBatch", "MacdBatch", "BbandsBatch",
]

EXEC_AUTO = 0
EXEC_CPU = 1
EXEC_GPU = 2


class TacudaStreamError(RuntimeError):
    """A streaming call returned a non-success ``ctStatus_t``."""

    def __init__(self, function: str, status: int) -> None:
        detail = _lib.ct_get_last_error()
        text = detail.decode("utf-8", "replace") if detail else ""
        super().__init__(f"{function} failed (status {status}): {text}")
        self.function = function
        self.status = status


_c_float_p = ctypes.POINTER(ctypes.c_float)


def _check(status: int, function: str) -> None:
    if status != 0:
        raise TacudaStreamError(function, status)


def _prototype(name: str, argtypes: Sequence[Any]) -> Any:
    fn = getattr(_lib, name)
    fn.argtypes = list(argtypes)
    fn.restype = ctypes.c_int
    return fn


# --- shared lifetime / introspection -----------------------------------------

_destroy = _prototype("ct_stream_destroy", [ctypes.c_void_p])
_reset = _prototype("ct_stream_reset", [ctypes.c_void_p])
_lookback = _prototype("ct_stream_lookback", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
_bars = _prototype("ct_stream_bars", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)])
_outputs = _prototype("ct_stream_outputs", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])

_batch_destroy = _prototype("ct_stream_batch_destroy", [ctypes.c_void_p])
_batch_reset = _prototype("ct_stream_batch_reset", [ctypes.c_void_p])
_batch_lookback = _prototype("ct_stream_batch_lookback", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
_batch_bars = _prototype("ct_stream_batch_bars", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)])
_batch_outputs = _prototype("ct_stream_batch_outputs", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
_batch_series = _prototype("ct_stream_batch_series", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
_batch_set_exec = _prototype("ct_stream_batch_set_exec", [ctypes.c_void_p, ctypes.c_int])


# Per family: (name, C stem, input components, output count, create params).
# `inputs` names the OHLCV components the append call takes, in C order.
_FAMILIES = (
    ("Sma", "sma", ("close",), 1, ("period",)),
    ("Ema", "ema", ("close",), 1, ("period",)),
    ("Wma", "wma", ("close",), 1, ("period",)),
    ("Rsi", "rsi", ("close",), 1, ("period",)),
    ("Atr", "atr", ("high", "low", "close"), 1, ("period",)),
    ("Natr", "natr", ("high", "low", "close"), 1, ("period",)),
    ("Obv", "obv", ("close", "volume"), 1, ()),
    ("Ad", "ad", ("high", "low", "close", "volume"), 1, ()),
    ("Macd", "macd", ("close",), 3, ("fast_period", "slow_period", "signal_period")),
    ("Bbands", "bbands", ("close",), 3, ("period", "upper_mul", "lower_mul")),
)

_CREATE_CTYPES = {
    "period": ctypes.c_int,
    "fast_period": ctypes.c_int,
    "slow_period": ctypes.c_int,
    "signal_period": ctypes.c_int,
    "upper_mul": ctypes.c_float,
    "lower_mul": ctypes.c_float,
}


def _as_input(arr: Any) -> Tuple[np.ndarray, Any]:
    """Contiguous float32 view plus a pointer to it (the copy, if any, is kept
    alive by the returned array)."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return a, a.ctypes.data_as(_c_float_p)


class _StreamBase:
    """Common handle plumbing. Subclasses are generated below."""

    _outputs_count = 1
    _inputs: Tuple[str, ...] = ("close",)
    _create_fn = None
    _append_fn = None
    _is_batch = False

    def __init__(self) -> None:
        self._handle = ctypes.c_void_p()

    # -- lifetime ------------------------------------------------------------

    def close(self) -> None:
        """Release the native handle. Safe to call more than once."""
        if getattr(self, "_handle", None) and self._handle.value:
            (_batch_destroy if self._is_batch else _destroy)(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "_StreamBase":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown order
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if not self._handle or not self._handle.value:
            raise TacudaStreamError(f"{type(self).__name__}.append", 4)

    # -- introspection -------------------------------------------------------

    def reset(self) -> None:
        """Drop state, keep parameters — equivalent to a fresh stream."""
        self._require_open()
        fn = _batch_reset if self._is_batch else _reset
        _check(fn(self._handle), "ct_stream_reset")

    @property
    def lookback(self) -> int:
        """Bars needed before the first non-NaN output."""
        self._require_open()
        out = ctypes.c_int()
        fn = _batch_lookback if self._is_batch else _lookback
        _check(fn(self._handle, ctypes.byref(out)), "ct_stream_lookback")
        return out.value

    @property
    def bars(self) -> int:
        """Bars appended since creation or the last reset."""
        self._require_open()
        out = ctypes.c_longlong()
        fn = _batch_bars if self._is_batch else _bars
        _check(fn(self._handle, ctypes.byref(out)), "ct_stream_bars")
        return out.value

    @property
    def outputs(self) -> int:
        """Number of output lines ``append`` returns."""
        self._require_open()
        out = ctypes.c_int()
        fn = _batch_outputs if self._is_batch else _outputs
        _check(fn(self._handle, ctypes.byref(out)), "ct_stream_outputs")
        return out.value


class _SingleStream(_StreamBase):
    """One indicator over one series."""

    def append(self, *arrays: Any) -> Any:
        """Append bars and return the matching outputs.

        Takes one array per input component (close, or high/low/close, ...);
        scalars are accepted for the single-bar case. Returns one ndarray, or
        a tuple of them for multi-line families.
        """
        self._require_open()
        if len(arrays) != len(self._inputs):
            raise TypeError(
                f"{type(self).__name__}.append expects {len(self._inputs)} array(s) "
                f"({', '.join(self._inputs)}), got {len(arrays)}"
            )

        kept, ptrs = [], []
        n = -1
        for name, arr in zip(self._inputs, arrays):
            a, p = _as_input(np.atleast_1d(arr))
            if a.ndim != 1:
                raise ValueError(f"{type(self).__name__}.append: {name} must be 1-D")
            if n < 0:
                n = a.size
            elif a.size != n:
                raise ValueError(
                    f"{type(self).__name__}.append: inputs must have equal length "
                    f"(got {n} and {a.size} for {name})"
                )
            kept.append(a)
            ptrs.append(p)
        if n <= 0:
            raise ValueError(f"{type(self).__name__}.append: need at least one bar")

        outs = [np.empty(n, dtype=np.float32) for _ in range(self._outputs_count)]
        out_ptrs = [o.ctypes.data_as(_c_float_p) for o in outs]
        _check(type(self)._append_fn(self._handle, *ptrs, n, *out_ptrs),
               f"ct_{type(self).__name__.lower()}_stream_append")
        return outs[0] if self._outputs_count == 1 else tuple(outs)


class _BatchStream(_StreamBase):
    """One indicator over `num_series` independent series."""

    _is_batch = True

    @property
    def num_series(self) -> int:
        self._require_open()
        out = ctypes.c_int()
        _check(_batch_series(self._handle, ctypes.byref(out)), "ct_stream_batch_series")
        return out.value

    def set_exec(self, exec_policy: int) -> None:
        """Pick where appends run: EXEC_AUTO (default), EXEC_CPU or EXEC_GPU."""
        self._require_open()
        _check(_batch_set_exec(self._handle, int(exec_policy)),
               "ct_stream_batch_set_exec")

    def append(self, *matrices: Any) -> Any:
        """Append `n` bars to every series.

        Each matrix is series-major, shaped ``(num_series, n)``. Returns one
        array of the same shape per output line.
        """
        self._require_open()
        if len(matrices) != len(self._inputs):
            raise TypeError(
                f"{type(self).__name__}.append expects {len(self._inputs)} matrix/matrices "
                f"({', '.join(self._inputs)}), got {len(matrices)}"
            )

        series = self.num_series
        kept, ptrs = [], []
        shape = None
        for name, mat in zip(self._inputs, matrices):
            a = np.ascontiguousarray(mat, dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(series, -1)
            if a.ndim != 2:
                raise ValueError(f"{type(self).__name__}.append: {name} must be 2-D "
                                 f"(num_series, n)")
            if a.shape[0] != series:
                raise ValueError(
                    f"{type(self).__name__}.append: {name} has {a.shape[0]} rows, "
                    f"stream was created for {series} series"
                )
            if shape is None:
                shape = a.shape
            elif a.shape != shape:
                raise ValueError(
                    f"{type(self).__name__}.append: inputs must have equal shape "
                    f"(got {shape} and {a.shape} for {name})"
                )
            kept.append(a)
            ptrs.append(a.ctypes.data_as(_c_float_p))

        n = shape[1]
        if n <= 0:
            raise ValueError(f"{type(self).__name__}.append: need at least one bar")

        outs = [np.empty(shape, dtype=np.float32) for _ in range(self._outputs_count)]
        out_ptrs = [o.ctypes.data_as(_c_float_p) for o in outs]
        _check(type(self)._append_fn(self._handle, *ptrs, n, *out_ptrs),
               "ct_stream_batch_append")
        return outs[0] if self._outputs_count == 1 else tuple(outs)


def _make_single(cls_name: str, stem: str, inputs: Tuple[str, ...],
                 out_count: int, create_params: Tuple[str, ...]) -> type:
    create = _prototype(
        f"ct_{stem}_stream_create",
        [_CREATE_CTYPES[p] for p in create_params] + [ctypes.POINTER(ctypes.c_void_p)],
    )
    append = _prototype(
        f"ct_{stem}_stream_append",
        [ctypes.c_void_p] + [_c_float_p] * len(inputs) + [ctypes.c_int] +
        [_c_float_p] * out_count,
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _StreamBase.__init__(self)
        values = _bind_create_args(cls_name, create_params, args, kwargs)
        handle = ctypes.c_void_p()
        _check(create(*values, ctypes.byref(handle)), f"ct_{stem}_stream_create")
        self._handle = handle

    return type(cls_name, (_SingleStream,), {
        "__init__": __init__,
        "__doc__": f"Incremental {stem.upper()} over one series.",
        "_inputs": inputs,
        "_outputs_count": out_count,
        "_append_fn": staticmethod(append),
    })


def _make_batch(cls_name: str, stem: str, inputs: Tuple[str, ...],
                out_count: int, create_params: Tuple[str, ...]) -> type:
    create = _prototype(
        f"ct_{stem}_stream_batch_create",
        [_CREATE_CTYPES[p] for p in create_params] + [ctypes.c_int,
                                                      ctypes.POINTER(ctypes.c_void_p)],
    )
    append = _prototype(
        f"ct_{stem}_stream_batch_append",
        [ctypes.c_void_p] + [_c_float_p] * len(inputs) + [ctypes.c_int] +
        [_c_float_p] * out_count,
    )

    def __init__(self, *args: Any, num_series: int = 0, **kwargs: Any) -> None:
        _StreamBase.__init__(self)
        if num_series <= 0:
            raise ValueError(f"{cls_name}: num_series must be > 0")
        values = _bind_create_args(cls_name, create_params, args, kwargs)
        handle = ctypes.c_void_p()
        _check(create(*values, int(num_series), ctypes.byref(handle)),
               f"ct_{stem}_stream_batch_create")
        self._handle = handle

    return type(cls_name + "Batch", (_BatchStream,), {
        "__init__": __init__,
        "__doc__": f"Incremental {stem.upper()} over a portfolio of series.",
        "_inputs": inputs,
        "_outputs_count": out_count,
        "_append_fn": staticmethod(append),
    })


def _bind_create_args(cls_name: str, params: Tuple[str, ...],
                      args: Tuple[Any, ...], kwargs: dict) -> list:
    """Map positional/keyword constructor arguments onto the C parameters."""
    if len(args) > len(params):
        raise TypeError(f"{cls_name}: expected at most {len(params)} positional "
                        f"argument(s) {params}, got {len(args)}")
    values = list(args)
    for name in params[len(args):]:
        if name not in kwargs:
            raise TypeError(f"{cls_name}: missing argument '{name}'")
        values.append(kwargs.pop(name))
    if kwargs:
        raise TypeError(f"{cls_name}: unexpected argument(s) "
                        f"{', '.join(sorted(kwargs))}")
    return [_CREATE_CTYPES[p](v) for p, v in zip(params, values)]


for _name, _stem, _inputs, _outc, _params in _FAMILIES:
    globals()[_name] = _make_single(_name, _stem, _inputs, _outc, _params)
    globals()[_name + "Batch"] = _make_batch(_name, _stem, _inputs, _outc, _params)
