# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Python binding contract, checked against an FFI recorder.

The native library is replaced before import, so these tests need neither a GPU
nor CUDA. They pin what must be decided in Python because the native side only
receives a pointer and a length: shapes, element counts and dtype spelling.
"""
import ctypes
import importlib
import os
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


class RecordedFunction:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.argtypes = None
        self.restype = ctypes.c_int

    def __call__(self, *args):
        self.calls.append((self.name, args))
        return b"" if self.restype is ctypes.c_char_p else 0


class RecorderLibrary:
    def __init__(self):
        self.calls = []
        self._functions = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._functions.setdefault(name, RecordedFunction(name, self.calls))


def scalar(value):
    return getattr(value, "value", value)


def load_tacuda():
    recorder = RecorderLibrary()
    sys.path.insert(0, str(ROOT / "bindings" / "python"))
    for name in [m for m in sys.modules if m == "tacuda" or m.startswith("tacuda.")]:
        del sys.modules[name]
    with mock.patch.dict(os.environ, {"TACUDA_LIBRARY": __file__}), \
            mock.patch("ctypes.CDLL", return_value=recorder):
        module = importlib.import_module("tacuda")
    return module, recorder


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tacuda, cls.recorder = load_tacuda()

    def setUp(self):
        self.recorder.calls.clear()

    def assert_no_native_call(self):
        self.assertEqual(self.recorder.calls, [])

    def last_call(self, name):
        calls = [args for called, args in self.recorder.calls if called == name]
        self.assertEqual(len(calls), 1, f"expected one {name} call")
        return calls[0]


class TransferCountTests(ContractTest):
    def upload(self, *args, **kwargs):
        return self.tacuda.device_upload(np.zeros(3, dtype=np.float32), 0x1000, *args, **kwargs)

    def test_count_beyond_host_array_never_reaches_native(self):
        with self.assertRaisesRegex(ValueError, "exceeds the host array size 3"):
            self.upload(100)
        self.assert_no_native_call()

    def test_count_defaults_to_host_array_size(self):
        self.upload()
        self.assertEqual(scalar(self.last_call("ct_device_upload")[2]), 3)

    def test_partial_transfer_is_allowed(self):
        self.upload(2)
        self.assertEqual(scalar(self.last_call("ct_device_upload")[2]), 2)

    def test_count_must_be_a_representable_integer(self):
        for bad, error in ((1.5, TypeError), (True, TypeError), (-1, ValueError), (2**40, ValueError)):
            with self.subTest(bad=bad), self.assertRaises(error):
                self.upload(bad)
        self.assert_no_native_call()

    def test_matrix_upload_counts_all_elements(self):
        # Batch device calls take one flat upload of a (numSeries, seriesLen) matrix.
        self.tacuda.device_upload(np.zeros((2, 3), dtype=np.float32), 0x1000)
        self.assertEqual(scalar(self.last_call("ct_device_upload")[2]), 6)

    def test_typed_upload_is_bounded_in_elements(self):
        host = np.zeros(4, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "exceeds the host array size 4"):
            self.tacuda.device_upload_ex(host, 0x1000, 5, "float64")
        self.assert_no_native_call()
        self.tacuda.device_upload_ex(host, 0x1000, None, "float64")
        args = self.last_call("ct_device_upload_ex")
        self.assertEqual(scalar(args[2]), 4)
        self.assertEqual(scalar(args[3]), self.tacuda.CT_DTYPE_F64)

    def test_download_count_sizes_the_result(self):
        out = self.tacuda.device_download(0x1000, 5)
        self.assertEqual(out.shape, (5,))
        with self.assertRaises(ValueError):
            self.tacuda.device_download(0x1000, -1)


class DtypeTests(ContractTest):
    def assert_storage(self, value, expected):
        self.recorder.calls.clear()
        self.tacuda.sma_ex([1, 2, 3], 2, value, 2)
        args = self.last_call("ct_sma_ex")
        self.assertEqual(scalar(args[4]), expected)

    def test_every_spelling_resolves_to_the_same_enum(self):
        f64 = self.tacuda.CT_DTYPE_F64
        for spelling in ("float64", "F64", np.float64, np.dtype("float64"), float, f64):
            with self.subTest(spelling=spelling):
                self.assert_storage(spelling, f64)

    def test_compute_type_is_normalized(self):
        self.tacuda.sma_ex([1, 2, 3], 2, "float64", "float32")
        args = self.last_call("ct_sma_ex")
        self.assertEqual(scalar(args[5]), self.tacuda.CT_DTYPE_F32)

    def test_output_uses_storage_dtype(self):
        self.assertEqual(self.tacuda.sma_ex([1, 2, 3], 2, "float64", 2).dtype, np.float64)

    def test_bad_values_fail_before_native(self):
        cases = (("float128", 2, ValueError), (7, 2, ValueError), (True, 2, TypeError),
                 ("float64", "nope", ValueError), ("float16", "float64", ValueError))
        for storage, compute, error in cases:
            with self.subTest(storage=storage, compute=compute), self.assertRaises(error):
                self.tacuda.sma_ex([1, 2, 3], 2, storage, compute)
        self.assert_no_native_call()


class SingleSeriesShapeTests(ContractTest):
    MATRIX = [[1.0, 2.0], [3.0, 4.0]]

    def test_matrix_is_rejected_in_every_precision_path(self):
        calls = {
            "sma f32": lambda: self.tacuda.sma(self.MATRIX, 2),
            "sma f64": lambda: self.tacuda.sma(self.MATRIX, 2, dtype="float64"),
            "ema f64": lambda: self.tacuda.ema(self.MATRIX, 2, dtype="float64"),
            "rsi f64": lambda: self.tacuda.rsi(self.MATRIX, 2, dtype="float64"),
            "macd f64": lambda: self.tacuda.macd(self.MATRIX, 2, 3, 2, dtype="float64"),
            "bbands f64": lambda: self.tacuda.bbands(self.MATRIX, 2, dtype="float64"),
            "atr f64": lambda: self.tacuda.atr(self.MATRIX, self.MATRIX, self.MATRIX, 2, dtype="float64"),
            "sma_ex": lambda: self.tacuda.sma_ex(self.MATRIX, 2, "float64", 2),
        }
        for label, call in calls.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, "must be a 1D array"):
                call()
        self.assert_no_native_call()

    def test_scalar_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a 1D array"):
            self.tacuda.sma(3.0, 2)
        self.assert_no_native_call()

    def test_series_keeps_its_length_and_entry_point(self):
        out = self.tacuda.sma([1.0, 2.0, 3.0], 2)
        self.assertEqual(out.shape, (3,))
        self.last_call("ct_sma")
        self.recorder.calls.clear()
        out = self.tacuda.sma([1.0, 2.0, 3.0], 2, dtype="float64")
        self.assertEqual((out.shape, out.dtype), ((3,), np.float64))
        self.last_call("ct_sma_ex")

    def test_batch_entry_point_still_takes_matrices(self):
        out = self.tacuda.sma_batch(self.MATRIX, 2)
        self.assertEqual(out.shape, (2, 2))
        self.last_call("ct_sma_batch")


if __name__ == "__main__":
    unittest.main()
