"""Release gates must fail for mismatched versions and non-exported strings."""
import importlib.util
import pathlib
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load("check_release")
exports = load("check_dll_exports")


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        for name in ("include/tacuda.h", "bindings/python/pyproject.toml", "bindings/nodejs/package.json", "bindings/csharp/Tacuda.Bindings.csproj", "docs/CHANGELOG.md"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
        self.version, _ = release.release_state(self.root)

    def test_rejects_mismatched_package(self):
        (self.root / "bindings/nodejs/package.json").write_text('{"version":"99.0.0"}')
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            release.release_state(self.root)

    def test_rejects_mismatched_tag(self):
        with self.assertRaisesRegex(ValueError, "tag"):
            release.release_state(self.root, tag="v99.0.0")

    def test_unreleased_cannot_fall_back_to_old_notes(self):
        (self.root / "docs/CHANGELOG.md").write_text("Unreleased\n----------\nnew\n\n0.7.1 - old\n------------\nold\n")
        with self.assertRaisesRegex(ValueError, "finalize Unreleased"):
            release.release_state(self.root, tag="v" + self.version)

    def test_exact_version_notes(self):
        (self.root / "docs/CHANGELOG.md").write_text(f"{self.version} - release\n----------\nnew\n\n0.7.1 - old\n------------\nold\n")
        _, notes = release.release_state(self.root, tag="v" + self.version)
        self.assertIn("new", notes)
        self.assertNotIn("old", notes)


class ExportTests(unittest.TestCase):
    def test_pe_only_counts_export_table_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "fixture.dll"
            path.write_bytes(b"MZ\0\0ct_fake\0")
            table = "  1    0 00001000 ct_real\n debug string ct_fake\n"
            with patch.object(exports, "dumpbin_path", return_value="dumpbin"), patch.object(exports.subprocess, "check_output", return_value=table):
                self.assertTrue(exports.check(path, {"ct_real"}))
                self.assertFalse(exports.check(path, {"ct_fake"}))

    def test_elf_undefined_import_is_not_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "fixture.so"
            path.write_bytes(b"\x7fELF")
            table = "000000001000 T ct_real\n                 U ct_missing\n"
            with patch.object(exports.subprocess, "check_output", return_value=table):
                self.assertEqual(exports.exported_symbols(path), {"ct_real"})

    def test_text_with_symbol_names_is_not_a_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "fake.dll"
            path.write_text("ct_sma ct_version ct_stream_destroy")
            self.assertFalse(exports.check(path, {"ct_sma"}))


if __name__ == "__main__":
    unittest.main()
