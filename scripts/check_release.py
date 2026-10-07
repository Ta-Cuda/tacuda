#!/usr/bin/env python3
"""Check package/native versions and select release notes for the exact tag."""
import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def release_state(root=ROOT, tag=None, release=False):
    header = (root / "include/tacuda.h").read_text(encoding="utf-8")
    native = re.search(r'#define CT_VERSION_STRING "([^"]+)"', header).group(1)
    macros = ".".join(re.search(r"#define CT_VERSION_" + part + r" (\d+)", header).group(1)
                      for part in ("MAJOR", "MINOR", "PATCH"))
    python = re.search(r'^version = "([^"]+)"', (root / "bindings/python/pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    node = json.loads((root / "bindings/nodejs/package.json").read_text(encoding="utf-8"))["version"]
    csharp = re.search(r"<Version>([^<]+)</Version>", (root / "bindings/csharp/Tacuda.Bindings.csproj").read_text(encoding="utf-8")).group(1)
    versions = dict(native=native, macros=macros, python=python, node=node, csharp=csharp)
    if len(set(versions.values())) != 1:
        raise ValueError(f"version mismatch: {versions}")
    if tag is not None and tag != "v" + native:
        raise ValueError(f"tag {tag!r} does not match native/package version v{native}")
    changelog = (root / "docs/CHANGELOG.md").read_text(encoding="utf-8")
    # Only a numbered release heading is eligible; Unreleased must never
    # silently produce notes for the preceding version.
    match = re.search(r"^" + re.escape(native) + r" [^\n]*\n-+\n(.*?)(?=^\d+\.\d+\.\d+ [^\n]*\n-+\n|\Z)", changelog, re.M | re.S)
    notes = match.group(0).strip() + "\n" if match else None
    if (tag is not None or release) and not notes:
        raise ValueError(f"CHANGELOG has no numbered {native} release section; finalize Unreleased first")
    return native, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--notes", type=pathlib.Path)
    args = parser.parse_args()
    try:
        version, notes = release_state(tag=args.tag, release=args.release or bool(args.notes))
        if args.notes:
            args.notes.write_text(notes, encoding="utf-8")
        print(f"PASS: native and all packages use {version}")
        return 0
    except (ValueError, OSError, AttributeError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
