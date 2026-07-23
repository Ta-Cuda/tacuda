# Contributing to TaCuda

Thanks for considering a contribution! This page describes the local
workflow; the hard gates a change must pass are the same ones CI enforces.

## Build and test

```bash
git clone https://github.com/Ta-Cuda/tacuda.git
cd tacuda

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Windows: `scripts\build_win_dll.bat` (MSVC + Ninja; set `CUDA_PATH` first).
The default configuration also builds the `tacuda-opt` optimizer and
`example_app` — keep them compiling.

With an NVIDIA GPU and the `TA-Lib` Python package installed, run the
strict differential corpus — it is the source of truth for every
compatibility claim:

```bash
export TACUDA_LIBRARY=$PWD/build/libtacuda.so
export PYTHONPATH=$PWD/bindings/python
python benchmarks/differential_corpus.py
```

Binding pipelines:

```bash
python bindings/python/test_pipeline.py
node bindings/nodejs/test_pipeline.js          # npm install first
dotnet run --project bindings/csharp/ConsoleExample -c Release -- --test
```

## Generated code

Two generators produce checked-in artifacts. Never edit generated files by
hand — change the generator and re-run it:

```bash
python scripts/generate_batch_api.py     # batch/device C surface + Node/C# layers
python bindings/generate_bindings.py     # Python ctypes + C# P/Invoke from headers
python bindings/audit_bindings.py        # must exit 0: full C ABI coverage
```

CTest contains a `bindings_up_to_date` guard that fails when checked-in
bindings are stale.

## Adding an indicator

1. `include/indicators/MyIndicator.h` + `src/indicators/MyIndicator.cu`:
   a class with `calculate(const float* in, float* out, int size,
   cudaStream_t stream)`. Follow TA-Lib semantics: warm-up region filled
   with `NaN`, double precision where TA-Lib uses it, parameter bounds like
   TA-Lib's `BAD_PARAM`.
2. C ABI: add a family entry in `scripts/generate_batch_api.py` (gives you
   `ct_*_d` + `ct_*_batch[_d]` for free) or a hand-written wrapper in
   `src/api/` for host-only functions.
3. Regenerate bindings (commands above); add a Node.js wrapper in
   `bindings/nodejs/index.js` only if the generated layer does not cover it.
4. Tests: a GoogleTest case in `tests/cpp/`; if TA-Lib implements the same
   function, add a spec in `benchmarks/talib_compare_helpers.py` — the
   corpus enforces exact coverage of all installed TA-Lib functions.

## Rules

- **ABI**: append-only within a MAJOR version — never remove/rename exported
  symbols or change shipped signatures (full policy below).
- **Versions**: the native version macros and all three package versions move
  together (release criteria below).
- **Numerical changes** that alter outputs must be listed in
  [CHANGELOG.md](CHANGELOG.md) with the affected functions.
- **Performance claims** need a reproducible benchmark under `benchmarks/`.
- Open an issue before large architectural changes.

## Submitting

- Keep PRs focused; include tests for behavior changes.
- Make sure `ctest`, the corpus (GPU), the binding pipelines and
  `audit_bindings.py` are green.
- CI builds all targets on Linux and Windows and runs the GPU suite on
  tagged releases — a release cannot ship if any gate fails.

---

## ABI stability policy

Applies to the native C ABI exported by `tacuda.dll` / `libtacuda.so` and
declared in `include/tacuda.h` + `include/tacuda_device.h`. Semantic
versioning is surfaced in three places which must always agree: the
`CT_VERSION_*` macros (compile-time), `ct_version()` (run-time) and the
package versions of the Python/Node.js/.NET bindings.

### Within a MAJOR version (append-only ABI)

- Exported symbols are never removed or renamed.
- Existing signatures never change: parameter types, order, defaults, and
  return type are frozen once shipped.
- Existing semantics never change silently. Numerical fixes that alter
  outputs are allowed only in MINOR releases and must be listed in
  [CHANGELOG.md](CHANGELOG.md) with the affected functions.
- Enum values (`ctStatus_t`, `ctMaType_t`, `ctDataType_t`) keep their
  numeric values; new enumerators may be appended.
- Structs are not part of the ABI (the surface is deliberately flat:
  pointers + scalars only) — OHLCV containers live in the bindings.
- New capabilities arrive as new symbols (`ct_*_ex`, `ct_*_batch`, ...) —
  never by repurposing existing ones.

**MINOR** releases may add symbols/enumerators/macros and fix numerical
behavior with changelog notice; binaries stay drop-in replaceable.
**PATCH** releases: bug fixes only — no new symbols, no output changes
outside the fixed defect. **MAJOR** releases may break the above; removed
symbols must have been documented as deprecated for at least one full MINOR
release beforehand (deprecation = `@deprecated` doc comment in `tacuda.h`
naming the replacement + changelog entry; the symbol stays functional until
the next MAJOR).

Enforcement: `bindings/audit_bindings.py` (bindings must track ABI growth in
the same release), `scripts/check_dll_exports.py` (representative symbols
present in both natives), the `bindings_up_to_date` CMake test (no drift
between headers and generated wrappers) and the release criteria below.

## Release criteria

Every tagged release (`vMAJOR.MINOR.PATCH`) ships native libraries
(win-x64 DLL, linux-x64 .so), PyPI wheel/sdist, npm tgz and NuGet nupkg.
Nothing is pushed to a registry until all gates pass — the
`release-packages` workflow enforces them:

1. **Version consistency (blocking).** `CT_VERSION_STRING` in
   `include/tacuda.h`, `project.version` (pyproject.toml), `version`
   (package.json), `<Version>` (Tacuda.Bindings.csproj) and the top
   [CHANGELOG.md](CHANGELOG.md) heading must all equal the tag;
   `ct_version()` of the built binaries must match.
2. **Native artifact validation (blocking).** Both binaries pass
   `python scripts/check_dll_exports.py <dll> <so>`; binaries link the
   shared CUDA runtime; nupkg `runtimes/` payloads are the artifacts of the
   same commit (CI downloads them from the `native-build` job).
3. **Test gates (blocking, GPU required).** `ctest` 100% pass, the three
   binding pipelines report `0 failed`, the differential corpus is green,
   `benchmark_batch_api.py --quick` passes (no pool growth explosion).
4. **Package validation (blocking).** Wheel installs into a clean venv and
   imports from site-packages; npm tarball installs and loads; nupkg
   contains both natives and a consumer app resolves P/Invoke.
5. **Signing & provenance.** Annotated signed tag; registry-scoped secrets
   (`PYPI_API_TOKEN`, `NPM_TOKEN`, `NUGET_API_KEY`); SHA-256 checksums of
   both natives in the release notes (automated by the workflow).
6. **Rollback.** Registries are immutable: ship `X.Y.(Z+1)` with the fix,
   yank/deprecate/unlist the broken version.

[CHANGELOG.md](CHANGELOG.md) is the single source of release notes
(one section per version); CI publishes the top section as the GitHub
release body.
