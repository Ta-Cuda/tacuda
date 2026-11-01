# Operations Guide

Deployment, monitoring, and upgrade runbooks for services embedding tacuda,
plus a production adoption checklist.

## Deployment

### Prerequisites

- [ ] NVIDIA driver installed and ≥ the minimum for CUDA 12.x
      (`nvidia-smi` works inside the target environment/container).
- [ ] For containers: NVIDIA Container Toolkit configured (`--gpus all` /
      `runtimeClassName: nvidia`), base image may be `-runtime` (not
      `-devel`) since tacuda links cudart shared.
- [ ] Native library resolvable: bundled by the PyPI wheel / NuGet package
      automatically; for npm set `TACUDA_LIBRARY=/path/libtacuda.so`.
- [ ] `ct_version()` (or `tacuda.version()`) logged at service start —
      confirms the loaded binary matches the expected release.

### Smoke test on a fresh host

```bash
python - <<'PY'
import numpy as np, tacuda
x = np.linspace(1, 100, 4096, dtype=np.float32)
assert not np.isnan(tacuda.sma(x, 20)[-1])
print("tacuda", tacuda.version(), "OK")
PY
```

Exit code 0 = driver, native library, and bindings all healthy.

## Monitoring

| Signal | Source | Alert on |
|--------|--------|----------|
| GPU utilization / memory | `nvidia-smi` / DCGM exporter | sustained > 90% mem |
| Buffer pool growth | `ct_device_buffer_pool_allocation_count()` | monotonic growth in steady state (leak) |
| Native errors | exceptions carrying `ct_get_last_error()` | any `CT_STATUS_ALLOC_FAILED` / `DEVICE_ERROR` burst |
| Library version | `ct_version()` at startup | mismatch across replicas |
| Batch latency | wrap `*_batch` calls with timers | p99 regression > 20% after upgrade |

`CT_STATUS_ALLOC_FAILED` under load usually means the buffer-pool cache plus
workload exceeds VRAM: lower `ct_device_buffer_pool_set_max_cache_bytes`,
reduce `maxConcurrency`, or shard the portfolio.

## Upgrade sequencing

1. Read [CHANGELOG.md](CHANGELOG.md) for the target version; MINOR bumps are
   append-only ABI (see the
   [ABI policy](CONTRIBUTING.md#abi-stability-policy)), MAJOR bumps may break.
2. Verify version consistency: package version == `ct_version()` of the
   bundled native (release criteria guarantee this, trust but log).
3. Canary one replica; compare indicator outputs against the previous
   version on a fixed dataset (`benchmarks/datasets.py` snapshots) —
   expected max drift within float32 tolerance (≤ 1e-4 relative) unless the
   changelog documents an algorithm fix.
4. Roll out; keep the previous package version pinned for instant rollback
   (registries are immutable, downgrade = reinstall previous version).

## Support escalation

1. Capture: `ct_version()`, GPU model, driver (`nvidia-smi`), OS,
   binding language + package version, `ct_get_last_error()` text, minimal
   reproducing series (lengths + parameters matter more than values).
2. Check known issues: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
   (parity corner cases) and the warm-up rules in
   [API.md](API.md#output-alignment-warmup-nan).
3. Reproduce with the pinned datasets
   (`python benchmarks/datasets.py --export /tmp/ds`) and file a GitHub
   issue with the bundle: https://github.com/Ta-Cuda/tacuda/issues.
4. Regressions blocking production: roll back to the previous package
   version (step 4 above) and mark the issue `severity:high`.

## Adoption checklist

A copy-paste checklist for teams putting tacuda into production. Work top
to bottom.

### 1. Environment

- [ ] GPU inventory confirmed: model, VRAM, compute capability ≥ 7.5
      (≥ 5.3 if float16 planned).
- [ ] Driver version compatible with CUDA 12.x runtime on every target
      host ([deployment](#deployment)).
- [ ] Containers: NVIDIA Container Toolkit configured; smoke test passes
      inside the image, not just on the host.
- [ ] Native library resolution decided per language (bundled wheel/nupkg
      vs `TACUDA_LIBRARY` for npm).

### 2. Integration

- [ ] Pinned package version (`tacuda==X.Y.Z` / lockfile / `PackageReference`).
- [ ] `ct_version()` logged at service startup.
- [ ] Error handling wraps native exceptions and logs
      `ct_get_last_error()` detail ([conventions](API.md#conventions)).
- [ ] NaN warmup regions handled downstream (never treat warmup as data).
- [ ] Warmup-alignment notes vs TA-Lib reviewed if migrating from TA-Lib
      ([IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)).
- [ ] OHLCV containers used instead of hand-packed arrays
      ([API.md](API.md#ohlcv-containers)).
- [ ] Portfolios use the batch API, not per-symbol loops
      ([BATCH_API.md](BATCH_API.md)).

### 3. Validation

- [ ] Binding test pipeline passes on a production-like GPU
      (`test_pipeline.py` / `test_pipeline.js` / `--pipeline`).
- [ ] Numerical spot-check against your incumbent implementation on real
      data; differences within float32 tolerance or explained.
- [ ] Load test at production portfolio size; p99 latency and VRAM headroom
      recorded ([PERFORMANCE.md](PERFORMANCE.md)).
- [ ] Buffer pool stable under sustained load (allocation count plateaus).

### 4. Monitoring (before go-live)

- [ ] GPU utilization/memory exported (DCGM or `nvidia-smi` scraping).
- [ ] Alert on `CT_STATUS_ALLOC_FAILED` / `CT_STATUS_DEVICE_ERROR`.
- [ ] Alert on buffer-pool allocation count growth in steady state.
- [ ] Indicator-latency dashboards segmented by batch size.

### 5. Upgrade & rollback readiness

- [ ] Upgrade sequencing documented for your service
      ([above](#upgrade-sequencing)): changelog review → canary vs fixed
      dataset → fleet rollout.
- [ ] Previous package version pinned and installable offline (registry
      outage must not block rollback).
- [ ] Fixed-dataset regression harness wired to
      `benchmarks/datasets.py` snapshots for canary comparison.

### 6. Support

- [ ] Escalation path known: capture bundle (version, GPU, driver, error
      text, repro shape) → known-issues docs → GitHub issue
      ([above](#support-escalation)).
- [ ] Team subscribed to release notifications
      (GitHub releases / [CHANGELOG.md](CHANGELOG.md)).
- [ ] ABI expectations understood: MINOR = drop-in, MAJOR = review
      ([ABI policy](CONTRIBUTING.md#abi-stability-policy)).

## Validation snapshot — 2026-10-07

The Windows GPU run tested the unchanged source snapshot at
`263f99bde46d3c93ad1734240ebc1cf7115c4b64`, including the unreleased streaming
API. This is validation evidence, not a release or a Linux acceptance result.
Logs, corpus JSON, CTest JUnit files and the isolated test harness are in
[windows-logs.zip](validation/2026-10-07/windows-logs.zip).

Every evidence archive under `validation/` is redacted before publication:
host addresses, SSH details, workstation paths and account names are removed
or replaced, and the evidence checksums in this document are those of the
redacted archives.

Environment: a Windows GPU workstation, Windows 11 Enterprise build 26300, RTX 3090 Ti,
driver 610.88, CUDA 12.9.41, MSVC 19.44.35228, CMake 4.1.0-rc2,
Release/Ninja targeting SM 86. Python 3.12.8 and NumPy 2.2.6 ran in isolated
virtual environments. No tracked source files changed on the remote host.

| Check | Result |
|-------|--------|
| Native build, including optimizer and examples | Passed |
| CTest, sequential | 418 passed, 1 skipped; the skipped `Backends.TalibMatchesReference` passed on a focused rerun after supplying the official TA-Lib 0.8.1 DLL via `TACUDA_OPT_TALIB` |
| Python pipeline | 34 passed, 0 failed |
| Node.js pipeline | 201 passed, 0 failed |
| C# pipeline / binding tests | 36 / 100 passed, 0 failed; net8.0 and net9.0 bindings built |
| Differential corpus, Python TA-Lib 0.7.0 / native TA-Lib 0.7.1 | 165/165 specs, 1319/1319 evaluated cells passed; one documented NATR cell excluded |
| Differential corpus, Python/native TA-Lib 0.8.1 | Failed: 40 reference functions lack corpus specs; IMI(14)/flat and WILLR(14)/zeros differ; 163/165 specs and 1317/1319 evaluated cells passed |
| Streaming corpus, TA-Lib 0.8.1 | 699/699 cells passed |
| Batch quick/stress check | Passed; outputs stable, allocation count 52 to 53 over three stress rounds |
| Native export resolution | All 811 declarations in the three public headers resolved; the existing 808-symbol audit excludes the three string-returning introspection functions |
| Streaming handle lifetime, AddressSanitizer | Failed as expected for the reported defect: `create -> destroy -> reset` reads freed memory in `src/api/api_stream.cpp:130`; normal create/append/destroy control passed |

The current TA-Lib 0.8.1 exposes 201 functions, whereas the earlier reference
exposes 161. Therefore the full-coverage claim must name its reference version.
The 40 missing corpus specifications are a coverage gap, not proof that all 40
implementations are absent: some functions already exist as TACUDA extras.
Do not loosen numerical tolerances to hide the IMI/WILLR differences; review
the reference-version semantics first. The Windows GPU result also confirms
that passing the existing suite does not establish the promised rejection of
destroyed raw C streaming handles.

Two initial inline Python preflight commands failed because Windows
PowerShell removed embedded quotes. File-based verification subsequently
passed (`verification-talib081.json` and `verification-talib070.json`); the
original harness errors remain in `results.json` for traceability.

Artifact provenance:

- Source archive SHA-256:
  `5a7e4b2738b92f93190ca540397b3ce1e1fe6c5ca5db595b004a651ee3cf0c83`.
- Built `tacuda.dll` SHA-256:
  `11e3fb5731eee931c8df32952c6b55eaf15001c24229a6b8ed7de48b427299b2`.
- Evidence ZIP SHA-256 (redacted archive):
  `48bdd580042459a4edf6372cc258a71ea015d9fe76a5f65ec766aa1010b562ff`.
- The run used an isolated directory on the workstation; the temporary
  scheduled task was removed after completion and existing host services
  were left running.

Linux remained unverified in this run: the Linux GPU host was unreachable
at the time. The 2026-10-14 run below closes that gap.


## Fix validation — 2026-10-07, 0.8.0 working tree

The corrections were built in a separate isolated directory on the same
Windows GPU workstation (RTX 3090 Ti, i5-13500, CUDA 12.9, MSVC 19.44). The
previous audit directory
and its evidence above were preserved. This is Windows validation of the
working tree based on `263f99b`, not a published release.

[fixes-windows-logs.zip](validation/2026-10-07/fixes-windows-logs.zip) contains
CTest JUnit, corpus JSON, binding logs, benchmarks, runner scripts and
`tested-source-manifest.json`. All 543 native/binding/test/build-tool input
hashes in that manifest matched the local working tree when collected.

| Check | Result |
| --- | --- |
| Release native build, optimizer and examples | Passed |
| Full CTest after handle correction | 423/423 passed |
| Focused CTest after adding ASan/version gates | 35/35 passed; overlaps the full run, not 35 additional tests |
| AddressSanitizer implementation test | Passed; retired handles, allocation churn and independent threads |
| Python pipeline including diagnostic error paths | 37/37 passed |
| Node.js pipeline including diagnostic helpers | 203/203 passed |
| C# pipeline / binding tests | 36/36 and 100/100 passed |
| Baseline differential corpus | 165/165 specs, 1319/1319 evaluated cells; one documented NATR exclusion |
| Streaming corpus | 699/699 cells passed |
| Actual PE export table | All 811 public declarations exported |
| Local release-tool regression suite | 7/7 passed, including malformed binaries and wrong/stale release notes |
| Local workflow validation | actionlint passed with the explicit custom `gpu` label |

The numerical reference was Python TA-Lib 0.7.0 / native TA-Lib 0.7.1,
NumPy 2.2.6. The optimizer reference used the existing official TA-Lib 0.8.1
DLL. No tolerances were widened. The TA-Lib 0.8.1 full-surface migration
remains open as recorded in the earlier snapshot.

Fresh Windows measurements: RSI append p50/p99 0.10/0.10 microseconds,
independent of 100K/1M history length at the timer's resolution. The portfolio
AUTO path at 256 symbols measured p50 34.9 and p99 57.1 microseconds. The
previous p99 < 1 ms claim through 65536 symbols did not reproduce: AUTO p99
was 1263.6 microseconds at 2048 and 3194.6 at 65536. These timings do not
isolate the registry change from OS, CPU and scheduling differences.
Vector math was rerun with separate NumPy and TA-Lib ratios in
`vector-math.json`; it is a best-of-five Windows run, not a general guarantee.

The initial run stopped at npm installation in its PowerShell harness.
Installation succeeded when run directly in the package directory, and the
second harness completed every remaining check. Both result logs are kept.
The temporary scheduled tasks were removed after completion.

Artifact SHA-256:

- Native DLL: `4f2ef88ae3efef4788f6127d7ebfb24d5e7e6cf858429022dea20b570de5d874`.
- Evidence ZIP (redacted archive): `9dcf3e868add6a1d1c89e7c60deb373ec999458cda3662c71941fc11fe0d47e6`.

Still required before publishing: hosted Windows/Linux CI, Linux GPU parity,
clean installation of the final packaged artifacts, finalized release notes,
and public tag/release. Registry publication and the separately maintained
website were not changed. Keep the changelog Unreleased until these gates
are satisfied; rollback remains the pinned public 0.7.1 packages.


## Linux GPU validation — 2026-10-14, source `4e8edd1`

This closes the Linux GPU parity gap left open above. The run used the exact
committed tree `4e8edd17576982cf17cf886f82906abb8818f1c3`, exported with
`git -c core.autocrlf=false archive` so the files have the LF endings a Linux CI
checkout would have. It is validation evidence, not a release: no tag, package
or registry changed.

[linux-logs.zip](validation/2026-10-14/linux-logs.zip) contains the gate script,
per-step logs, CTest JUnit, both corpus reports, `provenance.txt` and the logs
of two aborted attempts (described below).

Environment: the Linux GPU host, Ubuntu 24.04.4, RTX 3090 Ti,
driver 590.48.01, CUDA 12.8.93, GCC 13.3, CMake 3.29.6, Ninja, Release, SM 86.
Python 3.12.3 with the pinned reference (`tests/requirements-reference.txt`),
Node.js 22.23.3, .NET SDK 8.0.426 only (as in the CI `gpu-tests` job).
Two vLLM services held 21.2 of 23 GB VRAM throughout and were left running.

| Check | Result |
| --- | --- |
| Release build, all targets incl. optimizer and examples | Passed, 389/389 steps |
| Streaming lifetime under AddressSanitizer | Passed |
| ELF export table | All 811 public declarations exported |
| CTest, sequential, reference venv first on `PATH` | 425 tests, 0 failed; `Backends.TalibMatchesReference` skipped (needs `TACUDA_OPT_TALIB`) |
| Python pipeline | 37/37 passed |
| Differential corpus | 165/165 specs, 1319/1319 cells; one documented NATR exclusion |
| Streaming corpus | 699/699 cells |
| Batch quick/stress | Passed; outputs stable, allocation count 41 to 41 |
| Node.js pipeline | 203/203 passed |
| C# pipeline / binding tests | 36/36 and 100/100 passed (net8.0 with SDK 8 only) |

Deviations from the CI job, all environmental:

- The host has no system CUDA. The user-local toolkit lacks
  `libcudart_static`, so the build used `--cudart=shared` with
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared` and `LIBRARY_PATH` pointing at it.
  The first attempt failed at CMake compiler detection and the second at
  linking `libtacuda.so` for exactly these reasons; no source change was made.
- System Node 18 ships without npm; a checksum-verified Node 22 tarball was
  unpacked into the run directory instead, matching `setup-node`.

Reference version caveat: the same pinned wheel `TA-Lib==0.7.0` reports
native `__ta_version__` 0.6.4 on Linux, while on Windows (2026-10-07) it
reported 0.7.1. Both expose the same 161 functions and both pass the strict
corpus. Statements naming "native TA-Lib 0.7.1" are therefore exact for the
Windows reference only; the oracle that matters is the pinned wheel.

Artifact SHA-256:

- Source archive: `2529aafe422ec30860d7e387f894b38b7b83926ea32396fa899a536d295cad50`.
- Built `libtacuda.so`: `8539db5507672b387457b3f10eb884a28c1b7f2afb3b117a30046645396b53d2`.
- Evidence ZIP (redacted archive): `de406f56a30c4c4bed067d12b0ba99929f202669bacc0aa0f1cd8b1e70b9d871`.

Remote run directory: `~/work/tacuda-v080-gate-20261014`; nothing outside it
was modified, and the .NET build servers were shut down afterwards.

Still required before publishing: a registered `gpu` runner so the tagged
workflow can run this gate itself, a dry run of `release-packages` (it has
never run end to end), clean installation of the final artifacts on a GPU
host, finalized release notes and the public tag/release. The packaging
defects found during this review (both natives in the Windows wheel, a
validation step that could not fail, per-registry publishing) are fixed in
the workflow; see [CHANGELOG.md](CHANGELOG.md).

## Linux GPU validation — 2026-11-01, source `a5c8f35`

Revalidation after the `*_batch_d` stream-0 fix (`6540d9e`), the only
behavioral source change since `4e8edd1`; the rest is documentation and SPDX
headers. Same gate as on 2026-10-14, on the exact committed tree
`a5c8f35f4acc7eb995d9f667d848d24315d81f69`, exported with
`git -c core.autocrlf=false -c core.eol=lf archive` — on Windows
`core.autocrlf=false` alone still writes CRLF for `text=auto` files. It is
validation evidence, not a release: no tag, package or registry changed.

[linux-logs.zip](validation/2026-11-01/linux-logs.zip) contains the gate
script, per-step logs, CTest JUnit, both corpus reports, the batch-path sweep
script and `provenance.txt`.

Environment as on 2026-10-14: the same Linux GPU host, RTX 3090 Ti, driver 590.48.01,
CUDA 12.8.93 (shared cudart, for the reasons given above), GCC 13.3, CMake
3.29.6, Ninja, Release, SM 86; Python 3.12.3 with the pinned reference;
Node.js 22.23.3 and .NET SDK 8.0.426 reused from the previous run directory.
Two vLLM services again held 21.2 of 23 GB VRAM throughout.

| Check | Result |
| --- | --- |
| Release build, all targets incl. optimizer and examples | Passed, 389/389 steps |
| Streaming lifetime under AddressSanitizer | Passed |
| ELF export table | All 811 public declarations exported |
| CTest, sequential, reference venv first on `PATH` | 426 tests, 0 failed (new: `BatchAPI.DeviceBatchOnStreamZeroWaitsForLegacyStream`); `Backends.TalibMatchesReference` skipped |
| Python pipeline | 37/37 passed |
| Differential corpus | 165/165 specs, 1319/1319 cells; one documented NATR exclusion |
| Streaming corpus | 699/699 cells |
| Batch quick/stress | Passed; outputs stable, buffer pool allocations 42 to 42 |
| Node.js pipeline | 203/203 passed |
| C# pipeline / binding tests | 36/36 and 100/100 passed |

Batch-path sweep, not a CI gate: the differential corpus calls the host API
only. `batch_corpus_vs_talib.py` runs every corpus spec whose TaCuda side is
one host call with a `*_batch_d` family (143 of 165) through that family on
stream 0 while the input is still being produced on the legacy stream — a
20 ms host-function hold, then a device copy over a NaN-filled buffer — and
judges every batch row with the corpus oracle against TA-Lib:

| Build | Batch rows passing | Specs green |
| --- | --- | --- |
| `a5c8f35` | 3972/4002 | 140/143 |
| Control: `a23d4b7`, the parent without the fix | 2114/4002 | 0/143 |

The three remaining failures are HT_DCPHASE, HT_SINE and HT_TRENDMODE on the
`flat` and `candles` datasets: the device-path deviation listed in
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md#known-deviations-from-ta-lib-tracked).
The single-series `ct_ht_*_d` fails the same cells; the host API passes.

Artifact SHA-256:

- Source archive: `3c4a0489033b4319c6c500208abeea448b2e0ceaa2a7b6b5633d6956b0d55ae1`.
- Control archive: `5e1f740ddb642bad22e5b33070c2ea41c96a63f67bdf345dd98483587b455f63`.
- Built `libtacuda.so`: `a96d00312641a33d35c45d31dd0cf23e0bf8dfd851943636d9b8668ffc41a5c8`.
- Evidence ZIP (redacted archive): `9390de6e10be5632fadd2741ae3280d6e586cd66da28c4a64ec6c14a5b2d7ee0`.

Remote run directory: `~/work/tacuda-v080-gate-20261101`; nothing outside it
and the reused toolchains was modified, and the .NET build servers were shut
down afterwards.

Still required before publishing, unchanged since 2026-10-14:

- A runner labelled `[self-hosted, Linux, gpu]`. A tag push runs
  `native-build` through `workflow_call`; its `gpu-tests` job needs that
  runner and every package job waits for it, so without one the tag run stays
  queued and nothing is published. No runner is installed on that host.
- A `release-packages` dry run (`workflow_dispatch`, publish off). The Actions
  history has no run of that workflow yet.
- Clean installation of the final artifacts on a GPU host, the release branch
  merged into `main`, and the signed tag.
