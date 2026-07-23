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
