# Release gates

The native library and Python, npm and NuGet packages must share one version.
`scripts/check_release.py --tag vX.Y.Z --release` validates all version fields
and the numbered changelog section. Preserve existing C signatures; additions
must be represented in all three bindings and `scripts/check_dll_exports.py`.

Before tagging, run the public `release-packages` workflow with `publish=false`.
Require source checks, Linux/Windows native builds, GPU CTest, differential
and streaming corpora, and all binding pipelines to pass. Use the pinned
TA-Lib requirements; a floating or older oracle is not release evidence.
Typed Node/C# fixtures cover integer, mixed and four-array outputs.
Fault-injection builds are test-only and must not supply packaged natives.

Download the actual workflow artifacts and verify their archive digests.
Each platform wheel contains exactly its corresponding native; NuGet contains
both natives and net8/net9 assemblies; npm contains bindings without a native.
Native hashes must match the same run's native artifacts. All formats must
include `THIRD_PARTY_NOTICES.txt` and retain the upstream BSD formula notice.
Install these actual packages in clean environments on Linux and Windows GPUs,
checking versions, exports, typed outputs, full-series and incremental behavior.

Official CUDA 12.9 binaries target SM60, 61, 70, 75, 80, 86, 89 and 90.
Scratch-backed graph capture requires device memory-pool support independently
of the compute capability. Unsupported capture must return a clean diagnostic;
allocation-free fused capture remains a separate contract. Never substitute
an older or unknown-provenance binary for a failed build.

After acceptance, fast-forward public main, create and verify a signed annotated
version tag, and push it. The tag workflow rebuilds, tests and publishes to
PyPI, npm, NuGet and GitHub Releases. Verify the indexed registry payloads and
native checksums against this tag run. Versioned registry payloads and signed
tags are immutable; a correction requires a new version.

The website is deployed separately after registry verification. Prepare staging,
retain a complete before-copy and hash manifest, reject drift before replacing
live files, then verify live hashes and Playwright desktop/mobile flows.
Historical benchmark results retain their original dates and environments.
Rollback restores only files still matching the deployed manifest.
