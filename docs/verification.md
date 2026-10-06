# Verification

`make test` preserves the original 141 Python and 10 JavaScript regressions. Additional tests check relocated paths, immutable locks, all 140 suite IDs and their required behavior IDs, archive corruption and missing dependencies, restoration conflicts, real Git hooks and protected/generated-file exclusion. Numerical tolerances and browser/hardware requirements are unchanged.

`make check` runs Prettier, ESLint, Ruff and artifact hygiene. `node scripts/toolchain.mjs commit-check` validates every message on the reconstructed branch. Training-isolation regressions compare every unrounded history value and model parameter with an independent serial control extracted from the unchanged supplied code on the same host. They retain exact equality assertions. PyTorch does not guarantee bit identity across different processors; hosted Apple Silicon runners differed from the original M4 Pro fixture. The historical fixture retains its original SHA-256 and is also compared exactly on its original CPU family. Full certification's fixed fixtures, tolerances and hardware/browser obligations remain unchanged. [PyTorch numerical accuracy](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md).

The CI workflow runs setup, source checks, regressions, commit checks and two clean static builds. It uploads validation output and the full certification gate result even when certification is blocked.

Build reproducibility compares complete release manifests, including the shell and all application payloads:

```sh
make build
cp web/out/release-manifest.json artifacts/export-first.json
make build
cmp artifacts/export-first.json web/out/release-manifest.json
```

Each build validates all 218 original payload records and seven approved startup overrides, and deletes the previous Next.js build/export before rebuilding. The existing application archive, tag and hashes remain unchanged.

`make proof` regenerates the inventory from unchanged course sources and runs the full gate. It exits nonzero while required evidence is missing, stale or failed. `proof/contracts/` preserves suite/behavior identities and the historical dynamic-choice obligations. The command writes its current report to `artifacts/proof/evidence/gate.json`. `proof/status.json` is a dated summary, not acceptance evidence.

Installed-browser acceptance remains a separate full workload. Use the retained harnesses under `scripts/proof/` and `proof/harness/browser/`. Required browsers remain Chrome, Edge, Firefox and Safari; acceptance must satisfy the unchanged hardware and power-state contracts. Local selector smoke checks do not certify the outstanding numerical, pixel, timing, memory or hardware backlog.

Scientific rebuild entry points retain their recipes and flags:

```sh
.cache/env/native/bin/python scripts/proof/build_forest.py
.cache/env/native/bin/python scripts/proof/build_neural.py
.cache/env/native/bin/python scripts/proof/build_cnn.py
.cache/env/native/bin/python scripts/proof/build_tsne.py
.cache/env/native/bin/python scripts/proof/build_matplotlib.py
.cache/env/native/bin/python scripts/proof/reference_corrections.py
.cache/env/native/bin/python scripts/proof/build.py
```

These commands require the frozen Pyodide/Emscripten build environment and original scientific assets. They are independent of the fast production build, which assembles the verified immutable release. The archive preserves the original compiler caches and build evidence for restoration and audit; native or scientific reruns must generate new current evidence.
