# Architecture

The course applications execute in browser Python workers. Static hosting serves assets without an inference API or application server. `web/app/course.jsx` presents the seven-assignment selector; `web/lib/navigation.mjs` handles selection URLs, normalization and browser history. `runtime/launcher-session.mjs` owns the iframe lifecycle and is shared with the protocol harness.

The launcher waits for both a connected session and initialized inputs. Its identity binds the protocol, unit, build, source and session. It disposes the incumbent before mounting a replacement, rejects obsolete messages, and offers Retry after startup failure. The canonical startup generator in `scripts/proof/patch_runtime.py` keeps the recovery panel hidden during normal loading so the original Shinylive animation remains visible.

`runtime/` contains tabular, embedding, forest, neural and CNN compatibility adapters, plus owned training, image preparation and embedding workers. Training uses disposable workers with generation-bound cancellation. Corrected native training uses isolated processes. `wasm/packages/` retains the compiled extension sources, upstream licenses, arithmetic patches and reproducible recipes. Frozen scientific versions and numerical tolerances remain authoritative.

`assignments/` contains unchanged supplied sources. Confirmed application fixes are checked substitutions in `scripts/proof/reference_corrections.py`, shared by native reference generation and browser staging. Browser-specific changes remain in the adapters and staging utilities.

Python tools share `scripts/proof/repository.py`; JavaScript tools share `scripts/paths.mjs`. Paths derive from source locations. Active proof metadata stays in `proof/`; generated references, scientific builds and current evidence stay in `artifacts/`. Scientific download and compiler caches stay in `.cache/science`.

Production builds consume the existing immutable application archive. The archive's 218 path/size/hash records are checked before publication. Exactly seven approved entry-page overrides are then applied and checked. The Next.js shell is rebuilt from root-locked dependencies, and every exported file is recorded in `web/out/release-manifest.json`.
