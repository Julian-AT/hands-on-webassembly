# Native-compatible t-SNE arithmetic

The browser-only package preserves scikit-learn 1.6.1's estimator, PCA initialization, probabilities, Barnes–Hut algorithm, parameters, seeds, and optimizer. It does not change the supplied applications or native fixtures. Floating acceptance remains `rtol=1e-4`, `atol=1e-5`; discrete neighbor decisions must match exactly. Comprehensive release certification remains separate.

The source and retained BSD license are in `packages/native_tsne/`. `upstream.json` pins original files to scikit-learn commit `f159b78dc59f250cdde8fe391a21f0bc871960ad`, records source hashes, and documents the changes. Earlier wheels and rejected experiments remain retained.

## Arithmetic changes

- Explicit native fused operations in tree summaries, forces, and cost accumulation. A deterministic one-thread cost accumulator fixes an isolated candidate's private accumulator error.
- Positive normal float power uses the native range-reduction and polynomial evaluation order. Its 128-entry tables are generated from mathematical definitions at 100 decimal digits. There are no fixture-derived output tables or shipped system executables. The diagnostic matches 123,818 retained and independent native inputs bitwise.
- Cython `cdivision=True` matches the pinned native build directive. Without it, a zero-distance nonleaf tree query raised an ignored exception and truncated a Penguins summary. The corrected identical-input probe matches all 347 arrays bitwise.
- A restricted four-feature distance backend reproduces the pinned native environment's double dot products and matrix tails. It retains the original private heap, merge, sorting, and self-removal decisions. It dispatches only for float64 Euclidean brute queries with 150, 178, or 256 points, contiguous input, and default/single jobs. All other queries retain scikit-learn's existing backend. This prevents a native/browser arithmetic tie from selecting different Iris neighbors.

The distance arithmetic matches 11,972,000 independently generated native matrix entries. Installed Chrome's strict operation diagnostic matches all 48 arrays, including duplicate points and tied indices: `evidence/tsne-distance-tail-operations.json`. Earlier unsupported-width and incomplete-heap diagnostics remain retained as failures; their settings are excluded from dispatch. The complete seed/perplexity matrices and deployed verification are recorded separately.

## Reproducible build

```sh
.venv/bin/python proof/scripts/build_tsne.py
.venv/bin/python proof/scripts/build_tsne.py
.venv/bin/python proof/scripts/build.py
```

The toolchain is pinned to Python 3.12, Emscripten 3.1.58, Pyodide 0.27.7 / ABI 2024_0. The source epoch is fixed; implicit contraction is disabled and native fused operations are explicit. A repeated build refuses to replace an existing wheel with different bytes. Package source changes require a new version and lock. Runtime registration checks the scikit-learn version and rejects mismatched assets.

## Verification limits

Full traces cover Wine, Penguins, Iris, and Breast Cancer, both 2D/3D, scaled/unscaled inputs, seeds 0/42/123, and perplexities 5/30/50. Seeds 0/123 are explicit diagnostic constructor overrides of the application's fixed seed 42. Original calculations and historical expectations remain unchanged.

Only reports marked as deployed establish integration. Isolated operation and candidate matrices do not establish complete UI, every feature subset, all slider values, other installed browsers, lifecycle reliability, or hardware acceptance. Standalone double-power differences remain retained; complete force/optimization traces establish their observed effect within each exercised setting.
