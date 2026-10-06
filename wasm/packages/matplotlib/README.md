# Reference Matplotlib in WebAssembly

The accepted browser wheel is Matplotlib **3.10.8**, using the reference's bundled **FreeType 2.6.1**. Its SHA-256 is `480bd5f132cfaf5e059ef5ab3a8f94aa511f163b2e4a6aeaeb5256e93d26ffba`. Two clean builds produced identical bytes.

`../../scripts/build_matplotlib.py` builds candidates into `cache/matplotlib-wheels/`; compilation alone never publishes a candidate. The accepted wheel, manifest, source checksums, compiler settings, and build metadata live in `assets/v1/runtime-packages/`. Normal exports verify the accepted wheel's checksum and copy it into the local Pyodide package registry.

Build with the existing Python 3.12 / pyodide-build 0.29.3 environment, Pyodide 0.27.7 cross-build environment, and Emscripten 3.1.58:

```sh
.venv/bin/python proof/scripts/build_matplotlib.py
```

The source archive is fixed by SHA-256. FreeType's source URL, fallback URL and SHA-256 come from Matplotlib's pinned Meson wrap. `build-constraints.txt` fixes the Python build tools. Exceptions are enabled, LTO is disabled, and `SOURCE_DATE_EPOCH` and `PYTHONHASHSEED` are fixed.

Two compatibility patches are applied with source-contract checks:

- Disable only the background font-cache warning timer on Emscripten, which has no Python threads in this runtime.
- Correct FreeType indirect-call signatures, adapting [the upstream Emscripten port fix](https://github.com/emscripten-ports/FreeType/commit/40a760c963bc2b76575918ff91a515c8474db4e0) to FreeType 2.6.1's additional glyph-index argument. The dummy hinter accepts its unused metrics argument, the apply-hints typedef returns `FT_Error`, and CID callbacks match their `void` callback interface. Native callers already discarded those CID return values. The rasterization algorithms and font files are unchanged.

The isolated candidate passed all 142 pre-existing bounded application workflow checks. Its stricter console collection found a separate Unit 2 hidden-plot resize rejection; a checked browser staging patch fixes that, and all 24 Unit 2 checks then passed without page errors. Three identical native/browser Agg diagnostics (text/math, heatmap/colorbar, 3D scatter) produced pixel-identical images. These diagnostics do not establish exhaustive application visual parity or resolve the outstanding numerical/training blockers. Evidence is under `proof/evidence/matplotlib-*`; the content-bound release gate remains authoritative.
