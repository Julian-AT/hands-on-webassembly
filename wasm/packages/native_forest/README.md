# Frozen-reference Gini arithmetic

This extension retains scikit-learn 1.6.1's Gini implementation and overrides its split-score evaluation with the fused arithmetic observed in the frozen ARM64 build. The browser's unfused expression turns two distinct scores into a tie at Digits / seed 42 / tree 42 / node 2. This changes the selected feature and downstream probabilities. No parameters, data, fixture values, tie tolerances, or algorithms are changed.

`forest_criterion.pyx` includes the original BSD-licensed Gini implementation. The vendored `.pxd` declarations are from the unchanged reference installation. The build manifest records all their hashes, compiler configuration, SDK commit, and resulting wheel hash. The extension is loaded only in the browser; the frozen native installation is unchanged.

Recreate the isolated build tools from the repository root (these are build-time dependencies only):

```sh
uv venv --python 3.12 proof/cache/wasm-build-env
uv pip install --prerelease=allow --python proof/cache/wasm-build-env/bin/python pyodide-build==0.29.3 cython==3.0.12 wheel==0.45.1 pip==25.0.1
PATH="$PWD/proof/cache/wasm-build-env/bin:$PATH" proof/cache/wasm-build-env/bin/pyodide xbuildenv install --url https://github.com/pyodide/pyodide/releases/download/0.27.7/xbuildenv-0.27.7.tar.bz2
git clone --depth 1 --branch 3.1.58 https://github.com/emscripten-core/emsdk.git proof/cache/emsdk
proof/cache/emsdk/emsdk install 3.1.58
proof/cache/emsdk/emsdk activate 3.1.58
.venv/bin/python proof/scripts/build_forest.py
```

Use the explicit archive URL because the old build tool's default metadata endpoint has been removed upstream. Put the isolated environment first in `PATH` so its Python and pip are used for host dependencies. The SDK environment applies only to the build subprocess.

Normal static builds copy the already locked wheel; they do not compile it or fetch packages at runtime. `build_forest.py` rejects a rebuilt wheel whose bytes differ from the locked artifact.
