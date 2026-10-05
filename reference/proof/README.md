# Build and verify the browser replicas

The [continuation checklist](CONTINUATION-CHECKLIST.md) links every retained release suite,
legacy behavior ID and added runtime-discovered choice. The original baseline manifest
remains retained; verify its corrected bindings with
`.venv/bin/python proof/scripts/verify_continuation_baseline.py`.

The HTML correction now transforms bytes per response, handles closing tags and UTF-8
across chunk boundaries, flushes the final buffered bytes, and removes invalid length,
range and representation-checksum headers from HTML. Non-HTML response metadata stays
intact. Installed Chrome/Edge response checks cover all seven assignments. These checks
do not establish upgrade reliability; see [continuation evidence](CONTINUATION-STATUS.md).

Every staged build reserves 15 GiB free space plus a conservative estimate of its next footprint. Immutable candidate trees remain retained, and writable hardlinks are detached before a build.

Run commands from the repository root. The supplied apps remain unchanged. Staging copies and explicit runtime patches live under `proof/`; no application backend is part of the deployment.

Confirmed native defects now have [shared, documented reference corrections](REFERENCE-CORRECTIONS.md). Generate the corrected runnable applications with `.venv/bin/python proof/scripts/reference_corrections.py`; browser staging applies the same corrections. Historical native fixtures remain preserved. These fixes do not certify full neural-training parity.

## Reference environment

The user selected the frozen working local environment as the numerical target. `requirements-native.lock` records that environment. `iml_env.yaml` is course provenance; it does not pin Shiny and omits PyTorch/torchvision. [REFERENCE.md](REFERENCE.md) lists differences without upgrading the reference.

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python proof/requirements-native.lock
.venv/bin/python -m playwright install chromium firefox webkit
.venv/bin/shinylive assets download
```

## Asset preparation and build

```sh
.venv/bin/python proof/scripts/extract_assignment5.py
.venv/bin/python proof/scripts/reference_corrections.py
.venv/bin/python proof/scripts/reference_environment.py
.venv/bin/python proof/scripts/inventory.py
.venv/bin/python proof/scripts/prepare.py
.venv/bin/python proof/scripts/prepare_embeddings.py
.venv/bin/python proof/scripts/download_images.py
.venv/bin/python proof/scripts/prepare_images.py
.venv/bin/python proof/scripts/prepare_runtime_packages.py
# First-time compiled package preparation: see packages/native_forest/README.md
.venv/bin/python proof/scripts/build_forest.py
# Pinned neural extension/toolchain: packages/native_neural/README.md
.venv/bin/python proof/scripts/build_neural.py
# Restricted small-CNN extension: CNN-ARITHMETIC.md
.venv/bin/python proof/scripts/build_cnn.py
# Native-compatible Barnes–Hut arithmetic and restricted distance backend.
.venv/bin/python proof/scripts/build_tsne.py
.venv/bin/python proof/scripts/build.py
.venv/bin/python proof/scripts/neural.py
.venv/bin/python proof/scripts/training.py
.venv/bin/python proof/scripts/tabular.py
.venv/bin/python proof/scripts/supervised.py
.venv/bin/python proof/scripts/unit5.py
.venv/bin/python proof/scripts/unit5.py --full
.venv/bin/python proof/scripts/check_assignment5_materials.py
.venv/bin/python proof/scripts/check_assets.py
```

Large image archives may take a long time. The range downloader keeps completed chunks in `proof/cache/`; rerunning resumes them. It verifies the original torchvision MD5 before promoting an archive. `prepare_images.py` uses the frozen native torchvision parsers, checks complete counts/order/labels, stores uint8 images, and writes SHA-256 entries to `assets/v1/images/manifest.json`. Missing datasets raise explicit loading errors; they are never replaced with subsets. Check `evidence/image-packaging.json` and the manifest for actual completion.

Build stages all seven apps, removes each previous stage and the shared generated runtime, and exports to the original `/unitN/` addresses. UI ASTs are checked against the corrected reference, allowing equivalent static resource URLs. Shared application corrections and browser-only substitutions are separately checked. Builds overwrite generated files under `build/` and `site/`; do not rebuild while browser verification is running.

The runtime remains Shinylive Python 0.8.12 / assets 0.10.15 / Pyodide 0.27.7. `patch_runtime.py` checks the expected upstream code before patching service-worker readiness, reload control, and replacement proxy channels. Plotly and Ace are physical local assets so synchronous script requests and editor workers do not depend on virtual ASGI paths.

The loading/recovery panel lives outside the React mount point and waits for Shiny session initialization. Worker startup failures propagate to its Retry button, and page exit/startup failure terminates the Python worker. Ordinary static WASM downloads bypass service-worker interception; the signature guard requests four uncached bytes. Static hosts must revalidate assets that do not have immutable content-addressed URLs, as the preview server does.

## Browser runtime adapters

- `embedding_runtime.py` implements only the tokenizer/document-vector interface used by Assignment 2. It uses the actual model's regexes, exceptions, retokenization patterns, all 514,157 vocabulary keys, and all 20,000 × 300 vectors. Licenses ship with the model assets. It does not claim to implement spaCy's parser/tagger/NER APIs.
- `torch_rng.py` supplies the frozen CPU MT19937 uniform/permutation/Bernoulli behavior, independent generators, parameter initialization draws, sampler advancement, loader base-seed consumption, and dropout. Unsupported normal/replacement sampling is explicitly rejected. Those operations are not used by the supplied model builders.
- `browser_torch.py` pins Borch CPU, the process/thread adapter, and the required CPU/determinism API surface. Original builders, losses, optimizers, and training calculations remain in the staged apps.
- `image_data.py` verifies asset hashes, retains compact uint8 images, and transforms individual images. Assignment 6 retains its original FashionMNIST normalization constants. Assignment 7 retains its different normalization and weighted loss averaging.
- Units 5, 6 and 7 train in disposable workers with independent RNG state; the corrected native references use spawned processes. Calculation bodies remain supplied-source copies. Invocations snapshot inputs, progress is ownership checked, and reactive consumers commit results. Unit 5 additionally tags dataset/model generations and rejects abandoned errors. Exhaustive evaluation, inspection, resource recovery and release matrices remain acceptance requirements.

## Verification

Serve only static files:

```sh
node tools/preview.mjs
```

Then, in another terminal:

```sh
.venv/bin/python -m unittest discover -s proof/tests
.venv/bin/python proof/scripts/browser.py --unit 1 --browser chrome
.venv/bin/python proof/scripts/browser.py --unit 2 --browser chrome
.venv/bin/python proof/scripts/browser.py --unit 3 --browser chrome
.venv/bin/python proof/scripts/browser.py --unit 4 --browser chrome
.venv/bin/python proof/scripts/browser.py --unit 5 --browser chrome
.venv/bin/python proof/scripts/browser.py --unit unit5probe --browser chrome
.venv/bin/python proof/scripts/browser.py --unit unit5full --browser chrome
.venv/bin/python proof/scripts/browser.py --unit unit5life --browser chrome
.venv/bin/python proof/scripts/browser.py --unit 6 --browser chrome
.venv/bin/python proof/scripts/browser.py --unit 7 --browser chrome
.venv/bin/python proof/scripts/browser.py --unit embedding --browser chrome
.venv/bin/python proof/scripts/browser.py --unit neural --browser chrome
.venv/bin/python proof/scripts/browser.py --unit training --browser chrome
.venv/bin/python proof/scripts/browser.py --unit tabular --browser chrome
.venv/bin/python proof/scripts/browser.py --unit supervised --browser chrome
.venv/bin/python proof/scripts/full_mnist.py
.venv/bin/python proof/scripts/browser_startup.py --browser chrome
.venv/bin/python proof/scripts/browser_startup.py --browser firefox
.venv/bin/python proof/scripts/browser_startup.py --browser webkit
.venv/bin/python proof/scripts/browser_startup.py --browser edge
.venv/bin/python proof/scripts/browser_recovery.py --browser chrome
.venv/bin/python proof/scripts/browser_recovery.py --browser firefox
.venv/bin/python proof/scripts/native_screens.py
.venv/bin/python proof/scripts/native_defects.py
.venv/bin/python proof/scripts/gate.py
```

For native UI comparisons, launch the corresponding generated app under `proof/reference/unitN/`, with that directory as its working directory, then pass its address with `--native-url` and `--corrected-reference`. See [the exact commands and dependency policy](REFERENCE-CORRECTIONS.md). Do not launch the supplied app when claiming corrected-reference evidence. Historical original-source comparisons remain separately retained.

The supervised numerical harness executes original calculation bodies with their decorators removed and test input/value containers. Neural training fixtures execute the original training bodies, preserving different loss averaging, preview passes, reseeding and restoration. Unit 1 fixtures execute original calculation bodies and observe preprocessing, t-SNE initialization, probabilities and the first gradient. Instrumentation is distinct from reactive/UI verification.

Integer/discrete supervised comparisons are exact; floating tolerances remain `rtol=1e-4`, `atol=1e-5`. Failed comparisons remain failed. Diagnostic training uses deliberately small batches with incomplete final batches; full-data training is a separate test.

## Release gate

`coverage_contract.py` derives requirements from every inventoried input/output/tab/download plus explicit exercise, numerical, dataset, lifecycle, browser and hardware requirements. `gate.py` reads their reports; it can pass only when every required check is executed, passing, and current. It rejects missing cases, empty results, skips, failed checks, stale source/assets/runtime/test fingerprints, mislabeled browsers, and nonrepresentative hardware. Unit tests verify both rejection and the ability of complete current evidence to pass.

Browser reports bind the deployed artifact, source, assets, adapters, build inputs and the executing harness. Unrelated validators and other workflow harnesses do not invalidate that execution; the gate checks required coverage separately. Corrected-native reports have their own dependency fingerprints, so unrelated browser builds do not invalidate them. Reports may retain historically passing checks with `status: stale`; this never authorizes release. Replaced JSON evidence is archived under `evidence/history/`. Re-run affected suites after the final build.

Chrome/Edge checks use installed applications. Playwright Firefox and WebKit are labeled as such; WebKit is not actual Safari. The authorized certification device is this 24 GB M4 Pro MacBook (Mac16,7); [the hardware contract](hardware-contract.json) preserves legacy 8 GB IDs and their supersession. Reports bind native device observations and the active contract checksum. Concurrent functional checks are not performance benchmarks.

`browser_startup.py --cycles N --label NAME` retains request/controller/cache traces for each startup, reload and worker update. Readiness now inspects the current same-origin iframe document, connected Shiny client and bound inputs directly. Earlier Firefox protocol-frame timeouts remain retained, including screenshots where the application was already rendered. Three cycles are diagnostic coverage; the required 30-cycle, actual-browser matrix remains a separate gate.

The accepted t-SNE arithmetic is packaged and deployed. Complete scaled/unscaled matrices and default selected-feature runs pass within the scope recorded in [TSNE-ARITHMETIC.md](TSNE-ARITHMETIC.md). The retained standalone power-function diagnostic does not waive any required trajectory or tolerance. Remaining configurations and comprehensive interface/appearance acceptance are separate obligations.

The proxy lifecycle also cancels HTTP requests when their owning document closes, closes abandoned message ports, and releases pending responses before a service-worker upgrade. This addresses a reproduced upgrade stall: an unfinished `respondWith` promise holds the previous worker active even when its replacement calls `skipWaiting`. The controlled test deliberately strands one proxy request, observes the waiting worker, disposes the proxy, and verifies takeover within two seconds and a connected app after reload:

```sh
.venv/bin/python proof/scripts/browser_upgrade_fault.py --browser chrome
.venv/bin/python proof/scripts/browser_upgrade_fault.py --browser firefox
```

The native CNN [RNG interleaving diagnostic](experiments/cnn_rng_interference.py) runs complete original training bodies with seeds 0, 42 and 123. It retains unrounded output arrays separately; it does not regenerate expected fixtures. An interleaved original model-summary renderer changes global Torch RNG and final training results. The shared Unit 7 correction isolates training. The compiled Small CNN arithmetic separately passes its complete three-seed MNIST scope; other presets, datasets and settings remain uncertified. See [CNN-ARITHMETIC.md](CNN-ARITHMETIC.md).

`case_inventory.py` produces stable assignment/identifier/scenario requirements independent of source lines. Its 1,086 current behavior cases supplement every existing release requirement. Nested tab IDs include their ancestry. Dynamic controls still require runtime expansion, and executor/evidence coverage remains incomplete. Inventory presence is never a test pass. The release gate also requires 30 consecutive complete startup/reload/update cycles per assignment/browser. See [REMAINING-PLAN.md](REMAINING-PLAN.md) for the remaining implementation and acceptance sequence.

Genuine generation checks now share `browser_generation_sequence.py` across installed Chrome, Edge, Firefox and Safari. The preview resolves an atomically published whole-tree pointer per request; added, removed and changed paths switch together, and in-flight response streams retain their original tree. Runs bind two real generations, origin ownership, hardware observations and native host conditions. Use fresh labels to preserve failed attempts.

Actual Firefox/Safari checks use a separate environment, preserving the frozen native dependencies:

```sh
.venv/bin/python -m venv proof/cache/browser-verification-env
proof/cache/browser-verification-env/bin/python -m pip install -r proof/requirements-browser.lock
proof/cache/browser-verification-env/bin/python proof/scripts/webdriver_startup.py --browser firefox --cycles 30
proof/cache/browser-verification-env/bin/python proof/scripts/webdriver_startup.py --browser safari --cycles 30
```

The Firefox runner explicitly selects `/Applications/Firefox.app`; it does not use Playwright's bundled Firefox. Safari requires its Developer setting “Allow remote automation.” Unavailable automation is reported as unavailable, never passed.

### Compiled scientific runtime

The browser uses the reference NumPy 1.26.4 wheel from Pyodide 0.26.4, which shares Python 3.12, Emscripten 3.1.58 and ABI `2024_0` with this runtime. The upstream registry, recipe, wheel URL, and SHA-256 are locked. Plotly 7.1.0 and Narwhals 2.26.0 are local pure Python wheels.

The Gini compatibility extension reproduces the frozen ARM64 compiler's fused split-score expression. Its source, BSD license, declarations, build recipe and exact toolchain are under `packages/native_forest/`. Two builds produced the same wheel bytes. It changes the arithmetic evaluation order to match the reference; dataset values, classifier parameters, splits, and tolerance are unchanged.

The browser now uses native-matching Matplotlib 3.10.8 and FreeType 2.6.1. A checked adaptation of the upstream Emscripten FreeType callback fix resolves the indirect-call crash without changing the rasterization algorithms. Two clean builds produced identical wheel bytes. Text/math, heatmap, and 3D scatter diagnostics matched the native reference pixel for pixel; exhaustive application visual parity remains required. See [the build and validation details](packages/matplotlib/README.md). `build_matplotlib.py` produces isolated candidates; a normal clean export copies the accepted checksum-locked wheel without recompiling it.

### Assignment 5 and full datasets

`assignment5-materials.lock.json` binds the original archive, CSV, compressed/uncompressed Fashion-MNIST assets and extracted Python files. Re-extraction refuses any changed supplied or extracted file. `check_assignment5_materials.py` verifies that both supplied Fashion-MNIST splits equal the complete browser arrays, in order, pixel for pixel and label for label.

All ten image train/test assets are packaged, including complete CIFAR-10. Packaging does not certify training parity. Unit 5 keeps its original post-ToTensor `255 - x` inversion and its Fashion-MNIST use of MNIST normalization constants. CSV parsing, split sampling, coefficients and download names are unchanged. Its complete supplied training body runs in a disposable worker, using input snapshots and a reactive result consumer. Dataset replacement and session termination cancel the owner. [October 5 implementation evidence](IMPLEMENTATION-OCT05.md) records independent native coefficients, repeated-run reseeding and installed-browser ownership checks.

Unit 5 fixtures execute original application functions and complete original training/evaluation bodies. They include unrounded batch losses, first-batch inputs/initial parameters/logits/gradients/updates, final parameters, sample order, confusion matrices and misclassified examples. Floating comparisons use `rtol=1e-4`, `atol=1e-5`; integers are exact. The compiled [native linear extension](packages/native_neural/README.md) closes the Fashion-MNIST discrepancy: all 123 retained comparisons and all 210 expanded full-data comparisons pass through the production adapter. The extra matrix covers seeds 0/42/123, incomplete batches and combined flip/inversion without changing original fixtures. Other architectures remain separately uncertified.

Run the retained expanded matrix against the production adapter with `.venv/bin/python proof/scripts/neural_matrix_browser.py`. Its isolated site changes the diagnostic case inventory only; arithmetic modules and application calculation bodies come from the deployed artifact.

### Deployment and remaining acceptance

For any static host, publish all of `proof/site/` over HTTPS. Serve `.js` as JavaScript, `.wasm` as `application/wasm`, JSON as JSON, and leave relative paths intact. Avoid SPA fallback rewrites and avoid immutable caching for `shinylive-sw.js`, entry pages, and `app.json`; updates must be revalidated. Runtime computation stays in browser workers. `tools/preview.mjs` provides a dependency-free Node static server with GET/HEAD and byte ranges.

The gate separates application parity from release certification. Application functional/appearance/numerical and lifecycle evidence controls shell readiness. Additional installed-browser and authorized-device evidence controls full release certification. Neither gate currently passes.

Run `proof/scripts/benchmark.py` with other tests stopped for a development-machine Unit 5 benchmark. It records actual CPU/RAM, transfer bytes, full-data workflow durations, event-loop delay, and aggregate RSS for the entire launched browser process tree. RSS includes shared pages and is not a measurement of private physical memory. This scope does not certify every assignment or the complete benchmark matrix.

### Deferred embedding assets

Assignment 2 bundles only the pinned embedding manifest. The complete tokenizer, vocabulary mapping, and vector matrix are served from `assets/v1/embedding-runtime/` and loaded when the text exercise requests them. Image and audio exercise checks assert that no embedding assets were fetched. All 514,157 keys and 20,000 × 300 vectors are retained. Every asset is checksum-verified before use, and incomplete vector loads remain retryable. The startup `app.json` shrank by 37,682,292 bytes. The 2,106 tokenizer/vector comparisons remain unchanged.

## Full CNN arithmetic

New native complete five-epoch MNIST fixtures retain histories, parameters, split indices, sample order and all test predictions without changing historical expected results. Native observation is checked against unobserved control runs. Native Torch uses NNPACK for training batches and slow convolution for the two-image preview.

The compiled pinned NNPACK Fourier kernel and restricted adapter now ship in the browser runtime. Packaged candidates pass all 55 comparisons through complete five-epoch training at seeds 0/42/123; all 84 retained first-step forward, loss, backward, and update arrays match bitwise. Its checked LLVM contraction sites require fused rounding on WebAssembly. See [the implementation and verification scope](CNN-ARITHMETIC.md). Other presets/datasets and exhaustive UI/release acceptance remain separate requirements.

```sh
.venv/bin/python proof/scripts/build_cnn.py
.venv/bin/python proof/experiments/cnn-full-data/browser.py --seed 0 --mode deployed
.venv/bin/python proof/experiments/cnn-full-data/browser.py --seed 42 --mode deployed
.venv/bin/python proof/experiments/cnn-full-data/browser.py --seed 123 --mode deployed
```

The first-step native capture refuses to overwrite retained fixtures. Candidate operation output archives and report history remain available under `evidence/cnn-full-data/`. Pinned upstream revisions, sources, LLVM contraction sites, compiler, kernel bytes, and native fixtures are recorded with their hashes.

The accepted [t-SNE package](TSNE-ARITHMETIC.md) has reproducible source/toolchain locks. Reproduce both complete production matrices with:

```sh
.venv/bin/python proof/experiments/tsne/matrix.py --mode deployed
.venv/bin/python proof/experiments/tsne/matrix.py --mode deployed --standardize off
.venv/bin/python proof/experiments/tsne/matrix_selected_features.py --mode deployed --seeds 42 --perplexities 30
.venv/bin/python proof/experiments/tsne/matrix_selected_features.py --mode deployed --standardize off --seeds 42 --perplexities 30
.venv/bin/python proof/scripts/reproduce_static.py
```

`reproduce_static.py` runs two clean assignment/runtime staging builds, compares every deployed file, verifies checked local assets, and retains the final artifact manifest. Full delivery/hosting and lazy-resource acceptance remain separately required.
