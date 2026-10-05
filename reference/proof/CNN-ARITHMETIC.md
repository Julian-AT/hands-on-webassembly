# Compiled small-CNN arithmetic

The browser runtime now includes `course-native-cnn` 1.0.0. Its dispatch is restricted to the original Unit 7 small MNIST preset's verified float32 convolution and dense shapes, with batches of 2, 16, or 32. Unmarked dense networks and other convolution settings use the existing adapter. This implementation does not establish parity for other presets, architectures, datasets, or complete UI workflows.

## Reference and arithmetic

The complete original five-epoch training body is observed without creating additional loader iterators or consuming additional random values. Separate unobserved native runs reproduce the observed final parameters, history, and evaluation bitwise. Retained fixtures include all sample order, split indices, losses, initial/final parameters, first forward inputs/outputs, filter history, and complete test labels/logits/predictions/confusion. Historical expectations are unchanged.

The native reference selects slow convolution for the two-image preview and NNPACK convolution for training batches. Sequential convolution alone cannot reproduce the training result. The compiled package reproduces the pinned NNPACK Fourier transforms, fused complex arithmetic, shape-dependent dense reduction, backward gradients, bias reductions, and fused SGD updates. It requires the 44 pinned LLVM contraction references to become explicit FMA operations; arithmetic elsewhere disables implicit contraction. This preserves native fused rounding without requiring relaxed WebAssembly SIMD.

Pinned revisions are Torch `5c4886908584029761b579af026dcfb627c84070`, NNPACK `c07e3a0400713d546e0dea2d5466dd22ea389c73`, and PSimd `072586a71b55b7f8c584153d223e95687148a900`. Source hashes and licenses are retained in [the source lock](packages/native_cnn/upstream.json). The wheel SHA-256 is `bc528125c7385bdcd469433abd7c4384aca5487cb1782c5b865deb0396fd5eab`; two builds produced identical bytes. The package targets Python 3.12, Pyodide ABI 2024_0, and Emscripten 3.1.58, using the same separate pinned build environment as the existing [linear package](packages/native_neural/README.md). The frozen native environment is not modified.

## Build and verification

```sh
.venv/bin/python proof/scripts/build_cnn.py
.venv/bin/python proof/scripts/build.py
.venv/bin/python proof/scripts/verify_build.py
.venv/bin/python proof/experiments/cnn-full-data/browser.py --seed 0 --mode deployed
.venv/bin/python proof/experiments/cnn-full-data/browser.py --seed 42 --mode deployed
.venv/bin/python proof/experiments/cnn-full-data/browser.py --seed 123 --mode deployed
```

The builder refuses changed upstream hashes, changed contraction counts, or a wheel that differs from its existing lock. The deployed registry verifies wheel bytes; the training bundle identity includes the compatibility adapter and package lock. The diagnostic runner executes the retained original body using the deployed modules without candidate overrides, and retains content-addressed browser output artifacts. Run it against an unchanged site; changing its actual dependencies marks a result stale.

Before promotion, packaged candidates passed all 55 complete-training comparisons at each of seeds 0/42/123, including bitwise-equal final parameters, sample order, split indices, logits, predictions, and confusion. All 84 retained first-step arrays match bitwise. Production evidence is recorded separately as `evidence/cnn-full-data/browser-seed{seed}-deployed.json`. Inspect those actual reports for their current status, dependency fingerprints, installed browser/version, and hardware; these diagnostics cannot satisfy the exhaustive Unit 7 release suite.

Floating comparisons remain `rtol=1e-4`, `atol=1e-5`; discrete values require exact equality. Other batch shapes, presets, full image datasets, custom/rectangular architectures, augmentation, dropout, early stopping, repeated training, restored-model inspection, and preview/loader interaction require additional evidence before broader dispatch or certification.
