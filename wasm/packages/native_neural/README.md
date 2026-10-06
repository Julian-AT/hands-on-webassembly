# Native-compatible linear arithmetic

This extension implements the course's direct float32 784-input, 10-class
classifier: fused bias/linear accumulation, backward gradients, cross entropy,
ARM reduction order, and fused SGD updates. `runtime/neural_compat.py` selects
these operations only for that verified configuration. Other neural model shapes
continue through the existing adapter and require independent certification.

SLEEF sources are pinned to the commit used by the frozen native PyTorch build.
Their checksums and PyTorch commit are in `upstream.json`; redistribution notices
are in `vendor/LICENSE.txt` and `LICENSE-PYTORCH`. No proprietary native library
code is included.

From the repository root, with the existing pinned cross-build environment:

```sh
.venv/bin/python proof/scripts/build_neural.py
.venv/bin/python proof/scripts/build.py
```

The cross compiler is Emscripten 3.1.58, Python 3.12, Pyodide 0.27.7, ABI 2024_0.
The build fixes `SOURCE_DATE_EPOCH=1740787200`, `PYTHONHASHSEED=0`, and disables
implicit floating-point contraction. A changed wheel is rejected against the
existing lock. Two clean builds produced the same wheel SHA-256:
`68366247086354fbc6e7883210a0717a15490c4da333475e3f8fdf93ba309ee4`.

Diagnostic validation before promotion passed all 123 retained comparisons plus
210 additional comparisons over MNIST/Fashion-MNIST, seeds 0/42/123, incomplete
batches, and combined flip/inversion. These comparisons execute the complete
original training bodies and retain the historical native seed-42 arrays.
The deployed application and other neural architectures still need their own
current evidence; these diagnostics do not certify the whole course.
