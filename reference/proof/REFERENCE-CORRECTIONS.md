# Shared application corrections

The supplied applications, datasets, archives, environment YAML and frozen native packages are unchanged. The implementation plan authorizes confirmed application fixes shared by the runnable native reference and browser staging. `scripts/reference_corrections.py` applies checked substitutions to copies; `reference-patches/` contains their unified diffs. `evidence/reference-corrections.json` records original, corrected, runnable and patch checksums.

| Correction | Reproduction and resulting behavior |
|---|---|
| Unit 2 one-hot rows | `New York, cat` produces three whitespace tokens but the table supplied two row labels. The table now labels the actual token rows returned by the original helper. Tokenization and unknown-word behavior are preserved. |
| Unit 2 duplicate vocabulary | With `['zebra', 'zebra', 'cat']`, the original helper allocated two columns but indexed `cat` at column two, catching the resulting error as an unknown word. Deduplication now retains first-occurrence order and indexing uses that same vocabulary. Unique-vocabulary results are unchanged. |
| Unit 2 Plotly | Load the pinned Plotly script once in the document head before reactive fragments execute; fragments no longer independently load the CDN script. Handle resize promises and skip hidden plots. Native and browser workflows exercise audio plots and embedding projections. |
| Units 6 and 7 architecture lists | The parsers invoked `.get()` on advertised top-level JSON lists. Lists and `{ "layers": [...] }` now produce equivalent layer specifications; scalar roots produce a validation error. |
| Unit 7 rectangular layers | Convolution and pooling dimension calculations used the input height for both output axes. They now use the actual width for the width calculation. Tests execute both rectangular max-pooling and average-pooling models and assert the final dense input size and prediction shape. |
| Unit 7 training RNG | A controlled interleaving of the original model-summary renderer changes global Torch RNG and final training results at seeds 0, 42 and 123. The complete original training body now runs in a spawned native process or dedicated browser worker. UI previews and resets have independent RNG state. Accepted results restore the model and advance the captured loader generators; canceled results are discarded. |
| Units 5/6 inspection RNG | [24 independent native controls](evidence/native-inspection-rng-audit.json) reproduce interference in both Unit 5 dataset previews and Unit 6 dataset-info/model-summary outputs at seeds 0, 42 and 123 on MNIST and Fashion-MNIST. A synchronous scope restores Python, NumPy and CPU Torch state after inspection. Preview output bytes and the next independent training batch are preserved exactly. Unit 6's sample-index preview already preserves these streams and is unchanged. Data preparation completes before the inspection scope, retaining its original RNG consumption. |
| Unit 5 dataset replacement | The [original native audit with a full one-second deadline](evidence/native-unit5-training-replacement-oct05-original-full-deadline.json) reproduces delayed replacement at seeds 0/42/123. The [corrected reference](evidence/native-unit5-training-replacement-oct05-corrected-full-deadline.json) clears progress in approximately 29 milliseconds. The original body now runs in a disposable process/worker. Inputs, progress, errors and coefficient commits belong to the current task and dataset generation. Earlier one-millisecond audits remain retained as harness faults. |
| Unit 6 Reset | Native Reset delays are reproduced in `native-unit6-reset-audit-acknowledged-inputs-oct04.json`. The integrated correction runs the supplied body in a disposable process/worker and consumes model/history results reactively. Seven presets at three seeds retain independent native controls. Complete application matrices remain required. |

These corrections do not alter training algorithms, seeds, data order, labels, presets or tolerances. Three-seed process tests compare unrounded histories/parameters or coefficients against independent controls while the UI context consumes RNG. Existing numerical fixtures remain unchanged. Full preset/dataset trajectories, evaluation/inspection cancellation and comprehensive acceptance remain separate work.

The browser adds scheduling checkpoints after training and validation batches so a cancellation message can close the worker promptly. This is a browser-only scheduling adaptation; native execution retains the synchronous body. Checkpoint tests compare complete unrounded histories and restored parameters against the same three original control runs. The worker protocol binds progress/results/cancellation to source build, session and task IDs. `reset_model`, replacement training and session termination cancel the owning context and discard obsolete progress. See `scripts/cnn_lifecycle_browser.py` for the strict one-second acknowledgement/two-second disposal checks.

## Reproduce

From the repository root:

```sh
.venv/bin/python proof/scripts/reference_corrections.py
.venv/bin/python -m unittest discover -s proof/tests -p test_reference_corrections.py
.venv/bin/python proof/scripts/build.py
```

Run a corrected native application with its working directory set to that application, for example:

```sh
(cd proof/reference/unit2 && ../../../.venv/bin/shiny run --port 8112 app.py)
```

The generated neural references point at the complete existing dataset cache. Unit 2 serves its pinned Plotly asset locally. Before regeneration, the correction command verifies and preserves the original manifest-bound reference and patch bytes. The original manifests, supplied applications and numerical fixtures stay unchanged. Browser-only adapters remain in `runtime/`, `stage_neural.py` and `patch_runtime.py`.

For a native server launched from the generated reference:

```sh
.venv/bin/python proof/scripts/browser.py --unit 2 --browser chrome \
  --native-url http://127.0.0.1:8112/ --corrected-reference
```

The explicit `--corrected-reference` option binds evidence to that generated native source, shared corrections, frozen environment lock, datasets and relevant test code. It must only be used for a server actually launched from `proof/reference/unitN`. Browser-only runtime/deployment changes do not invalidate that native evidence. Native source, shared-correction and relevant test changes still invalidate it. Tests verify these dependency boundaries.
