# Remaining implementation and release plan

The supplied completion plan remains the acceptance contract. This document records the next executable work after the compiled linear classifier, shared reference corrections, isolated Unit 7 training, and startup lifecycle fixes. The authoritative decision is `evidence/gate.json`; an inventory or a successful bounded diagnostic cannot satisfy an exhaustive release suite.

## Rules that remain fixed

- Preserve supplied application files, archives, datasets, `iml_env.yaml`, native dependency lock, historical fixtures, labels, splits, sample order, and discrete decisions. Apply confirmed shared defects only through `scripts/reference_corrections.py` and retained patches.
- Compare floats with `rtol=1e-4`, `atol=1e-5`, and discrete arrays exactly. Never change fixtures to accommodate browser output.
- Bind a report to the actual corrected reference, runtime/assets, executing test, installed browser/version, and hardware. Retain previous failures. A retry does not replace a failed consecutive lifecycle sequence.
- Keep `/unit1/` through `/unit7/`, input/output identifiers, and upload/download formats. Computation stays inside browser workers.
- Build the static navigation shell only after every application-parity requirement passes. Physical 8 GB hardware remains a release prerequisite.

## 1. Close startup and resource lifecycle failures

**Implementation:** `runtime/proxy-lifecycle.js`, `runtime/service-worker-http.js`, startup/client patch generation, `tools/preview.mjs`.

The deployed proxy tracks owners, disposes old controllers, cancels abandoned
requests and reconnects ports. Guarded activation retry passed an isolated affected
30-cycle Edge Unit 6 sequence, then was integrated. The final Chrome/Edge seven-app
matrix across real generations df420/c0596 failed: waiting-worker takeover stalled,
and some applications stayed loading despite an activated controller. The input
artifacts are unchanged. Fresh isolated startup diagnostics pass, but do not explain
the sequence failures. Trace both actual generations, worker executions, native
lifecycle, promise settlement, cache work, response consumption and reconnection
before another correction or acceptance restart. Retain every failed sequence.

**Next work:** retain the completed 30-cycle installed Firefox sequences for all seven assignments; close the failing installed Chrome/Edge sequences, then repeat all affected sequences against the final artifact; investigate any failure before restarting the whole affected sequence. Execute interrupted runtime/dataset downloads, stale HTTP/SW caches, direct links, trailing-slash redirects, navigation, reset, replacement, and Retry. Extend the real Firefox runner to capture worker/cache/request diagnostics and injected faults. Extend download cancellation to every exercise, replacing synchronous uncancellable image fetches where necessary.

**Evidence:** `startup-matrix-*.json`, `startup-actual-firefox-*.json`, `startup-recovery-*.json`, `startup-upgrade-fault-*.json`, and strict worker target-disappearance checks. All original failed reports remain in history.

**Exit:** 30 consecutive startup/reload/update cycles per assignment/browser, real error-and-Retry recovery, cancellation acknowledgement within one second and abandoned worker disposal within two seconds, with no obsolete results accepted.

## 2. Complete Unit 1 behavior around accepted t-SNE arithmetic

**Implementation:** isolated candidates under `experiments/tsne*`; accepted kernels must become a pinned, reproducibly built package before deployment.

The verified arithmetic is now packaged and deployed. All four datasets, both dimensions, scaled/unscaled inputs, seeds 0/42/123, and perplexities 5/30/50 pass complete optimization. The default four UI features also pass at seed 42/default perplexity. These 160 complete runs provide 314,885 passing comparisons; see `TSNE-ARITHMETIC.md` and the deployed matrix reports. Original fixtures, algorithms, seeds, and tolerances remain unchanged.

The browser-only repairs cover native fused arithmetic and power evaluation, finite costs, native Cython division semantics at zero-distance nonleaf queries, and a restricted native-compatible distance backend that retains discrete neighbor/heap decisions. The two newly confirmed regressions are added to the required feature coverage. Rejected candidates remain retained. The standalone double-power difference remains a recorded diagnostic rather than a waived tolerance.

**Next work:** execute remaining feature selections, boundary/pairwise configurations, invalid-input states, complete UI assertions, and installed-browser numerical matrices. When a configuration diverges, trace initialization, probabilities, tree summaries, reductions, cost, gradients and optimizer updates to the first operation. Retain narrow compiled dispatch until independently verified for a new shape. Do not substitute seeds/objectives, fit fixture lookup corrections, or loosen tolerances.

**Exit:** every retained regression and required final optimization passes, together with every comprehensive Unit 1 behavior and appearance requirement.

## 3. Extend neural parity beyond the accepted linear and Small CNN scopes

**Implementation:** `packages/native_neural/`, `packages/native_cnn/`, `runtime/neural_compat.py`, `runtime/cnn_compat.py`, `runtime/browser_torch.py`, `runtime/isolated_training.py`, `runtime/course-training-worker.js`.

The direct float32 784-input/10-class linear classifier now passes the original 123 comparisons and the expanded 210 comparisons: both full image datasets, seeds 0/42/123, incomplete batches, and flip/inversion. Its compatibility dispatch deliberately excludes other shapes and losses until verified.

The small MNIST CNN now has complete five-epoch native fixtures at seeds 0/42/123, checked against unobserved control runs. The pinned NNPACK Fourier kernel, dense reductions, backward gradients, and fused SGD update are packaged and integrated through narrow dispatch. The deployed adapter passes all 55 comparisons for each seed; final parameters, logits/predictions/confusion, split indices, and sample order match bitwise. All 165 current complete-training comparisons pass against the final artifact. See `CNN-ARITHMETIC.md` for build locks and scope.

**Next work:** retain the current 165 CNN and 210 expanded linear comparisons, all 17 bounded passing suites, and cancellation/worker-disposal regressions. Extend native and browser fixtures to the other presets, compatible full datasets, and supported input settings. Trace each first differing operation before extending compiled dispatch. Disabling native NNPACK is not authorized. Carry exact dropout, augmentation, loader/global generator states, preview consumption, epoch reseeding, early stopping, and best-model restoration. Complete application UI training/evaluation/inspection evidence in addition to function-body comparisons.

Unit 7 model-summary interference with training's shared global RNG is confirmed for all three seeds and corrected through independent contexts in both references. Units 5/6 inspection interference is corrected across 24 native controls. Unit 6 disposable training is integrated and retains seven-preset/three-seed native controls. Unit 5 disposable training is also integrated: full native coefficient controls cover both datasets, seeds 0/42/123, flip/inversion, incomplete batches and repeated runs; installed Chrome/Edge cover bounded loading and ownership workflows. These scopes do not establish comprehensive Unit 5/6 acceptance. Snapshot timing, complete loader/global RNG trajectories, evaluation/inspection ownership and the final artifact's complete recovery matrix still require evidence. See [the October 5 implementation record](IMPLEMENTATION-OCT05.md).

**Exit:** all seven Unit 6 presets and three Unit 7 presets with compatible full datasets and default durations; every supported layer and rectangular custom architecture; repeated runs; validation on/off; augmentation variants; incomplete batches; early stopping; restored-model inspection; seeds 0/42/123 and existing seeds. The previous rounded full-CNN accuracy match is not numerical acceptance.

## 4. Execute exhaustive exercise and appearance cases

**Implementation:** `scripts/case_inventory.py`, `scripts/coverage_contract.py`, assignment-specific browser executors, native/browser captures.

The additional inventory has 1,086 stable behavior requirements. Nested tab identities include their ancestry, and IDs do not depend on source lines. Dynamic choices require runtime expansion. The gate requires observed and expected behavior assertions; element presence cannot pass. These cases are requirements, not 1,086 executed tests.

**Next work:** implement executors for the inventory and preserve every legacy requirement. For each assignment, execute every tab, dynamic control, dataset, preset, upload, download, output change, validation failure, and recoverable error. Assert numerical/content/workflow effects. Include audio playback, tokenizer/vocabulary arithmetic, clustering diagnostics, segmentation, classifier tuning/reports/curves, CSV/header variants, coefficient and architecture round trips, and image/network inspection. Use boundary and pairwise cases to supplement the required full runs.

The first same-browser Unit 1 diagnostic records unresolved heading/tab and correlation-figure pixel differences (`unit1-same-browser-appearance.json`). Trace native font formats, rendering parameters, glyph/affine arithmetic and plot dimensions before proposing a fix; an observed font-format difference alone does not establish the cause. Capture every corrected native and browser view in the same installed browser, viewport, display scale, font assets, and application state. Investigate every unexplained layout/control/table/figure difference. Retain Random Forest tree equality, Matplotlib pixel equality, and full embedding regressions.

**Exit:** no missing dynamic expansion or inventoried behavior/visual case; downloads retain their filenames and exact formats; all seven assignment-specific acceptance matrices pass.

## 5. Finish full-data, installed-browser, and hardware acceptance

Load, train, evaluate, and inspect MNIST, Fashion-MNIST, CIFAR-10, SVHN, and USPS completely. Validate lazy assets, compact image storage, batch transformations, interrupted downloads, replacement/reset/navigation disposal, and recoverable oversized-job/resource-exhaustion errors. Do not infer training success from loading complete datasets.

Run functional, visual, numerical, and lifecycle checks in installed Chrome, Edge, Firefox, and Safari. Keep Playwright Firefox/WebKit separately labeled. Edge 154.0.4258.53 is now installed from its verified official package; its bounded update sequence records a failure. Safari's remote automation is currently disabled. These are outstanding environment prerequisites, not waived requirements.

Benchmark the development machine and a physical 8 GB laptop separately, recording hardware, transferred bytes, total browser memory with its measurement definition, cold/warm startup, training-only duration, workflow duration, and responsiveness. Run hardware benchmarks without concurrent verification jobs. No suitable physical 8 GB machine is currently available.

**Exit:** every installed-browser matrix and physical 8 GB requirement passes. Missing hardware remains an explicit release blocker.

## 6. Build and certify delivery

Only after the application-parity gate authorizes it, add the Next.js static export shell with one full-size assignment iframe and `?unit=1..7`. Preserve standalone URLs; test Back/Forward, reload, selection, and worker disposal. Fetch current official documentation through Context7 before library-specific implementation.

Package runtime modules, datasets, models, fonts, styles, and scripts as local versioned assets. Content-addressed assets receive immutable caching; entry pages and SW/bootstrap files revalidate. Verify MIME types, base paths, byte ranges, static hosting and HTTPS. Produce identical deployment contents from two clean builds, and bind final evidence to those contents.

Deliver corrected runnable Python references, checked patch history, the static artifact, pinned reproducible build instructions, preview/deployment instructions, and one current report linking every required case and its evidence. All 133 original suites and 1,824 behavior IDs remain retained. Seven additional release-only suites explicitly require Next.js navigation, reproducible shell export, Vercel HTTPS delivery, benchmarks and release materials. The current gate therefore requires 140 suites. A final release is complete only when no requirement is missing, stale, skipped, failed, or unavailable.
