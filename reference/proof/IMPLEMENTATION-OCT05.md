# October 5 implementation and verification

Application parity and final release remain incomplete. The Next.js shell remains
deferred by the accepted application-parity gate. No GitHub release or Vercel
deployment has been published.

## Implemented

- Unit 5 training now runs in a disposable native process/browser worker, using
  the complete supplied calculation body. Inputs are snapshotted at Train;
  reactive consumers commit coefficients and progress. Replacement, session end
  and subsequent training cancel the old owner. Stale progress, errors and results
  cannot update the current task.
- The shared worker accepts Unit 5 and checks assignment, operation, build,
  source, session, task and dataset/model generation identities. Browser executor
  cancellation also settles its waiting future. Unit 6/7 model serialization is
  retained; Unit 5 returns its original coefficient tuple format.
- Independent native controls verify constructor reseeding/RNG neutrality and
  exact coefficients for both complete image datasets, seeds 0/42/123,
  flip/inversion, incomplete batches and repeated runs. Inspection activity in the
  invoking context does not change training coefficients.
- The gate retains every original suite and case, adds four Unit 5 defect
  obligations, and adds seven delivery suites. Lifecycle acceptance now requires
  real asserted build-generation upgrades, continuous sequences, owned origins,
  uninterrupted native host observations and no diagnostic instrumentation.
  Numerical coverage requires observed/expected assertions.
- Unit 5 loading verification now checks origin ownership and host conditions;
  its assertions compare actual values. Clean-build logs are archived before
  replacement. Stale isolation/arithmetic documentation is reconciled.

## Evidence and retained limitations

| Check | Current evidence |
|---|---|
| 2,541 original baseline bindings | [Baseline verification](evidence/continuation-baseline-verification.json) |
| Native replacement defect, full one-second deadline | [Original reference](evidence/native-unit5-training-replacement-oct05-original-full-deadline.json) |
| Corrected replacement, approximately 29 ms at all three seeds | [Corrected reference](evidence/native-unit5-training-replacement-oct05-corrected-full-deadline.json) |
| 12 bounded native combinations plus process cancellation | [Bounded controls](evidence/unit5-training-context-bounded-thread-oct05.json) |
| 12 full-data native combinations plus process cancellation | [Full-data controls](evidence/unit5-training-context-full-oct05.json) |
| Detached Chrome loading/training workflows | [20 bounded workflows](evidence/unit5-loading-ui-chrome-isolation-candidate-oct05.json) |
| Integrated Edge loading/training workflows | [20 bounded workflows](evidence/unit5-loading-ui-edge-integrated-isolation-oct05.json) |
| Final artifact Edge ownership/error recovery | [10 cases](evidence/unit5-owned-training-ui-edge-integrated-final-oct05.json) |
| Final artifact Chrome ownership/error recovery | [10 cases](evidence/unit5-owned-training-ui-chrome-integrated-final-oct05.json) |
| Unit 6 shared-worker Reset/retry regression | [Six cases](evidence/unit6-owned-training-ui-chrome-shared-unit5-final-oct05.json) |
| Two byte-identical clean static exports, 209 files | [Reproducibility](evidence/reproducible-build.json), [file manifest](evidence/static-artifact-manifest.json) |
| Original 17 bounded browser suites | [Checkpoint](evidence/verification-checkpoint.json), [dependency audit](evidence/bounded-evidence-audit.json) |
| Every release requirement | [Current gate](evidence/gate.json), [checklist](CONTINUATION-CHECKLIST.md) |

The final local run passes **104 tests** ([log-bound report](evidence/unit-tests-oct05.json)).
All **17 bounded browser suites** pass their recursive dependency audit. Both
final ownership reports pass ten cases, and the Unit 6 regression passes six.
The refreshed native Unit 4/5 reports pass 26 and 14 bounded workflows. The
current release gate is **BLOCKED: 11/140 suites passing**; the other required
reports are missing. Next.js implementation is not yet authorized by application
parity. Bounded successes do not satisfy those missing comprehensive reports.

The first native audit used a one-millisecond text assertion after seeing an
initially 0% progress bar. Its before/after reports remain retained as harness
faults. The corrected full-deadline audit independently reproduces the original
defect and verifies the corrected reference. One Unit 6 browser report is retained
as stale because a concurrent owned-preview marker changed the observed file map;
its complete functional sequence was rerun against an unchanged served tree.

The detached candidate and the pre-change deployment remain under
`cache/unit5-training-owned-oct05/` and `cache/site-before-unit5-isolation-oct05/`.
Final ownership verification serves an identical retained copy at
`cache/static-unit5-final-oct05/`, keeping diagnostic origin markers outside the
integrated deployment. Original source files, datasets, locks, native fixtures
and earlier reports remain preserved. Discrete comparisons stay exact; floating
tolerances remain `rtol=1e-4`, `atol=1e-5`.

These are bounded implementation checks. They do not establish complete browser
training trajectories, comprehensive Unit 5 acceptance, all native/browser
appearance views, lifecycle reliability, or physical-hardware performance.

## Outstanding work

Continue in the original order: shared startup/lifecycle and long-calculation
ownership, Units 1–7 comprehensive matrices, application parity, then the Next.js
shell and final delivery. Abortable embeddings, historical synchronous image
probes, evaluation/inspection and other long calculations remain open. Full
parent-state discovery and executable behavior coverage, additional neural
presets/datasets/settings, exact pixels and all 30-cycle browser sequences remain
required. Every benchmark combination must run independently.

The original 133 suites and all 1,824 behavior IDs remain obligations. The added
seven release suites bring the total to 140 and explicitly require the shell,
reproducible Next.js export, verified archive/GitHub release, Vercel HTTPS delivery,
zero runtime functions, complete benchmarks and release/rollback materials.

Safari automation authorization, a physical 8 GB integrated-graphics laptop, and
GitHub/Vercel repository/project access remain pending. These prerequisites do
not waive any missing software evidence.

## Reproduction

From the workspace root:

```sh
.venv/bin/python -m unittest discover -s proof/tests
.venv/bin/python proof/scripts/verify_continuation_baseline.py
.venv/bin/python proof/scripts/verify_unit5_training_context.py \
  --candidate proof/reference/unit5 --label fresh-native-controls --full
.venv/bin/python proof/scripts/reproduce_static.py
node tools/preview.mjs proof/site
```

After the preview is running, refresh bounded evidence without concurrent builds:

```sh
.venv/bin/python proof/scripts/verify_checkpoint.py --jobs 3
.venv/bin/python proof/scripts/verify_bounded_evidence.py
.venv/bin/python proof/scripts/gate.py
.venv/bin/python proof/scripts/continuation_checklist.py
```

The gate returns a blocked exit code until every applicable requirement passes.
Use fresh labels for audits and retained candidates. Execute marker-writing
ownership checks on their own served copy or sequentially; do not mutate a tree
while another report is binding its contents.
