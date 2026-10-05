# Shared bridge and evidence validation continuation

The requested release is **unfinished**. This continuation implements the
launcher bridge required before the Next.js shell and strengthens the evidence
gate. It does not complete application parity, authorize the shell, or publish
GitHub/Vercel artifacts. The integrated `site/` remains the previous deployment;
the bridge has been checked on separate retained candidates.

## Changes

- [`runtime/launcher-session.mjs`](runtime/launcher-session.mjs) serializes
  assignment changes. It keeps one same-origin iframe, retains the latest rapid
  selection, waits for a matching disposal acknowledgement, and enforces a
  one-second fallback. Only matching origin, frame window, protocol, assignment,
  build, source and session messages can change state. Reload creates a fresh
  session identity. Error/timeout remains visible until Retry creates a new owner.
- [`runtime/session-bridge.js`](runtime/session-bridge.js) pins the assignment,
  build and application-bundle hash before startup. Its parent handshake binds a
  session once; it buffers readiness/errors if startup precedes the handshake.
  Disposal closes registered resources, blocks obsolete readiness/errors, and
  reports whether every disposer succeeded. Standalone assignment pages work
  without a launcher handshake.
- [`runtime/startup-status.js`](runtime/startup-status.js) now requires the current
  Shiny instance's initialization promise, an open socket, a connected application
  and initialized inputs. An old frame/instance's initialization cannot certify
  a replacement. Session disposal stops startup polling and deadlines.
- [`scripts/patch_runtime.py`](scripts/patch_runtime.py) registers the main Python
  worker and proxy disposers, includes the bridge in build identity, injects it
  before worker construction and emits `assignment-sessions.json`. Its source-
  checked substitutions pass against actual cached Shinylive 0.10.15 bootstrap
  bytes. This assembly check uses a small temporary bootstrap tree rather than
  claiming a fresh complete export.
- [`scripts/assertions.py`](scripts/assertions.py),
  [`scripts/evidence_tree.py`](scripts/evidence_tree.py) and
  [`scripts/gate.py`](scripts/gate.py) check expected/observed values recursively,
  reject malformed or failed nested check maps, validate child browser claims
  even without parent labels, and apply comprehensive case bindings at every
  depth. Pixel acceptance now opens both checksum-bound captures and recomputes
  exact pixel equality. Capture paths are required alongside their existing hash
  fields. Fixed float tolerances and exact discrete comparisons are retained.
- Dynamic choice validation rejects contradictory observations, malformed parent
  records and duplicate choice identities. This does not resolve the outstanding
  native parent-state inventory.

## Verification

The final local run passes **114 tests**; its
[report](evidence/shared-foundation-unit-tests-oct05.json) binds the
[complete log](evidence/shared-foundation-unit-tests-oct05.log) and implementation
inputs. The deterministic launcher/child harness exercises nine ownership,
readiness, error, retry and disposal scenarios, including queued readiness from a
previous document after reload.

Two installed-browser reports each pass ten bounded cases: seven real assignment
readiness states, failed application-bundle download and Retry, a suppressed
cooperative disposal command with rapid replacement, and final shutdown:

- [Chrome](evidence/session-bridge-chrome-chrome-final-all-units-recovery-oct05.json)
- [Edge](evidence/session-bridge-edge-all-units-recovery-oct05.json)

Both runs use independently owned loopback origins and complete retained file
maps. Native host observations record no sleep interruptions. Readiness assertions
inspect actual Shiny sockets and initial inputs. Worker disappearance is checked
against the pre-navigation dedicated-worker target IDs within two seconds of the
selection request. The forced cases remove the prior worker in approximately
1.02 seconds; final shutdown is approximately 0.11–0.14 seconds. These are bounded
development-machine disposal observations, **not performance benchmarks**.

The forced case deliberately suppresses the cooperative command to exercise the
launcher's direct cleanup and frame removal. It is fault-injection evidence;
it cannot satisfy the required uninstrumented 30-sequence lifecycle suite.
Navigation during full training and nested calculation-worker workloads still
requires its separate comprehensive matrix.

Earlier attempts remain retained: the initial harness had a Playwright argument
error; the next exposed removal of the launcher's message listener before its
close acknowledgement; an initial fault injector failed to suppress the command.
Their reports and candidates are preserved. The close-listener defect and harness
faults were corrected before the final complete browser runs.

All **2,541 baseline bindings** still verify. Original sources, fixtures, datasets,
locks and the original baseline manifest remain unchanged. Candidates share
immutable assets through hardlinks and unlink mutable files before writes.
Preview markers are confined to their own served candidate.

The refreshed [gate](evidence/gate.json) remains **BLOCKED: 2/140 suites current**,
with `nextjs_authorized: false`. The previous 11/140 result is preserved in evidence
history. Browser evidence tied to the previous runtime sources is now stale;
the two unchanged bounded native reports remain current. New bridge checks are
bounded evidence and do not replace any of the 133 original or seven delivery
suites, or any of the 1,824 legacy behavior IDs.

## Reproduce and continue

From the workspace root, choose fresh labels for each candidate:

```sh
.venv/bin/python -m unittest discover -s proof/tests
.venv/bin/python proof/scripts/verify_continuation_baseline.py
.venv/bin/python proof/scripts/verify_session_bridge.py \
  --browser chrome --label fresh-chrome-bridge --faults
.venv/bin/python proof/scripts/verify_session_bridge.py \
  --browser edge --label fresh-edge-bridge --faults
.venv/bin/python proof/scripts/gate.py
```

Storage remains about **1 GB free**. No full rebuild or large verification matrix
was run. Obtain the planned working reserve before those runs. Once storage is
available, regenerate and verify complete standalone exports with the updated
patch, rerun every affected lifecycle/ownership regression, and then continue the
shared asynchronous-loading/long-calculation ownership and executable coverage
work. Abortable full embeddings, checksum/decompression workers, comprehensive
Unit 1–7 behavior/numerical/pixel matrices and broader training trajectories remain
open. The retained bridge harness is not the Next.js navigation shell.

GitHub/Vercel account targets, Vercel authentication, Safari automation and access
to a physical 8 GB integrated-graphics laptop remain unanswered execution inputs.
The complete release and physical-hardware certification cannot be delivered
without them. Preserve the original order in [REMAINING-PLAN.md](REMAINING-PLAN.md).
