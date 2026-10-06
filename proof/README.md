# Proof contracts

`source.lock.json`, `assignment5-materials.lock.json`, `assets.lock.json` and `hardware-contract.json` retain their original bytes. `contracts/` retains all suite IDs, behavior IDs and dynamic-choice obligations. Active executors live in `scripts/proof/`; browser harnesses live in `proof/harness/browser/`.

`make proof` writes current evidence to `artifacts/proof/evidence/` and returns a blocked result until every required suite passes in its final context. Historical reports, attempts, source snapshots and baseline dependencies are preserved by the checksum-verified release in `archives/history.json`.

Read [verification](../docs/verification.md), [archive restoration](../docs/provenance.md), and [current status](status.json). Historical checkpoints do not count as current passes.
