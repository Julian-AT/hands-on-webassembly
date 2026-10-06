# Provenance and archived evidence

The original source, material, scientific dependency, hardware and application release locks retain their original bytes. `proof/archives/immutable-inputs.json` verifies those inputs. Larger supplied arrays live in `artifacts/originals/` and are restored by `scripts/archive/originals.py` from the existing checksum-locked supplemental release.

The verified preservation release is `history-before-root-20261006-v1`. An initial empty `history-before-root-20261006` release became immutable before upload; its tag remains intact as a publication record. Use the `-v1` archive index for restoration.

The preservation release contains every retained report, failed attempt, screenshot, numerical fixture, candidate cache snapshot, historical source snapshot, baseline binding and the prior Git history. [The archive index](../proof/archives/history.json) records its immutable release, inventory digest and shard digests. The [verification receipt](../proof/archives/verification.json) records the downloaded-object, restored-baseline and complete-history checks performed before local historical cleanup. The compressed inventory records 132,526 original files, their modes, sizes and SHA-256 values; content deduplication retains all paths through 40,333 exact byte objects. Symlink targets are retained as metadata. Restoration materializes regular-file objects and rejects traversal, unexpected members, missing dependencies, corrupt bytes, existing symlinks and conflicting files.

With GitHub CLI authenticated to this private repository:

```sh
make archive-verify
.cache/env/native/bin/python scripts/archive/history.py restore \
  --directory .cache/history --destination .cache/history-restored --baseline
.cache/env/native/bin/python \
  .cache/history-restored/proof/scripts/verify_continuation_baseline.py
```

The superseded baseline-creation script and dated experiments are preserved in the archive. Active verification and diagnostic helpers remain in the repository.

Baseline restoration resolves the complete original binding closure and runs the original verifier in its original layout. It does not rewrite the historical manifest or evidence fingerprints. For a larger historical audit, omit `--baseline`, or select original path prefixes with repeated `--prefix` arguments. The inventory preserves external tool symlink metadata; historical runtime environments should be recreated from the frozen locks rather than executing external links.

The Git bundle is restored as `.cache/history-restored/.cache/restructure/old-history.bundle` by the baseline command. Verify it with `git bundle verify`, or clone it into a separate audit workspace. The existing immutable application tag remains attached to its original commit. The archive tag points to the prior published main.

Current evidence is generated under `artifacts/proof/evidence`. A new source or path fingerprint invalidates old execution evidence. Historical reports remain archive bytes and are never restamped to obtain a current pass. The 9/140 checkpoint is retained as history; it does not authorize current certification.
