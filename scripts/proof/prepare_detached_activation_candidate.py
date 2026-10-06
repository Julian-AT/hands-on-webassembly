"""Controlled activation repair with new, content-bound candidate generations."""

from repository import EVIDENCE

from repository import ROOT, CACHE
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from common import PROOF, sha, write_json
from browser_generation_sequence import site_manifest, generation
from storage_budget import require_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = CACHE / f"detached-activation-{args.label}"
    if root.exists():
        raise FileExistsError("Preserve candidate attempts")
    require_space(128 * 1024**2)
    records = {}
    for name, base in (("before", args.before), ("after", args.after)):
        site = root / name
        shutil.copytree(base, site, copy_function=os.link)
        inputs = site_manifest(base)
        old = generation(base)
        correction = (
            'self.skipWaiting().catch(error => console.error("Runtime activation failed", error));'
        )
        version = hashlib.sha256(
            json.dumps(dict(inputs=inputs, correction=correction), sort_keys=True).encode()
        ).hexdigest()[:20]
        worker = site / "shinylive-sw.js"
        source = worker.read_text()
        marker = "event.waitUntil(self.skipWaiting());"
        if source.count(marker) != 1:
            raise ValueError("Guarded activation source contract changed")
        worker.unlink()
        worker.write_text(source.replace(marker, correction, 1))
        changed = []
        for path in sorted(site.rglob("*")):
            if not path.is_file() or path.suffix not in (".html", ".js", ".mjs", ".json"):
                continue
            text = path.read_text()
            if old not in text:
                continue
            path.unlink()
            path.write_text(text.replace(old, version))
            changed.append(str(path.relative_to(site)))
        records[name] = dict(
            artifact=str(site.relative_to(ROOT)),
            base=str(base.resolve().relative_to(ROOT)),
            inputs=inputs,
            original_generation=old,
            candidate_generation=version,
            changed_paths=changed,
            manifest=site_manifest(site),
        )
    write_json(
        EVIDENCE / f"detached-activation-candidate-{args.label}.json",
        dict(
            status="candidate",
            executor_sha256=sha(Path(__file__)),
            candidates=records,
            hypothesis="Waiting-worker activation retries must not retain their activation promise as pending message work.",
            scope="Two detached candidate generations with guarded activation retry lifetime removed; no instrumentation. Full affected 30-sequence runs required before promotion.",
        ),
    )
    print(root, flush=True)


if __name__ == "__main__":
    main()
