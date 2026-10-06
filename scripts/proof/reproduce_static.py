"""Build twice from clean staging and retain the exact final file manifest."""

from repository import EVIDENCE, SCRIPTS, SITE
import json
import subprocess
import sys
from common import PROOF, sha, write_json, archive_file
from provenance import stamp


def manifest():
    return {str(p.relative_to(SITE)): sha(p) for p in sorted((SITE).rglob("*")) if p.is_file()}


def main():
    results = []
    for index in (1, 2):
        log = EVIDENCE / f"static-build-{index}.log"
        archive_file(log)
        with log.open("w") as output:
            subprocess.run(
                [sys.executable, str(SCRIPTS / "build.py")],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        results.append(manifest())
    changed = [
        name
        for name in sorted(set(results[0]) | set(results[1]))
        if results[0].get(name) != results[1].get(name)
    ]
    log = EVIDENCE / "static-asset-checks.log"
    archive_file(log)
    with log.open("w") as output:
        assets = subprocess.run(
            [sys.executable, str(SCRIPTS / "check_assets.py")],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    record = dict(
        status="pass" if not changed and assets.returncode == 0 else "fail",
        cases={
            "clean-staging": {"status": "pass"},
            "identical-builds": {"status": "fail" if changed else "pass", "changed_files": changed},
            "local-asset-checksums": {
                "status": "pass" if assets.returncode == 0 else "fail",
                "exit_code": assets.returncode,
            },
        },
        provenance=stamp(),
        files_compared=len(results[1]),
        scope="Two independent clean assignment/runtime staging builds, complete static contents and checked local assets. Lazy loading, hosting/HTTPS, and exhaustive release acceptance remain separate.",
    )
    write_json(EVIDENCE / "static-artifact-manifest.json", results[1])
    write_json(
        EVIDENCE / "static-artifact-files.json",
        dict(
            files={
                name: dict(bytes=(SITE / name).stat().st_size, sha256=digest)
                for name, digest in results[1].items()
            },
            file_count=len(results[1]),
            manifest_sha256=sha(EVIDENCE / "static-artifact-manifest.json"),
        ),
    )
    write_json(EVIDENCE / "reproducible-build.json", record)
    print(record["status"], record["files_compared"], "files; changed:", changed, flush=True)
    return int(record["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
