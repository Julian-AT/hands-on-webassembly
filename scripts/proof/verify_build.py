"""Rebuild from dirty staging and compare every static file byte-for-byte."""

from repository import ASSETS, BUILD, EVIDENCE, SCRIPTS, SITE
import json
import subprocess
import sys
from common import PROOF, UNITS, sha, write_json
from provenance import stamp


def main():
    site = SITE

    def snapshot():
        return {
            str(path.relative_to(site)): sha(path) for path in site.rglob("*") if path.is_file()
        }

    before = snapshot()
    for unit in UNITS:
        (BUILD / f"unit{unit}/obsolete-stage-file.py").write_text("obsolete")
    subprocess.run([sys.executable, str(SCRIPTS / "build.py")], check=True)
    after = snapshot()
    changed = sorted(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )
    lock = json.loads((PROOF / "assets.lock.json").read_text())
    invalid = [
        name for name, record in lock.items() if sha(ASSETS / "v1" / name) != record["sha256"]
    ]
    cases = {
        "clean-staging": {
            "status": "fail" if list((BUILD).rglob("obsolete-stage-file.py")) else "pass"
        },
        "identical-builds": {"status": "fail" if changed else "pass", "changed_files": changed},
        "local-asset-checksums": {
            "status": "fail" if invalid else "pass",
            "invalid_assets": invalid,
        },
    }
    passed = all(case["status"] == "pass" for case in cases.values())
    write_json(
        EVIDENCE / "reproducible-build.json",
        {
            "status": "pass" if passed else "fail",
            "cases": cases,
            "provenance": stamp(),
            "files_compared": len(after),
            "scope": "Static content reproducibility and staging; lazy network loading is a separate check.",
        },
    )
    print("PASS" if passed else "FAIL", cases)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
