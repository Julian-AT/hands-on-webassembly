"""Apply retained patches to untouched supplied files and verify runnable bytes."""

from repository import ROOT, CACHE, EVIDENCE, PATCHES, REFERENCE
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from common import ROOT, PROOF, UNITS, sha, write_json
from storage_budget import require_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("Safe label required")
    root = CACHE / f"reference-patch-audit-{args.label}"
    if root.exists():
        raise FileExistsError("Preserve prior audits")
    require_space(64 * 1024**2)
    root.mkdir()
    manifest = EVIDENCE / "reference-corrections.json"
    record = json.loads(manifest.read_text())
    result = dict(
        status="pass",
        cases={},
        executor_sha256=sha(Path(__file__)),
        correction_manifest_sha256=sha(manifest),
        patch_program_sha256=sha(Path("/usr/bin/patch")),
        scope="Actual retained patches applied to independent copies of all changed supplied Python files, then compared byte-for-byte with runnable references. Sources and references remain untouched.",
    )
    for unit, folder in UNITS.items():
        destination = root / f"unit{unit}"
        destination.mkdir()
        for original in (ROOT / folder).glob("*.py"):
            shutil.copy2(original, destination / original.name)
        for key, binding in record["files"].items():
            if not key.startswith(f"unit{unit}/"):
                continue
            filename = key.split("/")[1]
            original = ROOT / folder / filename
            reference = REFERENCE / key
            patch = PATCHES / f"unit{unit}-{Path(filename).stem}.patch"
            expected = dict(original=sha(original), corrected=sha(reference), patch=sha(patch))
            supplied = dict(
                original=binding["original_sha256"],
                corrected=binding["corrected_sha256"],
                patch=binding["patch_sha256"],
            )
            applied = subprocess.run(
                ["/usr/bin/patch", "--batch", "-p1", "-d", str(root)],
                input=patch.read_text(),
                capture_output=True,
                text=True,
            )
            log = root / f"unit{unit}-{filename}.log"
            log.write_text(applied.stdout + applied.stderr)
            observed = sha(destination / filename)
            passed = (
                applied.returncode == 0
                and expected == supplied
                and observed == expected["corrected"]
            )
            result["cases"][key] = dict(
                status="pass" if passed else "fail",
                exit_code=applied.returncode,
                assertion=dict(
                    kind="exact", expected=expected["corrected"], observed=observed, matched=passed
                ),
                bindings_match=expected == supplied,
                bindings=expected,
                log=str(log.relative_to(ROOT)),
                log_sha256=sha(log),
            )
        for original in (ROOT / folder).glob("*.py"):
            key = f"unit{unit}/{original.name}"
            if sha(original) != sha(REFERENCE / key) and key not in record["files"]:
                result["cases"][key] = dict(
                    status="fail", error="Changed runnable file has no retained patch"
                )
        app = REFERENCE / f"unit{unit}/app.py"
        expected = record["runnable_apps"][str(unit)]
        result["cases"][f"unit{unit}/runnable-identity"] = dict(
            status="pass" if sha(app) == expected else "fail",
            assertion=dict(
                kind="exact", expected=expected, observed=sha(app), matched=sha(app) == expected
            ),
        )
    result["status"] = (
        "pass" if all(c["status"] == "pass" for c in result["cases"].values()) else "fail"
    )
    write_json(EVIDENCE / f"reference-patch-audit-{args.label}.json", result)
    print(result["status"], len(result["cases"]), "applied patch/identity checks", flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
