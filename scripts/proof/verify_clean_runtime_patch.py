"""Exercise checked upstream substitutions on a fresh export, leaving deployment intact."""

from repository import BUILD, EVIDENCE

from repository import ROOT, CACHE, RUNTIME, SITE
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import PROOF, UNITS, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="html-correction")
    parser.add_argument("--candidate-site", type=Path, default=SITE)
    args = parser.parse_args()
    args.candidate_site = args.candidate_site.resolve()
    target = CACHE / f"clean-runtime-patch-{args.label}"
    if target.exists():
        raise FileExistsError("Retain the previous export; choose a new label")
    target.mkdir()
    (target / "runtime").symlink_to(RUNTIME, target_is_directory=True)
    (target / "evidence").mkdir()
    site = target / "site"
    report = dict(
        status="running",
        executor_sha256=sha(Path(__file__)),
        exports={},
        scope="Fresh pinned Shinylive exports and checked upstream patch; no live deployment mutation or lifecycle acceptance.",
    )
    output = EVIDENCE / f"clean-runtime-patch-{args.label}.json"
    try:
        for unit in UNITS:
            subprocess.run(
                [
                    str(Path(sys.executable).parent / "shinylive"),
                    "export",
                    str(BUILD / f"unit{unit}"),
                    str(site),
                    "--subdir",
                    f"unit{unit}",
                ],
                check=True,
            )
            report["exports"][str(unit)] = dict(app_json_sha256=sha(site / f"unit{unit}/app.json"))
        report["upstream"] = {
            name: sha(site / name)
            for name in (
                "shinylive-sw.js",
                "shinylive/load-shinylive-sw.js",
                "shinylive/shinylive.js",
            )
        }
        import patch_runtime

        patch_runtime.PROOF = target
        patch_runtime.patch_runtime()
        report["patched"] = {name: sha(site / name) for name in report["upstream"]}
        # The isolated candidate differs only by its already-bound cache version.
        candidate = args.candidate_site / "shinylive-sw.js"
        import re

        def normalize(text):
            return re.sub(
                r'var version = "course-[^"]+";',
                'var version = "course-__COURSE_RUNTIME_VERSION__";',
                text,
            )

        candidate_source = normalize(candidate.read_text())
        comment = (
            "// A proxy request belongs to one application document. Never leave an orphaned\n"
            "// respondWith promise keeping the old service worker active during an upgrade.\n"
        )
        duplicated = candidate_source.count(comment + comment)
        if duplicated == 1:
            candidate_source = candidate_source.replace(comment + comment, comment, 1)
        report["candidate_duplicate_comment_removed_for_comparison"] = duplicated == 1
        matches = candidate_source == (site / "shinylive-sw.js").read_text()
        report["candidate_worker_equivalent"] = matches
        candidate_loader = args.candidate_site / "shinylive/load-shinylive-sw.js"

        def normalize_loader(text):
            return re.sub(
                r'serviceWorkerPath \+= "\?v=[^"\n]+";',
                'serviceWorkerPath += "?v=__COURSE_RUNTIME_VERSION__";',
                text,
            )

        report["candidate_loader_equivalent"] = (
            normalize_loader(candidate_loader.read_text())
            == (site / "shinylive/load-shinylive-sw.js").read_text()
        )
        report["candidate_site"] = str(args.candidate_site.relative_to(ROOT))
        report["candidate_worker_sha256"] = sha(candidate)
        report["candidate_loader_sha256"] = sha(candidate_loader)
        if not matches:
            import difflib

            patch = target / "candidate-difference.patch"
            patch.write_text(
                "".join(
                    difflib.unified_diff(
                        normalize(candidate.read_text()).splitlines(True),
                        (site / "shinylive-sw.js").read_text().splitlines(True),
                    )
                )
            )
            report["difference_patch"] = str(patch.relative_to(ROOT))
        report["status"] = "pass" if matches and report["candidate_loader_equivalent"] else "fail"
    except Exception as error:
        report.update(status="fail", error=str(error))
    write_json(output, report)
    print(report["status"], report.get("error", ""))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
