"""Audit the fixed bounded suite set from current dependency-bound child reports."""

from repository import ROOT, EVIDENCE
import json
from pathlib import Path
from common import PROOF, sha, write_json
from provenance import browser_fingerprint, stamp
from evidence_tree import validate_tree

SUITES = [
    *map(str, range(1, 8)),
    "uploads",
    "unit5probe",
    "unit5full",
    "unit5life",
    "training",
    "neural",
    "embedding",
    "tabular",
    "supervised",
    "runtime",
]


def main():
    children = {}
    errors = []
    versions = set()
    for suite in SUITES:
        path = EVIDENCE / f"browser-chrome-unit{suite}.json"
        if not path.exists():
            errors.append(f"Missing suite {suite}")
            continue
        report = json.loads(path.read_text())
        versions.add(report.get("version"))
        children[suite] = dict(
            status=report.get("status"),
            report=str(path.relative_to(ROOT)),
            report_sha256=sha(path),
            input_fingerprint=report.get("provenance", {}).get("fingerprint"),
            version=report.get("version"),
            counts={
                section: len(report.get(section, {}))
                for section in ("cases", "checks", "comparisons")
            },
        )
    version = next(iter(versions)) if len(versions) == 1 else None
    if not version:
        errors.append("Installed Chrome identity is missing or mixed")
    report = dict(
        status="pass",
        browser="chrome",
        distribution="installed",
        version=version,
        suites=children,
        provenance=stamp(),
        executor_sha256=sha(Path(__file__)),
        scope="Read-only recursive audit of all 17 fixed bounded suites, using each actual harness dependency scope. No comprehensive or hardware acceptance.",
    )

    def expected(child, name):
        provenance = child.get("provenance", {})
        harness = provenance.get("harness")
        if (
            provenance.get("scope") != "browser"
            or harness not in SUITES
            or Path(name).name != f"browser-chrome-unit{harness}.json"
        ):
            raise ValueError("Unknown or mismatched bounded harness")
        return browser_fingerprint(harness)

    errors.extend(
        validate_tree(
            report,
            PROOF,
            report["provenance"]["fingerprint"],
            browser="chrome",
            version=version,
            fingerprint_resolver=expected,
        )
    )
    report.update(status="fail" if errors else "pass", errors=errors)
    write_json(EVIDENCE / "bounded-evidence-audit.json", report)
    print(report["status"], len(children), "bounded children;", len(errors), "errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
