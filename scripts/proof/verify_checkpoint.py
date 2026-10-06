"""Refresh existing bounded browser suites without hiding numerical failures."""

from repository import ROOT, EVIDENCE, SCRIPTS, SITE
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import urllib.request
import hashlib
from common import PROOF, sha, write_json, archive_file
from provenance import stamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("Use 1–4 independent browser processes")
    # Refuse missing or incorrectly served deployments before overwriting suite outputs.
    for name in ("shinylive-sw.js", "shinylive/shinylive.js", "probes/index.html"):
        with urllib.request.urlopen("http://127.0.0.1:8008/" + name, timeout=10) as response:
            observed = hashlib.sha256(response.read()).hexdigest()
        if observed != sha(SITE / name):
            raise ValueError(f"Checkpoint origin does not serve the bound artifact: {name}")
    suites = [
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
    result = dict(
        status="running",
        browser="chrome",
        distribution="installed",
        provenance=stamp(),
        suites={},
        scope="Existing bounded Chrome workflows and numerical probes; exhaustive release certification is separate.",
    )
    path = EVIDENCE / "verification-checkpoint.json"
    write_json(path, result)
    for capture in (EVIDENCE / "screenshots").glob("browser-chrome-*.png"):
        archive_file(capture)

    def run(suite):
        log = EVIDENCE / f"checkpoint-{suite}.log"
        archive_file(log)
        with log.open("w") as stream:
            code = subprocess.call(
                [
                    sys.executable,
                    str(SCRIPTS / "browser.py"),
                    "--browser",
                    "chrome",
                    "--unit",
                    suite,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        report = EVIDENCE / f"browser-chrome-unit{suite}.json"
        value = json.loads(report.read_text()) if report.exists() else {}
        return suite, dict(
            status=value.get("status", "error"),
            exit_code=code,
            report=str(report.relative_to(ROOT)),
            sha256=sha(report) if report.exists() else None,
            report_sha256=sha(report) if report.exists() else None,
            input_fingerprint=value.get("provenance", {}).get("fingerprint"),
            version=value.get("version"),
            counts={
                section: len(value.get(section, {}))
                for section in ("cases", "checks", "comparisons")
            },
            failures={
                section: [
                    key
                    for key, check in value.get(section, {}).items()
                    if check.get("status") != "pass"
                ]
                for section in ("cases", "checks", "comparisons")
            },
        )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, suite) for suite in suites]
        for future in as_completed(futures):
            suite, record = future.result()
            result["suites"][suite] = record
            write_json(path, result)
            print(suite, record["status"], record["counts"], flush=True)
    result["status"] = (
        "pass" if all(value["status"] == "pass" for value in result["suites"].values()) else "fail"
    )
    versions = {record["version"] for record in result["suites"].values()}
    result["version"] = next(iter(versions)) if len(versions) == 1 else None
    if not result["version"]:
        result.update(
            status="fail", browser_identity_error="Mixed or missing installed browser versions"
        )
    if result["provenance"]["fingerprint"] != stamp()["fingerprint"]:
        result["observed_status"] = result["status"]
        result["status"] = "stale"
    write_json(path, result)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
