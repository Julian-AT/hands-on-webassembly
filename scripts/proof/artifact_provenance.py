"""Resolve bounded evidence against every retained artifact and executor byte."""

import hashlib
import json
from pathlib import Path
from common import sha
from repository import evidence_locations


def fingerprint(report, proof):
    root, _ = evidence_locations(proof)
    proof = proof.resolve()
    executor = report.get("executor")
    if not isinstance(executor, str):
        raise ValueError("Artifact report lacks its executor path")
    path = (root / executor).resolve()
    if (
        not path.is_relative_to(root / "scripts/proof")
        or not path.is_file()
        or sha(path) != report.get("executor_sha256")
    ):
        raise ValueError("Artifact executor is missing, outside scripts or changed")
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("Artifact executor dependencies are missing")
    for name, digest in dependencies.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Artifact executor dependency is missing or changed: {name}")
    scope = report.get("provenance", {}).get("scope")
    if scope == "isolated-artifact":
        name = report.get("artifact")
        if not isinstance(name, str):
            raise ValueError("Artifact report lacks its retained directory")
        artifacts = {name: report.get("inputs")}
    elif scope == "two-isolated-artifacts":
        artifacts = report.get("inputs")
        if not isinstance(artifacts, dict) or len(artifacts) != 2:
            raise ValueError("Build transition must bind two retained artifacts")
    else:
        raise ValueError("Unknown retained artifact scope")
    for name, expected in artifacts.items():
        directory = (
            (root if name.startswith(("artifacts/", ".cache/")) else proof) / name
        ).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError("Artifact directory missing or outside proof")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Artifact file manifest missing")
        observed = {}
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError("Artifact contains an external file binding")
            observed[str(path.relative_to(directory))] = sha(path)
        if observed != expected:
            raise ValueError(f"Retained artifact is incomplete or changed: {name}")
    return hashlib.sha256(json.dumps(report["inputs"], sort_keys=True).encode()).hexdigest()
