"""Exercise real migrated paths instead of relying on test import order."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from repository import ROOT, PROOF, EVIDENCE, ARTIFACTS
from common import sha
from evidence_tree import validate_tree
from artifact_provenance import fingerprint


class MigratedPathsTests(unittest.TestCase):
    def test_large_resources_rehydrate_only_the_selected_assignment(self):
        import repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "archives").mkdir()
            (root / "archives/large-originals.json").write_text(
                json.dumps({"assignments/2/resources/piano.wav": {}, "assignments/5/other.bin": {}})
            )
            source = root / "original.wav"
            source.write_bytes(b"supplied bytes")
            destination = root / "native-reference"
            with (
                patch.object(repository, "PROOF", root),
                patch.object(repository, "original_path", return_value=source),
            ):
                repository.copy_original_resources(2, destination)
            self.assertEqual(
                (destination / "resources/piano.wav").read_bytes(), source.read_bytes()
            )
            self.assertFalse((destination / "other.bin").exists())

    def test_historical_reference_binding_uses_restored_original_paths(self):
        import preserve_reference_baseline as preservation

        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            historical_root = Path(temporary)
            baseline = historical_root / "proof/continuation-baseline"
            baseline.mkdir(parents=True)
            source = historical_root / "proof/reference/unit1/app.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"historical source")
            record = {
                "files": {
                    "proof/reference/unit1/app.py": {
                        "binding": "proof/reference/unit1/app.py",
                        "sha256": sha(source),
                        "bytes": source.stat().st_size,
                    }
                }
            }
            (baseline / "manifest.corrected.json").write_text(json.dumps(record))
            with (
                patch.object(preservation, "BASELINE", baseline),
                patch.object(preservation, "EVIDENCE", historical_root / "evidence"),
            ):
                preservation.main()
            self.assertEqual(
                (baseline / "proof/reference/unit1/app.py").read_bytes(), source.read_bytes()
            )

    def test_generated_paths_never_use_legacy_proof_prefix(self):
        failures = []
        legacy = ("evidence/", "site/", "build/", "assets/", "reference/", "native/", "cache/")
        for path in (ROOT / "scripts/proof").glob("*.py"):
            if path.name in ("continuation_baseline.py", "verify_continuation_baseline.py"):
                continue  # Historical manifest tools deliberately retain original paths.
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                    continue
                if not isinstance(node.left, ast.Name) or node.left.id != "PROOF":
                    continue
                value = node.right
                if isinstance(value, ast.JoinedStr) and value.values:
                    value = value.values[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if value.value.startswith(legacy):
                        failures.append(f"{path.name}:{node.lineno}: {value.value}")
        self.assertEqual(failures, [])

    def test_canonical_child_binds_artifact_and_rejects_changes(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as temporary:
            directory = Path(temporary)
            data = directory / "input"
            data.mkdir()
            (data / "data.bin").write_bytes(b"bound input")
            executor = ROOT / "scripts/proof/evidence_tree.py"
            report = dict(
                status="pass",
                executor=str(executor.relative_to(ROOT)),
                executor_sha256=sha(executor),
                dependencies={str(executor.relative_to(ROOT)): sha(executor)},
                artifact=str(data.relative_to(ROOT)),
                inputs={"data.bin": sha(data / "data.bin")},
                provenance={"scope": "isolated-artifact"},
                cases={"check": {"status": "pass"}},
            )
            current = hashlib.sha256(
                json.dumps(report["inputs"], sort_keys=True).encode()
            ).hexdigest()
            report["provenance"]["fingerprint"] = current
            child = directory / "child.json"
            child.write_text(json.dumps(report))
            parent = {
                "cases": {
                    "child": {
                        "status": "pass",
                        "report": str(child.relative_to(ROOT)),
                        "report_sha256": sha(child),
                        "input_fingerprint": current,
                    }
                }
            }

            def resolve(child, name):
                return fingerprint(child, PROOF)

            self.assertEqual(
                validate_tree(parent, PROOF, current, fingerprint_resolver=resolve), []
            )
            (data / "data.bin").write_bytes(b"changed input")
            self.assertTrue(validate_tree(parent, PROOF, current, fingerprint_resolver=resolve))
            parent["cases"]["child"]["report"] = "scripts/proof/evidence_tree.py"
            self.assertTrue(validate_tree(parent, PROOF, current, fingerprint_resolver=resolve))


if __name__ == "__main__":
    unittest.main()
