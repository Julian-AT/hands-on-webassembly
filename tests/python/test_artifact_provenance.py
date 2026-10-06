import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from common import sha
from artifact_provenance import fingerprint


class ArtifactProvenanceTests(unittest.TestCase):
    def example(self, root):
        proof = root / "proof"
        (root / "scripts/proof").mkdir(parents=True)
        (proof / "site").mkdir(parents=True)
        (proof / "site/app.json").write_bytes(b"app")
        executor = root / "scripts/proof/check.py"
        executor.write_bytes(b"checked executor")
        helper = root / "scripts/proof/helper.py"
        helper.write_bytes(b"checked helper")
        report = dict(
            executor="scripts/proof/check.py",
            executor_sha256=sha(executor),
            dependencies={"scripts/proof/helper.py": sha(helper)},
            artifact="site",
            provenance=dict(scope="isolated-artifact"),
            inputs={"app.json": sha(proof / "site/app.json")},
        )
        return proof, report

    def test_missing_or_changed_child_files_and_executors_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            proof, report = self.example(Path(temporary))
            self.assertEqual(
                fingerprint(report, proof),
                hashlib.sha256(json.dumps(report["inputs"], sort_keys=True).encode()).hexdigest(),
            )
            (proof / "site/app.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                fingerprint(report, proof)
            (proof / "site/app.json").write_bytes(b"app")
            (proof.parent / "scripts/proof/helper.py").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                fingerprint(report, proof)
            (proof.parent / "scripts/proof/helper.py").write_bytes(b"checked helper")
            (proof.parent / "scripts/proof/check.py").unlink()
            with self.assertRaises(ValueError):
                fingerprint(report, proof)

    def test_paths_cannot_escape_and_added_files_invalidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proof, report = self.example(root)
            report["artifact"] = "../outside"
            (root / "outside").mkdir()
            with self.assertRaises(ValueError):
                fingerprint(report, proof)
            report["artifact"] = "site"
            (proof / "site/unbound.txt").write_bytes(b"extra")
            with self.assertRaises(ValueError):
                fingerprint(report, proof)

    def test_two_generation_inputs_are_both_rechecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            proof, report = self.example(Path(temporary))
            (proof / "other").mkdir()
            (proof / "other/app.json").write_bytes(b"next generation")
            report["provenance"]["scope"] = "two-isolated-artifacts"
            report["inputs"] = {
                "site": report["inputs"],
                "other": {"app.json": sha(proof / "other/app.json")},
            }
            fingerprint(report, proof)
            (proof / "other/app.json").unlink()
            with self.assertRaises(ValueError):
                fingerprint(report, proof)


if __name__ == "__main__":
    unittest.main()
