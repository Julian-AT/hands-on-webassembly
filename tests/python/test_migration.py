"""Migration contracts protect original bytes and require canonical source paths."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/proof"))


class MigrationTests(unittest.TestCase):
    def test_canonical_layout_has_no_nested_release(self):
        for folder in (
            "assignments",
            "web",
            "runtime",
            "wasm",
            "scripts",
            "tests",
            "proof",
            "docs",
            "requirements",
        ):
            self.assertTrue((ROOT / folder).is_dir(), folder)
        self.assertFalse((ROOT / "release").exists())
        self.assertTrue((ROOT / ".git").is_dir())

    def test_paths_resolve_without_current_directory_assumptions(self):
        import repository

        self.assertEqual(repository.ROOT, ROOT)
        self.assertEqual(repository.RUNTIME, ROOT / "runtime")
        self.assertEqual(repository.SCRIPTS, ROOT / "scripts/proof")
        self.assertEqual(repository.EVIDENCE, ROOT / "artifacts/proof/evidence")

    def test_supplied_source_and_material_locks_are_preserved(self):
        locks = [
            json.loads((ROOT / "proof/source.lock.json").read_text()),
            json.loads((ROOT / "proof/assignment5-materials.lock.json").read_text())["materials"],
        ]
        for lock in locks:
            for name, digest in lock.items():
                from repository import original_path

                with original_path(name).open("rb") as f:
                    self.assertEqual(hashlib.file_digest(f, "sha256").hexdigest(), digest, name)

    def test_frozen_native_and_application_manifests_retain_original_bytes(self):
        originals = json.loads((ROOT / "proof/archives/immutable-inputs.json").read_text())
        for name, digest in originals.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)

    def test_all_suites_and_legacy_behavior_ids_remain(self):
        from coverage_contract import requirements

        inventory = json.loads((ROOT / "proof/contracts/source-inventory.json").read_text())
        rows = requirements(inventory)
        expected = json.loads((ROOT / "proof/contracts/legacy-requirements.json").read_text())
        self.assertEqual(len(rows), 140)
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in expected])
        for row, old in zip(rows, expected):
            self.assertEqual(row["required_cases"], old["required_cases"], row["id"])


if __name__ == "__main__":
    unittest.main()
