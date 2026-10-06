import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
import provenance


class NativeProvenanceTests(unittest.TestCase):
    def test_browser_edits_preserve_native_but_reference_and_test_edits_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                "iml_env.yaml",
                "requirements/native.lock",
                "proof/source.lock.json",
                "proof/assignment5-materials.lock.json",
                "scripts/proof/reference_corrections.py",
                "scripts/proof/common.py",
                "scripts/proof/provenance.py",
                "scripts/proof/browser.py",
                "scripts/proof/browser_unit2.py",
                "artifacts/reference/unit2/app.py",
                "assignments/2/app.py",
                "runtime/startup-status.js",
                "artifacts/site/unit2/app.json",
            ]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            with patch.multiple(
                provenance,
                ROOT=root,
                ARTIFACTS=root / "artifacts",
                PROOF=root / "proof",
                RUNTIME=root / "runtime",
                SCRIPTS=root / "scripts/proof",
                TESTS=root / "tests/python",
                PACKAGES=root / "wasm/packages",
                HARNESS=root / "proof/harness/browser",
                REQUIREMENTS=root / "requirements",
                ASSETS=root / "artifacts/assets",
                SITE=root / "artifacts/site",
                REFERENCE=root / "artifacts/reference",
                CACHE=root / ".cache/science",
            ):
                expected = provenance.native_fingerprint(2)
                for name in [
                    "runtime/startup-status.js",
                    "artifacts/site/unit2/app.json",
                    "scripts/proof/provenance.py",
                ]:
                    (root / name).write_text("browser change")
                    self.assertEqual(provenance.native_fingerprint(2), expected)
                for name in [
                    "artifacts/reference/unit2/app.py",
                    "scripts/proof/browser_unit2.py",
                    "scripts/proof/reference_corrections.py",
                    "requirements/native.lock",
                ]:
                    (root / name).write_text("native dependency change")
                    self.assertNotEqual(provenance.native_fingerprint(2), expected)
                    (root / name).write_text("original")

    def test_browser_scope_tracks_artifact_and_actual_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                "artifacts/site/unit2/app.json",
                "runtime/startup-status.js",
                "scripts/proof/browser.py",
                "scripts/proof/browser_unit2.py",
                "scripts/proof/browser_unit4.py",
                "scripts/proof/check_assets.py",
                "scripts/proof/build.py",
                "scripts/proof/reference_corrections.py",
                "tests/python/test_unrelated.py",
            ]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            with patch.multiple(
                provenance,
                ROOT=root,
                ARTIFACTS=root / "artifacts",
                PROOF=root / "proof",
                RUNTIME=root / "runtime",
                SCRIPTS=root / "scripts/proof",
                TESTS=root / "tests/python",
                PACKAGES=root / "wasm/packages",
                HARNESS=root / "proof/harness/browser",
                REQUIREMENTS=root / "requirements",
                ASSETS=root / "artifacts/assets",
                SITE=root / "artifacts/site",
                REFERENCE=root / "artifacts/reference",
                CACHE=root / ".cache/science",
            ):
                expected = provenance.browser_fingerprint("2")
                for name in [
                    "scripts/proof/check_assets.py",
                    "tests/python/test_unrelated.py",
                    "scripts/proof/browser_unit4.py",
                ]:
                    (root / name).write_text("unrelated checker change")
                    self.assertEqual(provenance.browser_fingerprint("2"), expected)
                for name in [
                    "artifacts/site/unit2/app.json",
                    "runtime/startup-status.js",
                    "scripts/proof/browser_unit2.py",
                    "scripts/proof/build.py",
                    "scripts/proof/reference_corrections.py",
                ]:
                    (root / name).write_text("relevant dependency change")
                    self.assertNotEqual(provenance.browser_fingerprint("2"), expected)
                    (root / name).write_text("original")
                with self.assertRaises(ValueError):
                    provenance.browser_fingerprint("unknown")


if __name__ == "__main__":
    unittest.main()
