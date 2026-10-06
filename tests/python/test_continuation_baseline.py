import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from common import sha
from verify_continuation_baseline import repair, verify


class BaselineTests(unittest.TestCase):
    def test_reference_extension_uses_verified_copy_but_original_sources_cannot_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in (
                "proof/reference/unit6/app.py",
                "assignments/6/app.py",
                "proof/native/fixture.npz",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original")
                files[name] = dict(
                    binding=name, storage="checksum-bound", bytes=8, sha256=sha(path)
                )
                saved = root / "proof/continuation-baseline" / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(path.read_bytes())
                path.write_bytes(b"extended")
            self.assertEqual(
                verify(root, {"files": files}), ["assignments/6/app.py", "proof/native/fixture.npz"]
            )
            (root / "proof/continuation-baseline/proof/reference/unit6/app.py").write_bytes(
                b"tampered"
            )
            self.assertEqual(verify(root, {"files": files}), list(files))

    def test_preserved_bytes_are_authoritative_and_original_manifest_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "proof/continuation-baseline"
            saved = baseline / "proof/runtime/http.js"
            saved.parent.mkdir(parents=True)
            saved.write_text("incumbent")
            source = root / "proof/runtime/http.js"
            source.parent.mkdir(parents=True)
            source.write_text("candidate")
            environment = baseline / "native-environment.txt"
            environment.write_text("pinned")
            manifest = baseline / "manifest.json"
            manifest.write_text(
                json.dumps(
                    dict(
                        environment_sha256=sha(environment),
                        files={
                            "proof/runtime/http.js": dict(
                                sha256=sha(source), bytes=source.stat().st_size
                            )
                        },
                    )
                )
            )
            original = manifest.read_bytes()
            record = repair(root, baseline)
            self.assertEqual(record["status"], "pass")
            self.assertEqual(record["files"]["proof/runtime/http.js"]["sha256"], sha(saved))
            self.assertEqual(verify(root, record), [])
            self.assertEqual(manifest.read_bytes(), original)
            saved.write_text("tampered")
            self.assertEqual(verify(root, record), ["proof/runtime/http.js"])

    def test_changed_uncopied_inputs_do_not_get_new_expected_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "proof/continuation-baseline"
            baseline.mkdir(parents=True)
            environment = baseline / "native-environment.txt"
            environment.write_text("pinned")
            (root / "original.py").write_text("changed")
            (baseline / "manifest.json").write_text(
                json.dumps(
                    dict(
                        environment_sha256=sha(environment),
                        files={"original.py": dict(sha256="old", bytes=3)},
                    )
                )
            )
            self.assertEqual(repair(root, baseline)["status"], "fail")

    def test_archived_original_evidence_remains_bound_after_report_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "proof/evidence/report.json"
            current.parent.mkdir(parents=True)
            current.write_text("original report")
            record = {
                "files": {
                    "proof/evidence/report.json": dict(
                        binding="proof/evidence/report.json",
                        storage="checksum-bound",
                        bytes=current.stat().st_size,
                        sha256=sha(current),
                    )
                }
            }
            history = current.parent / "history"
            history.mkdir()
            (history / "report-timestamp.json").write_bytes(current.read_bytes())
            current.write_text("new report")
            self.assertEqual(verify(root, record), [])
            (history / "report-timestamp.json").write_text("wrong report")
            self.assertEqual(verify(root, record), ["proof/evidence/report.json"])
