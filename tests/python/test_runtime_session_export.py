"""Check bridge assembly against actual pinned upstream bootstrap bytes."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/proof"))
import patch_runtime
from common import sha


class RuntimeSessionExportTests(unittest.TestCase):
    def test_checked_upstream_patch_and_versioned_assignment_handshakes(self):
        upstream = ROOT / "tests/fixtures/shinylive-0.10.15"
        if not upstream.is_dir():
            self.skipTest("Requires the pinned upstream Shinylive 0.10.15 assets")
        with tempfile.TemporaryDirectory() as temporary:
            proof = Path(temporary)
            site = proof / "site"
            (proof / "runtime").symlink_to(ROOT / "runtime", target_is_directory=True)
            for name in (
                "shinylive-sw.js",
                "shinylive/load-shinylive-sw.js",
                "shinylive/shinylive.js",
            ):
                destination = site / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(upstream / name, destination)
            (proof / "assets.lock.json").write_text("{}")
            for name in (
                "shinylive/pyodide/pyodide-lock.json",
                "assets/v1/neural-runtime/manifest.json",
            ):
                path = site / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            shutil.copy2(
                ROOT / "runtime/course-training-worker.js",
                site / "shinylive/course-training-worker.js",
            )
            template = (
                (upstream / "export_template/index.html")
                .read_text()
                .replace("{{REL_PATH}}", "../")
                .replace("{{APP_ENGINE}}", "python")
            )
            for unit in range(1, 8):
                (site / f"unit{unit}").mkdir()
                (site / f"unit{unit}/index.html").write_text(template)
                (site / f"unit{unit}/app.json").write_text(json.dumps({"unit": unit}))
            with patch.multiple(
                patch_runtime,
                PROOF=proof,
                SITE=site,
                RUNTIME=ROOT / "runtime",
                EVIDENCE=proof / "evidence",
            ):
                patch_runtime.patch_runtime()
                patch_runtime.version_runtime()
            manifest = json.loads((site / "assignment-sessions.json").read_text())
            self.assertEqual(manifest["protocol"], 1)
            self.assertEqual(set(manifest["sources"]), {str(unit) for unit in range(1, 8)})
            for unit in range(1, 8):
                html = (site / f"unit{unit}/index.html").read_text()
                identity = re.search(
                    r"window.courseAssignmentIdentity = Object.freeze\(([^\n]+)\);", html
                )
                self.assertIsNotNone(identity)
                self.assertEqual(
                    json.loads(identity[1]),
                    dict(
                        unit=unit,
                        build_id=manifest["build_id"],
                        source_id=sha(site / f"unit{unit}/app.json"),
                    ),
                )
                self.assertIn(f"?v={manifest['build_id']}", html)
                self.assertLess(
                    html.index("window.courseSessionBridge ="), html.index("await runExportedApp")
                )
                script = re.search(r'<script type="module">(.*?)</script>', html, re.S)[1]
                result = subprocess.run(
                    ["node", "--input-type=module", "--check"],
                    input=script,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            client = (site / "shinylive/shinylive.js").read_text()
            self.assertIn("registerDisposer(() => this.pyWorker.terminate())", client)
            self.assertIn("registerDisposer(dispose)", client)
            for name in ("shinylive/shinylive.js", "shinylive-sw.js"):
                result = subprocess.run(
                    ["node", "--input-type=module", "--check"],
                    input=(site / name).read_text(),
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
