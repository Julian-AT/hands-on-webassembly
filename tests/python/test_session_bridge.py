"""Exercise launcher/assignment ownership, disposal and failure recovery together."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/proof"))
from patch_runtime import session_startup_script


class SessionBridgeTests(unittest.TestCase):
    def test_launcher_and_child_protocol(self):
        result = subprocess.run(
            ["node", "proof/harness/browser/session-bridge-harness.mjs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["cases"]), 9)

    def test_staged_identity_precedes_startup_and_is_assignment_specific(self):
        for unit in range(1, 8):
            source = session_startup_script(unit, "build-id", f"bundle-{unit}")
            self.assertTrue(source.startswith("window.courseAssignmentIdentity = Object.freeze("))
            self.assertIn(f'"unit":{unit}', source)
            self.assertIn(f'"source_id":"bundle-{unit}"', source)
            self.assertLess(
                source.index("window.courseSessionBridge ="), source.index("const startupPanel =")
            )


if __name__ == "__main__":
    unittest.main()
