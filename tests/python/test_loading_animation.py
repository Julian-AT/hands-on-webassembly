"""The canonical generator preserves the approved loading presentation."""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class LoadingAnimationTests(unittest.TestCase):
    def test_canonical_generator_and_seven_immutable_overrides(self):
        record = json.loads((ROOT / "proof/archives/generator-relocation.json").read_text())
        source = (ROOT / record["current_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), record["current_sha256"])
        self.assertIn(b'<section id="course-startup" hidden role="status"', source)
        manifest = json.loads((ROOT / "manifests/startup-ui.json").read_text())
        self.assertEqual(set(manifest["files"]), {f"unit{unit}/index.html" for unit in range(1, 8)})
        self.assertEqual(record["historical_patch"], manifest["reference_source_patch"])


if __name__ == "__main__":
    unittest.main()
