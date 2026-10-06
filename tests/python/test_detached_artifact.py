import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from detached_artifact import detach_shared_artifact


class DetachedArtifactTests(unittest.TestCase):
    def test_rebuild_cannot_modify_the_retained_candidate_or_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "retained.py"
            source.write_bytes(b"original candidate")
            site = root / "site"
            (site / "probes").mkdir(parents=True)
            destination = site / "probes/body.py"
            os.link(source, destination)
            (site / "unique.js").write_bytes(b"already detached")
            self.assertEqual(detach_shared_artifact(site), ["probes/body.py"])
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            destination.write_bytes(b"next generation")
            self.assertEqual(source.read_bytes(), b"original candidate")
            self.assertEqual((site / "unique.js").read_bytes(), b"already detached")


if __name__ == "__main__":
    unittest.main()
