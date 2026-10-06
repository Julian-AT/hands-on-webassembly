"""History verification rejects corruption, missing dependencies and unsafe extraction."""

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "history_archive", ROOT / "scripts/archive/history.py"
)
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)


class HistoryArchiveTests(unittest.TestCase):
    def fixture(self, root, data=b"proof", member=None, declared=None):
        digest = hashlib.sha256(data).hexdigest()
        shard = root / "history-001.tar.gz"
        with tarfile.open(shard, "w:gz") as archive:
            entry = tarfile.TarInfo(member or "objects/" + digest)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        inventory = {
            "files": {
                "proof/evidence/report.json": {"bytes": len(data), "sha256": digest, "mode": 0o644}
            },
            "shards": [
                {
                    "name": shard.name,
                    "bytes": shard.stat().st_size,
                    "sha256": history.sha(shard),
                    "objects": [declared or digest],
                }
            ],
        }
        path = root / "inventory.json.gz"
        with gzip.open(path, "wt") as stream:
            json.dump(inventory, stream)
        record = {
            "old_main": "historical",
            "inventory": {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": history.sha(path),
            },
            "shards": [{k: v for k, v in inventory["shards"][0].items() if k != "objects"}],
        }
        index = root / "index.json"
        index.write_text(json.dumps(record))
        return index, shard

    def test_restore_recreates_exact_original_path_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, _ = self.fixture(root)
            result = history.verify(index, root, root / "restored")
            self.assertEqual(result["status"], "pass")
            self.assertEqual((root / "restored/proof/evidence/report.json").read_bytes(), b"proof")

    def test_corrupt_or_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, shard = self.fixture(root)
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Missing or corrupt"):
                history.verify(index, root)
            shard.unlink()
            with self.assertRaisesRegex(ValueError, "Missing or corrupt"):
                history.verify(index, root)

    def test_missing_dependency_and_unsafe_member_fail_closed(self):
        for options in ({"declared": "0" * 64}, {"member": "../escape"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                index, _ = self.fixture(root, **options)
                with self.assertRaises(ValueError):
                    history.verify(index, root, root / "restored")
                self.assertFalse((root / "escape").exists())

    def test_restoration_does_not_follow_existing_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, _ = self.fixture(root)
            outside = root / "outside"
            outside.mkdir()
            destination = root / "restored"
            destination.mkdir()
            (destination / "proof").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                history.verify(index, root, destination)
            self.assertEqual(list(outside.iterdir()), [])

    def test_conflicting_existing_bytes_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, _ = self.fixture(root)
            target = root / "restored/proof/evidence/report.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                history.verify(index, root, root / "restored")
            self.assertEqual(target.read_bytes(), b"different")


if __name__ == "__main__":
    unittest.main()
