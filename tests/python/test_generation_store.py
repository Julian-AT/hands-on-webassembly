"""Exercise real HTTP publication, with added, removed and changed files."""

import sys
from pathlib import Path
import tempfile
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/proof"))
from generation_store import GenerationStore
from owned_preview import start


class GenerationStoreTests(unittest.TestCase):
    def test_atomic_complete_tree_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            (before / "index.html").write_text("old")
            (after / "index.html").write_text("new")
            (before / "removed.txt").write_text("removed")
            (after / "nested").mkdir()
            (after / "nested/added.txt").write_text("added")
            store = GenerationStore(root / "served", [before, after])
            server, identity = start(store.current, 0, dynamic_root=True)
            store.retain_origin_marker(identity)
            base = f"http://127.0.0.1:{identity['port']}"

            def read(path):
                with urlopen(base + path, timeout=5) as response:
                    return response.read()

            try:
                for index in [0, 1] * 5:
                    store.publish(index)
                    self.assertEqual(read("/"), b"old" if index == 0 else b"new")
                    self.assertEqual(read("/" + identity["marker"]), identity["identity"].encode())
                    present = "/removed.txt" if index == 0 else "/nested/added.txt"
                    missing = "/nested/added.txt" if index == 0 else "/removed.txt"
                    self.assertTrue(read(present))
                    with self.assertRaises(HTTPError) as error:
                        read(missing)
                    self.assertEqual(error.exception.code, 404)
                self.assertEqual((before / "index.html").read_text(), "old")
                self.assertEqual((after / "index.html").read_text(), "new")
                self.assertEqual(
                    (store.trees[0] / "index.html").stat().st_ino,
                    (before / "index.html").stat().st_ino,
                )
            finally:
                server.terminate()
                server.wait(timeout=10)

    def test_failed_attempts_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            GenerationStore(root / "served", [source])
            with self.assertRaises(FileExistsError):
                GenerationStore(root / "served", [source])


if __name__ == "__main__":
    unittest.main()
