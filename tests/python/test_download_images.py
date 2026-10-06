"""Archive integrity and interrupted-range recovery against a local server."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
import download_images


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = bytes(range(256)) * 1100
        self.ranges = []
        self.ignore = False
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(parent.payload)))
                self.end_headers()

            def do_GET(self):
                value = self.headers["Range"]
                parent.ranges.append(value)
                first, last = map(int, value.removeprefix("bytes=").split("-"))
                self.send_response(200 if parent.ignore else 206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(parent.payload)}")
                self.end_headers()
                self.wfile.write(parent.payload[first : last + 1])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/data"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def fetch(self, expected=None):
        with (
            patch.object(download_images, "CACHE", self.root / "cache"),
            patch.object(download_images.time, "sleep", lambda _: None),
        ):
            download_images.fetch(
                self.url,
                "archive.bin",
                expected or hashlib.md5(self.payload).hexdigest(),
                chunk_bytes=128 * 1024,
                workers=2,
            )

    def test_resume_partial_and_reuse_complete_range(self):
        chunks = self.root / "cache/torchvision/archive.bin.chunks"
        chunks.mkdir(parents=True)
        step = 128 * 1024
        (chunks / "0").write_bytes(self.payload[:step])
        (chunks / f"{step}.part").write_bytes(self.payload[step : step + 90000])
        self.fetch()
        self.assertIn(f"bytes={step + 90000}-{2 * step - 1}", self.ranges)
        self.assertFalse(any(value.startswith("bytes=0-") for value in self.ranges))
        self.assertEqual((self.root / "cache/torchvision/archive.bin").read_bytes(), self.payload)

    def test_ignored_range_is_rejected(self):
        self.ignore = True
        with self.assertRaisesRegex(ValueError, "exact byte range"):
            self.fetch()
        self.assertFalse((self.root / "cache/torchvision/archive.bin").exists())

    def test_bad_checksum_never_replaces_destination(self):
        destination = self.root / "cache/torchvision/archive.bin"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous file")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.fetch("0" * 32)
        self.assertEqual(destination.read_bytes(), b"previous file")
