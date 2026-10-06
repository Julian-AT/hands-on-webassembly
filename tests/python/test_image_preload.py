import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
import image_preload
import image_data


class ImagePreloadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous = dict(image_data._cache)
        image_data._cache.clear()
        self.addCleanup(
            lambda: (image_data._cache.clear(), image_data._cache.update(self.previous))
        )
        self.assets = {}
        entries = {}
        for split, count in [("train", 3), ("test", 2)]:
            payload = io.BytesIO()
            np.savez_compressed(
                payload,
                images=np.arange(count * 6, dtype=np.uint8).reshape(count, 2, 3),
                labels=np.arange(count, dtype=np.int64),
            )
            raw = payload.getvalue()
            self.assets[f"{split}.npz"] = raw
            entries[f"MNIST/{split}"] = dict(
                file=f"{split}.npz",
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                image_shape=[count, 2, 3],
                count=count,
                label_dtype="int64",
                layout="NHW",
            )
        self.assets["manifest.json"] = json.dumps({"datasets": entries}).encode()
        self.owner = image_preload.Owner("build", "source", "session")

    async def fetch(self, name, tags):
        await asyncio.sleep(0)
        self.assertEqual(tags["build_id"], "build")
        return self.assets[name]

    async def test_pair_is_installed_atomically_and_compact(self):
        events = []
        result = await self.owner.preload("MNIST", fetch_asset=self.fetch, notify=events.append)
        self.assertEqual(result["splits"], ["train", "test"])
        self.assertEqual(set(image_data._cache), {"MNIST/train", "MNIST/test"})
        self.assertEqual(image_data.load("MNIST", "train")[0].dtype, np.uint8)
        self.assertEqual(len(events), 2)

    async def test_corrupt_download_does_not_install_partial_pair_and_retry_works(self):
        valid = self.assets["test.npz"]
        self.assets["test.npz"] = valid[:-1]
        with self.assertRaisesRegex(ValueError, "checksum"):
            await self.owner.preload("MNIST", fetch_asset=self.fetch)
        self.assertEqual(image_data._cache, {})
        self.assets["test.npz"] = valid
        await self.owner.preload("MNIST", fetch_asset=self.fetch)
        self.assertEqual(len(image_data._cache), 2)

    async def test_cancel_acknowledges_without_cache_or_obsolete_error(self):
        started = asyncio.Event()

        async def blocked(name, tags):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                # Even a transport that turns cancellation into an error must be rejected as obsolete.
                raise OSError("obsolete network error")

        task = asyncio.create_task(self.owner.preload("MNIST", fetch_asset=blocked))
        await started.wait()
        self.owner.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(image_data._cache, {})

    async def test_replacement_rejects_transport_that_ignores_cancellation(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def stubborn(name, tags):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return self.assets[name]

        obsolete = asyncio.create_task(self.owner.preload("MNIST", fetch_asset=stubborn))
        await started.wait()
        current = await self.owner.preload("MNIST", fetch_asset=self.fetch)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await obsolete
        self.assertEqual(current["tags"]["task_id"], self.owner.generation)
        self.assertEqual(len(image_data._cache), 2)

    async def test_wrong_manifest_build_is_recoverable(self):
        with self.assertRaisesRegex(ValueError, "another build"):
            await self.owner.preload("MNIST", fetch_asset=self.fetch, manifest_sha256="wrong")
        self.assertEqual(image_data._cache, {})
