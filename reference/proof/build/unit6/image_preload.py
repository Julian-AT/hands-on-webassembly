"""Abortable image preparation with atomic, ownership-checked compact-array commits.

Dataset constructors remain synchronous. Callers must await this preparation
before constructing them, and cancel their owner on reset/replacement/session end.
"""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import uuid
import numpy as np


def unpack(payload, entry, key):
    if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
        raise ValueError(f'Dataset checksum or length mismatch: {key}. Retry loading.')
    with np.load(io.BytesIO(payload), allow_pickle=False) as packed:
        images, labels = packed['images'], packed['labels']
    if (images.dtype != np.uint8 or list(images.shape) != entry['image_shape']
            or len(images) != entry['count'] or labels.shape != (entry['count'],)
            or str(labels.dtype) != entry['label_dtype'] or labels.dtype.kind not in 'iu'
            or entry['layout'] not in ('NHW','NCHW','NHWC')):
        raise ValueError(f'Dataset shape, label or layout mismatch: {key}. Retry loading.')
    return images, labels, entry['layout']


class Owner:
    def __init__(self, build_id, source_id, session_id=None):
        if not build_id or not source_id:
            raise ValueError('Image loading requires build and source identities')
        self.build_id, self.source_id = build_id, source_id
        self.session_id = session_id or uuid.uuid4().hex
        self.generation = 0
        self.task = None
        # Pyodide's JavaScript proxies are unhashable. Retain controllers by
        # Python identity so cancellation can still abort every outstanding fetch.
        self.controllers = {}

    def cancel(self):
        self.generation += 1
        for controller in tuple(self.controllers.values()):
            controller.abort()
        self.controllers.clear()
        active, self.task = self.task, None
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if active is not None and active is not current:
            active.cancel()

    async def _fetch(self, name, tags):
        local = Path(__file__).parent/'image-assets'/name
        if local.exists():
            await asyncio.sleep(0)
            return local.read_bytes()
        if sys.platform != 'emscripten':
            raise FileNotFoundError(f'Dataset asset missing: {name}')
        from js import AbortController, Object, Uint8Array, fetch, location
        from pyodide.ffi import to_js
        from urllib.parse import urljoin
        controller = AbortController.new()
        token = id(controller)
        self.controllers[token] = controller
        try:
            url = urljoin(str(location.href), '../assets/v1/images/'+name)
            options = to_js(dict(signal=controller.signal, cache='no-cache'), dict_converter=Object.fromEntries)
            response = await fetch(url, options)
            if response.status != 200:
                raise OSError(f'Dataset asset request failed ({response.status}): {name}. Retry loading.')
            return Uint8Array.new(await response.arrayBuffer()).to_py().tobytes()
        finally:
            # Abort closes any response body still outstanding after cancellation or error.
            controller.abort()
            self.controllers.pop(token, None)

    async def preload(self, name, *, fetch_asset=None, manifest_sha256=None, notify=None):
        self.cancel()
        self.task = asyncio.current_task()
        generation = self.generation
        tags = dict(build_id=self.build_id, source_id=self.source_id,
                    session_id=self.session_id, task_id=generation, operation='image-loading')
        fetch_asset = fetch_asset or self._fetch
        def current():
            if self.generation != generation:
                raise asyncio.CancelledError()
        try:
            payload = await fetch_asset('manifest.json', tags)
            current()
            if manifest_sha256 and hashlib.sha256(payload).hexdigest() != manifest_sha256:
                raise ValueError('Dataset manifest belongs to another build. Reload and retry.')
            manifest = json.loads(payload)
            validated = {}
            for split in ('train', 'test'):
                key = f'{name}/{split}'
                entry = manifest['datasets'].get(key)
                if entry is None:
                    raise ValueError(f'The complete {key} dataset has not been packaged')
                raw = await fetch_asset(entry['file'], tags)
                current()
                validated[key] = unpack(raw, entry, key)
                del raw
                current()
                if notify:
                    notify(dict(tags=tags, dataset=name, split=split, status='validated'))
                # Let reset/replacement run before committing either split.
                await asyncio.sleep(0)
                current()
            import image_data
            # No await between the final ownership check and atomic installation.
            current()
            image_data._cache.update(validated)
            return dict(tags=tags, dataset=name, splits=['train','test'])
        except asyncio.CancelledError:
            raise
        except Exception:
            current()  # Obsolete failures must not surface in the replacement task.
            raise
        finally:
            if self.generation == generation:
                self.task = None
