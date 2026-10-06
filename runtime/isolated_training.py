"""Session-owned training contexts with isolated random state and cancellation.

The numerical body is generated verbatim from the shared corrected application.
Native execution uses a spawned process; Emscripten uses a dedicated Worker.
"""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import uuid
import cloudpickle

PROTOCOL = 2


def _torch():
    if sys.platform == 'emscripten':
        from browser_torch import torch
    else:
        import torch
    return torch


def _snapshot(params):
    result = dict(params)
    # Native loaders retain a local worker_init_fn even with num_workers=0.
    # It never executes in this mode and need not retain a UI module closure.
    result['loaders'] = tuple(copy.copy(loader) if loader is not None else None
                              for loader in params['loaders'])
    for loader in result['loaders']:
        if loader is not None:
            if getattr(loader, 'num_workers', 0) != 0:
                raise ValueError('Isolated course training requires num_workers=0')
            loader.worker_init_fn = None
            if hasattr(loader, '_iterator'): loader._iterator = None
    if 'bundle' in result:
        result['bundle'] = dict(result['bundle'])
        for name, loader in zip(('train','val','test'), result['loaders']):
            result['bundle'][name] = loader
    return cloudpickle.dumps(result)


def _generator_states(loaders):
    records = []
    for loader in loaders:
        generator = None if loader is None else getattr(loader, 'generator', None)
        if generator is None:
            records.append(None)
        elif hasattr(generator, 'get_state'):
            records.append(('torch', generator.get_state().cpu().numpy().copy()))
        else:
            records.append(('borch', generator._rng.seed, generator._rng.state.get_state()))
    return records


def _commit_generators(loaders, records):
    torch = _torch()
    for loader, record in zip(loaders, records):
        if record is None: continue
        generator = loader.generator
        if record[0] == 'torch':
            generator.set_state(torch.tensor(record[1], dtype=torch.uint8))
        else:
            generator._rng.seed = record[1]
            generator._rng.state.set_state(record[2])


def _prepare(source, payload, notify):
    import math
    import random
    import types
    import typing
    import dataclasses
    import numpy as np
    import pandas as pd
    torch = _torch()
    torch.set_num_threads(1)
    module = types.ModuleType('_course_training_body')
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, nn=torch.nn, np=np, pd=pd, json=json,
        math=math, random=random, copy=copy, asyncio=asyncio, dataclass=dataclasses.dataclass,
        DataLoader=torch.utils.data.DataLoader, _q_put=notify)
    module.__dict__.update({name:getattr(typing,name) for name in
                           ('Any','List','Dict','Tuple','Optional','Union')})
    exec(source, module.__dict__)
    params = cloudpickle.loads(payload)
    shapes = []
    if hasattr(module, 'build_model'):
        build = module.build_model
        def observe_build(specs, shape):
            shapes.append(tuple(shape))
            return build(specs, shape)
        module.build_model = observe_build
    return module, params, shapes


def _serialize(result, params, shapes):
    if 'model' in result:
        model = result.pop('model')
        result['model_record'] = dict(shape=shapes[-1], training=model.training,
            state={name:value.detach().cpu().numpy().copy() for name,value in model.state_dict().items()})
    result['generator_states'] = _generator_states(params['loaders'])
    return cloudpickle.dumps(result)


def execute_payload(source, payload, notify):
    """Run the complete original body without cooperative scheduling on native."""
    module,params,shapes=_prepare(source,payload,notify)
    if hasattr(module, 'run_training_async'):
        async def flush():
            pass
        module._course_checkpoint = flush
        result = asyncio.run(module.run_training_async(params))
    else:
        result = module.run_training_sync(params)
    return _serialize(result,params,shapes)


async def execute_payload_async(source, payload, notify, cancelled):
    """Yield after batch updates; arithmetic, data and RNG calls are unchanged."""
    if 'async def run_training_async(' in source:
        if 'async def _train(' in source:
            validation='                    val_sum += vloss.item()'
            if source.count(validation)!=1:
                raise ValueError('Unit 6 validation cancellation source changed')
            source=source.replace(validation,validation+'\n                    await _course_checkpoint()')
        elif 'async def _mn_train(' not in source:
            raise ValueError('Unsupported asynchronous training calculation')
        module,params,shapes=_prepare(source,payload,notify)
        async def checkpoint():
            await asyncio.sleep(.001)
            if cancelled():raise asyncio.CancelledError()
        module._course_checkpoint=checkpoint
        return _serialize(await module.run_training_async(params),params,shapes)
    declaration='def run_training_sync('
    update='            optimizer.step()'
    validation='                    nval += int(xv.size(0))'
    if any(source.count(marker)!=1 for marker in (declaration,update,validation)):
        raise ValueError('Training cancellation source contract changed')
    source=source.replace(declaration,'async def run_training_sync(')
    source=source.replace(update,update+'\n            await _checkpoint()')
    source=source.replace(validation,validation+'\n                    await _checkpoint()')
    module,params,shapes=_prepare(source,payload,notify)
    async def checkpoint():
        # sleep(0) only schedules Python microtasks. A timed turn also admits
        # incoming Worker cancellation messages during synchronous WASM work.
        await asyncio.sleep(.001)
        if cancelled():raise asyncio.CancelledError()
    module._checkpoint=checkpoint
    return _serialize(await module.run_training_sync(params),params,shapes)


def _process_entry(connection, tags, source, payload):
    try:
        def notify(message): connection.send(('progress', tags, cloudpickle.dumps(message)))
        result = execute_payload(source, payload, notify)
        connection.send(('result', tags, result))
    except BaseException as error:
        connection.send(('error', tags, f'{type(error).__name__}: {error}'))
    finally:
        connection.close()


class Executor:
    def __init__(self, unit):
        if unit not in (5,6,7): raise ValueError('Unsupported isolated training assignment')
        self.unit = unit
        config = json.loads(Path(__file__).with_name('course-training.json').read_text())
        self.source = config['source']
        self.source_id = hashlib.sha256(self.source.encode()).hexdigest()
        self.build_id = config['build_id']
        self.session_id = uuid.uuid4().hex
        self.generation = 0
        self.active = None
        self._browser_future = None

    def cancel(self):
        self.generation += 1
        active, self.active = self.active, None
        future, self._browser_future = self._browser_future, None
        if future is not None and not future.done(): future.cancel()
        if active is None: return
        if sys.platform == 'emscripten':
            from js import Function
            from pyodide.ffi import to_js
            # A blocked WASM worker takes ~2s to hard-terminate in Chrome.
            # First close it on its next batch turn; retain a hard-stop fallback.
            Function.new('worker','tags',
                "worker.postMessage({kind:'cancel',tags});setTimeout(()=>worker.terminate(),1000)")(
                    active,to_js(self.active_tags,dict_converter=__import__('js').Object.fromEntries))
        else:
            if active.is_alive(): active.terminate()
            active.join(.2)
            if active.is_alive():
                active.kill()
                active.join(.2)

    async def run(self, params, notify):
        self.cancel()
        generation = self.generation
        tags = dict(protocol=PROTOCOL, build_id=self.build_id, source_id=self.source_id,
                    session_id=self.session_id, task_id=generation, unit=self.unit,
                    operation='training', dataset_generation=params.get('dataset_generation', 0),
                    model_generation=params.get('model_generation', 0))
        payload = _snapshot(params)
        if len(payload) > 2 * 1024**3:
            raise ValueError('Training data exceeds the supported memory budget. Reduce the job and retry.')
        try:
            if sys.platform == 'emscripten':
                result = await self._browser(tags, payload, notify)
            else:
                result = await self._native(tags, payload, notify)
            if self.generation != generation: raise asyncio.CancelledError()
            restored = cloudpickle.loads(result)
            _commit_generators(params['loaders'], restored.pop('generator_states'))
            return restored
        finally:
            if self.generation == generation: self.cancel()

    async def _native(self, tags, payload, notify):
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_process_entry,args=(child,tags,self.source,payload),daemon=True)
        process.start()
        child.close()
        self.active = process
        try:
            while self.generation == tags['task_id']:
                if parent.poll():
                    kind, actual, value = parent.recv()
                    if actual != tags: continue
                    if kind == 'progress': notify(cloudpickle.loads(value))
                    elif kind == 'result': return value
                    elif kind == 'error': raise RuntimeError(value)
                elif not process.is_alive():
                    raise RuntimeError('Training process stopped unexpectedly. Retry training.')
                await asyncio.sleep(.01)
            raise asyncio.CancelledError()
        finally:
            parent.close()

    async def _browser(self, tags, payload, notify):
        from js import Worker, location
        from urllib.parse import urljoin
        from pyodide.ffi import create_proxy, to_js
        worker = Worker.new(urljoin(str(location.href), 'course-training-worker.js')+'?v='+self.build_id)
        self.active = worker
        self.active_tags = tags
        future = asyncio.get_running_loop().create_future()
        self._browser_future = future
        def message(event):
            data = event.data
            actual = data.tags.to_py()
            if actual != tags or self.generation != tags['task_id'] or future.done(): return
            if data.kind == 'progress': notify(cloudpickle.loads(data.payload.to_py().tobytes()))
            elif data.kind == 'result': future.set_result(data.payload.to_py().tobytes())
            elif data.kind == 'error': future.set_exception(RuntimeError(str(data.error)))
        def failed(event):
            if not future.done(): future.set_exception(RuntimeError(str(event.message) or 'Training worker failed. Retry training.'))
        listener, error_listener = create_proxy(message), create_proxy(failed)
        worker.addEventListener('message', listener)
        worker.addEventListener('error', error_listener)
        try:
            worker.postMessage(to_js(dict(tags=tags, source=self.source, payload=payload), dict_converter=__import__('js').Object.fromEntries))
            return await future
        finally:
            if self._browser_future is future: self._browser_future = None
            worker.removeEventListener('message', listener)
            worker.removeEventListener('error', error_listener)
            listener.destroy()
            error_listener.destroy()


def restore_model(record, build_model, specs):
    torch = _torch()
    model = build_model(specs, tuple(record['shape']))
    current = model.state_dict()
    model.load_state_dict({name:torch.tensor(value, dtype=current[name].dtype)
                           for name,value in record['state'].items()})
    model.train(record['training'])
    return model
