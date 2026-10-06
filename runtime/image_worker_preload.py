"""Disposable image preparation candidate; compact synchronous consumers stay intact."""
import asyncio
import hashlib
from pathlib import Path
import sys
import uuid
import numpy as np


class Owner:
    def __init__(self,unit,build_id,source_id,session_id=None,*,worker_url=None):
        if unit not in (5,6,7) or not build_id or not source_id:
            raise ValueError('Image preparation requires assignment, build and source identities')
        self.unit,self.build_id,self.source_id=unit,build_id,source_id
        self.session_id=session_id or uuid.uuid4().hex
        self.worker_url=worker_url
        self.generation=0;self.active=None;self.future=None;self.active_tags=None
        self.cancelled_tasks=set();self.last_disposal=None

    def cancel(self):
        self.generation+=1
        if self.future is not None and not self.future.done():self.future.cancel()
        worker,self.active=self.active,None
        if worker is not None:
            self.cancelled_tasks.add(self.active_tags['task_id'])
            from js import Function,Object
            from pyodide.ffi import to_js
            Function.new('worker','tags',
                "worker.postMessage({kind:'cancel',tags});setTimeout(()=>worker.terminate(),1000)")(
                worker,to_js(self.active_tags,dict_converter=Object.fromEntries))

    async def preload(self,name,*,manifest_sha256,notify=None,dataset_generation=None,model_generation=0):
        self.cancel()
        tags=dict(protocol=1,unit=self.unit,operation='image-loading',build_id=self.build_id,
            source_id=self.source_id,session_id=self.session_id,task_id=self.generation,
            dataset_generation=self.generation if dataset_generation is None else dataset_generation,
            model_generation=model_generation)
        if sys.platform!='emscripten':raise RuntimeError('Use the original native loader outside the browser')
        from js import Worker,location,Object
        from urllib.parse import urljoin
        from pyodide.ffi import create_proxy,to_js
        worker=Worker.new(urljoin(str(location.href),self.worker_url or 'course-image-worker.js')+'?v='+self.build_id)
        self.active,self.active_tags=worker,tags
        future=asyncio.get_running_loop().create_future();self.future=future
        acknowledgement=asyncio.get_running_loop().create_future()
        def message(event):
            data=event.data
            if data.tags.to_py()!=tags:return
            if data.kind=='cancelled':
                if tags['task_id'] in self.cancelled_tasks and not acknowledgement.done():acknowledgement.set_result(True)
                return
            if self.generation!=tags['task_id'] or future.done():return
            if data.kind=='progress':
                if notify:notify(dict(tags=tags,**data.payload.to_py()))
            elif data.kind=='error':future.set_exception(RuntimeError(str(data.error)))
            elif data.kind=='result':future.set_result(data)
        def failed(event):
            if self.generation==tags['task_id'] and not future.done():future.set_exception(RuntimeError(str(event.message)))
        listener,error_listener=create_proxy(message),create_proxy(failed)
        worker.addEventListener('message',listener);worker.addEventListener('error',error_listener)
        unpack_source=Path(__file__).with_name('image_preload.py').read_text()
        try:
            worker.postMessage(to_js(dict(kind='prepare',tags=tags,dataset=name,
                manifest_sha256=manifest_sha256,unpack_source=unpack_source,
                unpack_sha256=hashlib.sha256(unpack_source.encode()).hexdigest()),dict_converter=Object.fromEntries))
            result=await future
            if self.generation!=tags['task_id']:raise asyncio.CancelledError()
            validated={}
            for split in result.splits:
                key=str(split.key);shape=list(split.shape.to_py())
                images=np.frombuffer(split.images.to_py().tobytes(),dtype=np.uint8).reshape(shape)
                labels=np.frombuffer(split.labels.to_py().tobytes(),dtype=str(split.label_dtype))
                if key not in (f'{name}/train',f'{name}/test') or key in validated or labels.shape!=(shape[0],):
                    raise ValueError('Image preparation returned an incomplete or contradictory set')
                validated[key]=(images,labels,str(split.layout))
            if set(validated)!={f'{name}/train',f'{name}/test'}:raise ValueError('Both complete dataset splits are required')
            import image_data
            if self.generation!=tags['task_id']:raise asyncio.CancelledError()
            image_data._cache.update(validated)
            return dict(tags=tags,dataset=name,splits=['train','test'])
        finally:
            if tags['task_id'] in self.cancelled_tasks:
                started=asyncio.get_running_loop().time();acknowledged=False
                try:
                    await asyncio.wait_for(asyncio.shield(acknowledgement),timeout=1);acknowledged=True
                except (asyncio.TimeoutError,asyncio.CancelledError):pass
                self.last_disposal=dict(tags=tags,acknowledged=acknowledged,seconds=asyncio.get_running_loop().time()-started)
                self.cancelled_tasks.discard(tags['task_id'])
            worker.removeEventListener('message',listener);worker.removeEventListener('error',error_listener)
            listener.destroy();error_listener.destroy();worker.terminate()
            if self.future is future:self.future=None
            if self.active is worker:self.active=None
