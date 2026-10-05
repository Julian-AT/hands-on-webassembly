"""Abortable complete model preparation; only the current owner may commit."""
import asyncio
import sys
import uuid
import numpy as np


class Rows:
    """Keep the worker-validated full vocabulary in JavaScript, without a bulk copy."""
    def __init__(self,values,count):self.values,self.count=values,count
    def __len__(self):return self.count
    def __contains__(self,key):
        from js import Object
        return bool(Object.hasOwn(self.values,key))
    def __getitem__(self,key):
        from js import Reflect
        if key not in self:raise KeyError(key)
        return int(Reflect.get(self.values,key))


class Owner:
    def __init__(self,model,config):
        self.model,self.config=model,dict(config)
        self.session_id=uuid.uuid4().hex
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

    async def prepare(self,notify=None):
        self.cancel()
        tags=dict(protocol=1,unit=2,operation='embedding-loading',build_id=self.config['build_id'],
                  source_id=self.config['source_id'],session_id=self.session_id,task_id=self.generation,
                  dataset_generation=0,model_generation=0)
        if self.model._vectors is not None and self.model._tokenizer_loaded:
            return dict(tags=tags,prepared=True)
        if sys.platform!='emscripten':
            raise RuntimeError('This preparation owner requires the browser worker; native controls load local assets.')
        from js import Worker,location,Object
        from urllib.parse import urljoin
        from pyodide.ffi import create_proxy,to_js
        worker=Worker.new(urljoin(str(location.href),'course-embedding-worker.js')+'?v='+tags['build_id'])
        self.active,self.active_tags=worker,tags
        future=asyncio.get_running_loop().create_future();self.future=future
        acknowledgement=asyncio.get_running_loop().create_future()
        def message(event):
            data=event.data
            if data.tags.to_py()!=tags:return
            if data.kind=='cancelled':
                if tags['task_id'] in self.cancelled_tasks and not acknowledgement.done():
                    acknowledgement.set_result(asyncio.get_running_loop().time())
                return
            if self.generation!=tags['task_id'] or future.done():return
            if data.kind=='progress':
                if notify:notify(data.payload.to_py())
            elif data.kind=='error':future.set_exception(RuntimeError(str(data.error)))
            elif data.kind=='result':future.set_result(data)
        def failed(event):
            if self.generation==tags['task_id'] and not future.done():
                future.set_exception(RuntimeError(str(event.message) or 'Embedding worker failed. Retry loading.'))
        listener,error_listener=create_proxy(message),create_proxy(failed)
        worker.addEventListener('message',listener);worker.addEventListener('error',error_listener)
        try:
            worker.postMessage(to_js(dict(kind='prepare',tags=tags,manifest_sha256=self.config['manifest_sha256']),
                                     dict_converter=Object.fromEntries))
            result=await future
            if self.generation!=tags['task_id']:raise asyncio.CancelledError()
            vectors=np.frombuffer(result.vectors.to_py().tobytes(),dtype='<f4').reshape(20000,300)
            rows=Rows(result.rows,514157)
            # No await between the final owner check and installation of the set.
            if self.generation!=tags['task_id']:raise asyncio.CancelledError()
            self.model.install_prepared(str(result.tokenizer),rows,vectors)
            return dict(tags=tags,prepared=True)
        finally:
            if tags['task_id'] in self.cancelled_tasks:
                disposal_started=asyncio.get_running_loop().time()
                acknowledged=False
                try:
                    await asyncio.wait_for(asyncio.shield(acknowledgement),timeout=1)
                    acknowledged=True
                except (asyncio.TimeoutError,asyncio.CancelledError):
                    pass
                self.last_disposal=dict(tags=tags,acknowledged=acknowledged,
                    seconds=asyncio.get_running_loop().time()-disposal_started)
                self.cancelled_tasks.discard(tags['task_id'])
            worker.removeEventListener('message',listener);worker.removeEventListener('error',error_listener)
            listener.destroy();error_listener.destroy();worker.terminate()
            if self.future is future:self.future=None
            if self.active is worker:self.active=None
