// Isolated verification only: production application entry points are separate.
importScripts('./pyodide/pyodide.js');
self.onmessage = async ({data}) => {
  const send = value => self.postMessage(value);
  try {
    const py = await loadPyodide({indexURL:'./pyodide/'});
    await py.loadPackage(['numpy','pillow']);
    for (const [name, expected] of Object.entries(data.files)) {
      const response = await fetch('./'+name,{cache:'no-store'});
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
      const raw = new Uint8Array(await response.arrayBuffer());
      const actual = [...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(x=>x.toString(16).padStart(2,'0')).join('');
      if (actual !== expected) throw new Error(`${name}: source checksum changed`);
      py.FS.writeFile(name,raw);
    }
    py.globals.set('manifest_digest',data.manifest_sha256);
    py.globals.set('probe_build_id',data.build_id);
    py.globals.set('probe_source_id',data.files['image_preload.py']);
    const result = await py.runPythonAsync(`
import asyncio, json, time
import numpy as np
import image_data
from image_preload import Owner
owner = Owner(probe_build_id, probe_source_id, 'installed-browser-preload-probe')
cases = {}

async def fetch_case(name, tags, case):
    return await owner._fetch(name + '?loading-probe=' + case, tags)

# The browser harness delays this actual request. Python cancellation must
# abort the JavaScript response as well as settle the owning asyncio task.
task = asyncio.create_task(owner.preload('MNIST', fetch_asset=lambda n,t: fetch_case(n,t,'cancel'), manifest_sha256=manifest_digest))
await asyncio.sleep(.1)
started = time.perf_counter()
owner.cancel()
try:
    await task
    raise AssertionError('Cancelled image operation produced a result')
except asyncio.CancelledError:
    elapsed = time.perf_counter() - started
    assert elapsed <= 1 and not owner.controllers and not image_data._cache
    cases['cancel-download'] = dict(status='pass', seconds=elapsed, active_requests=len(owner.controllers), cache_entries=len(image_data._cache))

# The test response is corrupted after a valid train split. Neither split may
# be installed until both checksums and layouts have passed.
try:
    await owner.preload('MNIST', fetch_asset=lambda n,t: fetch_case(n,t,'corrupt'), manifest_sha256=manifest_digest)
    raise AssertionError('Corrupt dataset was accepted')
except ValueError as error:
    assert 'checksum' in str(error).lower() and not image_data._cache
    cases['corrupt-download'] = dict(status='pass', error=str(error), cache_entries=0)

for repetition in range(2):
    for dataset, counts in [('MNIST',(60000,10000)), ('FashionMNIST',(60000,10000))]:
        result = await owner.preload(dataset, manifest_sha256=manifest_digest)
        assert result['tags']['task_id'] == owner.generation
        assert result['tags']['source_id'] == probe_source_id
        observed = []
        for split, count in zip(('train','test'),counts):
            images, labels, layout = image_data.load(dataset,split)
            assert images.dtype == np.uint8 and len(images) == count and len(labels) == count
            observed.append(dict(split=split,count=count,shape=list(images.shape),dtype=str(images.dtype),layout=layout))
        cases[f'complete-data/{dataset}/repeat-{repetition+1}'] = dict(status='pass', tags=result['tags'], observed=observed)
json.dumps(dict(done=True,cases=cases))
`);
    send(JSON.parse(result));
  } catch (error) {send({error:String(error)});}
};
