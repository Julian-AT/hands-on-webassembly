importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage=async({data})=>{
  try{
    const start=performance.now();
    const py=await loadPyodide({indexURL:'../shinylive/pyodide/'});
    await py.loadPackage(['numpy','pandas','pillow','matplotlib','scipy','scikit-learn','seaborn','ipython','setuptools','course-native-neural','course-native-cnn']);
    const bytes=async name=>{
      const response=await fetch(name);
      if(!response.ok)throw new Error(name+': '+response.status);
      return new Uint8Array(await response.arrayBuffer());
    };
    py.unpackArchive(await bytes('pyborch-1.14.1-py3-none-any.whl'),'zip',{extractDir:'/home/pyodide'});
    const full=data==='unit5full';
    const fixture=full?'unit5-full-native.npz':'unit5-native.npz';
    for(const name of ['browser_torch.py','neural_compat.py','cnn_compat.py','module_hooks.py','image_transforms.py','image_data.py','image_preload.py','image_worker_preload.py','torch_rng.py','borch_compat.py','unit5_probe.py','u5_utils.py','unit5-definitions.py','unit5-data.csv',fixture,'course-probe-image-loading.json'])py.FS.writeFile(name,await bytes(name));
    py.globals.set('full',full);py.globals.set('fixture',fixture);
    const result=await py.runPythonAsync(`
import json,numpy as np,u5_utils
from browser_torch import torch
from unit5_probe import run_unit5,compare
if full:
    from image_worker_preload import Owner
    config=json.load(open('course-probe-image-loading.json'))
    owner=Owner(5,config['build_id'],config['source_id'],worker_url='../shinylive/course-image-worker.js')
    try:
        for name in ('MNIST','FashionMNIST'):
            await owner.preload(name,manifest_sha256=config['manifest_sha256'])
    finally:
        owner.cancel()
arrays=await run_unit5(torch,u5_utils,open('unit5-definitions.py').read(),'unit5-data.csv',full)
json.dumps({'comparisons':compare(arrays,np.load(fixture)),'full_data':full})
`);
    self.postMessage({done:true,elapsed_ms:performance.now()-start,...JSON.parse(result)});
  }catch(error){self.postMessage({error:String(error)});}
};
