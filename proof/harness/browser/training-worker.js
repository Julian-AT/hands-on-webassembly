importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage=async()=>{
  try{
    const start=performance.now();
    const py=await loadPyodide({indexURL:'../shinylive/pyodide/'});
    await py.loadPackage(['numpy','pandas','pillow','course-native-neural','course-native-cnn']);
    const bytes=async name=>{
      const response=await fetch(name);
      if(!response.ok)throw new Error(name+': '+response.status);
      return new Uint8Array(await response.arrayBuffer());
    };
    py.unpackArchive(await bytes('pyborch-1.14.1-py3-none-any.whl'),'zip',{extractDir:'/home/pyodide'});
    for(const name of ['browser_torch.py','neural_compat.py','cnn_compat.py','module_hooks.py','image_transforms.py','image_data.py','torch_rng.py','borch_compat.py','training_probe.py','definitions.json','training-definitions.json','training-native.npz'])py.FS.writeFile(name,await bytes(name));
    const result=await py.runPythonAsync(`
import json,numpy as np
from browser_torch import torch
from training_probe import run_training
arrays=await run_training(torch,json.load(open('definitions.json')),json.load(open('training-definitions.json')))
reference=np.load('training-native.npz')
if set(arrays)!=set(reference.files):raise AssertionError('Training fixture keys differ')
comparisons={}
for key in reference.files:
    a,b=arrays[key],reference[key]
    match=a.shape==b.shape and (np.array_equal(a,b) if b.dtype.kind in 'biu' else np.allclose(a,b,rtol=1e-4,atol=1e-5))
    comparisons[key]={'status':'pass' if match else 'fail','max_abs_error':float(np.max(np.abs(a-b))) if a.shape==b.shape else None}
json.dumps({'comparisons':comparisons})
`);
    self.postMessage({done:true,elapsed_ms:performance.now()-start,...JSON.parse(result)});
  }catch(error){self.postMessage({error:String(error)});}
};
