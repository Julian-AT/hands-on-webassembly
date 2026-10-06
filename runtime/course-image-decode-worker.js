// CPU-bound checksum validation/decompression can always be terminated by owner.
importScripts('./pyodide/pyodide.js');
self.onmessage=async ({data})=>{
  try{
    const py=await loadPyodide({indexURL:'./pyodide/'});
    const packageErrors=[];
    await py.loadPackage(['numpy'],{errorCallback:message=>packageErrors.push(String(message))});
    try{py.runPython('import numpy');}
    catch(error){throw new Error('Numerical preparation package NumPy did not load. Retry loading. '+
      packageErrors.join('; ')+' '+String(error));}
    py.FS.writeFile('image_preload.py',data.unpack_source);
    const splits=[];
    for(let i=0;i<data.keys.length;i++){
      py.FS.writeFile('archive.npz',data.archives[i]);
      py.globals.set('entry_json',JSON.stringify(data.entries[i]));
      py.globals.set('dataset_key',data.keys[i]);
      const metadata=JSON.parse(await py.runPythonAsync(`
import json
from image_preload import unpack
images,labels,layout=unpack(open('archive.npz','rb').read(),json.loads(entry_json),dataset_key)
open('images.bin','wb').write(images.tobytes(order='C'))
open('labels.bin','wb').write(labels.tobytes(order='C'))
json.dumps(dict(shape=list(images.shape),label_dtype=labels.dtype.str,layout=layout))
`));
      splits.push({key:data.keys[i],...metadata,images:py.FS.readFile('images.bin'),labels:py.FS.readFile('labels.bin')});
      py.FS.unlink('archive.npz');py.FS.unlink('images.bin');py.FS.unlink('labels.bin');
      data.archives[i]=null;
      py.runPython('del images,labels');
    }
    self.postMessage({kind:'result',tags:data.tags,splits},splits.flatMap(split=>[split.images.buffer,split.labels.buffer]));
  }catch(error){self.postMessage({kind:'error',tags:data.tags,error:String(error)});}
  finally{self.close();}
};
