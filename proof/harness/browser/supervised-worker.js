importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage = async () => {
  try {
    const start = performance.now();
    const py = await loadPyodide({indexURL: '../shinylive/pyodide/'});
    await py.loadPackage(['numpy', 'pandas', 'scikit-learn','course-forest-criterion']);
    for (const name of ['forest_runtime.py','supervised_probe.py','supervised-definitions.py','supervised-native.npz','pima.csv','banknote.csv']) {
      const response = await fetch(name);
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
      py.FS.writeFile(name, new Uint8Array(await response.arrayBuffer()));
    }
    const result = await py.runPythonAsync(`
import json, numpy as np
import forest_runtime
from supervised_probe import run_supervised
arrays = run_supervised(open('supervised-definitions.py').read())
reference = np.load('supervised-native.npz')
comparison = {}
if set(arrays) != set(reference.files):
    raise AssertionError('Native/browser fixture keys differ')
for key in reference.files:
    a, b = arrays[key], reference[key]
    discrete = b.dtype.kind in 'biu'
    match = a.shape == b.shape and (np.array_equal(a,b) if discrete else np.allclose(a,b,rtol=1e-4,atol=1e-5))
    comparison[key] = {'status':'pass' if match else 'fail', 'max_abs_error':float(np.max(np.abs(a-b))) if a.shape == b.shape else None, 'shape':list(a.shape), 'mode':'exact' if discrete else 'rtol=1e-4,atol=1e-5'}
    if not match and a.shape == b.shape:
        bad = np.argwhere(a != b if discrete else ~np.isclose(a,b,rtol=1e-4,atol=1e-5))
        comparison[key]['first_differences'] = [{'index':idx.tolist(),'native':float(b[tuple(idx)]),'browser':float(a[tuple(idx)])} for idx in bad[:5]]
json.dumps({'comparisons':comparison})
`);
    self.postMessage({done:true,elapsed_ms:performance.now()-start,wasm_heap_bytes:py._module.HEAP8.buffer.byteLength,...JSON.parse(result)});
  } catch (error) { self.postMessage({error:String(error)}); }
};
