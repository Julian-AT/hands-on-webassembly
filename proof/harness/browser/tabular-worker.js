importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage = async () => {
  try {
    const start = performance.now();
    const py = await loadPyodide({indexURL: '../shinylive/pyodide/'});
    await py.loadPackage(['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn','course-native-tsne']);
    for (const name of ['tsne_runtime.py','tabular_probe.py','tabular_regressions.py','tabular-definitions.py','tabular-native.npz','penguins.csv']) {
      const response = await fetch(name);
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
      py.FS.writeFile(name, new Uint8Array(await response.arrayBuffer()));
    }
    const result = await py.runPythonAsync(`
import json, numpy as np
import tsne_runtime
from tabular_probe import run_tabular
arrays = run_tabular(open('tabular-definitions.py').read())
from tabular_regressions import run_regressions
arrays.update(run_regressions(open('tabular-definitions.py').read()))
reference = np.load('tabular-native.npz')
comparison = {}
for key in reference.files:
    a, b = arrays[key], reference[key]
    match = a.shape == b.shape and (np.array_equal(a,b) if b.dtype.kind in 'biu' else np.allclose(a,b,rtol=1e-4,atol=1e-5))
    comparison[key] = {'status':'pass' if match else 'fail', 'max_abs_error':float(np.max(np.abs(a-b))), 'shape':list(a.shape)}
json.dumps({'comparisons':comparison})
`);
    self.postMessage({done:true,elapsed_ms:performance.now()-start,wasm_heap_bytes:py._module.HEAP8.buffer.byteLength,...JSON.parse(result)});
  } catch (error) { self.postMessage({error:String(error)}); }
};
