importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage = async () => {
  try {
    const start = performance.now();
    const py = await loadPyodide({indexURL: '../shinylive/pyodide/'});
    await py.loadPackage(['numpy','pillow']);
    const fetchBytes = async (name) => {
      const response = await fetch(name);
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    };
    py.unpackArchive(await fetchBytes('pyborch-1.14.1-py3-none-any.whl'), 'zip', {extractDir: '/home/pyodide'});
    for (const name of ['neural_probe.py', 'borch_compat.py', 'torch_rng.py', 'image_transforms.py', 'definitions.json', 'neural-native.npz']) py.FS.writeFile(name, await fetchBytes(name));
    const result = await py.runPythonAsync(`
import json, io, numpy as np, borch, asyncio
from torch_rng import install
install(borch)
from neural_probe import run_probe
checks, arrays = run_probe(borch, json.load(open('definitions.json')))
reference = np.load('neural-native.npz')
comparison = {}
for key in reference.files:
    if key not in arrays:
        comparison[key] = {'status': 'missing'}
        continue
    a, b = arrays[key], reference[key]
    match = a.shape == b.shape and (np.array_equal(a,b) if b.dtype.kind in 'biu' else np.allclose(a, b, rtol=1e-4, atol=1e-5))
    comparison[key] = {'status': 'pass' if match else 'fail', 'max_abs_error': float(np.max(np.abs(a-b))) if a.shape == b.shape else None}
try:
    await asyncio.to_thread(lambda: 1)
    checks['unit7_asyncio_to_thread'] = {'status': 'pass'}
except Exception as e:
    checks['unit7_asyncio_to_thread'] = {'status': 'fail', 'error': str(e)}
json.dumps({'checks':checks, 'comparisons':comparison})
`);
    self.postMessage({done: true, elapsed_ms: performance.now() - start, wasm_heap_bytes: py._module.HEAP8.buffer.byteLength, ...JSON.parse(result)});
  } catch (error) { self.postMessage({error: String(error)}); }
};
