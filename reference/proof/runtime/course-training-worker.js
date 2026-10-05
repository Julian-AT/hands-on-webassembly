// Dedicated numerical context: UI previews never share this worker's RNG.
importScripts('./pyodide/pyodide.js');
let current, activeAbort;
self.onmessage = async ({data}) => {
  if (data.kind === 'cancel') {
    if (current && ['protocol','unit','build_id','source_id','session_id','task_id','operation','dataset_generation','model_generation'].every(key=>data.tags?.[key]===current[key])) {
      activeAbort?.abort();
      self.close();
    }
    return;
  }
  if (current) return;
  const tags = data.tags;
  if (tags?.protocol !== 2 || ![5,6,7].includes(tags.unit) || !tags.session_id || !Number.isInteger(tags.task_id)
      || tags.operation !== 'training' || !Number.isInteger(tags.dataset_generation) || !Number.isInteger(tags.model_generation)) {
    if (tags) self.postMessage({kind:'error',tags,error:'Training protocol changed. Reload and retry.'});
    return;
  }
  current = tags;
  const send = (kind, payload) => self.postMessage({kind, tags, payload});
  const abort = new AbortController();
  activeAbort = abort;
  try {
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(data.source)))].map(x=>x.toString(16).padStart(2,'0')).join('');
    if (digest !== tags.source_id) throw new Error('Training source version changed. Reload and retry.');
    const root = '../assets/v1/neural-runtime/';
    const response = await fetch(root+'manifest.json',{signal:abort.signal,cache:'no-cache'});
    if (!response.ok) throw new Error('Training runtime could not load. Retry training.');
    const manifest = await response.json();
    if (manifest.sources[String(tags.unit)] !== digest || manifest.build_ids[String(tags.unit)] !== tags.build_id) throw new Error('Training runtime is obsolete. Reload and retry.');
    async function bytes(name) {
      const response = await fetch(root+name,{signal:abort.signal});
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}. Retry training.`);
      const result = new Uint8Array(await response.arrayBuffer());
      const actual = [...new Uint8Array(await crypto.subtle.digest('SHA-256',result))].map(x=>x.toString(16).padStart(2,'0')).join('');
      if (actual !== manifest.files[name]) throw new Error(`${name}: checksum mismatch. Reload and retry.`);
      return result;
    }
    const py = await loadPyodide({indexURL:'./pyodide/'});
    await py.loadPackage(['numpy','pandas','pillow','scipy','course-native-neural','course-native-cnn']);
    py.unpackArchive(await bytes(manifest.borch),'zip',{extractDir:'/home/pyodide'});
    py.FS.mkdirTree('/home/pyodide/cloudpickle');
    for (const name of Object.keys(manifest.files)) {
      if (name === manifest.borch) continue;
      py.FS.writeFile('/home/pyodide/'+name,await bytes(name));
    }
    py.FS.writeFile('/home/pyodide/training-payload.pkl',data.payload);
    py.globals.set('course_source',data.source);
    py.globals.set('course_notify', payload => send('progress',payload.toJs ? payload.toJs() : payload));
    py.globals.set('course_cancelled', () => false);
    await py.runPythonAsync(`
import cloudpickle
from isolated_training import execute_payload_async
result = await execute_payload_async(course_source, open('/home/pyodide/training-payload.pkl','rb').read(), lambda message: course_notify(cloudpickle.dumps(message)), course_cancelled)
open('/home/pyodide/training-result.pkl','wb').write(result)
`);
    const result = py.FS.readFile('/home/pyodide/training-result.pkl');
    self.postMessage({kind:'result',tags,payload:result},[result.buffer]);
  } catch (error) {
    self.postMessage({kind:'error',tags,error:String(error)});
  } finally {abort.abort();}
};
