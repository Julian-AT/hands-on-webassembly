importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage=async()=>{
  try {
    const py=await loadPyodide({indexURL:'../shinylive/pyodide/'});
    await py.loadPackage(['numpy']);
    const response=await fetch('../unit2/app.json');
    if(!response.ok)throw new Error('Missing Unit 2 export');
    const files=await response.json();
    py.FS.mkdirTree('/home/pyodide/embedding-assets');
    for(const file of files){
      if(['embedding_runtime.py','embedding_preload.py','course-embedding.json'].includes(file.name)||file.name.startsWith('embedding-assets/')){
        const content=file.type==='binary'?Uint8Array.from(atob(file.content),c=>c.charCodeAt(0)):file.content;
        py.FS.writeFile(file.name,content);
      }
    }
    for(const name of ['embedding-cases.json','embedding-vectors.npy']){
      const response=await fetch(name);
      if(!response.ok)throw new Error(name+': '+response.status);
      py.FS.writeFile(name,new Uint8Array(await response.arrayBuffer()));
    }
    const result=await py.runPythonAsync(`
import json,numpy as np
from embedding_runtime import English
nlp=English()
from embedding_preload import Owner
await Owner(nlp,json.load(open('course-embedding.json'))).prepare()
cases=json.load(open('embedding-cases.json'))
reference=np.load('embedding-vectors.npy')
comparisons={}
for i,case in enumerate(cases):
    doc=nlp(case['text'])
    tokens=[[t.text,t.idx,t.text_with_ws] for t in doc]
    match=tokens==case['tokens'] and np.allclose(doc.vector,reference[i],rtol=1e-4,atol=1e-5) and np.isclose(doc.vector_norm,case['norm'],rtol=1e-4,atol=1e-5) and np.isclose(doc.similarity(nlp(case['other_text'])),case['similarity'],rtol=1e-4,atol=1e-5)
    comparisons[str(i)]={'status':'pass' if match else 'fail','text':case['text']}
json.dumps({'comparisons':comparisons,'vocabulary_keys':len(nlp.rows),'vector_shape':list(nlp.vectors.shape)})
`);
    self.postMessage({done:true,...JSON.parse(result)});
  }catch(error){self.postMessage({error:`${error?.name || ''}: ${error?.message || String(error)} (errno ${error?.errno || ''})`});}
};
