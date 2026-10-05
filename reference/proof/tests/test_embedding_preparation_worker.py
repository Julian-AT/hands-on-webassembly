"""Execute the actual worker against complete pinned model bytes and faults."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


class EmbeddingPreparationWorkerTests(unittest.TestCase):
    def test_complete_vectors_faults_and_owned_cancellation(self):
        script=r'''
import fs from 'node:fs';import vm from 'node:vm';import {webcrypto} from 'node:crypto';
const root=process.cwd(),assets=root+'/proof/assets/v1/embedding-runtime/';
const source=fs.readFileSync(root+'/proof/runtime/course-embedding-worker.js','utf8');
const hash=async bytes=>Buffer.from(await webcrypto.subtle.digest('SHA-256',bytes)).toString('hex');
const tags={protocol:1,unit:2,operation:'embedding-loading',build_id:'build',source_id:'source',
  session_id:'session',task_id:1,dataset_generation:0,model_generation:0};
async function run(mode){
  const messages=[];let closed=false,aborted=false,heldResolve;
  const held=new Promise(resolve=>heldResolve=resolve);
  const self={location:{href:'https://example.test/shinylive/course-embedding-worker.js'},
    postMessage:(data,transfer)=>messages.push(structuredClone(data,{transfer:transfer||[]})),close:()=>closed=true};
  const fetch=async (url,options)=>{
    const name=new URL(url).pathname.split('/').at(-1);
    if(mode==='cancel'&&name==='vectors.npy'){
      heldResolve();return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>{
        aborted=true;reject(new DOMException('Cancelled','AbortError'));},{once:true}));
    }
    if(mode==='failed'&&name==='vectors.npy')return new Response('Unavailable',{status:503});
    return new Response(mode==='corrupt'&&name==='vectors.npy'?Buffer.from('corrupt'):fs.readFileSync(assets+name));
  };
  vm.runInNewContext(source,{self,crypto:webcrypto,AbortController,URL,Uint8Array,DataView,TextDecoder,
    Blob,Response,DecompressionStream,fetch,console});
  const expected=await hash(fs.readFileSync(assets+'manifest.json'));
  const pending=self.onmessage({data:{kind:'prepare',tags,manifest_sha256:mode==='manifest'?'stale':expected}});
  if(mode==='cancel'){
    await held;
    await self.onmessage({data:{kind:'cancel',tags:{...tags,session_id:'obsolete'}}});
    if(aborted||closed)throw new Error('Obsolete cancellation affected current owner');
    await self.onmessage({data:{kind:'cancel',tags}});
  }
  await pending;
  const result=messages.find(m=>m.kind==='result'),error=messages.find(m=>m.kind==='error');
  if(!closed)throw new Error('Completed worker did not close');
  if(result)return {result:true,vector_hash:await hash(result.vectors),keys:Object.keys(result.rows).length,
    shape:result.shape,tags:result.tags};
  return {result:false,error:error?.error||null,cancelled:messages.some(m=>m.kind==='cancelled'),aborted};
}
const cases={};for(const mode of ['valid','manifest','corrupt','failed','cancel'])cases[mode]=await run(mode);
process.stdout.write(JSON.stringify(cases));
'''
        result=subprocess.run(['node','--input-type=module'],input=script,text=True,cwd=ROOT,
            capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        cases=json.loads(result.stdout)
        raw=(ROOT/'proof/assets/v1/embedding-runtime/vectors.npy').read_bytes()
        expected=hashlib.sha256(np.load(io.BytesIO(raw),allow_pickle=False).tobytes()).hexdigest()
        self.assertEqual(cases['valid']['vector_hash'],expected)
        self.assertEqual(cases['valid']['keys'],514157)
        self.assertEqual(cases['valid']['shape'],[20000,300])
        self.assertEqual(cases['valid']['tags']['session_id'],'session')
        self.assertIn('manifest changed',cases['manifest']['error'])
        self.assertIn('checksum mismatch',cases['corrupt']['error'])
        self.assertIn('HTTP 503',cases['failed']['error'])
        self.assertEqual(cases['cancel'],dict(result=False,error=None,cancelled=True,aborted=True))


if __name__=='__main__':unittest.main()
