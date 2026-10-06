"""Actual supervisor protocol: complete bytes, stale owners and nested disposal."""

import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ImagePreparationWorkerTests(unittest.TestCase):
    def test_faults_and_nested_owned_cancellation(self):
        script = r"""
import fs from 'node:fs';import vm from 'node:vm';import {webcrypto} from 'node:crypto';
const root=process.cwd(),assets=root+'/artifacts/assets/v1/images/';
const source=fs.readFileSync(root+'/runtime/course-image-worker.js','utf8');
const unpack=fs.readFileSync(root+'/runtime/image_preload.py','utf8');
const hash=async bytes=>Buffer.from(await webcrypto.subtle.digest('SHA-256',bytes)).toString('hex');
const tags={protocol:1,unit:7,operation:'image-loading',build_id:'build',source_id:'source',
 session_id:'session',task_id:1,dataset_generation:2,model_generation:3};
async function run(mode){
 let closed=false,aborted=false,childTerminated=false,heldResolve;
 const held=new Promise(resolve=>heldResolve=resolve),messages=[],reads=[];
 const self={location:{href:'https://example.test/shinylive/course-image-worker.js'},
  postMessage:(data,transfer)=>messages.push(structuredClone(data,{transfer:transfer||[]})),close:()=>closed=true};
 class Child{
  constructor(url){if(!url.href.endsWith('/course-image-decode-worker.js'))throw new Error('Wrong decoder');}
  terminate(){childTerminated=true;}
  postMessage(data){
   if(data.keys.join(',')!=='MNIST/train,MNIST/test')throw new Error('Incomplete child set');
   if(mode==='nested'){heldResolve();return;}
   // An obsolete child must not be able to complete another task.
   this.onmessage({data:{kind:'error',tags:{...tags,task_id:0},error:'obsolete'}});
   this.onmessage({data:{kind:'result',tags,splits:[]}});
  }
 }
 const fetch=async (url,options)=>{
  const name=new URL(url).pathname.split('/').at(-1);reads.push(name);
  if(mode==='download'&&name==='MNIST-test.npz'){
   heldResolve();return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>{
    aborted=true;reject(new DOMException('Cancelled','AbortError'));},{once:true}));
  }
  return new Response(mode==='corrupt'&&name==='MNIST-test.npz'?Buffer.from('corrupt'):fs.readFileSync(assets+name));
 };
 vm.runInNewContext(source,{self,crypto:webcrypto,AbortController,URL,Uint8Array,TextEncoder,
  TextDecoder,Worker:Child,fetch,console});
 const pending=self.onmessage({data:{kind:'prepare',tags,dataset:'MNIST',unpack_source:unpack,
  unpack_sha256:mode==='source'?'0'.repeat(64):await hash(Buffer.from(unpack)),
  manifest_sha256:await hash(fs.readFileSync(assets+'manifest.json'))}});
 if(mode==='download'||mode==='nested'){
  await held;
  await self.onmessage({data:{kind:'cancel',tags:{...tags,session_id:'obsolete'}}});
  if(aborted||closed||childTerminated)throw new Error('Stale owner cancelled current work');
  await self.onmessage({data:{kind:'cancel',tags}});
  if(!closed||!messages.some(m=>m.kind==='cancelled'))throw new Error('No cancellation acknowledgement');
  if(mode==='nested'&&!childTerminated)throw new Error('Nested worker survived cancellation');
  // The terminated child promise cannot settle, just as in the browser.
  if(mode==='nested')return {cancelled:true,closed,childTerminated};
 }
 await pending;
 if(!closed)throw new Error('Finished supervisor survived');
 return {reads,result:messages.some(m=>m.kind==='result'),
  error:messages.find(m=>m.kind==='error')?.error||null,
  cancelled:messages.some(m=>m.kind==='cancelled'),aborted,childTerminated};
}
const cases={};for(const mode of ['valid','source','corrupt','download','nested'])cases[mode]=await run(mode);
process.stdout.write(JSON.stringify(cases));
"""
        result = subprocess.run(
            ["node", "--input-type=module"],
            input=script,
            text=True,
            cwd=ROOT,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        cases = json.loads(result.stdout)
        self.assertEqual(
            cases["valid"]["reads"], ["manifest.json", "MNIST-train.npz", "MNIST-test.npz"]
        )
        self.assertTrue(cases["valid"]["result"])
        self.assertTrue(cases["valid"]["childTerminated"])
        self.assertIn("identity changed", cases["source"]["error"])
        self.assertEqual(cases["source"]["reads"], [])
        self.assertIn("checksum or length mismatch", cases["corrupt"]["error"])
        self.assertFalse(cases["corrupt"]["result"])
        self.assertTrue(cases["download"]["aborted"])
        self.assertTrue(cases["download"]["cancelled"])
        self.assertFalse(cases["download"]["result"])
        self.assertEqual(cases["nested"], dict(cancelled=True, closed=True, childTerminated=True))


if __name__ == "__main__":
    unittest.main()
