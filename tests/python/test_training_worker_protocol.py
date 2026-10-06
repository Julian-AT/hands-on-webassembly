"""Cancellation must match the complete calculation owner, including generations."""

from pathlib import Path
import subprocess
import unittest


class TrainingWorkerProtocolTests(unittest.TestCase):
    def test_unit5_and_generation_checked_cancellation(self):
        worker = Path(__file__).resolve().parents[2] / "runtime/course-training-worker.js"
        program = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {webcrypto}=require('node:crypto');
const source=fs.readFileSync(process.argv[1],'utf8');
function environment(){
  const messages=[],status={closed:0,fetches:0,aborted:false};
  const self={postMessage:m=>messages.push(m),close:()=>status.closed++};
  const ctx=vm.createContext({self,importScripts:()=>{},crypto:webcrypto,TextEncoder,
    Uint8Array,AbortController,fetch:(_url,{signal})=>{
      status.fetches++;signal.addEventListener('abort',()=>status.aborted=true);
      return new Promise(()=>{});
    }});
  vm.runInContext(source,ctx);
  return {self,messages,status};
}
(async()=>{
  const body='worker calculation';
  const digest=Buffer.from(await webcrypto.subtle.digest('SHA-256',new TextEncoder().encode(body))).toString('hex');
  const tags={protocol:2,unit:5,operation:'training',build_id:'build',source_id:digest,
    session_id:'session',task_id:3,dataset_generation:4,model_generation:5};
  for(const update of [{unit:1},{operation:'inspection'},{dataset_generation:1.5},{model_generation:null}]){
    const e=environment();await e.self.onmessage({data:{tags:{...tags,...update},source:body}});
    assert.equal(e.messages[0].kind,'error');assert.equal(e.status.fetches,0);
  }
  const e=environment();e.self.onmessage({data:{tags,source:body}});
  for(let i=0;i<100 && !e.status.fetches;i++)await new Promise(resolve=>setImmediate(resolve));
  assert.equal(e.status.fetches,1);
  for(const [key,value] of Object.entries(tags)){
    const changed={...tags,[key]:typeof value==='number'?value+1:value+'wrong'};
    await e.self.onmessage({data:{kind:'cancel',tags:changed}});
    assert.equal(e.status.closed,0,key);assert.equal(e.status.aborted,false,key);
  }
  await e.self.onmessage({data:{kind:'cancel',tags:{...tags}}});
  assert.equal(e.status.closed,1);assert.equal(e.status.aborted,true);
  const stale=environment();await stale.self.onmessage({data:{tags,source:body+'changed'}});
  assert.equal(stale.status.fetches,0);assert.equal(stale.messages[0].kind,'error');
  assert.match(stale.messages[0].error,/source version changed/);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
        result = subprocess.run(
            ["node", "-e", program, str(worker)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
