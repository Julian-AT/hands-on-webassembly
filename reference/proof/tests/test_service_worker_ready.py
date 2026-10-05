"""An old worker controlling the page must not unlock the new application."""
import subprocess
import ast
import json
import unittest
from pathlib import Path


class ServiceWorkerReadyTests(unittest.TestCase):
    def test_only_expected_waiting_build_receives_retries_and_poll_stops(self):
        root=Path(__file__).resolve().parents[2]
        script=r'''
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
(async()=>{
 const events=new Map(),polls=new Map(),messages=[];
 const registration={update:async()=>{},active:{postMessage:()=>{}},
   waiting:{scriptURL:'http://localhost/shinylive-sw.js?v=unrelated',postMessage:message=>messages.push(message)}};
 const workers={controller:{scriptURL:'http://localhost/shinylive-sw.js?v=old'},
   register:async()=>registration,ready:Promise.resolve(registration),
   addEventListener:(name,fn)=>events.set(name,fn),removeEventListener:name=>events.delete(name)};
 const context={navigator:{serviceWorker:workers},location:{href:'http://localhost/unit6/'},
   serviceWorkerPath:'/shinylive-sw.js',URL,setTimeout,clearTimeout,
   setInterval:(fn,ms)=>{assert.equal(ms,200);polls.set(1,fn);return 1},clearInterval:id=>polls.delete(id)};
 vm.runInNewContext(fs.readFileSync('proof/runtime/service-worker-ready.js','utf8').replace('export const serviceWorkerReady','globalThis.result'),context);
 await new Promise(setImmediate);
 polls.get(1)();assert.equal(messages.length,0,'An unrelated waiting build must not be activated');
 const expected='http://localhost/shinylive-sw.js?v=__COURSE_RUNTIME_VERSION__';
 registration.waiting.scriptURL=expected;
 polls.get(1)();polls.get(1)();
 assert.equal(messages.length,2);
 assert.equal(messages[0].type,'course:activate-runtime');
 assert.equal(messages[0].version,'course-__COURSE_RUNTIME_VERSION__');
 workers.controller={scriptURL:expected};registration.waiting=null;
 events.get('controllerchange')();await context.result;
 assert.equal(polls.size,0);assert.equal(events.size,0);
})().catch(error=>{console.error(error);process.exit(1)});
'''
        subprocess.run(['node','-e',script],cwd=root,check=True,timeout=10,capture_output=True,text=True)

    def test_worker_rejects_other_build_activation_requests(self):
        root=Path(__file__).resolve().parents[2]
        constants=[node.value for node in ast.walk(ast.parse((root/'proof/scripts/patch_runtime.py').read_text()))
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('\n// Reclaim an uncontrolled')]
        self.assertEqual(len(constants),1)
        script=r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
let message,activations=0;const promises=[];
const context={version:'course-expected',courseRequests:new Map(),apps:{},courseAppOwners:{},
 self:{addEventListener:(name,fn)=>message=fn,skipWaiting:()=>{activations++;return Promise.resolve()},clients:{claim:async()=>{}}}};
vm.runInNewContext(SOURCE,context);
for(const version of ['course-other',null,undefined])message({data:{type:'course:activate-runtime',version},waitUntil:p=>promises.push(p)});
assert.equal(activations,0);assert.equal(promises.length,0);
message({data:{type:'course:activate-runtime',version:'course-expected'},waitUntil:p=>promises.push(p)});
assert.equal(activations,1);assert.equal(promises.length,1);
'''.replace('SOURCE',json.dumps(constants[0]))
        subprocess.run(['node','-e',script],cwd=root,check=True,timeout=10,capture_output=True,text=True)

    def test_waits_for_current_version_before_loading_application(self):
        root = Path(__file__).resolve().parents[2]
        script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
(async () => {
  const events = new Map();
  let registered, options, resolved = false, claims = 0;
  const registration = {update: async () => {}, active: {postMessage: () => claims++}};
  const workers = {
    controller: {scriptURL: 'http://localhost/shinylive-sw.js'},
    register: async (url, opts) => {registered = url; options = opts; return registration},
    ready: Promise.resolve(registration),
    addEventListener: (name, fn) => events.set(name, fn),
    removeEventListener: name => events.delete(name),
  };
  const context = {navigator: {serviceWorker: workers}, location: {href: 'http://localhost/unit5/'},
    serviceWorkerPath: '/shinylive-sw.js', URL, setTimeout, clearTimeout, setInterval, clearInterval};
  const source = fs.readFileSync('proof/runtime/service-worker-ready.js', 'utf8')
    .replace('export const serviceWorkerReady', 'globalThis.result');
  vm.runInNewContext(source, context);
  context.result.then(() => {resolved = true});
  await new Promise(setImmediate);
  assert.equal(resolved, false, 'An old controller must not unlock startup');
  assert.equal(options.updateViaCache, 'none');
  assert.ok(registered.includes('?v='));
  assert.equal(claims, 1);
  workers.controller = {scriptURL: 'http://localhost/shinylive-sw.js?v=another-build'};
  events.get('controllerchange')();
  await new Promise(setImmediate);
  assert.equal(resolved, false, 'An unrelated version must not unlock startup');
  workers.controller = {scriptURL: new URL(registered, context.location.href).href};
  events.get('controllerchange')();
  assert.equal(await context.result, registration);
  assert.equal(events.size, 0);
})().catch(error => {console.error(error); process.exit(1)});
'''
        subprocess.run(['node', '-e', script], cwd=root, check=True, timeout=10,
                       capture_output=True, text=True)

    def test_installation_does_not_wait_for_activation_to_finish(self):
        root=Path(__file__).resolve().parents[2]
        script=r'''
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
let install,finishActivation,skipCalls=0;
const activation=new Promise(resolve=>finishActivation=resolve);
const context={version:'build',cacheName:'cache',caches:{open:async()=>({})},
 self:{addEventListener:(name,fn)=>{if(name==='install')install=fn},skipWaiting:()=>{skipCalls++;return activation}}};
vm.runInNewContext(fs.readFileSync('proof/experiments/startup-handoff/service-worker-install-candidate.js','utf8'),context);
(async()=>{
 let installWork;install({waitUntil:promise=>installWork=promise});
 assert.equal(skipCalls,1);
 const installed=await Promise.race([installWork.then(()=>true),new Promise(resolve=>setTimeout(()=>resolve(false),100))]);
 assert.equal(installed,true,'Installation must finish while activation is still pending');
 finishActivation();
})().catch(error=>{console.error(error);process.exit(1)});
'''
        subprocess.run(['node','-e',script],cwd=root,check=True,timeout=10,capture_output=True,text=True)


if __name__ == '__main__':
    unittest.main()
