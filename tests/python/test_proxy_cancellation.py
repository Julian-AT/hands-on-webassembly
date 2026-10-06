import subprocess
import unittest
from pathlib import Path


class ProxyCancellationTests(unittest.TestCase):
    def test_replacing_an_owned_port_cancels_its_pending_http_requests(self):
        script = r"""
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const pending=new Map(),owners={'app/':'owner'},calls=[];
const old={close:()=>calls.push('old-closed')},next={close:()=>calls.push('next-closed')};
const apps={'app/':old};
pending.set('app/',new Set([()=>calls.push('request-cancelled')]));
const context={courseRequests:pending,courseAppOwners:owners,apps};
vm.runInNewContext(fs.readFileSync('runtime/service-worker-proxy.js','utf8'),context);
assert.equal(context.courseConfigureProxy({data:{path:'app/'},ports:[next],source:{id:'owner'}}),true);
assert.deepEqual(calls,['request-cancelled','old-closed']);
assert.equal(apps['app/'],next);assert.equal(owners['app/'],'owner');
const foreign={close:()=>calls.push('foreign-closed')};
assert.equal(context.courseConfigureProxy({data:{path:'app/'},ports:[foreign],source:{id:'different'}}),false);
assert.equal(apps['app/'],next);assert.equal(calls.at(-1),'foreign-closed');
"""
        subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_takeover_disposes_the_worker_that_owned_each_proxy_port(self):
        script = r"""
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const calls=[],listeners={},windowEvents={};
const old={postMessage:message=>calls.push(['old',message.type])};
const next={postMessage:message=>calls.push(['next',message.type])};
const serviceWorker={controller:old,addEventListener:(name,fn)=>listeners[name]=fn,removeEventListener:()=>{}};
const context={navigator:{serviceWorker},window:{addEventListener:(name,fn)=>windowEvents[name]=fn},
 makeRandomKey:()=> 'fixed',createHttpRequestChannel:()=>({port1:{close:()=>calls.push(['port','close'])}})};
vm.runInNewContext(fs.readFileSync('runtime/proxy-lifecycle.js','utf8'),context);
context.setupAppProxyPath({});
serviceWorker.controller=next;listeners.controllerchange();
assert.deepEqual(calls.slice(0,2),[['old','course:dispose-proxy'],['port','close']]);
windowEvents.pagehide();
assert.ok(calls.some(x=>x[0]==='next'&&x[1]==='course:dispose-proxy'));
assert.equal(calls.filter(x=>x[0]==='old'&&x[1]==='course:dispose-proxy').length,2);
// A redundant controller cannot prevent connecting/disposal of its successor.
old.postMessage=()=>{throw new Error('redundant worker')};
const another={postMessage:message=>calls.push(['another',message.type])};
const again={...context,navigator:{serviceWorker:{...serviceWorker,controller:old}}};
vm.runInNewContext(fs.readFileSync('runtime/proxy-lifecycle.js','utf8'),again);
again.setupAppProxyPath({});
again.navigator.serviceWorker.controller=another;listeners.controllerchange();windowEvents.pagehide();
assert.ok(calls.some(x=>x[0]==='another'&&x[1]==='course:dispose-proxy'));
"""
        subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_stream_completion_abort_and_vanished_owner_settle_requests(self):
        script = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const {MessageChannel} = require('node:worker_threads');
const pending = new Map(), owners = {}, apps = {};
let ownerExists = true, ownerControlled = true;
const context = {MessageChannel, Request, Response, ReadableStream, DOMException,
  setTimeout, clearTimeout, setInterval, clearInterval, courseRequests:pending,
  courseAppOwners:owners, apps, self:{clients:{get:async () => ownerExists ? {} : undefined,
    matchAll:async () => ownerExists && ownerControlled ? [{id:'gone-client'}] : []}},
  reqToASGI:() => ({}), asgiToRes:(msg,body) => new Response(body,{status:msg.status})};
vm.runInNewContext(fs.readFileSync('runtime/service-worker-http.js','utf8'),context);
const ports=[];
const client = {postMessage: (msg, transferred) => ports.push(transferred[0])};
(async () => {
  const success=context.fetchASGI(client,new Request('http://localhost/app/'),undefined,
    chunk => chunk,'app/');
  ports.at(-1).postMessage({type:'http.response.start',status:200,headers:[]});
  ports.at(-1).postMessage({type:'http.response.body',body:new TextEncoder().encode('complete'),more_body:false});
  assert.equal(await (await success).text(),'complete');
  assert.equal(pending.size,0);

  const abort=new AbortController();
  const cancelled=context.fetchASGI(client,new Request('http://localhost/app/',{signal:abort.signal}),undefined,undefined,'app/');
  abort.abort();
  await assert.rejects(cancelled,error => error.name==='AbortError');
  assert.equal(pending.size,0);

  owners['app/']='gone-client';ownerExists=false;
  let closed=0;apps['app/']={close:()=>closed++};
  const first=context.fetchASGI(client,new Request('http://localhost/app/1'),undefined,undefined,'app/');
  const second=context.fetchASGI(client,new Request('http://localhost/app/2'),undefined,undefined,'app/');
  const started=Date.now();
  await Promise.all([assert.rejects(first,error=>error.name==='AbortError'),assert.rejects(second,error=>error.name==='AbortError')]);
  assert.ok(Date.now()-started<1000,'Orphaned requests must settle within one second');
  assert.equal(pending.size,0);assert.equal(closed,1);assert.equal(owners['app/'],undefined);
  // The old document still exists after takeover, but is controlled by a new SW.
  ownerExists=true;ownerControlled=false;owners['app/']='gone-client';apps['app/']={close:()=>closed++};
  const displaced=context.fetchASGI(client,new Request('http://localhost/app/3'),undefined,undefined,'app/');
  const displacedStart=Date.now();
  await assert.rejects(displaced,error=>error.name==='AbortError');
  assert.ok(Date.now()-displacedStart<1000);
  assert.equal(pending.size,0);assert.equal(closed,2);
  for(const port of ports)port.close();
})().catch(error=>{console.error(error);process.exit(1)});
"""
        subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )


if __name__ == "__main__":
    unittest.main()
