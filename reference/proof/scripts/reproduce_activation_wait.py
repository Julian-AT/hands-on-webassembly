"""Independent control for activation requested from an extended message event."""
import argparse
import json
from playwright.sync_api import sync_playwright
from common import PROOF,sha,write_json
from owned_preview import start
from host_conditions import snapshot,apply


WORKER='''self.addEventListener('install',event=>event.waitUntil(caches.open('control-'+self.location.search)));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
let release;
self.addEventListener('fetch',event=>{
 if(new URL(event.request.url).pathname==='/pending')event.respondWith(new Promise(resolve=>release=()=>resolve(new Response('settled'))));
});
self.addEventListener('message',event=>{
 if(event.data.kind==='release')release?.();
 if(event.data.kind==='activate'){
   const activation=self.skipWaiting();
   if(new URL(self.location.href).searchParams.get('mode')==='coupled')event.waitUntil(activation);
   else activation.catch(()=>{});
 }
 if(event.data.kind==='state'){event.ports[0].postMessage({pending:!!release});event.ports[0].close();}
});'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--browser',choices=['chrome','edge'],required=True)
    parser.add_argument('--label',required=True);args=parser.parse_args()
    root=PROOF/'cache'/f'activation-wait-control-{args.browser}-{args.label}'
    if root.exists():raise FileExistsError('Preserve independent controls')
    root.mkdir();(root/'index.html').write_text('<!doctype html><title>Independent activation control</title>')
    (root/'worker.js').write_text(WORKER)
    before=snapshot();server,identity=start(root,0)
    report=dict(status='diagnostic',browser=args.browser,distribution='installed',cases={},
        executor_sha256=sha(__import__('pathlib').Path(__file__)),worker_sha256=sha(root/'worker.js'),
        preview_identity=identity,scope='Independent lifecycle causality experiment; no application acceptance.')
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge');report['version']=browser.version
            for mode in ('coupled','detached'):
                context=browser.new_context();page=context.new_page();page.goto(f'http://127.0.0.1:{identity["port"]}/')
                result=page.evaluate('''async mode=>{
                  const until=async(predicate,seconds)=>{const end=performance.now()+1000*seconds;while(!predicate()){
                    if(performance.now()>end)return false;await new Promise(r=>setTimeout(r,10));}return true;};
                  const registration=await navigator.serviceWorker.register('/worker.js?generation=1&mode='+mode);
                  await navigator.serviceWorker.ready;if(!await until(()=>navigator.serviceWorker.controller,5))throw new Error('No initial controller');
                  const old=navigator.serviceWorker.controller;
                  const response=fetch('/pending');
                  const state=()=>new Promise(resolve=>{const channel=new MessageChannel();channel.port1.onmessage=e=>{
                    channel.port1.close();resolve(e.data)};old.postMessage({kind:'state'},[channel.port2]);});
                  let pending=false;for(let i=0;i<100;i++){pending=(await state()).pending;if(pending)break;await new Promise(r=>setTimeout(r,10));}
                  if(!pending)throw new Error('No actual extendable fetch was started');
                  const next='/worker.js?generation=2&mode='+mode;
                  await navigator.serviceWorker.register(next);
                  if(!await until(()=>registration.waiting?.state==='installed',5))throw new Error('No waiting worker');
                  const started=performance.now();registration.waiting.postMessage({kind:'activate'});
                  await new Promise(r=>setTimeout(r,50));old.postMessage({kind:'release'});
                  const body=await (await response).text();
                  const activated=await until(()=>navigator.serviceWorker.controller?.scriptURL.endsWith(next),3);
                  return {mode,pending_fetch_settled:body==='settled',activated,seconds:(performance.now()-started)/1000,
                    active:registration.active?.scriptURL,waiting:registration.waiting?.scriptURL};
                }''',mode)
                report['cases'][mode]=result;context.close();print(args.browser,mode,result,flush=True)
            browser.close()
    except Exception as error:report.update(status='fail',error=str(error))
    finally:server.terminate();server.wait(timeout=10);apply(report,before,snapshot())
    write_json(PROOF/f'evidence/activation-wait-control-{args.browser}-{args.label}.json',report)


if __name__=='__main__':main()
