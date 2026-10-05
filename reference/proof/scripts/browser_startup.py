"""Startup, reload and real service-worker replacement regressions."""
import argparse
import time
from common import PROOF,UNITS,write_json
from provenance import browser_stamp


def stamp():
    return browser_stamp('startup')


def application_ready(page):
    # Inspect the current same-origin document, rather than a protocol frame
    # handle. Firefox's frame tracker can retain about:blank after a SW reload.
    predicate = '''() => {
      const frame=document.querySelector('#root iframe');
      const doc=frame?.contentDocument;
      const title=doc?.querySelector('h2');
      const app=frame?.contentWindow?.Shiny?.shinyapp;
      const loading=document.getElementById('course-startup');
      return !!(title?.getBoundingClientRect().width && doc.querySelector('.shiny-bound-input') && app?.isConnected() && (!loading || loading.hidden));
    }'''
    page.wait_for_function(predicate, timeout=120000)
    return page.evaluate('''() => {
      const frame=document.querySelector('#root iframe');
      return {title:frame.contentDocument.querySelector('h2').textContent,
        document_url:frame.contentWindow.location.href,
        bound_inputs:frame.contentDocument.querySelectorAll('.shiny-bound-input').length,
        connected:frame.contentWindow.Shiny.shinyapp.isConnected(),
        loading_visible:!document.getElementById('course-startup')?.hidden};
    }''')


def main():
    from playwright.sync_api import sync_playwright
    parser=argparse.ArgumentParser();parser.add_argument('--browser',choices=['chrome','edge','firefox','webkit'],default='chrome')
    parser.add_argument('--units', nargs='+', type=int, choices=list(UNITS), default=list(UNITS))
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--label', default='')
    args=parser.parse_args()
    if args.cycles < 1:
        parser.error('--cycles must be positive')
    result={'browser':args.browser,'distribution':'installed' if args.browser in ('chrome','edge') else 'playwright','provenance':stamp(),'cases':{},'status':'error'}
    with sync_playwright() as p:
        browser=None
        try:
            browser=p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge') if args.browser in ('chrome','edge') else getattr(p,args.browser).launch()
            result['version']=browser.version
            context=browser.new_context(viewport={'width':1440,'height':1000})
            for cycle, unit in ((cycle, unit) for cycle in range(args.cycles) for unit in args.units):
                page=context.new_page();errors=[];console_errors=[];trace=[]
                key=f'unit{unit}' if args.cycles == 1 else f'unit{unit}/cycle-{cycle+1:02}'
                page.on('requestfailed',lambda request:trace.append({'event':'requestfailed','url':request.url,'failure':request.failure}))
                page.on('response',lambda response:trace.append({'event':'response','url':response.url,'status':response.status,'service_worker':response.from_service_worker}) if response.url.endswith('.wasm') or response.status>=400 else None)
                page.on('worker',lambda worker:trace.append({'event':'worker','url':worker.url}))
                page.add_init_script('''window.__startupTrace=[];
                    navigator.serviceWorker?.addEventListener('controllerchange',()=>window.__startupTrace.push({event:'controllerchange',url:navigator.serviceWorker.controller?.scriptURL,time:performance.now()}));
                    navigator.serviceWorker?.addEventListener('message',event=>{if(event.data?.type?.startsWith('course:')) window.__startupTrace.push(event.data)});
                ''')
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.on('console',lambda message:console_errors.append(message.text) if message.type=='error' else None)
                operation='startup'
                try:
                    start=time.perf_counter();page.goto(f'http://127.0.0.1:8008/unit{unit}/',wait_until='domcontentloaded')
                    state=application_ready(page)
                    result['cases'][f'{key}/startup']={'status':'pass','seconds':time.perf_counter()-start,'application':state}
                    for operation in ('reload','service-worker-update'):
                        if operation=='service-worker-update':
                            page.evaluate('''async()=>{
                              const registration=await navigator.serviceWorker.register('../shinylive-sw.js?regression-update=1',{type:'module'});
                              const desired=new URL('../shinylive-sw.js?regression-update=1',location.href).href;
                              await new Promise((resolve,reject)=>{
                                const end=Date.now()+30000;
                                function poll(){
                                  if(navigator.serviceWorker.controller?.scriptURL===desired)return resolve();
                                  if(Date.now()>end)return reject(new Error('Updated service worker did not take control'));setTimeout(poll,30);
                                }poll();
                              });
                            }''')
                        page.reload(wait_until='domcontentloaded')
                        state=application_ready(page)
                        result['cases'][f'{key}/{operation}']={'status':'pass','application':state}
                except Exception as e:
                    result['cases'][f'{key}/lifecycle']={'status':'fail','operation':operation,'error':str(e),
                        'console_errors':console_errors,'page_text':page.locator('body').inner_text(),
                        'frames':[frame.url for frame in page.frames]}
                    page.screenshot(path=str(PROOF/f'evidence/screenshots/startup-{args.browser}-unit{unit}-failure.png'))
                if errors:result['cases'][f'{key}/page-errors']={'status':'fail','errors':errors}
                try:
                    state=page.evaluate('''async()=>({controller:navigator.serviceWorker.controller?.scriptURL,
                        registrations:(await navigator.serviceWorker.getRegistrations()).map(r=>({scope:r.scope,active:r.active?.scriptURL,waiting:r.waiting?.scriptURL,installing:r.installing?.scriptURL})),
                        caches:await caches.keys(),events:window.__startupTrace})''')
                except Exception as e:state={'error':str(e)}
                result.setdefault('traces',{})[key]={'network':trace,'state':state,'console_errors':console_errors}
                page.close()
                print(args.browser,key,'checked',flush=True)
            result['status']='pass' if all(v['status']=='pass' for v in result['cases'].values()) else 'fail'
        except Exception as e:result.update(status='unavailable' if browser is None else 'error',error=str(e))
        finally:
            if browser:browser.close()
    if result['provenance']['fingerprint']!=stamp()['fingerprint']:
        result['original_status']=result['status'];result['status']='stale'
    suffix='' if args.units==list(UNITS) else '-units-'+'-'.join(map(str,args.units))
    if args.label:suffix+='-'+args.label
    path=PROOF/f'evidence/startup-{args.browser}{suffix}.json'
    write_json(path,result)
    print(result['status']);return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
