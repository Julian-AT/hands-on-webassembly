"""Hold an ASGI fetch open and verify disposal releases a waiting SW upgrade."""
import argparse
from playwright.sync_api import sync_playwright
from browser_startup import application_ready
from common import PROOF, write_json
from provenance import browser_stamp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--browser',choices=['chrome','firefox'],default='chrome')
    args=parser.parse_args()
    result={'browser':args.browser,'distribution':'installed' if args.browser=='chrome' else 'playwright',
            'provenance':browser_stamp('upgrade'),'cases':{},'status':'fail'}
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome') if args.browser=='chrome' else p.firefox.launch()
        result['version']=browser.version
        page=browser.new_page()
        try:
            page.goto('http://127.0.0.1:8008/unit1/',wait_until='domcontentloaded')
            application_ready(page)
            result['cases'].update(page.evaluate('''async () => {
                const path='app_pending_upgrade_probe/';
                const previous=navigator.serviceWorker.controller;
                const channel=new MessageChannel();
                window.__orphanPort=channel.port1;
                channel.port1.onmessage=()=>{}; // Simulate an unresponsive Python worker.
                previous.postMessage({type:'configureProxyPath',path},[channel.port2]);
                const request=fetch('/'+path).then(()=>({resolved:true}),error=>({rejected:true,error:String(error)}));
                async function diagnostics(worker) {
                    const channel=new MessageChannel();
                    return new Promise((resolve,reject)=>{
                        const timer=setTimeout(()=>{channel.port1.close();reject(new Error('Worker diagnostics timeout'))},3000);
                        channel.port1.onmessage=e=>{clearTimeout(timer);channel.port1.close();resolve(e.data)};
                        worker.postMessage({type:'course:diagnostics'},[channel.port2]);
                    });
                }
                async function until(predicate,timeout) {
                    const end=performance.now()+timeout;
                    while(!await predicate()) {
                        if(performance.now()>end) throw new Error('Lifecycle transition timed out');
                        await new Promise(resolve=>setTimeout(resolve,20));
                    }
                }
                await until(async()=> (await diagnostics(previous)).requests.some(r=>r.path===path && r.pending===1),10000);
                const before=await diagnostics(previous);
                const registration=await navigator.serviceWorker.register('../shinylive-sw.js?upgrade-orphan-test=1',{type:'module',updateViaCache:'none'});
                await until(()=>registration.waiting,10000);
                if(navigator.serviceWorker.controller!==previous) throw new Error('The open request did not hold the previous worker active');
                const started=performance.now();
                previous.postMessage({type:'course:dispose-proxy',path});
                await until(()=>navigator.serviceWorker.controller!==previous,2000);
                const elapsed=(performance.now()-started)/1000;
                const completion=await request;
                if(!completion.rejected) throw new Error('The abandoned proxy fetch was not rejected');
                channel.port1.close();
                return {
                    'orphan-request-held':{status:'pass',diagnostics:before},
                    'upgrade-waits-for-request':{status:'pass'},
                    'disposal-unblocks-upgrade':{status:'pass',seconds:elapsed,request:completion}
                };
            }'''))
            page.reload(wait_until='domcontentloaded')
            result['cases']['reload-after-upgrade']={'status':'pass','application':application_ready(page)}
            result['status']='pass'
        except Exception as error:
            result['cases']['upgrade-fault']={'status':'fail','error':str(error),'text':page.locator('body').inner_text()}
        finally:
            browser.close()
    if result['provenance']['fingerprint']!=browser_stamp('upgrade')['fingerprint']:
        result['original_status']=result['status'];result['status']='stale'
    write_json(PROOF/f'evidence/startup-upgrade-fault-{args.browser}.json',result)
    print(args.browser,result['status'],flush=True)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
