"""Verify a replaced proxy cannot strand HTTP requests or break its successor."""

from repository import ROOT, EVIDENCE, SCRIPTS, SITE
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from common import PROOF, sha, write_json
from browser_startup import application_ready


def inputs():
    paths = [Path(__file__), SCRIPTS / "browser_startup.py", SCRIPTS / "common.py"]
    paths += [path for path in (SITE).rglob("*") if path.is_file()]
    return hashlib.sha256(
        json.dumps(
            {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}, sort_keys=True
        ).encode()
    ).hexdigest()


def main():
    before = inputs()
    result = dict(
        status="fail",
        browser="chrome",
        distribution="installed",
        input_fingerprint=before,
        cases={},
        scope="Bounded same-owner proxy port replacement; real HTTP cancellation and successor response.",
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        result["version"] = browser.version
        page = browser.new_page()
        try:
            page.goto("http://127.0.0.1:8008/unit1/")
            application_ready(page)
            result["cases"] = page.evaluate("""async()=>{
 const worker=navigator.serviceWorker.controller,path='app_replaced_port_probe/';
 const first=new MessageChannel();first.port1.onmessage=()=>{};
 worker.postMessage({type:'configureProxyPath',path},[first.port2]);
 const diagnostic=()=>new Promise((resolve,reject)=>{const c=new MessageChannel();const t=setTimeout(()=>reject(new Error('Diagnostics timed out')),2000);c.port1.onmessage=e=>{clearTimeout(t);c.port1.close();resolve(e.data)};worker.postMessage({type:'course:diagnostics'},[c.port2])});
 const request=fetch('/'+path).then(()=>({resolved:true}),e=>({rejected:true,error:String(e)}));
 const deadline=performance.now()+3000;
 while(!(await diagnostic()).requests.some(r=>r.path===path)){
  if(performance.now()>deadline)throw new Error('Pending request was not registered');
  await new Promise(r=>setTimeout(r,10));
 }
 const next=new MessageChannel();next.port1.onmessage=e=>{
  if(e.data.type==='makeRequest'){
   const port=e.ports[0];port.postMessage({type:'http.response.start',status:200,headers:[['content-type','text/plain']]});
   port.postMessage({type:'http.response.body',body:new TextEncoder().encode('replacement works'),more_body:false});
  }
 };
 const start=performance.now();worker.postMessage({type:'configureProxyPath',path},[next.port2]);
 const outcome=await Promise.race([request,new Promise(resolve=>setTimeout(()=>resolve({still_pending:true}),1000))]);
 const seconds=(performance.now()-start)/1000;
 if(!outcome.rejected)throw new Error('Replaced proxy still owns a pending request');
 const response=await fetch('/'+path+'successor'),text=await response.text();
 if(response.status!==200||text!=='replacement works')throw new Error('Replacement proxy cannot serve a new request');
 const state=await diagnostic();if(state.requests.some(r=>r.path===path))throw new Error('Finished requests remain registered');
 worker.postMessage({type:'course:dispose-proxy',path});first.port1.close();next.port1.close();
 return {'old-port-request-cancelled':{status:'pass',seconds,outcome},'new-port-serves-response':{status:'pass',status_code:response.status,expected:'replacement works',observed:text,diagnostics:state}};
}""")
            result["status"] = "pass"
        except Exception as error:
            result.update(status="fail", error=str(error))
        finally:
            browser.close()
    if before != inputs():
        result["observed_status"] = result["status"]
        result["status"] = "stale"
    write_json(EVIDENCE / "proxy-port-replacement-chrome.json", result)
    print(result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
