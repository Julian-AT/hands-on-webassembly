"""Instrumented diagnostic across two actual static build generations.

The served copy shares immutable assets and atomically replaces changed files.
Source artifacts and failed served copies are retained. No diagnostic hooks run
inside an accepting sequence; native worker states are read only after failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments/startup-handoff"))
from cdp_reproduce import DIAGNOSTICS
from playwright.sync_api import sync_playwright
from browser_html_proxy import site_manifest
from browser_startup import application_ready
from common import ROOT, PROOF, sha, write_json
from generation_store import GenerationStore
from owned_preview import start
from host_conditions import snapshot,apply
from hardware_contract import collect


def generation(site):
    worker = (site / 'shinylive-sw.js').read_text()
    match = re.search(r'var version = "course-([^"\n]+)";', worker)
    if not match:
        raise ValueError('Artifact lacks its content-bound runtime generation')
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', choices=['chrome', 'edge'], required=True)
    parser.add_argument('--unit', type=int, choices=range(1, 8), required=True)
    parser.add_argument('--before', type=Path, default=PROOF/'continuation-baseline/proof/site')
    parser.add_argument('--after', type=Path, default=PROOF/'site')
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--port', type=int, default=8062)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('Positive cycle count required')
    sites = [args.before.resolve(), args.after.resolve()]
    manifests = [site_manifest(site) for site in sites]
    versions = [generation(site) for site in sites]
    if versions[0] == versions[1]:
        raise ValueError('Query changes alone do not constitute a build upgrade')
    changed = sorted(key for key in manifests[0].keys() | manifests[1].keys() if manifests[0].get(key) != manifests[1].get(key))
    served = PROOF/'cache'/f'generation-passive-diagnostic-{args.browser}-unit{args.unit}-{args.label}'
    if served.exists():
        raise FileExistsError('Retain previous sequences; choose a fresh label')
    store=GenerationStore(served,sites)
    power_before=snapshot()
    inputs = {str(site.relative_to(PROOF)): manifest for site, manifest in zip(sites, manifests)}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    result = dict(status='running', browser=args.browser, distribution='installed',
        executor='proof/scripts/diagnose_passive_generation.py',
        dependencies={'proof/scripts/browser_startup.py':sha(PROOF/'scripts/browser_startup.py'),
                      'proof/scripts/browser_html_proxy.py':sha(PROOF/'scripts/browser_html_proxy.py'),
                      'proof/scripts/common.py':sha(PROOF/'scripts/common.py')},
        provenance=dict(scope='two-isolated-artifacts', fingerprint=fingerprint), inputs=inputs,
        executor_sha256=sha(Path(__file__)), generations=versions, changed_files=changed,
        cycles_requested=args.cycles, cycle_evidence=[], cases={},
        scope='Instrumented failure investigation only. Worker tracing, CDP network/lifecycle and diagnostic messages affect timing. No reliability or release certification claim.')
    output = PROOF/f'evidence/generation-passive-diagnostic-{args.browser}-unit{args.unit}-{args.label}.json'
    write_json(output, result)
    def page_state(page):
        return dict(python_wall_time=time.time(),python_monotonic=time.perf_counter(),
            browser=page.evaluate("()=>{\n                const f=document.querySelector('#root iframe'), w=f?.contentWindow,d=f?.contentDocument,a=w?.Shiny?.shinyapp;\n                const title=d?.querySelector('h2'),loading=document.getElementById('course-startup');\n                return {wall_time:Date.now(),monotonic:performance.now(),visibility:document.visibilityState,\n                    frame_url:w?.location.href,frame_ready:d?.readyState,title:title?.textContent,\n                    title_width:title?.getBoundingClientRect().width,bound_inputs:d?.querySelectorAll('.shiny-bound-input').length,\n                    shiny_connected:a?.isConnected(),socket_state:a?.$socket?.readyState,\n                    frame_text:d?.body?.innerText?.slice(0,1000),loading_hidden:loading?.hidden,\n                    loading_role:loading?.getAttribute('role'),loading_text:loading?.innerText};\n            }"))
    def ready_case(page):
        application=application_ready(page)
        expected=dict(connected=True,loading_visible=False)
        observed={key:application[key] for key in expected}
        if observed!=expected:raise AssertionError(observed)
        return dict(status='pass',application=application,
            assertion=dict(kind='workflow',expected=expected,observed=observed,matched=True))
    server,identity=start(store.current,args.port,dynamic_root=True)
    store.retain_origin_marker(identity)
    result['preview_identity']=identity
    result['hardware']=collect()
    result['diagnostic_instrumentation']=True
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError('Isolated server failed to start')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{args.port}/unit{args.unit}/', timeout=1).close()
                break
            except OSError:
                time.sleep(.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
            result['version'] = browser.version
            context = browser.new_context()
            try:
                current = 0
                for index in range(1, args.cycles+1):
                    page = context.new_page()
                    record = dict(index=index, before_generation=versions[current],begin_wall_time=time.time(),begin_monotonic=time.perf_counter())
                    operation = 'startup'
                    console=[]; errors=[]; pending={}; completed=[]; lifecycle=[]
                    page.on('console',lambda message:console.append(dict(type=message.type,text=message.text)))
                    page.on('pageerror',lambda error:errors.append(str(error)))
                    cdp=context.new_cdp_session(page)
                    def requested(event):
                        pending[event['requestId']]=dict(url=event['request']['url'],type=event.get('type'),started=event['timestamp'])
                    def finished(event,failed=False):
                        completed.append(dict(pending.pop(event['requestId'],{}),status='failed' if failed else 'finished',timestamp=event['timestamp'],error=event.get('errorText'),bytes=event.get('encodedDataLength')))
                    cdp.on('Network.requestWillBeSent',requested)
                    cdp.on('Network.loadingFinished',finished)
                    cdp.on('Network.loadingFailed',lambda event:finished(event,True))
                    cdp.on('ServiceWorker.workerVersionUpdated',lambda event:lifecycle.append(event))
                    cdp.send('Network.enable');cdp.send('ServiceWorker.enable')
                    try:
                        page.goto(f'http://127.0.0.1:{args.port}/unit{args.unit}/', wait_until='domcontentloaded')
                        record['startup'] = ready_case(page)
                        operation = 'reload'
                        page.reload(wait_until='domcontentloaded')
                        record['reload'] = ready_case(page)
                        operation = 'service-worker-update'
                        page.evaluate('''async()=>{
                          const target=new URL('../shinylive-sw.js?regression-update=1',location.href).href;
                          await navigator.serviceWorker.register(target,{type:'module',updateViaCache:'none'});
                          const deadline=performance.now()+30000;
                          while(navigator.serviceWorker.controller?.scriptURL!==target){
                            if(performance.now()>deadline)throw new Error('Query replacement did not activate');
                            await new Promise(resolve=>setTimeout(resolve,25));
                          }
                        }''')
                        page.reload(wait_until='domcontentloaded')
                        record['service-worker-update'] = ready_case(page)
                        operation = 'build-generation-update'
                        current = 1-current
                        store.publish(current)
                        page.reload(wait_until='domcontentloaded')
                        application = application_ready(page)
                        controller = page.evaluate('navigator.serviceWorker.controller?.scriptURL')
                        observed = controller and controller.split('?v=')[-1]
                        expected = versions[current]
                        if observed != expected:
                            raise AssertionError(f'Controller generation {observed!r} differs from {expected!r}')
                        record[operation] = dict(status='pass', application=application,
                            assertion=dict(kind='workflow', expected=expected, observed=observed, matched=True))
                        record['after_generation'] = expected
                    except Exception as error:
                        record[operation] = dict(status='fail', error=str(error),page_state=page_state(page))
                        try:
                            record['failure_state'] = page.evaluate('''async()=>({
                              controller:navigator.serviceWorker.controller?.scriptURL,
                              workers:(await navigator.serviceWorker.getRegistrations()).map(r=>({
                                active:r.active?.scriptURL,active_state:r.active?.state,
                                waiting:r.waiting?.scriptURL,waiting_state:r.waiting?.state,
                                installing:r.installing?.scriptURL,installing_state:r.installing?.state})),
                              text:document.body.innerText})''')
                        except Exception as diagnostic_error:
                            record['diagnostic_error'] = str(diagnostic_error)
                    if any(isinstance(v,dict) and v.get('status')=='fail' for v in record.values()):
                        try:record['failure_diagnostics']=page.evaluate(DIAGNOSTICS)
                        except Exception as error:record['diagnostics_error']=str(error)
                    record['end_page_state']=page_state(page)
                    record.update(console=console,page_errors=errors,network_pending=pending,network_completed=completed,worker_lifecycle=lifecycle,dedicated_workers=[w.url for w in page.workers],memory_pressure=subprocess.run(['vm_stat'],capture_output=True,text=True).stdout)
                    result['cycle_evidence'].append(record)
                    operations = ('startup','reload','service-worker-update','build-generation-update')
                    for name in operations:
                        if name in record:
                            record[name]['executor']=dict(path=result['executor'],sha256=result['executor_sha256'])
                            record[name]['dependencies']=dict(result['dependencies'])
                            result['cases'][f'cycle-{index}/{name}'] = record[name]
                    write_json(output, result)
                    page.close()
                    passed = all(record.get(name, {}).get('status')=='pass' for name in operations)
                    print(args.browser, args.unit, index, 'pass' if passed else 'fail', flush=True)
                    if not passed:
                        break
                result['status'] = 'pass' if len(result['cycle_evidence'])==args.cycles and all(
                    case['status']=='pass' for case in result['cases'].values()) else 'fail'
            finally:
                browser.close()
        if manifests != [site_manifest(site) for site in sites]:
            result.update(observed_status=result['status'], status='stale')
    except Exception as error:
        result.update(status='error', error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        apply(result,power_before,snapshot())
        write_json(output, result)
    return int(result['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
