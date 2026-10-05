"""Fail-fast installed-browser sequences across two actual static build generations.

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
from playwright.sync_api import sync_playwright
from browser_html_proxy import site_manifest
from browser_startup import application_ready
from common import ROOT, PROOF, sha, write_json
from host_conditions import snapshot as power_snapshot, apply as apply_power_conditions
from owned_preview import start as start_preview


def generation(site):
    worker = (site / 'shinylive-sw.js').read_text()
    match = re.search(r'var version = "course-([^"\n]+)";', worker)
    if not match:
        raise ValueError('Artifact lacks its content-bound runtime generation')
    return match.group(1)


def switch_files(served, target, changed):
    for relative in changed:
        destination = served / relative
        temporary = destination.with_name(destination.name + '.generation-next')
        os.link(target / relative, temporary)
        os.replace(temporary, destination)


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
    if manifests[0].keys() != manifests[1].keys():
        raise ValueError('This sequence requires identical asset paths in both generations')
    versions = [generation(site) for site in sites]
    if versions[0] == versions[1]:
        raise ValueError('Query changes alone do not constitute a build upgrade')
    changed = sorted(key for key in manifests[0] if manifests[0][key] != manifests[1][key])
    served = PROOF/'cache'/f'generation-verified-{args.browser}-unit{args.unit}-{args.label}'
    if served.exists():
        raise FileExistsError('Retain previous sequences; choose a fresh label')
    shutil.copytree(sites[0], served, copy_function=os.link)
    inputs = {str(site.relative_to(PROOF)): manifest for site, manifest in zip(sites, manifests)}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    result = dict(status='running', browser=args.browser, distribution='installed',
        executor='proof/scripts/browser_generation_verified_origin.py',
        dependencies={'proof/scripts/browser_startup.py':sha(PROOF/'scripts/browser_startup.py'),
                      'proof/scripts/browser_html_proxy.py':sha(PROOF/'scripts/browser_html_proxy.py'),
                      'proof/scripts/common.py':sha(PROOF/'scripts/common.py')},
        provenance=dict(scope='two-isolated-artifacts', fingerprint=fingerprint), inputs=inputs,
        executor_sha256=sha(Path(__file__)), generations=versions, changed_files=changed,
        cycles_requested=args.cycles, cycle_evidence=[], cases={}, diagnostic_instrumentation=False,
        scope='Uninstrumented startup, reload, fixed-query replacement and actual build-generation reload, stopped on the first failure. Each cycle alternates the two recorded artifacts.')
    output = PROOF/f'evidence/generation-verified-{args.browser}-unit{args.unit}-{args.label}.json'
    result['dependencies'].update({f'proof/scripts/{name}':sha(PROOF/'scripts'/name)
        for name in ('host_conditions.py','audit_host_sleep.py','owned_preview.py')})
    write_json(output, result)
    try:
        power_before = power_snapshot()
    except Exception as error:
        result.update(status='unavailable', host_conditions_error=str(error))
        write_json(output, result)
        return 1
    def ready_case(page):
        application=application_ready(page)
        expected=dict(connected=True,loading_visible=False)
        observed={key:application[key] for key in expected}
        if observed!=expected:raise AssertionError(observed)
        return dict(status='pass',application=application,
            assertion=dict(kind='workflow',expected=expected,observed=observed,matched=True))
    server = None
    try:
        server,result['preview_identity']=start_preview(served,args.port)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
            result['version'] = browser.version
            context = browser.new_context()
            try:
                current = 0
                for index in range(1, args.cycles+1):
                    page = context.new_page()
                    record = dict(index=index, before_generation=versions[current])
                    operation = 'startup'
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
                        switch_files(served, sites[current], changed)
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
                        record[operation] = dict(status='fail', error=str(error))
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
        if server is not None:
            server.terminate()
            server.wait(timeout=10)
        try:
            apply_power_conditions(result, power_before, power_snapshot())
        except Exception as error:
            result.update(observed_status=result['status'], status='unavailable', host_conditions_error=str(error))
        write_json(output, result)
    return int(result['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
