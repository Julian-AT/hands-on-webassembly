"""Observed owned errors, retry, and rejection of abandoned worker failures."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from playwright.async_api import async_playwright,expect
from common import ROOT,PROOF,sha,write_json
from browser_html_proxy import site_manifest
from host_conditions import snapshot as power_snapshot,apply as apply_power_conditions


async def run(args,result,output):
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
        result['version']=browser.version
        context=await browser.new_context();page=await context.new_page()
        try:
            await page.goto(f'http://127.0.0.1:{args.port}/unit6/',wait_until='domcontentloaded')
            app=page.frame_locator('iframe')
            await expect(app.locator('#dataset.shiny-bound-input')).to_be_attached(timeout=120000)
            async def acknowledged(name,value):
                await app.locator('body').evaluate('''(el,[name,value])=>new Promise((resolve,reject)=>{
                    const end=performance.now()+10000;
                    const poll=()=>{const values=window.Shiny.shinyapp.$inputValues;
                        const key=Object.keys(values).find(key=>key===name||key.startsWith(name+':'));
                        if(values[key]===value)return resolve();
                        if(performance.now()>end)return reject(new Error('Input '+name+' was not acknowledged'));
                        setTimeout(poll,20)};poll();
                })''',[name,value])
            async def tab(name):await app.get_by_role('tab',name=name,exact=True).click()
            await tab('FNN: Data');await app.locator('#dataset').select_option('toy_reg')
            await acknowledged('dataset','toy_reg');await app.locator('#load').click()
            await expect(app.locator('#data_info')).to_contain_text('N=200',timeout=30000)
            await tab('FNN:Architecture')
            await app.locator('#preset').select_option('Toy Regression – 2×Hidden (ReLU)')
            await app.locator('#load_preset').click()
            await expect(app.locator('#arch_text')).to_have_value(re.compile('"type": "relu"'),timeout=30000)
            await acknowledged('arch_text',await app.locator('#arch_text').input_value())
            await app.locator('#apply_arch').click()
            await expect(app.locator('#model_summary')).to_contain_text('Total parameters',timeout=30000)
            await tab('FNN: Training')
            await app.locator('#epochs').evaluate("el=>{window.jQuery(el).data('ionRangeSlider').update({from:2});window.jQuery(el).trigger('change');}")
            await acknowledged('epochs',2)
            pattern='**/assets/v1/neural-runtime/manifest.json'
            async def failed(route):await route.fulfill(status=503,body=b'Unavailable diagnostic runtime')
            await context.route(pattern,failed);await app.locator('#train').click()
            await expect(app.locator('#train_progress')).to_contain_text('Training runtime could not load',timeout=60000)
            await expect(app.locator('#best_model_info')).to_contain_text('Error:',timeout=30000)
            deadline=time.perf_counter()+2
            while any('course-training-worker.js' in w.url for w in page.workers) and time.perf_counter()<deadline:await asyncio.sleep(.01)
            workers=len([w for w in page.workers if 'course-training-worker.js' in w.url])
            if workers:raise AssertionError('Failed task retained its worker')
            result['cases']['current-error']=dict(status='pass',message=await app.locator('#best_model_info').inner_text(),
                assertion=dict(kind='workflow',expected=dict(error_visible=True,workers=0),observed=dict(error_visible=True,workers=workers),matched=True))
            await context.unroute(pattern,failed);await app.locator('#train').click()
            await expect(app.locator('#train_progress')).to_contain_text('Training complete',timeout=60000)
            await expect(app.locator('#best_model_info')).to_contain_text('Using last epoch model',timeout=30000)
            result['cases']['retry']=dict(status='pass',assertion=dict(kind='workflow',expected='Training complete',observed=(await app.locator('#train_progress').inner_text()).splitlines()[-1],matched=True))
            started=asyncio.Event();release=asyncio.Event()
            async def hold(route):
                started.set();await release.wait()
                try:await route.fulfill(status=503,body=b'Abandoned runtime failure')
                except Exception:pass # The canceled owning worker closes the request.
            await context.route(pattern,hold);await app.locator('#train').click()
            await asyncio.wait_for(started.wait(),30)
            old=[w for w in page.workers if 'course-training-worker.js' in w.url]
            if len(old)!=1:raise AssertionError('No single owning worker before replacement')
            closed=[];old[0].on('close',lambda w:closed.append(time.perf_counter()))
            start=time.perf_counter()
            await tab('FNN: Data');await app.locator('#dataset').select_option('toy_sine')
            await acknowledged('dataset','toy_sine')
            release.set();await context.unroute(pattern,hold)
            while not closed and time.perf_counter()-start<2:await asyncio.sleep(.01)
            if not closed:raise AssertionError('Abandoned worker survived replacement')
            await app.locator('#load').click();await expect(app.locator('#data_info')).to_contain_text('Sine-wave regression',timeout=30000)
            await tab('FNN: Training')
            await expect(app.locator('#best_model_info')).to_contain_text('not set',timeout=30000)
            await expect(app.locator('#train_progress .progress-bar')).to_have_attribute('aria-valuenow','0',timeout=1000)
            text=await app.locator('#train_progress').inner_text()
            observed=dict(obsolete_error_absent='Error:' not in text,obsolete_completion_absent='Training complete' not in text,worker_disposed=closed[0]-start<=2)
            expected={key:True for key in observed}
            if observed!=expected:raise AssertionError(observed)
            result['cases']['abandoned-error']=dict(status='pass',disposal_seconds=closed[0]-start,assertion=dict(kind='workflow',expected=expected,observed=observed,matched=True))
            await app.locator('#train').click();await expect(app.locator('#train_progress')).to_contain_text('Training complete',timeout=60000)
            result['cases']['replacement-retry']=dict(status='pass',assertion=dict(kind='workflow',expected='Training complete',observed=(await app.locator('#train_progress').inner_text()).splitlines()[-1],matched=True))
            result['status']='pass'
        finally:await browser.close();write_json(output,result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--browser',choices=('chrome','edge'),required=True);parser.add_argument('--port',type=int,default=8085)
    parser.add_argument('--label',required=True);args=parser.parse_args()
    output=PROOF/f'evidence/unit6-owned-recovery-ui-{args.browser}-{args.label}.json'
    if output.exists():raise FileExistsError('Retain previous evidence; choose a new label')
    inputs=site_manifest(args.site)
    result=dict(status='running',cases={},browser=args.browser,distribution='installed',inputs=inputs,
        artifact=str(args.site.resolve().relative_to(PROOF)),executor='proof/scripts/verify_unit6_owned_recovery_ui.py',executor_sha256=sha(Path(__file__)),
        dependencies={f'proof/scripts/{name}':sha(PROOF/'scripts'/name) for name in ('browser_html_proxy.py','common.py','host_conditions.py','audit_host_sleep.py')},
        provenance=dict(scope='isolated-artifact',fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()),
        scope='Bounded Unit 6 owned runtime error, retry, cancellation and abandoned-failure rejection; no comprehensive parity claim.')
    power_before=power_snapshot()
    server=subprocess.Popen(['node',str(ROOT/'tools/preview.mjs'),str(args.site)],env=dict(os.environ,PORT=str(args.port)),stdout=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if server.poll() is not None:raise RuntimeError('Candidate server exited')
            try:urllib.request.urlopen(f'http://127.0.0.1:{args.port}/unit6/',timeout=1).close();break
            except OSError:time.sleep(.1)
        asyncio.run(run(args,result,output))
    except Exception as error:result.update(status='fail',error=str(error))
    finally:
        server.terminate();server.wait(timeout=10)
        apply_power_conditions(result,power_before,power_snapshot())
        if inputs!=site_manifest(args.site):result.update(observed_status=result['status'],status='stale')
        write_json(output,result)
    print(args.browser,result['status'],result.get('error',''),flush=True)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
