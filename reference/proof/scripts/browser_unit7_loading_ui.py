"""Installed-browser assertions for Unit 7 preparation and queued training."""
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
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright, expect
from browser_html_proxy import site_manifest
from browser_neural_ui import neural_ui
from common import ROOT, PROOF, sha, write_json
from image_cancellation import workers as image_workers,acknowledged as cancellation_acknowledged
from hardware_contract import collect as collect_hardware
from host_conditions import snapshot as host_snapshot,apply as apply_host_conditions


async def sequences(args, result, retain):
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
        context=await browser.new_context()
        page=await context.new_page()
        console=[]
        context.on('console',lambda message:console.append(dict(text=message.text,type=message.type,
            worker=message.worker.url if message.worker else None,location=message.location)))
        result['console']=console
        url=f'http://127.0.0.1:{args.port}/unit7/'
        await page.goto(url,wait_until='domcontentloaded')
        app=page.frame_locator('iframe')
        await expect(app.locator('#variant.shiny-bound-input')).to_be_attached(timeout=120000)
        async def tab(name):await app.get_by_role('tab',name=name,exact=True).click()
        async def acknowledged(name,value):
            await app.locator('body').evaluate('''(el,[name,value])=>new Promise((resolve,reject)=>{
                const deadline=performance.now()+10000;
                const poll=()=>{
                  const values=window.Shiny.shinyapp.$inputValues;
                  const key=Object.keys(values).find(key=>key===name||key.startsWith(name+':'));
                  if(values[key]===value)return resolve();
                  if(performance.now()>deadline)return reject(new Error('Input '+name+' did not reach '+value));
                  setTimeout(poll,20);
                };poll();
            })''',[name,value])
        async def choose(value):
            await app.locator('#variant').select_option(value)
            await acknowledged('variant',value)
        async def slider(name,value):
            await app.locator('#'+name).evaluate('''(el,value)=>{
                const slider=window.jQuery(el).data('ionRangeSlider');
                if(!slider)throw new Error('Slider widget missing');
                slider.update({from:value});window.jQuery(el).trigger('change');
            }''',value)
            await acknowledged(name,value)
        async def loaded(name,counts):
            await expect(app.locator('#data_info')).to_contain_text('Loaded '+name+' dataset.',timeout=120000)
            text=await app.locator('#data_info').inner_text()
            observed={key:int(re.search(pattern,text).group(1)) for key,pattern in [
                ('train',r'training set:\s*(\d+)'),('validation',r'validation set:\s*(\d+)'),('test',r'test set:\s*(\d+)')]}
            if observed!=counts:raise AssertionError(dict(expected=counts,observed=observed,text=text))
            return dict(assertion=dict(kind='workflow',expected=counts,observed=observed,matched=True),data_info=text)
        async def case(name,detail):retain(name,dict(status='pass',**detail))
        started,release=asyncio.Event(),asyncio.Event()
        async def hold(route):
            started.set();await release.wait()
            try:await route.continue_()
            except Exception:pass
        async def hold_train(name):
            started.clear();release.clear()
            await context.route(f'**/images/{name}-train.npz',hold)
        async def release_train(name):
            release.set();await context.unroute(f'**/images/{name}-train.npz',hold)
        failed=[]
        context.on('requestfailed',lambda request:failed.append((request.url,time.perf_counter(),request.failure)))
        async def cancellation(name,start,owners):
            return await cancellation_acknowledged(context,page,owners,failed,name,start)
        await tab('CNN: Data')
        await choose('MNIST');await hold_train('MNIST')
        await app.locator('#load_data').click();await asyncio.wait_for(started.wait(),30)
        await slider('valid',.3)
        await release_train('MNIST')
        await case('loading/input-snapshot',await loaded('MNIST',dict(train=54000,validation=6000,test=10000)))
        await slider('valid',.1)
        await hold_train('MNIST');await app.locator('#load_data').click()
        await asyncio.wait_for(started.wait(),30)
        owners=await image_workers(context,page)
        start=time.perf_counter();await choose('FashionMNIST')
        await case('loading/cancel-on-selection',await cancellation('MNIST',start,owners))
        await release_train('MNIST');await app.locator('#load_data').click()
        await case('loading/retry-after-cancel',await loaded('FashionMNIST',dict(train=54000,validation=6000,test=10000)))
        async def corrupt(route):await route.fulfill(status=200,body=b'incomplete test split')
        await choose('MNIST');await context.route('**/images/MNIST-test.npz',corrupt)
        await app.locator('#load_data').click()
        await expect(app.locator('#data_info')).to_contain_text('checksum',timeout=120000)
        error=await app.locator('#data_info').inner_text()
        await case('loading/corrupt-pair',dict(message=error,assertion=dict(kind='recoverable-error',expected=True,observed='checksum' in error,matched=True)))
        await context.unroute('**/images/MNIST-test.npz',corrupt)
        await app.locator('#load_data').click()
        await case('loading/retry-after-corruption',await loaded('MNIST',dict(train=54000,validation=6000,test=10000)))
        # Reset must abort loading even when the active panel is Training.
        await hold_train('MNIST');await app.locator('#load_data').click()
        await asyncio.wait_for(started.wait(),30);await tab('CNN: Training')
        owners=await image_workers(context,page)
        start=time.perf_counter();await app.locator('#reset_model').click()
        await case('loading/cancel-on-reset',await cancellation('MNIST',start,owners))
        await release_train('MNIST')
        # A Train click on an unprepared dataset must resume with both original
        # snapshots. This small custom architecture exercises the dispatch path;
        # numerical training parity is certified by its separate matrices.
        await tab('CNN: Data');await choose('USPS');await slider('batch',256)
        await tab('CNN: Architecture')
        architecture=json.dumps({'layers':[{'type':'flatten'},{'type':'linear','out_features':10}]})
        await app.locator('#arch_editor').evaluate("(el,text)=>window.ace.edit(el).setValue(text,-1)",architecture)
        await acknowledged('arch_text',architecture)
        await app.locator('#apply_arch').click()
        await expect(app.locator('#arch_table_block')).to_contain_text('linear',timeout=30000)
        await tab('CNN: Training');await slider('epochs',1)
        await app.locator('#seed').fill('42');await app.locator('#seed').press('Tab');await acknowledged('seed',42)
        await hold_train('USPS');await app.locator('#start_train').click()
        await asyncio.wait_for(started.wait(),30)
        await slider('epochs',2)
        await release_train('USPS')
        await expect(app.locator('#train_progress')).to_contain_text('Training complete',timeout=180000)
        await expect(app.locator('#early_stop_info')).to_contain_text('epoch 1,',timeout=30000)
        info=await app.locator('#early_stop_info').inner_text()
        await case('training/queued-input-snapshot',dict(message=info,assertion=dict(kind='workflow',expected=True,observed='epoch 1,' in info,matched=True)))
        await tab('CNN: Data')
        await case('loading/queued-training-data',await loaded('USPS',dict(train=6562,validation=729,test=2007)))
        # Disposal is observed on the actual owning workers, not only the DOM.
        await choose('MNIST');await hold_train('MNIST');await app.locator('#load_data').click()
        await asyncio.wait_for(started.wait(),30)
        workers=list(page.workers)
        if not workers:raise AssertionError('No owning worker observed')
        disposed=[]
        for worker in workers:worker.on('close',lambda closed:disposed.append(time.perf_counter()))
        start=time.perf_counter();await page.goto(url.replace('/unit7/','/probes/'),wait_until='domcontentloaded')
        deadline=start+2
        while len(disposed)<len(workers) and time.perf_counter()<deadline:await asyncio.sleep(.01)
        release.set()
        if len(disposed)!=len(workers):raise AssertionError('The abandoned workers survived two seconds')
        seconds=max(disposed)-start
        await case('loading/navigation-disposal',dict(seconds=seconds,workers=len(workers),
            assertion=dict(kind='workflow',expected=True,observed=seconds<=2 and len(disposed)==len(workers),matched=True)))
        await browser.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--browser',choices=('chrome','edge'),required=True)
    parser.add_argument('--port',type=int,default=8070)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    inputs=site_manifest(args.site)
    host_before=host_snapshot()
    output=PROOF/f'evidence/unit7-loading-ui-{args.browser}-{args.label}.json'
    result=dict(status='running',browser=args.browser,distribution='installed',inputs=inputs,cases={},hardware=collect_hardware(),
        artifact=str(args.site.resolve().relative_to(PROOF)),executor='proof/scripts/browser_unit7_loading_ui.py',
        executor_sha256=sha(Path(__file__)),dependencies={f'proof/scripts/{name}':sha(PROOF/'scripts'/name)
            for name in ('browser_neural_ui.py','image_cancellation.py','common.py','owned_preview.py','host_conditions.py','audit_host_sleep.py','hardware_contract.py','browser_html_proxy.py')},
        provenance=dict(scope='isolated-artifact',fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()),
        scope='Unit 7 all five full image datasets, filter upload, snapshot loading, cancellation, reset, corrupt-pair retry, queued training snapshot and owning worker disposal. Full numerical, layer and gesture parity remain separate obligations.')
    def retain(name,case):result['cases'][name]=case;write_json(output,result)
    from owned_preview import start
    server,result['preview_identity']=start(args.site,args.port)
    args.port=result['preview_identity']['port']
    try:
        url=f'http://127.0.0.1:{args.port}/unit7/'
        for _ in range(100):
            if server.poll() is not None:raise RuntimeError('Isolated server failed to start')
            try:urllib.request.urlopen(url,timeout=1).close();break
            except OSError:time.sleep(.1)
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
            result['version']=browser.version
            context=browser.new_context();page=context.new_page()
            result['full_data_console']=[];result['failed_requests']=[]
            context.on('console',lambda message:result['full_data_console'].append(dict(text=message.text,
                type=message.type,worker=message.worker.url if message.worker else None,location=message.location)))
            context.on('requestfailed',lambda request:result['failed_requests'].append(dict(url=request.url,failure=request.failure)))
            neural_ui(page,url,f'candidate-{args.browser}-unit7-{args.label}',7,fail_fast=True,on_case=retain)
            browser.close()
        asyncio.run(sequences(args,result,retain))
        result['status']='pass'
    except Exception as error:result.update(status='fail',error=str(error))
    finally:
        server.terminate();server.wait(timeout=10)
        (args.site/result['preview_identity']['marker']).unlink()
        apply_host_conditions(result,host_before,host_snapshot())
        if inputs!=site_manifest(args.site):result.update(observed_status=result['status'],status='stale')
        write_json(output,result)
    print(args.browser,result['status'],result.get('error',''),flush=True)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
