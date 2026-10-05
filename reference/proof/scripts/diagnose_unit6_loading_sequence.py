"""Network, console and WebSocket investigation of the complete loading precursor sequence. Diagnostic timing; no acceptance claim."""
import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
import urllib.request
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright, expect
from browser_html_proxy import site_manifest
from browser_neural_ui import neural_ui
from common import ROOT,PROOF,sha,write_json


async def loading_sequences(browser_name,url,result):
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome' if browser_name=='chrome' else 'msedge')
        context=await browser.new_context()
        page=await context.new_page()
        trace=[];console=[];messages=[]
        result.update(network_trace=trace,console=console,websocket_messages=messages)
        page.on('console',lambda message:console.append(dict(type=message.type,text=message.text)))
        page.on('pageerror',lambda error:console.append(dict(type='pageerror',text=str(error))))
        context.on('request',lambda request:trace.append(dict(event='request',url=request.url,time=time.perf_counter())) if '/images/' in request.url else None)
        context.on('requestfailed',lambda request:trace.append(dict(event='failed',url=request.url,error=request.failure,time=time.perf_counter())))
        context.on('response',lambda response:trace.append(dict(event='response',url=response.url,status=response.status,headers=response.headers,time=time.perf_counter())) if '/images/' in response.url else None)
        def websocket(socket):
            def frame(direction,data):
                messages.append(dict(direction=direction,time=time.perf_counter(),data=data.decode(errors='replace') if isinstance(data,bytes) else data))
            socket.on('framesent',lambda data:frame('sent',data))
            socket.on('framereceived',lambda data:frame('received',data))
        page.on('websocket',websocket)
        await page.goto(url,wait_until='domcontentloaded')
        app=page.frame_locator('iframe')
        await expect(app.locator('#dataset.shiny-bound-input')).to_be_attached(timeout=120000)
        await app.get_by_role('tab',name='FNN: Data',exact=True).click()
        async def choose(dataset):
            await app.locator('#dataset').select_option(dataset)
            await app.locator('body').evaluate('''(el,value)=>new Promise(resolve=>{
                const poll=()=>window.Shiny.shinyapp.$inputValues.dataset===value?resolve():setTimeout(poll,20);poll();
            })''',dataset)
        async def loaded(dataset,train=54000):
            await expect(app.locator('#data_info')).to_contain_text('Loaded '+dataset+' data.',timeout=120000)
            await expect(app.locator('#data_info')).to_contain_text(str(train),timeout=30000)
            await expect(app.locator('#data_info')).to_contain_text('10000',timeout=30000)
            errors=await app.locator('.shiny-output-error:visible').all_text_contents()
            if errors:raise AssertionError(errors)
            text=await app.locator('#data_info').inner_text()
            counts={name:int(re.search(pattern,text).group(1)) for name,pattern in
                [('train',r'Training set\s*=\s*(\d+)'),('validation',r'Validation set\s*=\s*(\d+)'),('test',r'Test set\s*=\s*(\d+)')]}
            if counts!=dict(train=train,validation=60000-train,test=10000):raise AssertionError(counts)
            return dict(text=text,counts=counts)
        async def validation(value):
            # Exercise the real widget/input binding while the owning task is
            # suspended. This scenario tests server-side input snapshots;
            # pointer and keyboard gestures remain separate control cases.
            widget=await app.locator('#valid').evaluate('''(element,value)=>{
                const slider=window.jQuery(element).data('ionRangeSlider');
                if(!slider)throw new Error('The native slider widget is missing');
                slider.update({from:value});
                window.jQuery(element).trigger('change');
                return {widget:slider.result.from, input:element.value};
            }''',value)
            if abs(widget['widget']-value)>1e-8:raise AssertionError(widget)
            await app.locator('body').evaluate('''(el,value)=>new Promise((resolve,reject)=>{
                const deadline=performance.now()+5000;
                const poll=()=>{
                  if(Math.abs(window.Shiny.shinyapp.$inputValues.valid-value)<1e-8)return resolve();
                  if(performance.now()>deadline)return reject(new Error('Validation slider did not reach '+value+'; observed '+window.Shiny.shinyapp.$inputValues.valid));
                  setTimeout(poll,20);
                };poll();
                })''',value)
        started,release=asyncio.Event(),asyncio.Event()
        async def hold(route):
            started.set()
            await release.wait()
            try:await route.continue_()
            except Exception:pass # The cancellation sequence intentionally closes it.
        async def hold_train():
            started.clear();release.clear()
            await context.route('**/images/MNIST-train.npz',hold)
        # Inputs change while the extended task waits on a real request. The
        # resulting split must use the invocation snapshot, not those new inputs.
        await hold_train()
        await choose('MNIST')
        await app.locator('#load').click()
        await asyncio.wait_for(started.wait(),30)
        await validation(.3)
        release.set()
        await context.unroute('**/images/MNIST-train.npz',hold)
        observed=await loaded('MNIST',54000)
        result['cases']['loading/input-snapshot']=dict(status='pass',assertion=dict(kind='workflow',
            expected=dict(train=54000,validation=6000,test=10000),observed=observed['counts'],matched=True),data_info=observed['text'])
        # Changing the selection cancels the request even before another load.
        await validation(.1)
        await hold_train()
        failed=[]
        context.on('requestfailed',lambda request:failed.append((request.url,time.perf_counter(),request.failure)))
        await app.locator('#load').click()
        await asyncio.wait_for(started.wait(),30)
        start=time.perf_counter()
        await choose('FashionMNIST')
        deadline=start+1
        while not any('/images/MNIST-train.npz' in item[0] for item in failed) and time.perf_counter()<deadline:
            await asyncio.sleep(.01)
        cancelled=[item for item in failed if '/images/MNIST-train.npz' in item[0]]
        release.set()
        await context.unroute('**/images/MNIST-train.npz',hold)
        if not cancelled:raise AssertionError('Dataset replacement failed to acknowledge actual download cancellation within one second')
        seconds=cancelled[0][1]-start
        if seconds>1:raise AssertionError(f'Cancellation took {seconds}s')
        result['cases']['loading/cancel-on-selection']=dict(status='pass',seconds=seconds,requests=cancelled)
        await app.locator('#load').click()
        result['cases']['loading/retry-after-cancel']=dict(status='pass',data_info=await loaded('FashionMNIST'))
        # An incomplete pair must surface a recoverable error. The same Load
        # control must then start a fresh invocation and clear that error.
        async def corrupt(route):await route.fulfill(status=200,body=b'corrupted test split')
        await context.route('**/images/MNIST-test.npz',corrupt)
        await choose('MNIST');await app.locator('#load').click()
        await expect(app.locator('#data_info')).to_contain_text('checksum',timeout=120000)
        result['cases']['loading/corrupt-download']=dict(status='pass',message=await app.locator('#data_info').inner_text())
        await context.unroute('**/images/MNIST-test.npz',corrupt)
        await app.locator('#load').click()
        result['cases']['loading/retry-after-corruption']=dict(status='pass',data_info=await loaded('MNIST'))
        # Navigation closes the owning app worker as well as the held request.
        await hold_train();await app.locator('#load').click()
        await asyncio.wait_for(started.wait(),30)
        workers=list(page.workers)
        if not workers:raise AssertionError('No owning application worker observed')
        disposed=[]
        for worker in workers:worker.on('close',lambda closed:disposed.append(time.perf_counter()))
        start=time.perf_counter()
        await page.goto(url.replace('/unit6/','/probes/'),wait_until='domcontentloaded')
        deadline=start+2
        while len(disposed)<len(workers) and time.perf_counter()<deadline:await asyncio.sleep(.01)
        release.set()
        if len(disposed)!=len(workers):raise AssertionError('The abandoned loading worker survived two seconds')
        result['cases']['loading/navigation-disposal']=dict(status='pass',seconds=max(disposed)-start,workers=len(workers))
        await browser.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--browser',choices=['chrome','edge'],required=True)
    parser.add_argument('--port',type=int,default=8066)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    inputs=site_manifest(args.site)
    output=PROOF/f'evidence/unit6-loading-sequence-diagnostic-{args.browser}-{args.label}.json'
    result=dict(status='running',browser=args.browser,distribution='installed',inputs=inputs,cases={},
        artifact=str(args.site.resolve().relative_to(PROOF)),executor='proof/scripts/diagnose_unit6_loading_sequence.py',
        executor_sha256=sha(Path(__file__)),provenance=dict(scope='isolated-artifact',fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()),
        dependencies={'proof/scripts/browser_neural_ui.py':sha(PROOF/'scripts/browser_neural_ui.py')},
        scope='Diagnostic of complete loading precursor sequence only. Network, console and WebSocket observers affect timing. No acceptance or reliability claim.')
    write_json(output,result)
    server=subprocess.Popen(['node',str(ROOT/'tools/preview.mjs'),str(args.site)],env=dict(os.environ,PORT=str(args.port)),stdout=subprocess.DEVNULL)
    try:
        url=f'http://127.0.0.1:{args.port}/unit6/'
        for _ in range(100):
            if server.poll() is not None:raise RuntimeError('Isolated server failed to start')
            try:urllib.request.urlopen(url,timeout=1).close();break
            except OSError:time.sleep(.1)
        asyncio.run(loading_sequences(args.browser,url,result))
        result['status']='pass'
    except Exception as error:
        result.update(status='fail',error=str(error))
    finally:
        server.terminate();server.wait(timeout=10)
        if inputs!=site_manifest(args.site):result.update(observed_status=result['status'],status='stale')
        write_json(output,result)
    print(args.browser,result['status'],result.get('error',''),flush=True)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
