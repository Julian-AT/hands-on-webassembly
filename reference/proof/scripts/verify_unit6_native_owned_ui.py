"""Bounded native Unit 6 Reset/disposable-process verification in installed browsers."""
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
from playwright.async_api import async_playwright,expect
from common import ROOT,PROOF,sha,write_json
from browser_html_proxy import site_manifest


async def run(args,result,output):
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
        result['version']=browser.version
        try:
            for seed in (0,42,123):
                page=await browser.new_page();console=[]
                page.on('console',lambda message:console.append(dict(type=message.type,text=message.text)))
                await page.goto(f'http://127.0.0.1:{args.port}/',wait_until='domcontentloaded')
                app=page
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
                async def slider(name,value):
                    await app.locator('#'+name).evaluate('''(el,value)=>{
                        window.jQuery(el).data('ionRangeSlider').update({from:value});
                        window.jQuery(el).trigger('change');
                    }''',value)
                    await acknowledged(name,value)
                await app.get_by_role('tab',name='FNN: Data',exact=True).click()
                await app.locator('#dataset').select_option('toy_reg');await acknowledged('dataset','toy_reg')
                await slider('n_pairs',2000)
                await app.locator('#load').click()
                await expect(app.locator('#data_info')).to_contain_text('N=2000',timeout=30000)
                await app.get_by_role('tab',name='FNN:Architecture',exact=True).click()
                await app.locator('#preset').select_option('Toy Regression – 2×Hidden (ReLU)')
                await app.locator('#load_preset').click()
                await expect(app.locator('#arch_text')).to_have_value(re.compile('"type": "relu"'),timeout=30000)
                architecture=await app.locator('#arch_text').input_value()
                await acknowledged('arch_text',architecture)
                await app.locator('#apply_arch').click()
                await expect(app.locator('#model_summary')).to_contain_text('Total parameters',timeout=30000)
                await app.get_by_role('tab',name='FNN: Training',exact=True).click()
                await app.locator('#train_seed').fill(str(seed));await app.locator('#train_seed').press('Tab')
                await acknowledged('train_seed',seed);await slider('epochs',100)
                await app.locator('#train').click()
                await expect(app.locator('#train_progress')).to_contain_text('Epoch',timeout=60000)
                before=await app.locator('#train_progress').inner_text()
                def processes():
                    rows=subprocess.run(['ps','-axo','pid,ppid,command'],capture_output=True,text=True,check=True).stdout.splitlines()
                    return [dict(pid=int(fields[0]),command=fields[2]) for row in rows[1:] if len(fields:=row.strip().split(None,2))==3 and fields[1]==str(args.native_pid) and 'multiprocessing.spawn' in fields[2]]
                workers=processes()
                if len(workers)!=1:raise AssertionError(f'Expected one owning native process, observed {workers}')
                disposed=[]
                start=time.perf_counter();await app.locator('#reset').click()
                await expect(app.locator('#train_progress .progress-bar')).to_have_attribute('aria-valuenow','0',timeout=1000)
                await expect(app.locator('#train_progress')).not_to_contain_text('Epoch',timeout=1000)
                acknowledged_seconds=time.perf_counter()-start
                if acknowledged_seconds>1:raise AssertionError(f'Reset acknowledgement took {acknowledged_seconds}s')
                while not disposed and time.perf_counter()-start<2:
                    if not any(item['pid']==workers[0]['pid'] for item in processes()):disposed.append(time.perf_counter())
                    else:await asyncio.sleep(.01)
                if not disposed:raise AssertionError('Owning training worker survived two seconds')
                await asyncio.sleep(.1)
                after=await app.locator('#train_progress').inner_text()
                info=await app.locator('#best_model_info').inner_text()
                observed=dict(reset_acknowledged=acknowledged_seconds<=1,worker_disposed=disposed[0]-start<=2,
                    obsolete_progress_absent='Epoch' not in after and 'Training complete' not in after,
                    obsolete_model_absent='not set' in info)
                expected={key:True for key in observed}
                if observed!=expected:raise AssertionError(observed)
                result['cases'][f'seed-{seed}/reset']=dict(status='pass',before=before,after=after,
                    acknowledged_seconds=acknowledged_seconds,disposal_seconds=disposed[0]-start,
                    assertion=dict(kind='workflow',expected=expected,observed=observed,matched=True),console=console)
                write_json(output,result)
                await slider('epochs',2);await app.locator('#train').click()
                await expect(app.locator('#train_progress')).to_contain_text('Training complete',timeout=60000)
                await expect(app.locator('#best_model_info')).to_contain_text('Using last epoch model',timeout=30000)
                # The old 100-epoch context must stay gone after replacement completes.
                current=processes()
                deadline=time.perf_counter()+2
                while current and time.perf_counter()<deadline:
                    await asyncio.sleep(.01);current=processes()
                if current:raise AssertionError('Completed task retained its worker')
                result['cases'][f'seed-{seed}/retry']=dict(status='pass',assertion=dict(kind='workflow',
                    expected=dict(complete=True,owning_workers=0),observed=dict(complete=True,owning_workers=len(current)),matched=True))
                write_json(output,result);await page.close()
                print(args.browser,seed,'reset and retry pass',flush=True)
            result['status']='pass'
        finally:await browser.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--browser',choices=('chrome','edge'),required=True);parser.add_argument('--port',type=int,default=8117)
    parser.add_argument('--label',required=True);args=parser.parse_args()
    output=PROOF/f'evidence/unit6-native-owned-training-ui-{args.browser}-{args.label}.json'
    if output.exists():raise FileExistsError('Retain previous evidence; choose another label')
    inputs={str(path.relative_to(args.reference)):sha(path) for path in sorted(args.reference.rglob('*')) if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts}
    result=dict(status='running',cases={},browser=args.browser,distribution='installed',inputs=inputs,
        artifact=str(args.reference.resolve().relative_to(PROOF)),executor='proof/scripts/verify_unit6_native_owned_ui.py',executor_sha256=sha(Path(__file__)),
        dependencies={'proof/scripts/browser_html_proxy.py':sha(PROOF/'scripts/browser_html_proxy.py'),'proof/scripts/common.py':sha(PROOF/'scripts/common.py')},
        provenance=dict(scope='isolated-artifact',fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()),
        scope='Bounded Unit 6 worker Reset and subsequent training at seeds 0/42/123. Full numerical, image-training and comprehensive control matrices remain required.')
    log=PROOF/f'evidence/unit6-native-owned-training-ui-{args.browser}-{args.label}.log'
    stream=log.open('w')
    server=subprocess.Popen([str(ROOT/'.venv/bin/shiny'),'run','--host','127.0.0.1','--port',str(args.port),'app.py'],cwd=args.reference,env=dict(os.environ,MPLBACKEND='Agg',OMP_NUM_THREADS='1'),stdout=stream,stderr=stream)
    args.native_pid=server.pid
    try:
        for _ in range(100):
            if server.poll() is not None:raise RuntimeError('Candidate server exited')
            try:urllib.request.urlopen(f'http://127.0.0.1:{args.port}/',timeout=1).close();break
            except OSError:time.sleep(.1)
        asyncio.run(run(args,result,output))
    except Exception as error:result.update(status='fail',error=str(error))
    finally:
        server.terminate();server.wait(timeout=10)
        stream.close();result['log_sha256']=sha(log)
        write_json(output,result)
    print(args.browser,result['status'],result.get('error',''),flush=True)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
