"""Full bounded UI/numerical regressions and real cancellation on a private snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright
from common import ROOT,PROOF,sha,write_json
from owned_preview import start
from host_conditions import snapshot,apply
from browser_generation_sequence import site_manifest
from storage_budget import require_space


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--browser',choices=['chrome','edge'],default='chrome')
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    root=PROOF/'cache'/f'embedding-full-{args.browser}-{args.label}'
    output=PROOF/f'evidence/embedding-full-{args.browser}-{args.label}.json'
    if root.exists() or output.exists():raise FileExistsError('Preserve attempts; choose a fresh label')
    require_space(64*1024**2)
    shutil.copytree(args.site,root,copy_function=os.link)
    probe=root/'probes/embedding-worker.js'
    original=probe.read_text()
    source=original.replace("file.name==='embedding_runtime.py'||", "['embedding_runtime.py','embedding_preload.py','course-embedding.json'].includes(file.name)||")
    if 'await Owner(' not in source:
        source=source.replace('nlp=English()','nlp=English()\nfrom embedding_preload import Owner\nawait Owner(nlp,json.load(open(\'course-embedding.json\'))).prepare()')
    probe.unlink();probe.write_text(source)
    shutil.copy2(root/'shinylive/course-embedding-worker.js',root/'probes/course-embedding-worker.js')
    cancel=source[:source.index('    const result=await py.runPythonAsync(`')]+'''    const result=await py.runPythonAsync(`
import asyncio,json
from embedding_runtime import English
from embedding_preload import Owner
owner=Owner(English(),json.load(open('course-embedding.json')))
task=asyncio.create_task(owner.prepare())
while owner.active is None:await asyncio.sleep(0.01)
await asyncio.sleep(0.5)
owner.cancel()
try:await task
except asyncio.CancelledError:pass
json.dumps(owner.last_disposal)
`);
    self.postMessage({done:true,...JSON.parse(result)});
  }catch(error){self.postMessage({error:String(error)});}
};
'''
    (root/'probes/embedding-cancel-worker.js').write_text(cancel)
    helper=PROOF/'scripts/browser_unit2.py'
    helper_source=helper.read_text()
    adapted=helper_source.replace("{'tokenizer.json','word-rows.json.gz','vectors.npy'}","{'manifest.json','tokenizer.json','word-rows.json.gz','vectors.npy'}")
    if "{'manifest.json','tokenizer.json','word-rows.json.gz','vectors.npy'}" not in adapted:
        raise AssertionError('UI helper asset assertion changed')
    namespace={};exec(compile(adapted,str(helper),'exec'),namespace)
    before=snapshot();server,identity=start(root,0)
    inputs=site_manifest(root)
    report=dict(status='running',browser=args.browser,distribution='installed',artifact=str(root.relative_to(PROOF)),
        executor=str(Path(__file__).relative_to(ROOT)),executor_sha256=sha(Path(__file__)),
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in [helper,PROOF/'web/embedding-worker.js',
            PROOF/'scripts/common.py',PROOF/'scripts/owned_preview.py',PROOF/'scripts/host_conditions.py',
            PROOF/'scripts/audit_host_sleep.py',PROOF/'scripts/browser_generation_sequence.py',
            PROOF/'scripts/storage_budget.py',ROOT/'tools/preview.mjs']},
        ui_helper_adaptation=dict(description='Require verified manifest plus three complete assets',
            sha256=hashlib.sha256(adapted.encode()).hexdigest()),
        inputs=inputs,preview_identity=identity,scope='Isolated candidate; bounded regression, no promotion or exhaustive acceptance.')
    write_json(output,report)
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
            report['version']=browser.version
            context=browser.new_context(viewport=dict(width=1440,height=1000));page=context.new_page()
            page.set_default_timeout(300000)
            base=f'http://127.0.0.1:{identity["port"]}'
            report['ui']=namespace['unit2'](page,base+'/unit2/',f'embedding-full-{args.browser}-{args.label}')
            if any(c['status']!='pass' for c in report['ui']['cases'].values()):raise AssertionError('Bounded Unit 2 UI regression failed')
            page.goto(base+'/probes/',wait_until='domcontentloaded')
            numerical=page.evaluate("()=>window.runProof('embedding')")
            report['numerical']=numerical
            if len(numerical['comparisons'])!=2106 or any(c['status']!='pass' for c in numerical['comparisons'].values()):
                raise AssertionError('Numerical regression failed')
            if numerical['vocabulary_keys']!=514157 or numerical['vector_shape']!=[20000,300]:
                raise AssertionError('Incomplete model')
            print(args.browser,'2106 numerical comparisons pass',flush=True)
            pending=[]
            context.route('**/assets/v1/embedding-runtime/vectors.npy',lambda route:pending.append(route))
            cancellation=page.evaluate('''()=>new Promise((resolve,reject)=>{
              const worker=new Worker('./embedding-cancel-worker.js');
              const timer=setTimeout(()=>{worker.terminate();reject(new Error('Cancellation timeout'));},60000);
              worker.onmessage=({data})=>{clearTimeout(timer);worker.terminate();data.error?reject(new Error(data.error)):resolve(data)};
              worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(new Error(e.message))};
              worker.postMessage('cancel');
            })''')
            report['cancellation']=cancellation
            if not pending or not cancellation['acknowledged'] or cancellation['seconds']>=1:
                raise AssertionError('Real blocked-download cancellation did not acknowledge within one second')
            cdp=context.new_cdp_session(page)
            remaining=[t for t in cdp.send('Target.getTargets')['targetInfos'] if t['type']=='worker' and 'course-embedding-worker.js' in t['url']]
            if remaining:raise AssertionError('Preparation worker remained after cancellation')
            report['cancellation']['workers_remaining']=len(remaining)
            for route in pending:
                try:route.abort('failed')
                except Exception:pass
            print(args.browser,'blocked download cancellation acknowledged',cancellation['seconds'],flush=True)
            context.close();browser.close();report['status']='pass'
    except Exception as error:
        report.update(status='fail',error=str(error))
    finally:
        server.terminate();server.wait(timeout=10)
        apply(report,before,snapshot())
        if inputs!=site_manifest(root):report.update(observed_status=report['status'],status='stale')
        write_json(output,report)
    print(report['status'],report.get('error',''),flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
