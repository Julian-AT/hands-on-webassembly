"""Bounded Unit 7 cancellation/worker disposal checks in installed Chrome."""
import re
import time
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
from common import PROOF,write_json,sha
from provenance import browser_stamp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8008/unit7/')
    parser.add_argument('--label',default='')
    parser.add_argument('--candidate-dir',type=Path)
    args=parser.parse_args()
    candidate_paths=[] if args.candidate_dir is None else [args.candidate_dir/path for path in
        ('unit7/app.json','shinylive/course-training-worker.js','assets/v1/neural-runtime/manifest.json')]
    candidate_hashes={str(path):sha(path) for path in candidate_paths}
    result=dict(status='error',browser='chrome',distribution='installed',cases={},
                provenance=browser_stamp('7'),test_sha256=sha(Path(__file__)),
                url=args.url,candidate_inputs=candidate_hashes,
                scope='Bounded dedicated CNN worker reset, stale-result rejection, retry, navigation and termination.')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome')
        try:
            result['version']=browser.version
            page=browser.new_page(viewport={'width':1440,'height':1000})
            cdp=browser.new_browser_cdp_session()
            page.goto(args.url,wait_until='domcontentloaded')
            app=page.frame_locator('iframe');app.locator('h2').wait_for(timeout=120000)
            parent=next(worker for worker in page.workers if 'shinylive' in worker.url)
            parent.evaluate('''()=>{
 const Original=globalThis.Worker;globalThis.__courseWorkerAudit=[];
 globalThis.Worker=class extends Original {
  constructor(...args){super(...args);this.courseUrl=String(args[0]);__courseWorkerAudit.push({event:'create',url:this.courseUrl,time:performance.now()})}
  terminate(){__courseWorkerAudit.push({event:'terminate',url:this.courseUrl,time:performance.now()});return super.terminate()}
 };
}''')
            def tab(name):app.get_by_role('tab',name=name,exact=True).click()
            def workers():
                return {x['targetId'] for x in cdp.send('Target.getTargets')['targetInfos']
                        if 'course-training-worker.js' in x['url']}
            def wait_dead(before,started):
                deadline=started+2
                while workers() & before:
                    if time.perf_counter()>deadline:raise AssertionError('Abandoned compute worker remains alive after two seconds')
                    page.wait_for_timeout(20)
                if time.perf_counter()>deadline:
                    raise AssertionError('Compute worker disposal within two seconds was not established')
            def begin():
                app.locator('#start_train').click()
                expect(app.locator('#train_progress')).to_contain_text('batch',timeout=120000)
                live=workers()
                if not live:raise AssertionError('Dedicated compute worker identity was not observed')
                return live
            tab('CNN: Data');app.locator('#load_data').click()
            expect(app.locator('#data_info')).to_contain_text('54000',timeout=120000)
            tab('CNN: Architecture');app.locator('#load_preset').click()
            expect(app.locator('#arch_text')).not_to_have_value('',timeout=30000)
            app.locator('#apply_arch').click()
            expect(app.locator('#model_summary')).to_contain_text('Conv2d',timeout=30000)
            tab('CNN: Training')
            # A page with a different compute bundle must fail recoverably,
            # before any obsolete numerical result can be accepted.
            intercepted=[]
            def obsolete(route):
                response=route.fetch()
                manifest=response.json();manifest['build_id']='0'*64
                intercepted.append(route.request.url)
                route.fulfill(response=response,body=json.dumps(manifest))
            page.context.route('**/assets/v1/neural-runtime/manifest.json',obsolete)
            app.locator('#start_train').click()
            expect(app.locator('#train_progress')).to_contain_text('Training runtime is obsolete',timeout=120000)
            if not intercepted:raise AssertionError('Obsolete compute bundle was not actually injected')
            result['cases']['stale-build-rejected']=dict(status='pass',interrupted_requests=intercepted,
                message=app.locator('#train_progress').inner_text())
            page.context.unroute('**/assets/v1/neural-runtime/manifest.json',obsolete)
            live=begin();start=time.perf_counter();app.locator('#reset_model').click()
            expect(app.locator('#train_progress .progress-bar')).to_have_attribute('aria-valuenow','0',timeout=1000)
            acknowledgement=time.perf_counter()-start
            if acknowledgement>1:raise AssertionError(f'Cancellation acknowledgement took {acknowledgement:.3f}s')
            wait_dead(live,start)
            result['cases']['reset-during-training']=dict(status='pass',acknowledgement_seconds=acknowledgement,
                                                        termination_seconds=time.perf_counter()-start)
            page.wait_for_timeout(2000)
            expect(app.locator('#train_progress')).not_to_contain_text('batch')
            expect(app.locator('#early_stop_info')).to_contain_text('Model reset.')
            result['cases']['stale-result-rejected']=dict(status='pass',observation_seconds=2)
            live=begin()
            result['cases']['retry-after-cancellation']=dict(status='pass',worker_targets=list(live))
            start=time.perf_counter()
            page.goto('http://127.0.0.1:8008/unit1/',wait_until='domcontentloaded')
            wait_dead(live,start)
            result['cases']['navigate-during-task']=dict(status='pass')
            result['cases']['worker-terminated']=dict(status='pass',termination_seconds=time.perf_counter()-start)
            result['status']='pass'
        except Exception as error:
            result.update(status='fail',error=str(error))
            try:
                result['worker_audit']=parent.evaluate('globalThis.__courseWorkerAudit')
                result['remaining_targets']=cdp.send('Target.getTargets')['targetInfos']
            except Exception as diagnostic_error:result['diagnostic_error']=str(diagnostic_error)
        finally:browser.close()
    if (browser_stamp('7')['fingerprint']!=result['provenance']['fingerprint'] or
        result['test_sha256']!=sha(Path(__file__)) or
        candidate_hashes!={str(path):sha(path) for path in candidate_paths}):
        result['observed_status']=result['status'];result['status']='stale'
    write_json(PROOF/f'evidence/cnn-lifecycle-chrome{("-"+args.label) if args.label else ""}.json',result)
    print(result['status'],result.get('error',''),flush=True)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
