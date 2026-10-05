"""Bounded Unit 5 measurement on the explicitly authorized certification device."""
import json
import subprocess
import threading
import time
from playwright.sync_api import sync_playwright
from common import PROOF, write_json
from provenance import stamp
from browser_unit5 import unit5
from hardware_contract import collect


def main():
    samples=[];stop=threading.Event()
    hardware=collect()
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome')
        cdp=browser.new_browser_cdp_session()
        pid=next(x['id'] for x in cdp.send('SystemInfo.getProcessInfo')['processInfo'] if x['type']=='browser')
        def memory():
            while not stop.is_set():
                records=[list(map(int,line.split())) for line in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines()]
                children={pid};old=None
                while old!=children:
                    old=children.copy();children.update(row[0] for row in records if row[1] in children)
                samples.append({'seconds':time.perf_counter()-start,'processes':len(children),
                    'sum_process_rss_bytes':sum(row[2]*1024 for row in records if row[0] in children)})
                stop.wait(.25)
        context=browser.new_context(viewport={'width':1440,'height':1000})
        context.add_init_script('''window.courseLags=[];let previous=performance.now();setInterval(()=>{const now=performance.now();window.courseLags.push(Math.max(0,now-previous-50));previous=now;},50);''')
        network=[]
        def transfer(request):
            try: network.append(request.sizes())
            except Exception: pass
        context.on('requestfinished',transfer)
        page=context.new_page();page.set_default_timeout(15000);page.set_default_navigation_timeout(120000)
        result={'browser':'chrome','distribution':'installed','version':browser.version,'hardware':hardware,'provenance':stamp()}
        start=time.perf_counter();thread=threading.Thread(target=memory);thread.start()
        try:
            result.update(unit5(page,'http://127.0.0.1:8008/unit5/','benchmark-chrome-unit5'))
            result['status']='pass' if all(v['status']=='pass' for v in result['cases'].values()) else 'fail'
            result['event_loop_lag_ms']=page.evaluate('window.courseLags')
        except Exception as error: result.update(status='error',error=str(error))
        finally:
            stop.set();thread.join();result['duration_seconds']=time.perf_counter()-start
            result['memory_samples']=samples
            result['peak_sum_process_rss_bytes']=max(x['sum_process_rss_bytes'] for x in samples)
            result['response_body_bytes']=sum(x['responseBodySize'] for x in network)
            result['scope']='Unit 5 cold startup and 14 listed workflows including full MNIST and Fashion-MNIST three-epoch runs. RSS sums include shared pages and are not private physical memory. This is not 8 GB certification; other assignments and warm benchmarks remain required.'
            browser.close()
    if result['provenance']['fingerprint']!=stamp()['fingerprint']:
        result['original_status']=result['status'];result['status']='stale'
    write_json(PROOF/'evidence/benchmark-development-unit5.json',result)
    print(result['status'],result['duration_seconds'],result['peak_sum_process_rss_bytes'])


if __name__=='__main__':main()
