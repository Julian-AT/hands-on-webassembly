"""Same-browser, same-viewport native/browser appearance diagnostic."""
import base64,json,sys,hashlib,argparse,os,re,socket,subprocess,time,urllib.request,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright,expect
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from common import PROOF,ROOT,sha,write_json
from browser_generation_sequence import site_manifest
from owned_preview import start
from hardware_contract import collect
from host_conditions import snapshot,apply

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True)
    parser.add_argument('--native-port',type=int,default=8291);args=parser.parse_args()
    root=PROOF/'evidence'/f'appearance-unit1-{args.label}'
    if root.exists():raise FileExistsError('Preserve prior matched-font diagnostics')
    root.mkdir()
    before=snapshot();hardware=collect()
    with socket.socket() as reservation:reservation.bind(('127.0.0.1',args.native_port))
    artifact=PROOF/'cache'/f'appearance-unit1-{args.label}'
    shutil.copytree(PROOF/'site',artifact,copy_function=os.link)
    server,identity=start(artifact,0)
    native_log=root/'native.log';log=native_log.open('w')
    native=subprocess.Popen([str(Path(sys.executable).parent/'shiny'),'run','--host','127.0.0.1',
        '--port',str(args.native_port),str(PROOF/'reference/unit1/app.py')],cwd=PROOF/'reference/unit1',
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=log)
    for _ in range(120):
        if native.poll() is not None:raise RuntimeError('Native reference exited; see retained log')
        try:urllib.request.urlopen(f'http://127.0.0.1:{args.native_port}/',timeout=1).close();break
        except OSError:time.sleep(.1)
    fonts=PROOF/'assets/v1/www/vendor/zephyr'
    font_sources={p.name:sha(p) for p in fonts.glob('*.ttf')}
    font_css=(fonts/'inter.css').read_text()
    font_css=re.sub(r'url\(([^)]+)\)',r'url(https://fonts.gstatic.com/course-proof/\1)',font_css)
    records={};images={};plots={}
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome');version=browser.version
        try:
            context=browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1)
            for kind in ['native','browser']:
                page=context.new_page()
                if kind=='native':
                    url=f'http://127.0.0.1:{args.native_port}/unit1-native-comparison/'
                    page.route(url,lambda route:route.fulfill(content_type='text/html',body=f'<!doctype html><html style="height:100%;margin:0"><body style="height:100%;margin:0"><iframe style="width:100%;height:100%;border:0" src="http://127.0.0.1:{args.native_port}/"></iframe></body></html>'))
                    page.route('https://fonts.googleapis.com/**',lambda route:route.fulfill(content_type='text/css',body=font_css))
                    page.route('https://fonts.gstatic.com/course-proof/**',lambda route:route.fulfill(
                        content_type='font/ttf',body=(fonts/route.request.url.rsplit('/',1)[1]).read_bytes(),
                        headers={'Access-Control-Allow-Origin':'*'}))
                else:url=f'http://127.0.0.1:{identity["port"]}/unit1/'
                print(kind,'navigation',flush=True)
                page.goto(url,wait_until='domcontentloaded');app=page.frame_locator('iframe')
                app.locator('#viz_features option').first.wait_for(state='attached',timeout=120000)
                for state in ['intro','correlation']:
                    print(kind,state,'begin',flush=True)
                    if state=='correlation':
                        app.get_by_role('tab',name='Dataset Overview',exact=True).click();app.locator('#dataset').select_option('breast');app.locator('#load_data').click()
                        expect(app.locator('#dataset_info')).to_contain_text('Samples: 569',timeout=120000)
                        app.get_by_role('tab',name='Correlation Analysis',exact=True).click();app.locator('#run_analysis').click()
                        expect(app.locator('#analysis_results')).to_contain_text('Strongest Correlations',timeout=120000)
                        expect(app.locator('#analysis_plot img')).to_have_attribute('src',__import__('re').compile('^data:image/'),timeout=120000)
                        el=app.locator('#analysis_plot img');encoded=el.get_attribute('src').split(',')[1]
                        raw=base64.b64decode(encoded);name=f'{kind}-correlation-{hashlib.sha256(raw).hexdigest()[:12]}.png';(root/name).write_bytes(raw);plots[kind]=name
                    data=app.locator('body').evaluate('''async body=>{await Promise.race([document.fonts.ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('Font readiness exceeded 30 seconds')),30000))]);return {faces:[...document.fonts].map(f=>({family:f.family,weight:f.weight,status:f.status})),styles:[...document.querySelectorAll('h2,.nav-link.active')].map(e=>{const s=getComputedStyle(e);return {text:e.innerText,font:s.fontFamily,size:s.fontSize,weight:s.fontWeight,bounds:e.getBoundingClientRect().toJSON()}}),width:innerWidth,height:innerHeight,scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight}}''')
                    print(kind,state,'fonts ready',flush=True)
                    raw=page.screenshot(full_page=False);name=f'{kind}-{state}-{hashlib.sha256(raw).hexdigest()[:12]}.png';(root/name).write_bytes(raw)
                    images[kind,state]=name;records[kind+'/'+state]=dict(artifact=name,sha256=sha(root/name),metrics=data)
                page.close()
        finally:
            browser.close();native.terminate();native.wait(timeout=10);log.close()
            server.terminate();server.wait(timeout=10);(artifact/identity['marker']).unlink()
    comparisons={}
    for state in ['intro','correlation']:
        a=np.asarray(Image.open(root/images['native',state]).convert('RGB'));b=np.asarray(Image.open(root/images['browser',state]).convert('RGB'))
        comparisons[state]=dict(status='pass' if np.array_equal(a,b) else 'fail',different_pixels=int(np.any(a!=b,axis=2).sum()),max_channel_difference=int(abs(a.astype(int)-b.astype(int)).max()))
    a=np.asarray(Image.open(root/plots['native']).convert('RGBA'));b=np.asarray(Image.open(root/plots['browser']).convert('RGBA'))
    comparisons['plot']=dict(status='pass' if np.array_equal(a,b) else 'fail',native_shape=list(a.shape),browser_shape=list(b.shape),different_pixels=int(np.any(a!=b,axis=2).sum()) if a.shape==b.shape else None)
    report=dict(status='pass' if all(v['status']=='pass' for v in comparisons.values()) else 'fail',browser='chrome',distribution='installed',version=version,script_sha256=sha(Path(__file__)),records=records,comparisons=comparisons,
        hardware=hardware,font_sources=font_sources,native_reference=site_manifest(PROOF/'reference/unit1'),
        browser_artifact=str(artifact.relative_to(PROOF)),browser_inputs=site_manifest(artifact),preview_identity=identity,
        scope='Same installed Chrome context and exact pinned Inter font bytes, 1440x1000 viewport and scale one, equivalent iframes. Two-state diagnostic, without masks or pixel tolerance; not complete appearance acceptance.')
    apply(report,before,snapshot())
    write_json(PROOF/f'evidence/unit1-matched-fonts-{args.label}.json',report);print(comparisons,flush=True)
if __name__=='__main__':main()
