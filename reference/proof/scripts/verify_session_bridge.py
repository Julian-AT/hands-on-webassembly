"""Bounded bridge checks on real assignments, with a private hardlink candidate.

This is a launcher harness, not the deferred Next.js shell or lifecycle release
matrix. Candidate changes never write through shared inode bindings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from playwright.sync_api import sync_playwright
from common import ROOT, PROOF, sha, write_json
from owned_preview import start
from patch_runtime import session_startup_script
from host_conditions import snapshot, apply


def replace_file(path, content):
    # Unlink before writing: the candidate shares only immutable input bytes.
    path.unlink(missing_ok=True)
    path.write_text(content)


def candidate(site, label):
    target = PROOF/'cache'/f'session-bridge-{label}'
    if target.exists(): raise FileExistsError(f'Use a fresh label: {target}')
    from storage_budget import require_space
    require_space(512 * 1024**2)
    manifest = json.loads((site/'assignment-sessions.json').read_text())
    if manifest.get('protocol') != 1 or not manifest.get('build_id'):
        raise ValueError('Rebuild the integrated bridge before verification')
    client = (site/'shinylive/shinylive.js').read_text()
    for disposer in ('registerDisposer(() => this.pyWorker.terminate())','registerDisposer(dispose)'):
        if disposer not in client: raise ValueError('Integrated runtime disposer missing')
    for unit in range(1,8):
        html = (site/f'unit{unit}/index.html').read_text()
        identity = re.search(r'window.courseAssignmentIdentity = Object.freeze\(([^\n]+)\);',html)
        expected = dict(unit=unit,build_id=manifest['build_id'],source_id=sha(site/f'unit{unit}/app.json'))
        if identity is None or json.loads(identity[1]) != expected:
            raise ValueError(f'Unit {unit}: integrated bridge identity differs')
        if manifest['sources'].get(str(unit)) != expected['source_id']:
            raise ValueError(f'Unit {unit}: integrated manifest source differs')
    shutil.copytree(site, target, copy_function=os.link)
    replace_file(target/'launcher-session.mjs', (PROOF/'runtime/launcher-session.mjs').read_text())
    replace_file(target/'index.html', '''<!doctype html><meta charset="utf-8"><title>Session bridge verification harness</title>
<div id="frames"></div><script type="module">
import {AssignmentLauncher} from './launcher-session.mjs';
window.bridgeStatuses=[];
const manifest=await (await fetch('./assignment-sessions.json')).json();
window.bridgeLauncher=new AssignmentLauncher({window, container:document.getElementById('frames'), manifest,
  onStatus: status=>window.bridgeStatuses.push({...status, at:performance.now()})});
</script>''')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, default=PROOF/'site')
    parser.add_argument('--browser', choices=['chrome', 'edge'], default='chrome')
    parser.add_argument('--label', required=True)
    parser.add_argument('--units', type=int, nargs='+', default=list(range(1, 8)))
    parser.add_argument('--faults', action='store_true', help='Also exercise failed fetch, Retry and missing disposal acknowledgement')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.label): parser.error('Label must be a safe filename')
    if not args.units or any(unit not in range(1, 8) for unit in args.units): parser.error('Units must be 1–7')
    target = candidate(args.site, args.label)
    output = PROOF/'evidence'/f'session-bridge-{args.browser}-{args.label}.json'
    before = snapshot()
    server, identity = start(target, 0)
    inputs = {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    report = dict(status='running', browser=args.browser, distribution='installed',
        artifact=str(target.relative_to(PROOF)), inputs=inputs, preview_identity=identity,
        executor=str(Path(__file__).relative_to(ROOT)), executor_sha256=sha(Path(__file__)),
        dependencies={str((PROOF/'scripts'/name).relative_to(ROOT)): sha(PROOF/'scripts'/name)
            for name in ('common.py', 'owned_preview.py', 'host_conditions.py', 'audit_host_sleep.py', 'patch_runtime.py')},
        provenance=dict(scope='isolated-artifact', fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()),
        cases={}, requested_units=args.units,
        scope='Bounded readiness and disposal bridge harness on real applications; no shell, generation matrix or training cancellation certification.')
    write_json(output, report)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome' if args.browser == 'chrome' else 'msedge')
            report['version'] = browser.version
            context = browser.new_context()
            page = context.new_page()
            try:
                page.goto(f'http://127.0.0.1:{identity["port"]}/', wait_until='domcontentloaded')
                page.wait_for_function('()=>!!window.bridgeLauncher')
                cdp = context.new_cdp_session(page)
                for unit in args.units:
                    old_workers = {t['targetId'] for t in cdp.send('Target.getTargets')['targetInfos'] if t['type'] == 'worker'}
                    started = time.monotonic()
                    page.evaluate('(unit)=>window.bridgeLauncher.select(unit)', unit)
                    deadline = started + 2
                    while old_workers & {t['targetId'] for t in cdp.send('Target.getTargets')['targetInfos'] if t['type'] == 'worker'}:
                        if time.monotonic() > deadline: raise AssertionError('Abandoned dedicated workers remained after two seconds')
                        page.wait_for_timeout(25)
                    disposal_ms = (time.monotonic() - started)*1000
                    page.wait_for_function('(unit)=>window.bridgeLauncher.active?.identity.unit===unit && window.bridgeLauncher.active.state==="ready"', arg=unit,
                                           timeout=180000)
                    observed = page.evaluate('''()=>{
                      const outer=window.bridgeLauncher.active.frame;
                      const app=outer.contentWindow.document.querySelector('#root iframe').contentWindow.Shiny.shinyapp;
                      return {unit:window.bridgeLauncher.active.identity.unit,connected:app.isConnected(),
                        socket_open:app.$socket.readyState===1,initialized_inputs:Object.keys(app.$initialInput).length>0,
                        one_frame:document.querySelectorAll('#frames>iframe').length===1,
                        startup_hidden:outer.contentWindow.document.getElementById('course-startup').hidden};
                    }''')
                    expected = dict(unit=unit, connected=True, socket_open=True, initialized_inputs=True,
                                    one_frame=True, startup_hidden=True)
                    assert observed == expected, observed
                    report['cases'][f'unit{unit}/ready-inputs'] = dict(status='pass', disposal_ms=disposal_ms,
                        assertion=dict(kind='workflow', expected=expected, observed=observed, matched=True))
                    write_json(output, report)
                    print(args.browser, f'unit{unit}', 'ready with connected initialized inputs', flush=True)
                if args.faults:
                    context.route('**/unit1/app.json', lambda route: route.abort('failed'))
                    page.evaluate('()=>window.bridgeLauncher.select(1,{retry:true})')
                    page.wait_for_function('()=>window.bridgeLauncher.active.state==="error"', timeout=180000)
                    failed_identity = page.evaluate('()=>window.bridgeLauncher.active.identity')
                    error = page.evaluate('()=>window.bridgeStatuses.filter(s=>s.state==="error").at(-1).error')
                    assert error and 'timed out' not in error, error
                    context.unroute('**/unit1/app.json')
                    page.evaluate('()=>window.bridgeLauncher.retry()')
                    page.wait_for_function('()=>window.bridgeLauncher.active.state==="ready"', timeout=180000)
                    recovered_identity = page.evaluate('()=>window.bridgeLauncher.active.identity')
                    observed = dict(error_reported=bool(error), retry_ready=True,
                        replaced_session=failed_identity['session_id'] != recovered_identity['session_id'])
                    expected = dict(error_reported=True, retry_ready=True, replaced_session=True)
                    assert observed == expected
                    report['cases']['failed-fetch-and-retry'] = dict(status='pass', error=error,
                        assertion=dict(kind='recoverable-error', expected=expected, observed=observed, matched=True))
                    # Suppress only the cooperative command. The launcher must call
                    # the child's resource registry directly when its deadline expires.
                    page.evaluate('''()=>{
                      const child=window.bridgeLauncher.active.frame.contentWindow;
                      const post=child.postMessage.bind(child);
                      child.postMessage=(message,...args)=>{if(message?.type!=='course:dispose')post(message,...args)};
                    }''')
                    old_workers = {t['targetId'] for t in cdp.send('Target.getTargets')['targetInfos'] if t['type'] == 'worker'}
                    assert old_workers, 'Forced-disposal case must actually own a calculation worker'
                    started = time.monotonic()
                    page.evaluate('()=>{window.bridgeLauncher.select(2);return window.bridgeLauncher.select(3)}')
                    while old_workers & {t['targetId'] for t in cdp.send('Target.getTargets')['targetInfos'] if t['type'] == 'worker'}:
                        if time.monotonic() > started + 2: raise AssertionError('Forced disposal missed its two-second worker deadline')
                        page.wait_for_timeout(25)
                    elapsed = (time.monotonic()-started)*1000
                    observed = page.evaluate('''()=>({latest_unit:window.bridgeLauncher.active.identity.unit,
                      one_frame:document.querySelectorAll('#frames>iframe').length===1,
                      forced:window.bridgeStatuses.filter(s=>s.state==='disposed').at(-1).outcome==='deadline'})''')
                    expected = dict(latest_unit=3, one_frame=True, forced=True)
                    assert observed == expected, observed
                    report['cases']['missing-ack-forced-disposal-and-latest-selection'] = dict(status='pass', disposal_ms=elapsed,
                        terminated_worker_count=len(old_workers),
                        assertion=dict(kind='workflow', expected=expected, observed=observed, matched=True))
                    page.wait_for_function('()=>window.bridgeLauncher.active.state==="ready"', timeout=180000)
                    write_json(output, report)
                old_workers = {t['targetId'] for t in cdp.send('Target.getTargets')['targetInfos'] if t['type'] == 'worker'}
                started = time.monotonic()
                page.evaluate('()=>window.bridgeLauncher.close()')
                deadline = started + 2
                while old_workers & {t['targetId'] for t in cdp.send('Target.getTargets')['targetInfos'] if t['type'] == 'worker'}:
                    if time.monotonic() > deadline: raise AssertionError('Dedicated workers survived close deadline')
                    page.wait_for_timeout(25)
                elapsed = (time.monotonic()-started)*1000
                observed = page.evaluate('''()=>({remaining_frames:document.querySelectorAll('#frames>iframe').length,
                  cooperative:window.bridgeStatuses.filter(s=>s.state==='disposed').at(-1).outcome==='cooperative'})''')
                expected = dict(remaining_frames=0, cooperative=True)
                assert observed == expected, observed
                report['cases']['close-and-worker-disappearance'] = dict(status='pass', disposal_ms=elapsed,
                    assertion=dict(kind='workflow', expected=expected, observed=observed, matched=True))
                report['statuses'] = page.evaluate('()=>window.bridgeStatuses')
                report['status'] = 'pass'
            finally:
                try: report['statuses'] = page.evaluate('()=>window.bridgeStatuses')
                except Exception: pass
                context.close()
                browser.close()
    except Exception as error:
        report.update(status='fail', error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        apply(report, before, snapshot())
        observed = {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        if observed != inputs: report.update(observed_status=report['status'], status='stale')
        write_json(output, report)
    print(report['status'], output, flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__': sys.exit(main())
