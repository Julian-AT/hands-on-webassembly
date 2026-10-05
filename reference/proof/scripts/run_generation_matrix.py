"""Retain fail-fast, uninstrumented sequences for every assignment and browser."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from browser_generation_sequence import site_manifest
from common import PROOF,sha,write_json,archive_file
from artifact_provenance import fingerprint
from evidence_tree import validate_tree


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,default=PROOF/'site')
    parser.add_argument('--browsers',nargs='+',choices=('chrome','edge','firefox','safari'),default=['chrome','edge'])
    parser.add_argument('--webdriver-python',type=Path,default=PROOF/'cache/browser-verification-env/bin/python')
    parser.add_argument('--cycles',type=int,default=30)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    if args.cycles<1 or not 1<=args.jobs<=4:parser.error('Use positive cycles and 1–4 independent functional runners')
    sites=[args.before.resolve(),args.after.resolve()]
    inputs={str(site.relative_to(PROOF)):site_manifest(site) for site in sites}
    output=PROOF/f'evidence/generation-matrix-{args.label}.json'
    if output.exists():raise FileExistsError('Retain previous matrix; choose a fresh label')
    result=dict(status='running',cycles_requested=args.cycles,cases={},inputs=inputs,
        provenance=dict(scope='two-isolated-artifacts',fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()),
        executor='proof/scripts/run_generation_matrix.py',executor_sha256=sha(Path(__file__)),
        dependencies={f'proof/scripts/{name}':sha(PROOF/'scripts'/name) for name in
            ('browser_generation_sequence.py','browser_html_proxy.py','browser_startup.py','common.py')},
        concurrency=args.jobs,scope='All seven assignments in installed Chrome/Edge, consecutive startup/reload/query update/actual build changes, each child stops on the first failure. Functional stress coverage; no performance or hardware certification.')
    write_json(output,result)
    def run(browser,unit,index):
        log=PROOF/f'evidence/generation-{browser}-unit{unit}-{args.label}.log';archive_file(log)
        with log.open('w') as stream:
            python = str(args.webdriver_python) if browser in ('firefox','safari') else sys.executable
            code=subprocess.call([python,str(PROOF/'scripts/browser_generation_sequence.py'),
                '--browser',browser,'--unit',str(unit),'--before',str(sites[0]),'--after',str(sites[1]),
                '--cycles',str(args.cycles),'--port',str(8080+index),'--label',args.label],stdout=stream,stderr=subprocess.STDOUT)
        path=PROOF/f'evidence/generation-{browser}-unit{unit}-{args.label}.json'
        child=json.loads(path.read_text()) if path.exists() else {}
        count=len(child.get('cases',{}))
        passed=code==0 and child.get('status')=='pass' and count==args.cycles*4
        record=dict(status='pass' if passed else 'fail',exit_code=code,browser=browser,
            report=str(path.relative_to(PROOF)),report_sha256=sha(path) if path.exists() else None,
            input_fingerprint=child.get('provenance',{}).get('fingerprint'),version=child.get('version'),
            executed_cases=count,log=str(log.relative_to(PROOF)),log_sha256=sha(log),
            assertion=dict(kind='workflow',expected=dict(cycles=args.cycles,cases=args.cycles*4),
                observed=dict(cycles=len(child.get('cycle_evidence',[])),cases=count),matched=passed))
        return f'{browser}/unit{unit}',record
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks=[pool.submit(run,browser,unit,index) for index,(browser,unit) in
            enumerate((browser,unit) for browser in args.browsers for unit in range(1,8))]
        for task in as_completed(tasks):
            name,record=task.result();result['cases'][name]=record;write_json(output,result)
            print(name,record['status'],record['executed_cases'],flush=True)
    result['status']='pass' if all(case['status']=='pass' for case in result['cases'].values()) else 'fail'
    try:
        if fingerprint(result,PROOF)!=result['provenance']['fingerprint']:raise ValueError('Matrix artifact fingerprint differs')
        result['dependency_errors']=validate_tree(result,PROOF,result['provenance']['fingerprint'],
            fingerprint_resolver=lambda child,name:fingerprint(child,PROOF))
        versions={browser:{case['version'] for name,case in result['cases'].items() if name.startswith(browser+'/')}
            for browser in args.browsers}
        if any(len(values)!=1 or None in values for values in versions.values()):raise ValueError('Missing or mixed installed browser versions')
        result['versions']={browser:next(iter(values)) for browser,values in versions.items()}
        if result['dependency_errors']:result['status']='fail'
    except ValueError as error:result.update(observed_status=result['status'],status='stale',error=str(error))
    write_json(output,result)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
