"""Installed-browser lifecycle sequences with independent bounded concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import subprocess
import sys
from common import PROOF,sha,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--browser',choices=['firefox','safari'],default='firefox')
    parser.add_argument('--cycles',type=int,default=30)
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--label',default='installed-release')
    args=parser.parse_args()
    if args.cycles<1 or not 1<=args.jobs<=4:parser.error('Positive cycles and 1–4 jobs required')
    result=dict(status='running',browser=args.browser,distribution='installed',cycles=args.cycles,
        concurrency=args.jobs,cases={},scope='Independent sequential installed-browser startup/reload/update sequences; not a performance benchmark.')
    output=PROOF/f'evidence/startup-installed-matrix-{args.browser}-{args.label}.json'
    write_json(output,result)
    def run(unit):
        label=f'{args.label}-unit{unit}'
        log=PROOF/f'evidence/startup-actual-{args.browser}-{label}.log'
        with log.open('w') as stream:
            code=subprocess.call([sys.executable,str(PROOF/'scripts/webdriver_startup.py'),
                '--browser',args.browser,'--units',str(unit),'--cycles',str(args.cycles),'--label',label],
                stdout=stream,stderr=subprocess.STDOUT)
        report=PROOF/f'evidence/startup-actual-{args.browser}-{label}.json'
        record=json.loads(report.read_text()) if report.exists() else {}
        return unit,dict(status=record.get('status','error'),exit_code=code,
            report=str(report.relative_to(PROOF)),report_sha256=sha(report) if report.exists() else None,
            version=record.get('version'),input_fingerprint=record.get('input_fingerprint'),
            executed_cases=len(record.get('cases',{})),
            failed_cases=[key for key,value in record.get('cases',{}).items() if value['status']!='pass'])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run,unit) for unit in range(1,8)]
        for future in as_completed(futures):
            unit,record=future.result();result['cases'][f'unit{unit}']=record
            write_json(output,result);print(unit,record['status'],record['executed_cases'],flush=True)
    result['status']='pass' if all(c['status']=='pass' for c in result['cases'].values()) else 'fail'
    write_json(output,result)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
