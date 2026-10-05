"""Run independent per-assignment lifecycle sequences with bounded concurrency.

Each sequence owns its browser context and runs all requested cycles in order.
This is functional stress coverage, never a hardware/performance benchmark.
Individual reports survive interruption of the matrix orchestrator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from common import PROOF, write_json, sha, archive_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cycles',type=int,default=30)
    parser.add_argument('--jobs',type=int,default=7)
    parser.add_argument('--browsers',nargs='+',choices=['chrome','firefox','edge','webkit'],default=['chrome','firefox'])
    parser.add_argument('--label',default='parallel-release')
    args=parser.parse_args()
    if args.cycles < 1 or not 1 <= args.jobs <= 7:parser.error('Positive cycles and 1–7 jobs required')
    result=dict(status='running',cycles=args.cycles,concurrency=args.jobs,cases={},
                scope='Independent sequential per-assignment startup/reload/update cycles; functional stress, not performance benchmark.')
    path=PROOF/f'evidence/startup-matrix-{args.label}.json'
    write_json(path,result)

    def run(browser,unit):
        log=PROOF/f'evidence/startup-{browser}-unit{unit}-{args.label}.log'
        archive_file(log)
        with log.open('w') as output:
            code=subprocess.call([sys.executable,str(PROOF/'scripts/browser_startup.py'),
                '--browser',browser,'--units',str(unit),'--cycles',str(args.cycles),'--label',args.label],
                stdout=output,stderr=subprocess.STDOUT)
        report=PROOF/f'evidence/startup-{browser}-units-{unit}-{args.label}.json'
        content=json.loads(report.read_text()) if report.exists() else {}
        return browser,unit,dict(status=content.get('status','error'),exit_code=code,
            report=str(report.relative_to(PROOF)),executed_cases=len(content.get('cases',{})),
            report_sha256=sha(report) if report.exists() else None,
            input_fingerprint=content.get('provenance',{}).get('fingerprint'),
            version=content.get('version'),
            failed_cases=[key for key,value in content.get('cases',{}).items() if value.get('status')!='pass'])

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks=[pool.submit(run,browser,unit) for browser in args.browsers for unit in range(1,8)]
        for task in as_completed(tasks):
            browser,unit,record=task.result()
            result['cases'][f'{browser}/unit{unit}']=record
            write_json(path,result)
            print(browser,unit,record['status'],record['executed_cases'],flush=True)
    result['status']='pass' if all(value['status']=='pass' for value in result['cases'].values()) else 'fail'
    write_json(path,result)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
