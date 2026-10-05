"""Fresh private artifacts for the six bounded application integration runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from artifact_provenance import fingerprint
from browser_generation_sequence import site_manifest
from common import PROOF, sha, write_json
from evidence_tree import validate_tree
from hardware_contract import collect, errors
from storage_budget import require_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--reuse', type=Path,
                        help='Reuse individually passing, freshly validated children from this matrix')
    args = parser.parse_args()
    root = PROOF / 'cache' / f'image-integration-matrix-{args.label}'
    output = PROOF / 'evidence' / f'image-integration-matrix-{args.label}.json'
    if root.exists() or output.exists():
        raise FileExistsError('Retain previous attempts; choose a fresh label')
    require_space(128 * 1024**2)
    inputs = site_manifest(args.site)
    # The aggregate retains its own unchanged candidate to bind assembly bytes.
    site = root / 'source'
    shutil.copytree(args.site, site, copy_function=os.link)
    hardware = collect()
    if errors(hardware):
        raise ValueError(errors(hardware))
    executors = {5: 'browser_unit5_loading_ui.py', 6: 'browser_image_loading_ui.py',
                 7: 'browser_unit7_loading_ui.py'}
    dependencies = ['common.py', 'artifact_provenance.py', 'evidence_tree.py',
                    'assertions.py', 'hardware_contract.py', 'storage_budget.py',
                    'browser_generation_sequence.py', *executors.values()]
    report = dict(status='running', cases={}, hardware=hardware, inputs=inputs,
                  artifact=str(site.relative_to(PROOF)),
                  provenance=dict(scope='isolated-artifact', fingerprint=hashlib.sha256(
                      json.dumps(inputs, sort_keys=True).encode()).hexdigest()),
                  executor='proof/scripts/run_image_preparation_matrix.py',
                  executor_sha256=sha(Path(__file__)),
                  dependencies={f'proof/scripts/{name}': sha(PROOF/'scripts'/name)
                                for name in dependencies},
                  scope='Six bounded installed-browser application integrations. '
                        'Complete numerical, appearance and assignment acceptance remain required.')
    write_json(output, report)

    if args.reuse:
        previous = json.loads(args.reuse.read_text())
        if previous.get('inputs') != inputs:
            raise ValueError('Previous matrix tested a different complete candidate')
        for name, case in previous.get('cases', {}).items():
            if case.get('status') != 'pass':
                continue
            failures = validate_tree(dict(cases={name: case}), PROOF,
                report['provenance']['fingerprint'], hardware=hardware,
                fingerprint_resolver=lambda child, name: fingerprint(child, PROOF))
            if not failures:
                report['cases'][name] = dict(case, reused_from=str(args.reuse.resolve().relative_to(PROOF)))
        write_json(output, report)

    def run(unit, browser, index):
        private = root / f'{browser}-unit{unit}'
        shutil.copytree(site, private, copy_function=os.link)
        label = f'image-matrix-{args.label}'
        log = PROOF / 'evidence' / f'image-integration-{browser}-unit{unit}-{args.label}.log'
        with log.open('w') as stream:
            code = subprocess.call([sys.executable, str(PROOF/'scripts'/executors[unit]),
                                    '--site', str(private), '--browser', browser,
                                    '--port', str(8270 + index), '--label', label],
                                   stdout=stream, stderr=subprocess.STDOUT)
        path = PROOF/'evidence'/f'unit{unit}-loading-ui-{browser}-{label}.json'
        child = json.loads(path.read_text()) if path.exists() else {}
        passed = code == 0 and child.get('status') == 'pass'
        return f'{browser}/unit{unit}', dict(status='pass' if passed else 'fail',
            exit_code=code, report=str(path.relative_to(PROOF)),
            report_sha256=sha(path) if path.exists() else None, browser=browser,
            version=child.get('version'), input_fingerprint=child.get('provenance', {}).get('fingerprint'),
            executed_cases=len(child.get('cases', {})), log=str(log.relative_to(PROOF)),
            log_sha256=sha(log), assertion=dict(kind='workflow', expected=True, observed=passed, matched=passed))

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(run, unit, browser, index) for index, (unit, browser) in
                 enumerate((unit, browser) for unit in (5, 6, 7) for browser in ('chrome', 'edge'))
                 if f'{browser}/unit{unit}' not in report['cases']]
        for task in as_completed(tasks):
            name, case = task.result()
            report['cases'][name] = case
            write_json(output, report)
            print(name, case['status'], case['executed_cases'], flush=True)
    report['status'] = 'pass' if all(c['status']=='pass' for c in report['cases'].values()) else 'fail'
    try:
        if fingerprint(report, PROOF) != report['provenance']['fingerprint']:
            raise ValueError('Matrix assembly changed')
        report['dependency_errors'] = validate_tree(report, PROOF, report['provenance']['fingerprint'],
            hardware=hardware, fingerprint_resolver=lambda child, name: fingerprint(child, PROOF))
        if report['dependency_errors']:
            report['status'] = 'fail'
    except ValueError as error:
        report.update(status='stale', error=str(error))
    write_json(output, report)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
