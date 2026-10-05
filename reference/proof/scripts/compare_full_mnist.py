"""Compare visible native/browser full-data results without hiding divergence.

Displayed values are rounded by the original UI. Matching these would still
require unrounded histories, weights and predictions to certify numerical parity.
"""
import json
import re
from common import PROOF, write_json
from provenance import stamp


def main():
    paths = [PROOF/'evidence'/name for name in ('native-unit7-full-mnist.json','unit7-full-mnist.json')]
    reports = [json.loads(path.read_text()) for path in paths]
    comparisons = {}
    for case, field in [('full-default-training','best_info'),('evaluation','metrics')]:
        values = [report.get('cases',{}).get(case,{}).get(field) for report in reports]
        comparisons[case] = {'status':'pass' if all(values) and values[0]==values[1] else 'fail','native':values[0],'browser':values[1]}
    current = stamp()
    stale = any(report.get('provenance',{}).get('fingerprint') != current['fingerprint'] for report in reports)
    status = 'fail' if any(c['status']!='pass' for c in comparisons.values()) else 'pass'
    write_json(PROOF/'evidence/full-mnist-comparison.json',{
        'status':'stale' if stale else status,'observed_status':status,
        'provenance':current,'input_reports':[str(path.relative_to(PROOF)) for path in paths],
        'comparisons':comparisons,
        'scope':'Rounded original UI output comparison only; not full numerical certification.'})
    print(status, '(stale input evidence)' if stale else '')
    return int(stale or status!='pass')


if __name__=='__main__':raise SystemExit(main())
