"""Certification accepts only fresh observations of the authorized device."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from hardware_contract import contract, errors
from gate import evaluate
from coverage_contract import requirements
from common import UNITS


class HardwareContractTests(unittest.TestCase):
    def observed(self):
        value, digest = contract()
        return dict(value['active'], contract_id=value['id'],
                    contract_sha256=digest, measurement='native sysctl and system_profiler')

    def test_active_device_passes_existing_suite(self):
        from host_conditions import apply
        wake=dict(type='Wake',timestamp=9)
        conditions=apply(dict(status='pass'),
            dict(started_at=10,collected_at=11,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=11)),
            dict(started_at=19,collected_at=20,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=20)))['host_conditions']
        report = dict(status='pass', provenance={'fingerprint':'current'},
                      hardware=self.observed(), host_conditions=conditions,cases={'training':{'status':'pass'}})
        requirement = dict(hardware=True, section='cases', required_cases=['training'])
        self.assertEqual(evaluate(requirement, report, 'current')['status'], 'pass')
        del report['host_conditions']
        self.assertEqual(evaluate(requirement, report, 'current')['status'], 'fail')

    def test_legacy_device_and_incomplete_claims_fail(self):
        for field, wrong in [('physical_ram_bytes',8*1024**3), ('model','Mac16,6'),
                             ('cpu','Apple M4'), ('integrated_graphics',False),
                             ('contract_sha256','stale'), ('measurement',None)]:
            with self.subTest(field=field):
                observed = self.observed()
                observed[field] = wrong
                self.assertTrue(errors(observed))

    def test_suite_ids_and_supersession_retained(self):
        rows = requirements({str(unit):{'interface':[]} for unit in UNITS})
        self.assertEqual(len(rows),140)
        hardware = [r for r in rows if r.get('hardware') and r['id'].startswith('unit')]
        self.assertEqual(len(hardware),28)
        self.assertTrue(all('/8gb/' in r['id'] for r in hardware))
        self.assertTrue(all(r['legacy_hardware_id']=='physical-8gb-integrated-graphics' for r in hardware))


if __name__ == '__main__': unittest.main()
