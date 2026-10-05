import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import sha
from evidence_tree import validate_tree


class EvidenceTreeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.proof = Path(self.temp.name)
        (self.proof/'evidence').mkdir()
        self.child = dict(status='pass', browser='edge', distribution='installed', version='154',
            provenance={'fingerprint':'current'}, cases={'startup':{'status':'pass'}})
        self.path = self.proof/'evidence/child.json'
        self.root = dict(status='pass', browser='edge', version='154', cases={})
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.child))
        self.root['cases']['unit1'] = dict(status='pass', report='evidence/child.json',
            report_sha256=sha(self.path), input_fingerprint='current', version='154')

    def errors(self):
        return validate_tree(self.root, self.proof, 'current', browser='edge', version='154')

    def test_bound_child_passes(self):
        self.assertEqual(self.errors(), [])

    def test_missing_or_modified_child_fails(self):
        self.path.write_text('{}')
        self.assertTrue(self.errors())
        self.path.unlink()
        self.assertTrue(self.errors())

    def test_child_status_checks_browser_and_fingerprint_not_summary(self):
        for changes in ({'status':'fail'}, {'browser':'chrome'}, {'distribution':'playwright'},
                        {'version':'153'}, {'provenance':{'fingerprint':'old'}},
                        {'cases':{'startup':{'status':'skipped'}}}):
            original = dict(self.child)
            self.child.update(changes)
            self.save()
            self.assertTrue(self.errors(), changes)
            self.child = original

    def test_recursive_missing_grandchild_fails(self):
        self.child['cases']['nested'] = dict(status='pass', report='evidence/missing.json',
            report_sha256='none', input_fingerprint='current')
        self.save()
        self.assertTrue(self.errors())

    def test_empty_child_cannot_pass(self):
        self.child['cases']={}
        self.save()
        self.assertTrue(self.errors())

    def test_assertions_are_checked_recursively_with_fixed_numeric_tolerances(self):
        assertion = dict(kind='computation', expected=[1.0, 2], observed=[1.00001, 2], matched=True)
        self.child['cases']['startup']['assertion'] = assertion
        self.save()
        self.assertEqual(self.errors(), [])
        for observed in ([1.1, 2], [1.0, 2.0], [True, 2]):
            assertion['observed'] = observed
            self.save()
            self.assertTrue(self.errors())
        self.child['cases']['startup']['assertion'] = None
        self.save()
        self.assertTrue(self.errors())

    def test_nonpassing_suite_and_invalid_check_map_cannot_hide_in_child(self):
        self.child['suites'] = {'hidden': {'status': 'fail'}}
        self.save()
        self.assertTrue(self.errors())
        self.child['suites'] = {}
        self.child['comparisons'] = ['not-a-check-map']
        self.save()
        self.assertTrue(self.errors())

    def test_case_binding_validator_applies_at_every_depth(self):
        grandchild = self.proof/'evidence/grandchild.json'
        value = dict(self.child)
        value['cases'] = {'calculation': {'status': 'pass', 'bound': False}}
        grandchild.write_text(json.dumps(value))
        self.child['cases']['nested'] = dict(status='pass', bound=True, report='evidence/grandchild.json',
            report_sha256=sha(grandchild), input_fingerprint='current')
        self.child['cases']['startup']['bound'] = True
        self.save()
        errors = validate_tree(self.root, self.proof, 'current',
            case_validator=lambda check: [] if check.get('bound') else ['unbound calculation'])
        self.assertTrue(any('grandchild.json/calculation' in error for error in errors))

    def test_child_scope_can_bind_its_actual_dependencies(self):
        self.child['provenance']['fingerprint']='scoped-current'
        self.save()
        self.root['cases']['unit1']['input_fingerprint']='scoped-current'
        self.assertEqual(validate_tree(self.root,self.proof,'current',
            fingerprint_resolver=lambda child,name:'scoped-current'),[])
        self.assertTrue(validate_tree(self.root,self.proof,'current',
            fingerprint_resolver=lambda child,name:'scoped-old'))

    def test_unhashed_and_unbound_child_fail(self):
        del self.root['cases']['unit1']['report_sha256']
        self.assertTrue(self.errors())
        self.save()
        del self.root['cases']['unit1']['input_fingerprint']
        self.assertTrue(self.errors())

    def test_physical_hardware_claim_must_match_child(self):
        self.assertTrue(validate_tree(self.root, self.proof, 'current', hardware={'physical_ram_bytes':8*1024**3}))

    def test_mixed_browser_aggregate_still_checks_each_child_identity(self):
        del self.root['browser']
        del self.root['version']
        self.root['cases']['unit1']['browser']='edge'
        self.assertEqual(validate_tree(self.root,self.proof,'current'),[])
        self.child['distribution']='playwright'
        self.save()
        self.root['cases']['unit1']['browser']='edge'
        self.assertTrue(validate_tree(self.root,self.proof,'current'))

    def test_child_browser_claim_is_validated_without_parent_or_reference_labels(self):
        del self.root['browser']
        del self.root['version']
        self.assertEqual(validate_tree(self.root, self.proof, 'current'), [])
        self.child['distribution'] = 'playwright'
        self.save()
        self.assertTrue(validate_tree(self.root, self.proof, 'current'))

    def test_paths_cannot_escape_evidence_directory(self):
        self.root['cases']['unit1']['report'] = '../child.json'
        self.assertTrue(self.errors())

    def test_legacy_lifecycle_child_needs_native_awake_boundaries(self):
        self.child['cycle_evidence']=[{'index':1}]
        self.save()
        self.assertTrue(any('Native host observation' in error for error in self.errors()))

    def test_awake_host_requirement_propagates_to_grandchildren(self):
        from host_conditions import apply
        wake=dict(type='Wake',timestamp=9)
        conditions=apply(dict(status='pass'),
            dict(started_at=10,collected_at=11,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=11)),
            dict(started_at=19,collected_at=20,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=20)))['host_conditions']
        self.root['host_conditions']=conditions;self.child['host_conditions']=conditions
        grandchild=self.proof/'evidence/grandchild.json'
        grandchild.write_text(json.dumps(dict(status='pass',browser='edge',distribution='installed',version='154',
            provenance={'fingerprint':'current'},cases={'startup':{'status':'pass'}})))
        self.child['cases']['nested']=dict(status='pass',report='evidence/grandchild.json',
            report_sha256=sha(grandchild),input_fingerprint='current')
        self.save()
        self.assertEqual(self.errors(),[])
        errors=validate_tree(self.root,self.proof,'current',require_awake_host=True)
        self.assertTrue(any('Native host observation' in error for error in errors))
