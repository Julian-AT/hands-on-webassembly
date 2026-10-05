"""The release gate must reject stale, partial and mislabeled evidence."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gate import evaluate, readiness
from coverage_contract import requirements
from common import UNITS


class GateTests(unittest.TestCase):
    def setUp(self):
        from host_conditions import apply
        wake=dict(type='Wake',timestamp=9)
        self.host_conditions=apply(dict(status='pass'),
            dict(started_at=10,collected_at=11,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=11)),
            dict(started_at=19,collected_at=20,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=20)))['host_conditions']
        self.requirement={'section':'cases','required_cases':['startup','reload'],'browser':'safari'}
        self.report={'status':'pass','browser':'safari','distribution':'installed','version':'26.6','provenance':{'fingerprint':'current'},'host_conditions':self.host_conditions,'cases':{'startup':{'status':'pass'},'reload':{'status':'pass'}}}
    def result(self): return evaluate(self.requirement,self.report,'current')['status']
    def test_complete_current_evidence_can_pass(self): self.assertEqual(self.result(),'pass')
    def test_missing(self): self.assertEqual(evaluate(self.requirement,None,'current')['status'],'missing')
    def test_stale(self):
        self.report['provenance']['fingerprint']='old';self.assertEqual(self.result(),'stale')
    def test_empty(self):
        self.report['cases']={};self.assertEqual(self.result(),'fail')
    def test_partial(self):
        del self.report['cases']['reload'];self.assertEqual(self.result(),'fail')
    def test_failure_even_with_pass_summary(self):
        self.report['cases']['reload']['status']='fail';self.assertEqual(self.result(),'fail')
    def test_webkit_is_not_safari(self):
        self.report['browser']='webkit';self.assertEqual(self.result(),'fail')
    def test_playwright_firefox_is_not_installed_firefox(self):
        self.requirement['browser']='firefox';self.report['browser']='firefox'
        self.report['distribution']='playwright';self.assertEqual(self.result(),'fail')
    def test_large_machine_is_not_student_hardware(self):
        self.requirement['hardware']=True
        self.report['hardware']={'physical_ram_bytes':24*1024**3,'integrated_graphics':True,'machine':'M4 Pro'}
        self.assertEqual(self.result(),'fail')
    def test_seven_assignments_required(self):
        self.assertEqual(set(UNITS),set(range(1,8)))
        rows=requirements({str(unit):{'interface':[]} for unit in UNITS})
        self.assertEqual({r['id'] for r in rows if r['id'].endswith('/numerical')},
                         {f'unit{unit}/numerical' for unit in range(1,8)})
        self.assertTrue(all(r['acceptance_scope']=='release' for r in rows if r.get('hardware')))
    def test_shell_readiness_requires_application_parity(self):
        rows=[{'id':'app','acceptance_scope':'application'},{'id':'8gb','acceptance_scope':'release'}]
        app,release,_=readiness(rows,{'app':{'status':'pass'},'8gb':{'status':'missing'}},True)
        self.assertTrue(app);self.assertFalse(release)
        app,release,_=readiness(rows,{'app':{'status':'fail'},'8gb':{'status':'pass'}},True)
        self.assertFalse(app);self.assertFalse(release)
        app,release,_=readiness(rows,{'app':{'status':'pass'},'8gb':{'status':'pass'}},False)
        self.assertFalse(app);self.assertFalse(release)
    def test_skipped_is_not_pass(self):
        self.report['cases']['reload']['status']='skipped';self.assertEqual(self.result(),'fail')
    def test_lifecycle_requires_thirty_consecutive_complete_cycles(self):
        self.requirement['minimum_consecutive_cycles']=30
        self.assertEqual(self.result(),'fail')
        from host_conditions import apply
        wake=dict(type='Wake',timestamp=9)
        conditions=apply(dict(status='pass'),
            dict(started_at=10,collected_at=11,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=11)),
            dict(started_at=19,collected_at=20,events=[wake],native_power=dict(system_capabilities=15,state='FullWake',recorded_at=20)))['host_conditions']
        self.report.update(preview_identity=dict(identity='owned',marker='marker.txt',pid=123,port=8008),
            host_conditions=conditions,
            diagnostic_instrumentation=False)
        for case in self.report['cases'].values():case['assertion']=dict(kind='workflow',matched=True,expected=True,observed=True)
        self.report['cycle_evidence']=[dict(index=i+1,before_generation=f'build-{i}',after_generation=f'build-{i+1}',
            **{op:{'status':'pass','assertion':dict(kind='workflow',expected=f'build-{i+1}',observed=f'build-{i+1}',matched=True)} for op in
            ('startup','reload','service-worker-update','build-generation-update')}) for i in range(30)]
        self.assertEqual(self.result(),'pass')
        self.report['cycle_evidence'][14]['reload']['status']='fail'
        self.assertEqual(self.result(),'fail')
        self.report['cycle_evidence'][14]['reload']['status']='pass'
        self.report['cycle_evidence'][14]['index']=16
        self.assertEqual(self.result(),'fail')
        self.report['cycle_evidence'][14]['index']=15
        self.report['host_conditions']['uninterrupted']=False
        self.assertEqual(self.result(),'fail')
        self.report['host_conditions']['uninterrupted']=True
        native=self.report['host_conditions'].pop('start_native_power')
        self.assertEqual(self.result(),'fail')
        self.report['host_conditions']['start_native_power']=native
        self.report['diagnostic_instrumentation']=True
        self.assertEqual(self.result(),'fail')
        self.report['diagnostic_instrumentation']=False
        self.report['cycle_evidence'][14]['after_generation']='build-14'
        self.assertEqual(self.result(),'fail')

    def test_delivery_required_without_changing_shell_application_readiness(self):
        rows=requirements({str(unit):{'interface':[]} for unit in UNITS})
        delivery=[r for r in rows if r['id'].startswith(('nextjs/','vercel/','release/'))]
        self.assertEqual(len(delivery),7)
        self.assertTrue(all(r['acceptance_scope']=='release' for r in delivery))
        self.assertEqual(len(rows)-len(delivery),133)
        ids=[r['id'] for r in rows]
        self.assertEqual(len(ids),len(set(ids)))
    def test_dom_presence_cannot_satisfy_behavior_case(self):
        self.requirement['behavior_assertions']=['startup']
        self.assertEqual(self.result(),'fail')
        self.report['cases']['startup']['assertion']=dict(kind='element',matched=True,expected=True,observed=True)
        self.assertEqual(self.result(),'fail')
        self.report['cases']['startup']['assertion']=dict(kind='workflow',matched=True,
            expected={'connected':True},observed={'connected':True})
        self.assertEqual(self.result(),'pass')

    def test_matched_boolean_cannot_hide_wrong_observation(self):
        self.requirement['behavior_assertions']=['startup']
        self.report['cases']['startup']['assertion']=dict(kind='workflow',matched=True,
            expected={'connected':True},observed={'connected':False})
        self.assertEqual(self.result(),'fail')

    def test_optional_assertions_still_cannot_contradict_passing_status(self):
        self.report['cases']['startup']['assertion'] = dict(kind='workflow', matched=True,
            expected={'connected': True}, observed={'connected': False})
        self.assertEqual(self.result(), 'fail')

    def test_comprehensive_suite_needs_hashed_executor_and_dependencies(self):
        from common import ROOT,sha
        self.requirement.update(comprehensive=True,unit=1,dynamic_controls=[])
        for check in self.report['cases'].values():
            check.update(executor=dict(path='proof/scripts/gate.py',sha256=sha(ROOT/'proof/scripts/gate.py')),
                dependencies={'iml_env.yaml':sha(ROOT/'iml_env.yaml')},
                assertion=dict(kind='workflow',expected=True,observed=True,matched=True))
        self.report['dynamic_choice_matrix']=dict(status='pass',unresolved=[],controls={})
        self.assertEqual(self.result(),'pass')
        self.report['cases']['startup']['executor']['sha256']='unbound'
        self.assertEqual(self.result(),'fail')

    def test_comprehensive_suite_needs_complete_dynamic_matrix(self):
        from common import ROOT,sha
        self.requirement.update(comprehensive=True,unit=1,dynamic_controls=['features'])
        for check in self.report['cases'].values():
            check.update(executor=dict(path='proof/scripts/gate.py',sha256=sha(ROOT/'proof/scripts/gate.py')),
                dependencies={'iml_env.yaml':sha(ROOT/'iml_env.yaml')},
                assertion=dict(kind='workflow',expected=True,observed=True,matched=True))
        self.report['dynamic_choice_matrix']=dict(status='pass',unresolved=[],controls={})
        self.assertEqual(self.result(),'fail')

    def test_exact_pixels_are_recomputed_from_checksum_bound_captures(self):
        import tempfile
        from unittest.mock import patch
        from PIL import Image
        import gate
        from common import sha
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executor = root/'executor.py'
            executor.write_text('capture()')
            native, browser = root/'native.png', root/'browser.png'
            Image.new('RGBA', (2, 2), 'white').save(native)
            Image.new('RGBA', (2, 2), 'white').save(browser)
            check = dict(executor=dict(path=executor.name, sha256=sha(executor)),
                dependencies={executor.name: sha(executor)}, assertion=dict(kind='pixel-equality',
                    expected=0, observed=0, matched=True, different_pixels=0, masks=[], tolerance=0,
                    matched_state={'tab': 'Overview'}, viewport={'width': 2, 'height': 2}, display_scale=1,
                    native_capture_path=native.name, native_capture_sha256=sha(native),
                    browser_capture_path=browser.name, browser_capture_sha256=sha(browser)))
            with patch.object(gate, 'ROOT', root):
                self.assertEqual(gate.bound_case_errors(check, appearance=True), [])
                changed = Image.new('RGBA', (2, 2), 'white')
                changed.putpixel((0, 0), (254, 255, 255, 255))
                changed.save(browser)
                # A reporter cannot forge zero differing pixels even with a valid new hash.
                check['assertion']['browser_capture_sha256'] = sha(browser)
                self.assertTrue(gate.bound_case_errors(check, appearance=True))
                browser.unlink()
                self.assertTrue(gate.bound_case_errors(check, appearance=True))


if __name__=='__main__': unittest.main()
