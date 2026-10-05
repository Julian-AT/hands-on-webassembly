import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dynamic_choices import choice_cases, validate_matrix


class DynamicChoiceTests(unittest.TestCase):
    def test_identity_includes_parent_but_not_dictionary_order(self):
        a = choice_cases(1, 'features', {'dataset':'iris','loaded':True}, ['petal'])
        b = choice_cases(1, 'features', {'loaded':True,'dataset':'iris'}, ['petal'])
        c = choice_cases(1, 'features', {'loaded':True,'dataset':'wine'}, ['petal'])
        self.assertEqual(a[0]['id'], b[0]['id'])
        self.assertNotEqual(a[0]['id'], c[0]['id'])

    def test_matrix_cannot_pass_from_presence_or_omitted_parent(self):
        state = {'dataset':'iris'}
        observation = dict(status='pass',parent_state=state,choices=['petal'],
            reference_scope='corrected-native',reference_fingerprint='native',
            executor='discovery.py',dependencies={'app.py':'hash'})
        matrix = dict(status='pass',unresolved=[],controls={'features':dict(
            expected_parent_states=[state],observations=[observation])})
        case = choice_cases(1,'features',state,['petal'])[0]['id']
        checks = {case:dict(assertion=dict(kind='element',matched=True,expected=True,observed=True))}
        self.assertTrue(validate_matrix(matrix,1,['features'],checks))
        checks[case]['assertion']['kind']='workflow'
        self.assertEqual(validate_matrix(matrix,1,['features'],checks), [])
        checks[case]['assertion']['observed'] = False
        self.assertTrue(validate_matrix(matrix,1,['features'],checks))
        checks[case]['assertion']['observed'] = True
        observation['choices'].append('petal')
        self.assertTrue(validate_matrix(matrix,1,['features'],checks))
        observation['choices'].pop()
        matrix['controls']['features']['expected_parent_states'].append({'dataset':'wine'})
        self.assertTrue(validate_matrix(matrix,1,['features'],checks))

    def test_unobserved_control_cannot_pass(self):
        self.assertTrue(validate_matrix(dict(status='pass',unresolved=[],controls={}),7,['preset'],{}))

    def test_malformed_parent_or_control_fails_closed(self):
        for record in (None, [], {'expected_parent_states': ['not-a-state'], 'observations': []}):
            matrix = dict(status='pass', unresolved=[], controls={'preset': record})
            self.assertTrue(validate_matrix(matrix, 7, ['preset'], {}))
