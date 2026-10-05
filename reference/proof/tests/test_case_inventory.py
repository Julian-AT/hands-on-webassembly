import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from case_inventory import identities
from unittest.mock import patch


class CaseInventoryTests(unittest.TestCase):
    def test_id_does_not_depend_on_source_line(self):
        first={'call':'input_select','line':12,'args':"ui.input_select('dataset', 'Data', choices={'iris':'Iris','wine':'Wine'})"}
        second=dict(first,line=999)
        a=identities(1,[first]);b=identities(1,[second])
        self.assertEqual([c['id'] for c in a],[c['id'] for c in b])
        self.assertIn('unit1/control/dataset/choice-iris',[c['id'] for c in a])
    def test_dynamic_choices_remain_unresolved(self):
        result=identities(7,[{'call':'input_select','line':1,'args':"ui.input_select('layers','Layers',choices=layer_choices)"}])
        self.assertTrue(all(c['requires_runtime_expansion'] for c in result))
    def test_upload_and_download_content_requirements(self):
        result=identities(5,[{'call':'input_file','line':1,'args':"ui.input_file('csv','Upload')"},
                             {'call':'download_button','line':2,'args':"ui.download_button('coefficients','Save')"}])
        ids={c['id'] for c in result}
        self.assertIn('unit5/control/csv/invalid-format',ids)
        self.assertIn('unit5/download/coefficients/filename-and-roundtrip-content',ids)

    def test_nested_tabs_with_the_same_label_have_distinct_stable_ids(self):
        first={'call':'nav_panel','line':12,'args':"ui.nav_panel('Plot')"}
        second=dict(first,line=40)
        with patch('case_inventory.tab_paths',return_value={
            (12,first['args']):('Regression','Plot'),
            (40,second['args']):('Classification','Plot'),
        }):
            result=identities(5,[first,second])
        self.assertEqual({c['id'] for c in result},{
            'unit5/tab/Regression%2FPlot/default-behavior',
            'unit5/tab/Classification%2FPlot/default-behavior',
        })
