"""The ownership substitution preserves all existing operation bodies."""
import ast
from pathlib import Path
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import PROOF
from stage_image_workers import application, write_config
from stage_neural import stage_neural
from reference_corrections import corrected_source
from build import replace_once


class StageImageWorkersTests(unittest.TestCase):
    def test_all_three_existing_applications_keep_their_calculations(self):
        for unit in (5, 6, 7):
            with self.subTest(unit=unit):
                with tempfile.TemporaryDirectory() as folder:
                    stage = Path(folder)
                    for name in (f'u{unit}_utils.py',):
                        supplied = PROOF.parent/f'assignments/{unit}'/name
                        if supplied.exists():
                            shutil.copy2(supplied, stage/name)
                    source = stage_neural(unit, stage, corrected_source(unit), replace_once)
                expected = ast.parse(source)
                observed = ast.parse(application(source, unit))
                for node in ast.walk(observed):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == 'image_worker_preload':
                                alias.name = 'image_preload'; alias.asname = None
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == 'image_preload' and node.func.attr == 'Owner'):
                        self.assertEqual(node.args.pop(0).value, unit)
                self.assertEqual(ast.dump(expected), ast.dump(observed))

    def test_code_dataset_and_assignment_changes_invalidate_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder)
            for name in ('image_preload.py', 'image_data.py'):
                (stage/name).write_bytes((PROOF/'build/unit7'/name).read_bytes())
            first, inputs = write_config(stage, 'original application', 7)
            self.assertEqual(set(inputs), {'image_preload.py', 'image_worker_preload.py',
                'image_data.py', 'course-image-worker.js', 'course-image-decode-worker.js'})
            changed, _ = write_config(stage, 'changed application', 7)
            self.assertNotEqual(first['build_id'], changed['build_id'])
            changed, _ = write_config(stage, 'original application', 6)
            self.assertNotEqual(first['build_id'], changed['build_id'])
            (stage/'image_data.py').write_text('changed constructor')
            changed, _ = write_config(stage, 'original application', 7)
            self.assertNotEqual(first['build_id'], changed['build_id'])


if __name__ == '__main__':
    unittest.main()
