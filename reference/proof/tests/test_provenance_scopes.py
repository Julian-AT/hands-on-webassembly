import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import provenance


class NativeProvenanceTests(unittest.TestCase):
    def test_browser_edits_preserve_native_but_reference_and_test_edits_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ['iml_env.yaml', 'proof/requirements-native.lock', 'proof/source.lock.json',
                     'proof/assignment5-materials.lock.json', 'proof/scripts/reference_corrections.py',
                     'proof/scripts/common.py', 'proof/scripts/provenance.py', 'proof/scripts/browser.py',
                     'proof/scripts/browser_unit2.py', 'proof/reference/unit2/app.py',
                     'assignments/2/app.py', 'proof/runtime/startup-status.js', 'proof/site/unit2/app.json']
            for name in paths:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
            with patch.object(provenance, 'ROOT', root), patch.object(provenance, 'PROOF', root/'proof'):
                expected = provenance.native_fingerprint(2)
                for name in ['proof/runtime/startup-status.js', 'proof/site/unit2/app.json',
                             'proof/scripts/provenance.py']:
                    (root/name).write_text('browser change')
                    self.assertEqual(provenance.native_fingerprint(2), expected)
                for name in ['proof/reference/unit2/app.py', 'proof/scripts/browser_unit2.py',
                             'proof/scripts/reference_corrections.py', 'proof/requirements-native.lock']:
                    (root/name).write_text('native dependency change')
                    self.assertNotEqual(provenance.native_fingerprint(2), expected)
                    (root/name).write_text('original')

    def test_browser_scope_tracks_artifact_and_actual_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ['proof/site/unit2/app.json', 'proof/runtime/startup-status.js',
                     'proof/scripts/browser.py', 'proof/scripts/browser_unit2.py',
                     'proof/scripts/browser_unit4.py', 'proof/scripts/check_assets.py',
                     'proof/scripts/build.py', 'proof/scripts/reference_corrections.py',
                     'proof/tests/test_unrelated.py']
            for name in paths:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
            with patch.object(provenance, 'ROOT', root), patch.object(provenance, 'PROOF', root/'proof'):
                expected = provenance.browser_fingerprint('2')
                for name in ['proof/scripts/check_assets.py', 'proof/tests/test_unrelated.py',
                             'proof/scripts/browser_unit4.py']:
                    (root/name).write_text('unrelated checker change')
                    self.assertEqual(provenance.browser_fingerprint('2'), expected)
                for name in ['proof/site/unit2/app.json', 'proof/runtime/startup-status.js',
                             'proof/scripts/browser_unit2.py', 'proof/scripts/build.py',
                             'proof/scripts/reference_corrections.py']:
                    (root/name).write_text('relevant dependency change')
                    self.assertNotEqual(provenance.browser_fingerprint('2'), expected)
                    (root/name).write_text('original')
                with self.assertRaises(ValueError):
                    provenance.browser_fingerprint('unknown')


if __name__ == '__main__':
    unittest.main()
