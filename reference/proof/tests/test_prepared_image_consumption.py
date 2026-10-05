"""Browser consumers cannot start transport or discard another prepared dataset."""
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

PROOF=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROOF/'scripts'))
from stage_image_loading import prepared_only


def module():
    spec=importlib.util.spec_from_file_location('prepared_image_under_test',PROOF/'runtime/image_data.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class PreparedImageConsumptionTests(unittest.TestCase):
    def test_unprepared_browser_load_keeps_existing_arrays_and_never_reads_files(self):
        data=module()
        retained=(np.zeros((1,2,2),dtype=np.uint8),np.array([0]),'NHW')
        data._cache['MNIST/train']=retained
        with patch.object(data.sys,'platform','emscripten'),patch.object(data.Path,'read_bytes',side_effect=AssertionError('Blocking file read')):
            with self.assertRaisesRegex(RuntimeError,'Dataset is not ready'):
                data.load('FashionMNIST','train')
            with self.assertRaisesRegex(RuntimeError,'Dataset is not ready'):
                data.asset_bytes('manifest.json')
            self.assertIs(data.load('MNIST','train'),retained)
        self.assertEqual(list(data._cache),['MNIST/train'])

    def test_native_loader_retains_original_checksum_and_compact_array_behavior(self):
        data=module();images=np.arange(8,dtype=np.uint8).reshape(2,2,2);labels=np.array([3,4])
        packed=io.BytesIO();np.savez_compressed(packed,images=images,labels=labels);raw=packed.getvalue()
        manifest=dict(datasets={'MNIST/train':dict(file='mnist-train.npz',sha256=hashlib.sha256(raw).hexdigest(),
            image_shape=[2,2,2],count=2,layout='NHW')})
        with tempfile.TemporaryDirectory() as folder:
            data.__file__=str(Path(folder)/'image_data.py');assets=Path(folder)/'image-assets';assets.mkdir()
            (assets/'manifest.json').write_text(json.dumps(manifest));(assets/'mnist-train.npz').write_bytes(raw)
            actual=data.load('MNIST','train')
            np.testing.assert_array_equal(actual[0],images);np.testing.assert_array_equal(actual[1],labels)
            self.assertEqual(actual[2],'NHW')
            data._cache.clear();(assets/'mnist-train.npz').write_bytes(raw+b'corruption')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):data.load('MNIST','train')
            self.assertFalse(data._cache)

    def test_staging_preserves_the_checked_prepared_only_source(self):
        source=(PROOF/'runtime/image_data.py').read_text()
        self.assertEqual(prepared_only(source),source)
        with self.assertRaisesRegex(ValueError,'source contract changed'):
            prepared_only(source.replace("if sys.platform == 'emscripten':","if False:",1))


if __name__=='__main__':unittest.main()
