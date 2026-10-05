import random
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import borch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from inspection_rng import preserve, protected
from torch_rng import install
install(borch)


class InspectionRNGTests(unittest.TestCase):
    def test_draws_and_reseeding_restore_all_streams_even_on_exception(self):
        for library in (torch,borch):
            for seed in (0,42,123):
                random.seed(seed);np.random.seed(seed);library.manual_seed(seed)
                expected=(random.random(),np.random.random(),library.rand(20).numpy())
                random.seed(seed);np.random.seed(seed);library.manual_seed(seed)
                with self.assertRaisesRegex(ValueError,'inspection failed'):
                    with preserve(library):
                        random.seed(17);np.random.seed(17);library.manual_seed(17)
                        random.random();np.random.random();library.rand(50)
                        raise ValueError('inspection failed')
                self.assertEqual(random.random(),expected[0])
                self.assertEqual(np.random.random(),expected[1])
                np.testing.assert_array_equal(library.rand(20).numpy(),expected[2])

    def test_loader_order_survives_an_inspection_iterator(self):
        for library in (torch,borch):
            dataset=library.utils.data.TensorDataset(library.arange(101))
            loader=library.utils.data.DataLoader(dataset,batch_size=11,shuffle=True)
            library.manual_seed(123)
            expected=next(iter(loader))[0].numpy()
            library.manual_seed(123)
            with preserve(library):
                next(iter(loader))
            np.testing.assert_array_equal(next(iter(loader))[0].numpy(),expected)

    def test_preparation_consumption_is_preserved_before_scoping(self):
        for library in (torch,borch):
            def prepare():library.rand(7)
            @protected(library, prepare=prepare)
            def inspect():return library.rand(3).numpy()
            library.manual_seed(42)
            prepare()
            expected=library.rand(20).numpy()
            library.manual_seed(42)
            np.testing.assert_array_equal(inspect(),expected[:3])
            np.testing.assert_array_equal(library.rand(20).numpy(),expected)


if __name__=='__main__':unittest.main()
