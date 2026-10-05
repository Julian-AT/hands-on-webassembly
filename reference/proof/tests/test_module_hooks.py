import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import borch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from module_hooks import install
install(borch)


class HookTests(unittest.TestCase):
    def test_original_catalog_and_removal(self):
        records = []
        for lib in (torch, borch):
            model = lib.nn.Sequential(lib.nn.Conv2d(1, 2, 3), lib.nn.ReLU(), lib.nn.Flatten())
            rows = []
            handles = [child.register_forward_hook(lambda m, inputs, output: rows.append((type(m).__name__, tuple(inputs[0].shape), tuple(output.shape)))) for child in model.children()]
            model(lib.ones(1, 1, 5, 5))
            for handle in handles:
                handle.remove()
                handle.remove()
            model(lib.ones(1, 1, 5, 5))
            records.append(rows)
        self.assertEqual(records[0], records[1])
        self.assertEqual(len(records[1]), 3)

    def test_output_replacement_order_and_context(self):
        values = []
        for lib in (torch, borch):
            model = lib.nn.ReLU()
            with model.register_forward_hook(lambda m, args, output: output * 2):
                with model.register_forward_hook(lambda m, args, output: output + 1):
                    values.append(model(lib.tensor([-1., 2.])).numpy())
            values.append(model(lib.tensor([-1., 2.])).numpy())
        np.testing.assert_array_equal(values[0], values[2])
        np.testing.assert_array_equal(values[1], values[3])
