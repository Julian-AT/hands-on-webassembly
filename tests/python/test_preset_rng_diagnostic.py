"""Uniform-only diagnostic state encoding against independent native controls."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
from torch_rng import CPUStream
from preset_trajectory_probe import rng_bytes


class PresetRngDiagnosticTests(unittest.TestCase):
    def test_initial_and_advanced_uniform_states_match_native_bytes(self):
        for seed in (0, 42, 123):
            for count in (0, 1, 5, 623, 624, 625, 4096):
                with self.subTest(seed=seed, count=count):
                    torch.manual_seed(seed)
                    torch.rand(count)
                    stream = CPUStream(seed)
                    stream.random(count)
                    backend = SimpleNamespace(default_generator=SimpleNamespace(_rng=stream))
                    expected = torch.get_rng_state().numpy()
                    np.testing.assert_array_equal(rng_bytes(backend), expected)
                    # Observing the state must leave its future untouched.
                    before = copy.deepcopy(stream)
                    rng_bytes(backend)
                    np.testing.assert_array_equal(stream.random(32), before.random(32))


if __name__ == "__main__":
    unittest.main()
