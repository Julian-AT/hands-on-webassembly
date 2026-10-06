"""Numerical invariance under UI RNG activity and real process cancellation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from repository import EVIDENCE, REFERENCE, SCRIPTS
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
import typing
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proof"
sys.path.insert(0, str(ROOT / "artifacts/reference/unit7"))
import isolated_training


class IsolatedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from native_training_control import run

        cls.expected = {}
        for seed in (0, 42, 123):
            result = run(cls().params(seed))
            cls.expected[f"seed{seed}/control-a/history"] = result["history_df"].to_numpy()
            for i, value in enumerate(result["model"].state_dict().values()):
                cls.expected[f"seed{seed}/control-a/parameters/{i}"] = (
                    value.detach().cpu().numpy().copy()
                )

    def params(self, seed, epochs=4):
        rng = np.random.default_rng(187)
        x = torch.tensor(rng.normal(size=(30, 1, 28, 28)).astype(np.float32))
        y = torch.arange(30) % 2
        loaders = []
        for section in (slice(0, 23), slice(23, 30)):
            loaders.append(
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x[section], y[section]),
                    batch_size=8,
                    shuffle=section.start == 0,
                    generator=torch.Generator().manual_seed(seed),
                )
            )
        namespace = {"Dict": typing.Dict, "Any": typing.Any}
        # The architecture is an unchanged supplied preset.
        sys.path.insert(0, str(SCRIPTS))
        from common import extract

        exec(extract(7, ["PRESETS"]), namespace)
        arch = next(arch for name, arch in namespace["PRESETS"].items() if "MNIST" in name)
        return dict(
            seed=seed,
            epochs=epochs,
            lr=0.01,
            momentum=0.9,
            use_es=True,
            es_patience=1,
            use_cuda=False,
            arch_text=json.dumps(arch),
            loaders=(*loaders, loaders[1]),
        )

    def test_historical_fixture_retains_its_original_bytes(self):
        from common import sha
        import platform
        import subprocess

        path = ROOT / "tests/fixtures/cnn-rng-interference.npz"
        self.assertEqual(
            sha(path), "9b78d10afa32034befebc753feddbdcdc0d16b8d9c9a949b41839a0ca52826e3"
        )
        # Historical exact values are also checked on the original CPU family.
        # Other CPUs compare isolation against the independent serial control.
        if (
            platform.system() == "Darwin"
            and subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            == "Apple M4 Pro"
        ):
            with np.load(path) as historical:
                for name, value in self.expected.items():
                    np.testing.assert_array_equal(value, historical[name])

    def test_original_complete_body_is_unchanged_by_main_context_rng(self):
        async def run():
            for seed in (0, 42, 123):
                context = isolated_training.Executor(7)
                messages = []
                task = asyncio.create_task(context.run(self.params(seed), messages.append))
                while not task.done():
                    # Renderers and reset actions are allowed to consume/reseed
                    # main-context RNG while the complete original body runs.
                    torch.manual_seed(999)
                    torch.nn.Linear(784, 10)
                    np.random.seed(999)
                    np.random.random(19)
                    await asyncio.sleep(0.01)
                result = await task
                expected = self.expected
                np.testing.assert_array_equal(
                    result["history_df"].to_numpy(), expected[f"seed{seed}/control-a/history"]
                )
                for i, value in enumerate(result["model_record"]["state"].values()):
                    np.testing.assert_array_equal(
                        value, expected[f"seed{seed}/control-a/parameters/{i}"]
                    )
                self.assertTrue(any(message["type"] == "history_append" for message in messages))
                self.assertIsNone(context.active)

        asyncio.run(run())

    def test_cancel_terminates_process_and_preserves_parent_generator(self):
        async def run():
            params = self.params(42, epochs=1000)
            params["use_es"] = False
            original = params["loaders"][0].generator.get_state().clone()
            context = isolated_training.Executor(7)
            task = asyncio.create_task(context.run(params, lambda message: None))
            while context.active is None:
                await asyncio.sleep(0.01)
            process = context.active
            started = time.perf_counter()
            context.cancel()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertLess(time.perf_counter() - started, 1)
            self.assertFalse(process.is_alive())
            self.assertIsNone(context.active)
            self.assertTrue(torch.equal(original, params["loaders"][0].generator.get_state()))

        asyncio.run(run())

    def test_batch_yields_preserve_unrounded_original_results(self):
        async def run():
            from reference_corrections import training_source

            for seed in (0, 42, 123):
                params = self.params(seed)
                encoded = await isolated_training.execute_payload_async(
                    training_source(7),
                    isolated_training._snapshot(params),
                    lambda message: None,
                    lambda: False,
                )
                result = isolated_training.cloudpickle.loads(encoded)
                np.testing.assert_array_equal(
                    result["history_df"].to_numpy(), self.expected[f"seed{seed}/control-a/history"]
                )
                for i, value in enumerate(result["model_record"]["state"].values()):
                    np.testing.assert_array_equal(
                        value, self.expected[f"seed{seed}/control-a/parameters/{i}"]
                    )

        asyncio.run(run())

    def test_batch_checkpoint_cancels_before_returning_model(self):
        async def run():
            from reference_corrections import training_source

            params = self.params(42, epochs=100)
            with self.assertRaises(asyncio.CancelledError):
                await isolated_training.execute_payload_async(
                    training_source(7),
                    isolated_training._snapshot(params),
                    lambda message: None,
                    lambda: True,
                )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
