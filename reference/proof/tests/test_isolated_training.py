"""Numerical invariance under UI RNG activity and real process cancellation."""
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

PROOF=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROOF/'reference/unit7'))
import isolated_training


class IsolatedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.expected=np.load(PROOF/'evidence/cnn-rng-interference.npz')

    def params(self,seed,epochs=4):
        rng=np.random.default_rng(187)
        x=torch.tensor(rng.normal(size=(30,1,28,28)).astype(np.float32))
        y=torch.arange(30)%2
        loaders=[]
        for section in (slice(0,23),slice(23,30)):
            loaders.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[section],y[section]),
                batch_size=8,shuffle=section.start==0,generator=torch.Generator().manual_seed(seed)))
        namespace={'Dict':typing.Dict,'Any':typing.Any}
        # The architecture is an unchanged supplied preset.
        sys.path.insert(0,str(PROOF/'scripts'))
        from common import extract
        exec(extract(7,['PRESETS']),namespace)
        arch=next(arch for name,arch in namespace['PRESETS'].items() if 'MNIST' in name)
        return dict(seed=seed,epochs=epochs,lr=.01,momentum=.9,use_es=True,
                    es_patience=1,use_cuda=False,arch_text=json.dumps(arch),loaders=(*loaders,loaders[1]))

    def test_original_complete_body_is_unchanged_by_main_context_rng(self):
        async def run():
            for seed in (0,42,123):
                context=isolated_training.Executor(7)
                messages=[]
                task=asyncio.create_task(context.run(self.params(seed),messages.append))
                while not task.done():
                    # Renderers and reset actions are allowed to consume/reseed
                    # main-context RNG while the complete original body runs.
                    torch.manual_seed(999)
                    torch.nn.Linear(784,10)
                    np.random.seed(999)
                    np.random.random(19)
                    await asyncio.sleep(.01)
                result=await task
                expected=self.expected
                np.testing.assert_array_equal(result['history_df'].to_numpy(),expected[f'seed{seed}/control-a/history'])
                for i,value in enumerate(result['model_record']['state'].values()):
                    np.testing.assert_array_equal(value,expected[f'seed{seed}/control-a/parameters/{i}'])
                self.assertTrue(any(message['type']=='history_append' for message in messages))
                self.assertIsNone(context.active)
        asyncio.run(run())

    def test_cancel_terminates_process_and_preserves_parent_generator(self):
        async def run():
            params=self.params(42,epochs=1000)
            params['use_es']=False
            original=params['loaders'][0].generator.get_state().clone()
            context=isolated_training.Executor(7)
            task=asyncio.create_task(context.run(params,lambda message:None))
            while context.active is None: await asyncio.sleep(.01)
            process=context.active
            started=time.perf_counter()
            context.cancel()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertLess(time.perf_counter()-started,1)
            self.assertFalse(process.is_alive())
            self.assertIsNone(context.active)
            self.assertTrue(torch.equal(original,params['loaders'][0].generator.get_state()))
        asyncio.run(run())

    def test_batch_yields_preserve_unrounded_original_results(self):
        async def run():
            from reference_corrections import training_source
            for seed in (0,42,123):
                params=self.params(seed)
                encoded=await isolated_training.execute_payload_async(training_source(7),
                    isolated_training._snapshot(params),lambda message:None,lambda:False)
                result=isolated_training.cloudpickle.loads(encoded)
                np.testing.assert_array_equal(result['history_df'].to_numpy(),
                    self.expected[f'seed{seed}/control-a/history'])
                for i,value in enumerate(result['model_record']['state'].values()):
                    np.testing.assert_array_equal(value,self.expected[f'seed{seed}/control-a/parameters/{i}'])
        asyncio.run(run())

    def test_batch_checkpoint_cancels_before_returning_model(self):
        async def run():
            from reference_corrections import training_source
            params=self.params(42,epochs=100)
            with self.assertRaises(asyncio.CancelledError):
                await isolated_training.execute_payload_async(training_source(7),
                    isolated_training._snapshot(params),lambda message:None,lambda:True)
        asyncio.run(run())


if __name__=='__main__':unittest.main()
