"""Compare isolated Unit 5 calculations to independent supplied-body controls."""
import argparse
import ast
import asyncio
import hashlib
import importlib.util
import json
import random
import sys
import textwrap
import time
import types
from pathlib import Path
import numpy as np
import torch
from common import ROOT, PROOF, sha, write_json
from unit5_training_source import training_source


class Value:
    def __init__(self): self.value = None
    def get(self): return self.value
    def set(self, value): self.value = value


async def flush(): pass


def independent_body():
    source = (ROOT/'assignments/5/app.py').read_text()
    node = next(n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.AsyncFunctionDef) and n.name == '_mn_train')
    return textwrap.dedent(''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno]))


def loaders(helper, inputs, full):
    function = helper.get_dataset_mnist if inputs['mn_ds'] == 'mnist' else helper.get_dataset_fashionmnist
    result = function(batch_size=inputs['mn_batch'], horizontal_flip_p=inputs['mn_hflip'],
        invert=inputs['mn_invert'], seed=inputs['mn_seed'],
        train_store_path=str(PROOF/'cache/torchvision'), test_store_path=str(PROOF/'cache/torchvision'))
    # Constructors/transforms/loaders must consume no draws after reseeding.
    actual = torch.get_rng_state().clone()
    expected = torch.Generator().manual_seed(inputs['mn_seed']).get_state()
    np_state = np.random.get_state()
    random_state = random.getstate()
    if not torch.equal(actual, expected): raise AssertionError('Dataset construction consumed Torch RNG')
    if not np.array_equal(np_state[1], np.random.RandomState(inputs['mn_seed']).get_state()[1]):
        raise AssertionError('Dataset construction consumed NumPy RNG')
    if random_state != random.Random(inputs['mn_seed']).getstate():
        raise AssertionError('Dataset construction consumed Python RNG')
    if not full:
        result = tuple(torch.utils.data.DataLoader(torch.utils.data.Subset(loader.dataset, range(37)),
            batch_size=inputs['mn_batch'], shuffle=index == 0) for index, loader in enumerate(result))
    return result


async def control(body, helper, inputs, full):
    namespace = dict(torch=torch, nn=torch.nn, np=np,
        U5=helper, reactive=types.SimpleNamespace(flush=flush),
        input=types.SimpleNamespace(**{key:(lambda value=value:value) for key,value in inputs.items()}))
    for name in ('mn_coeffs', 'mn_progress_pct', 'mn_progress_msg'): namespace[name] = Value()
    namespace['_get_loaders'] = lambda: loaders(helper, inputs, full)
    exec(body, namespace)
    await namespace['_mn_train']()
    return dict(coefficients=namespace['mn_coeffs'].get(), progress_pct=namespace['mn_progress_pct'].get(),
        progress_msg=namespace['mn_progress_msg'].get())


def equal(actual, expected):
    np.testing.assert_array_equal(actual['coefficients'], expected['coefficients'])
    if (actual['progress_pct'], actual['progress_msg']) != (expected['progress_pct'], expected['progress_msg']):
        raise AssertionError('Progress/completion decisions differ')


async def verify(context, source, helper, full, result, output):
    body = independent_body()
    original = ast.parse(body).body[0]
    candidate = next(n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_mn_train')
    if ast.dump(original) != ast.dump(candidate): raise AssertionError('Supplied calculation body changed')
    result['original_body_unchanged'] = True
    for dataset in ('mnist', 'fashion'):
        for seed in (0, 42, 123):
            for invert in (False, True):
                inputs = dict(mn_seed=seed, mn_batch=257 if full else 8, mn_ds=dataset,
                    mn_epochs=1 if full else 3, mn_lr=.01, mn_mom=.9,
                    mn_hflip=.5 if invert else 0., mn_invert=invert)
                expected = await control(body, helper, inputs, full)
                parameters = dict(inputs=inputs, loaders=loaders(helper, inputs, full))
                payload = context._snapshot(parameters)
                sync = context.cloudpickle.loads(await asyncio.to_thread(context.execute_payload, source, payload, lambda message:None))
                equal(sync, expected)
                yielding = context.cloudpickle.loads(await context.execute_payload_async(source, payload, lambda message:None, lambda:False))
                equal(yielding, expected)
                owner = context.Executor(5)
                messages = []
                task = asyncio.create_task(owner.run(parameters, messages.append))
                while not task.done():
                    torch.manual_seed(999); torch.nn.Linear(784, 10)
                    np.random.seed(999); np.random.random(19); random.seed(999)
                    await asyncio.sleep(.01)
                equal(await task, expected)
                if owner.active is not None or not messages: raise AssertionError('Worker/progress ownership failed')
                # Repeat with the same loader objects. The original constructor
                # reseeding and global sampler semantics must still agree.
                equal(await owner.run(parameters, lambda message:None), expected)
                key = f'{dataset}/seed-{seed}/flip-invert-{invert}'
                observed = dict(coefficients_equal=True, completion_equal=True, constructor_rng_neutral=True,
                    repeated_training_equal=True, worker_disposed=True)
                result['cases'][key] = dict(status='pass',
                    assertion=dict(kind='computation', expected=observed, observed=observed, matched=True),
                    coefficient_sha256=hashlib.sha256(np.asarray(expected['coefficients']).tobytes()).hexdigest(),
                    samples=len(parameters['loaders'][0].dataset), epochs=inputs['mn_epochs'],
                    methods=['supplied independent body', 'synchronous candidate', 'yielding candidate',
                             'spawned worker with concurrent UI RNG draws', 'repeated spawned worker'])
                write_json(output, result); print(key, 'pass', flush=True)
    parameters['inputs'] = dict(parameters['inputs'], mn_epochs=10000)
    owner = context.Executor(5)
    task = asyncio.create_task(owner.run(parameters, lambda message:None))
    while owner.active is None: await asyncio.sleep(.01)
    process = owner.active; start = time.perf_counter(); owner.cancel()
    try: await task; raise AssertionError('Cancelled owner returned a result')
    except asyncio.CancelledError: pass
    seconds = time.perf_counter()-start
    if process.is_alive() or seconds > 1: raise AssertionError('Cancellation exceeded one second')
    observed = dict(cancelled=True, worker_disposed=True, deadline_met=seconds<=1)
    result['cases']['cancel-native-process'] = dict(status='pass', seconds=seconds,
        assertion=dict(kind='workflow', expected=observed, observed=observed, matched=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.candidate.resolve()))
    import isolated_training as context
    spec = importlib.util.spec_from_file_location('_unit5_supplied_helper', ROOT/'assignments/5/u5_utils.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    torch.set_num_threads(1)
    source = training_source()
    if json.loads((args.candidate/'course-training.json').read_text())['source'] != source:
        raise ValueError('Candidate source differs')
    output = PROOF/f'evidence/unit5-training-context-{args.label}.json'
    if output.exists(): raise FileExistsError('Choose a fresh label; retain previous observations')
    paths = [Path(__file__), args.candidate/'isolated_training.py', args.candidate/'course-training.json',
        PROOF/'scripts/unit5_training_source.py', ROOT/'assignments/5/app.py', ROOT/'assignments/5/u5_utils.py',
        PROOF/'requirements-native.lock']
    result = dict(status='running', cases={}, executor=str(Path(__file__).relative_to(ROOT)),
        dependencies={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}, torch_version=torch.__version__,
        scope='Exact original Unit 5 coefficients and constructor reseeding at seeds 0/42/123, both datasets, flip/inversion, incomplete batches, repeated training and cancellation. '
            + ('Complete native training data.' if args.full else '37-sample native controls.')
            + ' Full browser trajectories and comprehensive application parity remain separate requirements.')
    try:
        asyncio.run(verify(context, source, helper, args.full, result, output)); result['status']='pass'
    except Exception as error: result.update(status='fail', error=str(error))
    write_json(output, result)
    print(result['status'], result.get('error', ''), flush=True)
    return int(result['status'] != 'pass')


if __name__ == '__main__': raise SystemExit(main())
