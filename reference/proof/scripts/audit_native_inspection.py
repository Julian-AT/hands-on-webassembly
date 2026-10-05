"""Run exact native render bodies against independent RNG continuation controls.

No numerical fixtures or application references are rewritten. The proposed
scope is tested separately after reproducing each native interference defect.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import PROOF, sha, write_json
sys.path.insert(0, str(PROOF/'runtime'))
import inspection_rng


def rng_digest():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
        torch.get_rng_state().numpy().tobytes()))).hexdigest()


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def body(source, name, namespace):
    tree = ast.parse(source)
    server = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='server')
    definition = next(n for n in server.body if isinstance(n, ast.FunctionDef) and n.name==name)
    definition.decorator_list = []
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<exact-native-render>', 'exec'), namespace)
    return namespace[name]


def output_digest(value):
    if hasattr(value, 'canvas'):
        value.canvas.draw()
        pixels = np.asarray(value.canvas.buffer_rgba()).copy()
        plt.close(value)
        return hashlib.sha256(pixels.tobytes()).hexdigest()
    return hashlib.sha256(str(value).encode()).hexdigest()


def next_training_batch(loader):
    images, labels = next(iter(loader))
    return hashlib.sha256(images.numpy().tobytes()+labels.numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    result = dict(status='running', executor_sha256=sha(Path(__file__)), cases={},
        torch_version=torch.__version__, scope='Exact corrected-native Units 5/6 render bodies at seeds 0, 42 and 123, paired with independent no-inspection controls. Proposed synchronous RNG scope is evaluated separately; no shared reference changes or fixtures.')
    output = PROOF/'evidence/native-inspection-rng-audit.json'
    result['inputs'] = {str(p.relative_to(PROOF)):sha(p) for p in
        [PROOF/'requirements-native.lock', PROOF/'runtime/inspection_rng.py']+
        [PROOF/f'reference/unit{u}/{f}' for u in (5,6) for f in ('app.py',f'u{u}_utils.py')]}
    result['input_fingerprint'] = hashlib.sha256(json.dumps(result['inputs'],sort_keys=True).encode()).hexdigest()
    helpers = {}
    for unit in (5,6):
        spec = importlib.util.spec_from_file_location(f'inspection_u{unit}',PROOF/f'reference/unit{unit}/u{unit}_utils.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helpers[unit] = module
    old_cwd = Path.cwd()
    try:
        for unit in (5,6):
            os.chdir(PROOF/f'reference/unit{unit}')
            source = (PROOF/f'reference/unit{unit}/app.py').read_text()
            namespace = dict(np=np,pd=pd,plt=plt,torch=torch,nn=nn,json=json,U5=helpers[5],u6=helpers[6])
            if unit == 6:
                tree = ast.parse(source)
                names = {'ACTIVATIONS','parse_architecture','build_model'}
                definitions = [n for n in tree.body if
                    isinstance(n,ast.FunctionDef) and n.name in names or
                    isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
                exec(compile(ast.Module(body=definitions,type_ignores=[]),'<exact-native-model>', 'exec'),namespace)
            for variant in ('MNIST','FashionMNIST'):
                if unit == 5:
                    get_loader = helpers[5].get_dataset_mnist if variant=='MNIST' else helpers[5].get_dataset_fashionmnist
                    loader,_ = get_loader(batch_size=20,seed=123)
                    functions = [('mnist_samples' if variant=='MNIST' else 'fashionmnist_samples',True)]
                else:
                    train,test = helpers[6].get_dataset_mnist(batch_size=20,variant=variant,augment_train=False)
                    loader = train
                    namespace['data_bundle'] = lambda:dict(kind='img',dataset_name=variant,variant=variant,train=train,val=None,test=test)
                    namespace['input'] = SimpleNamespace(dataset=lambda:variant,sample_ix=lambda:0)
                    namespace['arch_state'] = lambda:json.dumps({'layers':[{'type':'flatten'},{'type':'linear','out_features':10}]})
                    functions = [('data_info',True),('model_summary',True),('sample_plot',False)]
                for name, expected_interference in functions:
                    render = body(source,name,namespace)
                    for value in (0,42,123):
                        key = f'unit{unit}/{variant}/{name}/seed-{value}'
                        seed(value)
                        before = rng_digest()
                        control = next_training_batch(loader)
                        seed(value)
                        first_output = output_digest(render())
                        after = rng_digest()
                        interrupted = next_training_batch(loader)
                        seed(value)
                        with inspection_rng.preserve(torch):
                            scoped_output = output_digest(render())
                        restored = rng_digest()
                        protected = next_training_batch(loader)
                        # A separate control also verifies the baseline is repeatable.
                        seed(value)
                        second_control = next_training_batch(loader)
                        interference = after != before and interrupted != control
                        assertions = dict(native_interference=interference==expected_interference,
                            repeatable_control=control==second_control,
                            inspection_output_preserved=first_output==scoped_output,
                            all_rng_restored=before==restored,
                            training_continuation_preserved=control==protected)
                        result['cases'][key] = dict(status='pass' if all(assertions.values()) else 'fail',
                            assertions=assertions, expected_interference=expected_interference,
                            observed_interference=interference, rng_before=before,rng_after=after,rng_restored=restored,
                            control_batch=control,interrupted_batch=interrupted,protected_batch=protected,
                            output=first_output,protected_output=scoped_output)
                        write_json(output,result)
                        print(key,result['cases'][key]['status'],flush=True)
                del loader
                if unit == 6:
                    del train,test
        result['status'] = 'pass' if all(c['status']=='pass' for c in result['cases'].values()) else 'fail'
    except Exception as error:
        result.update(status='error',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        os.chdir(old_cwd)
        write_json(output,result)
    return int(result['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
