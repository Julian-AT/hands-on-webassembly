"""Compare snapshotted loader construction to independent native controls.

The supplied/native fixtures are read only. Every dataset, three required seeds,
both split modes and default/augmented transforms are observed with real native
dataset constructors, exact subset order, generators and batch pixels/labels.
"""
import ast
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import typing
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader,Subset,ConcatDataset
from torchvision import transforms as T
from common import PROOF,sha,write_json
from build import replace_once
from stage_image_loading import unit7_loading


def definition(source,name):
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    return node


def compile_namespace(source,name):
    namespace=dict(torch=torch,np=np,random=random,torchvision=torchvision,T=T,
        DataLoader=DataLoader,Subset=Subset,ConcatDataset=ConcatDataset,reactive=SimpleNamespace(isolate=nullcontext))
    namespace.update({name:getattr(typing,name) for name in ('Any','List','Tuple','Union')})
    module=ast.Module(body=[definition(source,'get_dataset'),definition(source,name)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<exact-native-data-construction>','exec'),namespace)
    return namespace


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def rng_digest():
    return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes()))).hexdigest()


def observed(loaders):
    records=[]
    for loader in loaders:
        dataset=loader.dataset
        indices=np.asarray(getattr(dataset,'indices',[]),dtype=np.int64)
        record=dict(count=len(dataset),batch_size=loader.batch_size,
            indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            generator_before=hashlib.sha256(loader.generator.get_state().numpy().tobytes()).hexdigest(),batches=[])
        iterator=iter(loader)
        for _ in range(3):
            try:x,y=next(iterator)
            except StopIteration:break
            record['batches'].append(dict(shape=list(x.shape),dtype=str(x.dtype),
                pixels=hashlib.sha256(x.numpy().tobytes()).hexdigest(),labels=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
        record['generator_after']=hashlib.sha256(loader.generator.get_state().numpy().tobytes()).hexdigest()
        records.append(record)
    return dict(loaders=records,rng=rng_digest())


def main():
    native=PROOF/'reference/unit7/app.py'
    source=native.read_text()
    transformed=unit7_loading(source.replace('import torch\n','from browser_torch import torch\n',1),replace_once)
    control=compile_namespace(source,'loaders')
    proposed=compile_namespace(transformed,'_course_build_loaders')
    inputs={str(path.relative_to(PROOF)):sha(path) for path in
        (native,PROOF/'requirements-native.lock',PROOF/'scripts/stage_image_loading.py')}
    result=dict(status='running',executor_sha256=sha(Path(__file__)),inputs=inputs,cases={},
        input_fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        torch_version=torch.__version__,scope='Exact native Unit 7 data factory versus preserved loader body with invocation snapshots, all five complete datasets, required seeds, both split modes and default/augmented transformations. No browser training, appearance or gesture certification.')
    output=PROOF/'evidence/unit7-loading-native-calculation.json'
    torch.set_num_threads(1)
    old_cwd=Path.cwd();os.chdir(native.parent)
    try:
        for variant in ('MNIST','FashionMNIST','CIFAR10','SVHN','USPS'):
            for value in (0,42,123):
                for official in (True,False):
                    for augmented in (False,True):
                        params=dict(batch=31,variant=variant,valid=.1 if official else .2,seed=value,
                            use_official_test=official,augment=augmented,hflip=.35 if augmented else 0.,
                            vflip=.25 if augmented else 0.,invert=augmented)
                        if not official:params['test']=.15
                        key=f'{variant}/seed-{value}/{"official" if official else "custom"}/{"augmented" if augmented else "default"}'
                        seed(value)
                        control['input']=SimpleNamespace(**{k:(lambda v=v:v) for k,v in params.items()})
                        expected=observed(control['loaders']());gc.collect()
                        seed(value)
                        actual=observed(proposed['_course_build_loaders'](params));gc.collect()
                        equal=expected==actual
                        result['cases'][key]=dict(status='pass' if equal else 'fail',
                            assertion=dict(kind='computation',expected=expected,observed=actual,matched=equal))
                        write_json(output,result)
                        print(key,result['cases'][key]['status'],flush=True)
                        if not equal:raise AssertionError(key)
        result['status']='pass'
    except Exception as error:result.update(status='fail',error=str(error))
    finally:os.chdir(old_cwd);write_json(output,result)
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
