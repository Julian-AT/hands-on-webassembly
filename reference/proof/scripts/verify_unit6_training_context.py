"""Compare disposable native contexts to unobserved exact-body controls."""
import argparse
import ast
import asyncio
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import types
import typing
import copy
import numpy as np
import pandas as pd
import torch
from common import ROOT,PROOF,sha,write_json
from unit6_training_source import training_source


class Value:
    def __init__(self,value=None):self.value=value
    def get(self):return self.value
    def __call__(self):return self.value
    def set(self,value):self.value=value


async def flush():pass


def module(source):
    namespace=types.ModuleType('_unit6_independent_native_control')
    sys.modules[namespace.__name__]=namespace
    namespace.__dict__.update(torch=torch,nn=torch.nn,np=np,pd=pd,json=json,math=math,
        random=random,asyncio=asyncio,copy=copy,dataclass=dataclasses.dataclass)
    namespace.__dict__.update({name:getattr(typing,name) for name in ('Any','List','Dict','Tuple','Optional','Union')})
    exec(source,namespace.__dict__)
    return namespace


def params(name,architecture,seed,epochs=4):
    shape=(1,28,28) if 'MNIST' in name else (1,)
    regression='Regression' in name and 'Logistic' not in name
    rng=np.random.default_rng(187)
    x=rng.normal(size=(30,*shape)).astype(np.float32)
    y=rng.normal(size=(30,1)).astype(np.float32) if regression else np.arange(30,dtype=np.int64)%2
    if shape==(1,):
        bundle=dict(kind='toy_reg' if regression else 'toy_bin',df=pd.DataFrame({'x':x[:,0],'y':y.reshape(-1)}))
        loaders=()
    else:
        loaders=[]
        for section in (slice(0,23),slice(23,30)):
            dataset=torch.utils.data.TensorDataset(torch.tensor(x[section]),torch.tensor(y[section]))
            loaders.append(torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=section.start==0,
                generator=torch.Generator().manual_seed(seed)))
        loaders=(*loaders,loaders[1])
        bundle=dict(kind='img',train=loaders[0],val=loaders[1],test=loaders[2])
    return dict(inputs=dict(train_seed=seed,lr=.01,momentum=.9,epochs=epochs,early=True,patience=1),
        bundle=bundle,loaders=loaders,arch_text=json.dumps(architecture),device='cpu')


async def control(namespace,parameters):
    namespace.reactive=types.SimpleNamespace(flush=flush)
    namespace.input=types.SimpleNamespace(**{key:Value(value) for key,value in parameters['inputs'].items()})
    namespace.data_bundle=lambda:parameters['bundle']
    namespace.arch_state=Value(parameters['arch_text'])
    namespace.selected_device=Value('cpu')
    for name in ('train_progress_pct','train_progress_msg','best_info','history_df','trained_model'):
        setattr(namespace,name,Value())
    await namespace._train()
    model=namespace.trained_model()
    return dict(history=namespace.history_df().to_numpy(),state={name:value.detach().numpy().copy() for name,value in model.state_dict().items()},
        best_info=namespace.best_info())


def arrays_equal(result,expected):
    np.testing.assert_array_equal(result['history_df'].to_numpy(),expected['history'])
    if result['best_info']!=expected['best_info']:raise AssertionError((result['best_info'],expected['best_info']))
    if result['model_record']['state'].keys()!=expected['state'].keys():raise AssertionError('State keys differ')
    for name,value in result['model_record']['state'].items():np.testing.assert_array_equal(value,expected['state'][name])


def control_source(source):
    tree=ast.parse(source);lines=source.splitlines(keepends=True)
    names={'LayerSpec','parse_architecture','build_model','ACTIVATIONS','PRESETS'}
    pieces=[]
    for node in tree.body:
        found={node.name} if isinstance(node,(ast.FunctionDef,ast.ClassDef)) else set()
        if isinstance(node,ast.Assign):found={target.id for target in node.targets if isinstance(target,ast.Name)}
        if found & names:
            start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
            pieces.append(''.join(lines[start:node.end_lineno]))
    body=next(node for node in ast.walk(tree) if isinstance(node,ast.AsyncFunctionDef) and node.name=='_train')
    import textwrap
    pieces.append(textwrap.dedent(''.join(lines[body.lineno-1:body.end_lineno])))
    return '\n\n'.join(pieces)


async def verify(context,source,independent_source,result,output):
    original=module(independent_source)
    for name,architecture in original.PRESETS.items():
        for seed in (0,42,123):
            expected=await control(original,params(name,architecture,seed))
            native=context.cloudpickle.loads(await asyncio.to_thread(context.execute_payload,source,
                context._snapshot(params(name,architecture,seed)),lambda message:None))
            arrays_equal(native,expected)
            yielding=context.cloudpickle.loads(await context.execute_payload_async(source,context._snapshot(params(name,architecture,seed)),lambda message:None,lambda:False))
            arrays_equal(yielding,expected)
            owner=context.Executor(6);messages=[]
            task=asyncio.create_task(owner.run(params(name,architecture,seed),messages.append))
            while not task.done():
                random.seed(999);np.random.seed(999);torch.manual_seed(999)
                torch.nn.Linear(784,10);await asyncio.sleep(.01)
            disposable=await task;arrays_equal(disposable,expected)
            if not messages or owner.active is not None:raise AssertionError('Missing progress or abandoned process')
            key=f'{name}/seed-{seed}'
            result['cases'][key]=dict(status='pass',
                assertion=dict(kind='computation',expected=dict(history=True,parameters=True,best_model=True),
                    observed=dict(history=True,parameters=True,best_model=True),matched=True),
                native_control_history_sha256=hashlib.sha256(expected['history'].tobytes()).hexdigest(),
                native_control_parameters={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in expected['state'].items()},
                methods=['independent original body','synchronous candidate','yielding candidate','disposable spawned process with main-context RNG activity'])
            write_json(output,result);print(key,'pass',flush=True)
    name,architecture=next(iter(original.PRESETS.items()))
    parameters=params(name,architecture,42,epochs=10000);owner=context.Executor(6)
    task=asyncio.create_task(owner.run(parameters,lambda message:None))
    while owner.active is None:await asyncio.sleep(.01)
    process=owner.active;start=__import__('time').perf_counter()
    owner.cancel();task.cancel()
    try:await task;raise AssertionError('Cancelled context returned a result')
    except asyncio.CancelledError:pass
    seconds=__import__('time').perf_counter()-start
    if process.is_alive() or seconds>1:raise AssertionError('Cancelled calculation did not disappear within one second')
    result['cases']['cancel-native-process']=dict(status='pass',seconds=seconds,
        assertion=dict(kind='workflow',expected=dict(cancelled=True,deadline_met=True),observed=dict(cancelled=not process.is_alive(),deadline_met=seconds<=1),matched=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--label',required=True)
    parser.add_argument('--control',type=Path,default=ROOT/'assignments/6/app.py')
    args=parser.parse_args();sys.path.insert(0,str(args.candidate.resolve()));import isolated_training as context
    source=training_source();torch.set_num_threads(1)
    output=PROOF/f'evidence/unit6-training-context-{args.label}.json'
    if output.exists():raise FileExistsError('Retain previous observations; choose a fresh label')
    config=json.loads((args.candidate/'course-training.json').read_text())
    if source!=config['source']:raise ValueError('Candidate source identity differs')
    independent_source=control_source(args.control.read_text())
    original_body=next(node for node in ast.walk(ast.parse(independent_source)) if isinstance(node,ast.AsyncFunctionDef) and node.name=='_train')
    wrapper_body=next(node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.AsyncFunctionDef) and node.name=='_train')
    if [ast.dump(node) for node in original_body.body]!=[ast.dump(node) for node in wrapper_body.body]:raise ValueError('Original arithmetic body changed')
    result=dict(status='running',cases={},executor_sha256=sha(Path(__file__)),
        inputs={str(path.resolve().relative_to(PROOF)):sha(path) for path in (args.candidate/'isolated_training.py',args.candidate/'course-training.json',PROOF/'scripts/unit6_training_source.py',PROOF/'reference/unit6/app.py',PROOF/'requirements-native.lock')},
        original_body_unchanged=True,torch_version=torch.__version__,
        native_control=dict(path=str(args.control.resolve().relative_to(ROOT)),sha256=sha(args.control)),
        scope='All seven supplied Unit 6 presets at seeds 0,42,123 on original 30-sample diagnostic data, exact history/parameters/best-model decisions against fresh independent native controls. Disposable process cancellation. Complete full-data and browser numerical matrices remain separate requirements.')
    try:asyncio.run(verify(context,source,independent_source,result,output));result['status']='pass'
    except Exception as error:result.update(status='fail',error=str(error))
    write_json(output,result);return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
