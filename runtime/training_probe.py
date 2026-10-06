"""Run the ORIGINAL Unit 6/7 training bodies with deterministic diagnostic data.

Reactive scheduling and UI values are supplied by the harness. Computation,
loss averaging, epoch reseeding, preview passes and restoration stay verbatim.
"""
import asyncio
import copy
import json
import random
from types import SimpleNamespace,ModuleType
import sys
import numpy as np
import pandas as pd


class Value:
    def __init__(self,value=None):self.value=value
    def set(self,value):self.value=value
    def get(self):return self.value
    def __call__(self):return self.value


async def flush():pass


async def run_training(torch,definitions,training):
    arrays={}
    for unit in ('6','7'):
        module=ModuleType('original_training_'+unit)
        sys.modules[module.__name__]=module
        module.__dict__.update(torch=torch,nn=torch.nn,np=np,pd=pd,json=json,random=random,copy=copy,asyncio=asyncio,
            reactive=SimpleNamespace(flush=flush),Any=__import__('typing').Any,List=__import__('typing').List,
            Dict=__import__('typing').Dict,Tuple=__import__('typing').Tuple,Optional=__import__('typing').Optional,
            Union=__import__('typing').Union,dataclass=__import__('dataclasses').dataclass,math=__import__('math'))
        exec(definitions[unit],module.__dict__)
        exec(training[unit],module.__dict__)
        for seed in (0,42):
            for name,arch in module.PRESETS.items():
                # Real presets and full training loop; 23/7 samples deliberately
                # leave partial final batches to distinguish unit loss averaging.
                shape=(3,32,32) if 'CIFAR' in name else (1,28,28) if 'MNIST' in name else (1,)
                is_regression='Regression' in name and 'Logistic' not in name
                rng=np.random.default_rng(187)
                x=rng.normal(size=(30,*shape)).astype(np.float32)
                y=rng.normal(size=(30,1)).astype(np.float32) if is_regression else np.arange(30,dtype=np.int64)%2
                torch.manual_seed(seed)
                loaders=[]
                for sl in (slice(0,23),slice(23,30)):
                    ds=torch.utils.data.TensorDataset(torch.tensor(x[sl]),torch.tensor(y[sl]))
                    loaders.append(torch.utils.data.DataLoader(ds,batch_size=8,shuffle=sl.start==0,generator=torch.Generator().manual_seed(seed)))
                key=f'unit{unit}/{name}/seed{seed}'
                if unit=='6':
                    # Toy paths create their own native batch-64 loaders.
                    bundle={'kind':'img','train':loaders[0],'val':loaders[1]}
                    if shape==(1,):
                        bundle={'kind':'toy_reg' if is_regression else 'toy_bin','df':pd.DataFrame({'x':x[:,0],'y':y.reshape(-1)})}
                    module.data_bundle=lambda:bundle
                    module.arch_state=Value(json.dumps(arch))
                    module.input=SimpleNamespace(**{k:Value(v) for k,v in dict(train_seed=seed,lr=.01,momentum=.9,epochs=4,early=True,patience=1).items()})
                    for field in ('train_progress_pct','train_progress_msg','best_info','history_df','trained_model'):setattr(module,field,Value())
                    module.selected_device=Value('cpu')
                    await module._train()
                    model=module.trained_model()
                    history=module.history_df().to_numpy()
                else:
                    module.DataLoader=torch.utils.data.DataLoader
                    messages=[]
                    module._q_put=messages.append
                    result=module.run_training_sync(dict(seed=seed,epochs=4,lr=.01,momentum=.9,use_es=True,es_patience=1,use_cuda=False,arch_text=json.dumps(arch),loaders=(*loaders,loaders[1])))
                    model=result['model'];history=result['history_df'].to_numpy()
                    for i,weights in enumerate(result['filters']):arrays[f'{key}/filters/{i}']=weights
                arrays[f'{key}/history']=history
                for i,p in enumerate(model.parameters()):arrays[f'{key}/restored-parameters/{i}']=p.detach().numpy().copy()
                model.eval()
                arrays[f'{key}/predictions']=model(torch.tensor(x[:3])).detach().numpy().copy()
                arrays[f'{key}/next-random-draw']=torch.rand(10).numpy()
    return arrays
