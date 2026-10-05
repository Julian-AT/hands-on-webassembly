"""Original Assignment 5 calculation and full training/evaluation bodies.

Only reactive containers/scheduling and result observation are supplied here.
The original helper, split, models, optimizers and losses execute unchanged.
"""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd

class Value:
    def __init__(self, value=None): self.value=value
    def __call__(self): return self.value
    def get(self): return self.value
    def set(self,value): self.value=value

async def flush(): pass

async def run_unit5(torch, helper, definitions, csv_path, full=False):
    arrays={}
    scope=dict(torch=torch,nn=torch.nn,U5=helper,np=np,pd=pd,json=json,
               reactive=SimpleNamespace(flush=flush))
    exec(definitions,scope)
    def inputs(**kwargs): scope['input']=SimpleNamespace(**{k:Value(v) for k,v in kwargs.items()})
    def values(names):
        for name in names: scope[name]=Value()
    for seed in (0,42,17):
        for choice in ('Custom','Random','Classes'):
            key=f'linear/{choice}/seed{seed}'
            inputs(pr_seed=seed,pr_choice=choice,pr_n=200,pr_var=.5,pr_true_c='0.422;0.241',pr_deg=3)
            values(('pr_df','pr_true_coefficients','pr_coeffs'))
            scope['_pr_make'](); scope['_pr_fit']()
            arrays[key+'/data']=scope['pr_df']().to_numpy()
            arrays[key+'/coefficients']=np.array(scope['pr_coeffs']())
        inputs(lc_seed=seed,lc_n=200,lc_epochs=20,lc_lr=.1,lc_mom=.9)
        values(('lc_df','lc_coeffs','lc_acc'))
        scope['_lc_make'](); scope['_lc_train']()
        key=f'logistic1d/seed{seed}'
        arrays[key+'/data']=scope['lc_df']().to_numpy()
        arrays[key+'/coefficients']=np.array(scope['lc_coeffs']())
        arrays[key+'/accuracy']=np.array(scope['lc_acc']())
        arrays[key+'/predictions']=helper.predict_logistic(scope['lc_df']()[['x']],scope['lc_coeffs']())
        arrays[key+'/next-random']=torch.rand(10).numpy()
        for header in (False,True):
            key=f'logistic2d/seed{seed}/ignore-header{header}'
            inputs(lc_seed_2d=seed,lc_file_2d=[{'datapath':csv_path}],lc_ignore_header_2d=header,
                   lc_split_2d=.8,lc_epochs_2d=20,lc_lr_2d=.1,lc_mom_2d=.9)
            values(('lc_df_2d','lc_df_2d_train','lc_df_2d_test','lc_coeffs_2d','lc_acc_2d_train','lc_acc_2d_test'))
            scope['_lc_make_2d'](); scope['_lc_train_2d']()
            for part in ('train','test'):
                df=scope['lc_df_2d_'+part]()
                arrays[key+'/'+part+'/indices']=df.index.to_numpy()
                arrays[key+'/'+part+'/data']=df.to_numpy()
                arrays[key+'/'+part+'/predictions']=helper.predict_logistic(df[df.columns[:-1]],scope['lc_coeffs_2d']())
            arrays[key+'/coefficients']=np.array(scope['lc_coeffs_2d']())
            arrays[key+'/next-random']=torch.rand(10).numpy()
    if full:
        for dataset in ('mnist','fashion'):
            key=f'full/{dataset}/seed42'
            inputs(mn_seed=42,mn_batch=256,mn_ds=dataset,mn_epochs=3,mn_lr=.1,mn_mom=.9,mn_hflip=False,mn_invert=False)
            values(('mn_coeffs','mn_accuracy','mn_samples','mn_cm_mat','mn_mis','mn_progress_pct','mn_progress_msg'))
            losses=[]
            original=torch.nn.CrossEntropyLoss.forward
            original_linear=torch.nn.Linear.forward
            original_step=torch.optim.SGD.step
            observed_steps=[0]
            def observe_linear(self, input):
                result=original_linear(self,input)
                if not observed_steps[0]:
                    for name,value in [('input',input),('weight',self.weight),('bias',self.bias),('logits',result)]:
                        arrays[key+'/first-batch/'+name]=value.detach().numpy().copy()
                return result
            def observe_step(self,*args,**kwargs):
                if not observed_steps[0]:
                    for i,p in enumerate(self.param_groups[0]['params']):
                        arrays[key+'/first-batch/gradient'+str(i)]=p.grad.detach().numpy().copy()
                result=original_step(self,*args,**kwargs)
                if not observed_steps[0]:
                    for i,p in enumerate(self.param_groups[0]['params']):
                        arrays[key+'/first-batch/updated'+str(i)]=p.detach().numpy().copy()
                observed_steps[0]+=1
                return result
            def observe_loss(self,*args,**kwargs):
                loss=original(self,*args,**kwargs)
                losses.append(float(loss.item()))
                return loss
            torch.nn.CrossEntropyLoss.forward=observe_loss
            torch.nn.Linear.forward=observe_linear
            torch.optim.SGD.step=observe_step
            try: await scope['_mn_train']()
            finally:
                torch.nn.CrossEntropyLoss.forward=original
                torch.nn.Linear.forward=original_linear
                torch.optim.SGD.step=original_step
            arrays[key+'/batch-losses']=np.array(losses)
            arrays[key+'/parameters']=np.array(scope['mn_coeffs']())
            arrays[key+'/after-training-random']=torch.rand(10).numpy()
            scope['_mn_eval']()
            arrays[key+'/accuracy']=np.array(scope['mn_accuracy']())
            arrays[key+'/confusion']=scope['mn_cm_mat']()
            for i,part in enumerate(scope['mn_samples']()): arrays[key+'/samples/'+str(i)]=part
            for i,part in enumerate(scope['mn_mis']()): arrays[key+'/misclassified/'+str(i)]=part
            arrays[key+'/after-evaluation-random']=torch.rand(10).numpy()
    return arrays


def compare(arrays, reference):
    if set(arrays)!=set(reference.files): raise AssertionError('Unit 5 fixture keys differ')
    results={}
    for key in reference.files:
        a,b=np.asarray(arrays[key]),reference[key]
        exact=b.dtype.kind in 'biu'
        passed=a.shape==b.shape and (np.array_equal(a,b) if exact else np.allclose(a,b,rtol=1e-4,atol=1e-5))
        results[key]={'status':'pass' if passed else 'fail','exact':exact,
                      'max_abs_error':float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else None}
        if a.shape==b.shape:
            results[key]['bitwise_equal']=bool(np.array_equal(a,b))
        if not passed and a.shape==b.shape and a.ndim==1:
            bad=np.flatnonzero(a!=b if exact else ~np.isclose(a,b,rtol=1e-4,atol=1e-5))
            results[key]['first_divergent_index']=int(bad[0]) if len(bad) else None
    return results
