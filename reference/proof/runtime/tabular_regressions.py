"""Original application bodies at the two confirmed unscaled t-SNE regressions."""
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import _t_sne
import tabular_probe


def run_regressions(definitions):
    arrays={};scope=dict(vars(tabular_probe))
    class SeabornData:
        @staticmethod
        def load_dataset(name):
            assert name=='penguins'
            return pd.read_csv('penguins.csv')
    scope['sns']=SeabornData
    exec(compile(definitions,'<original-unit1-regressions>','exec'),scope)
    for name,perplexity in [('iris',30),('penguins',50)]:
        scope['input']=SimpleNamespace(dataset=lambda:name,standardize=lambda:False,
            reduction_method=lambda:'tsne',perplexity=lambda:perplexity)
        frame=scope['get_data']();target=scope['get_target_column']()
        features=[column for column in frame.columns if column!=target]
        scope['input'].analysis_features=lambda:features
        for dimensions in (2,3):
            prefix=f'regressions/{name}/unscaled/perplexity{perplexity}/tsne{dimensions}'
            scope['input'].n_components=lambda:dimensions
            probability=_t_sne._joint_probabilities_nn;objective=_t_sne._kl_divergence_bh
            previous=sys.getprofile();step=[0]
            def observed_probability(distances,*args):
                arrays[prefix+'/neighbor-indices']=distances.indices.copy()
                arrays[prefix+'/neighbor-indptr']=distances.indptr.copy()
                arrays[prefix+'/neighbor-distances']=distances.data.copy()
                result=probability(distances,*args)
                arrays[prefix+'/probabilities']=result.toarray()
                return result
            def observed_objective(*args,**kwargs):
                result=objective(*args,**kwargs)
                if step[0] in (0,85):
                    arrays[prefix+f'/step{step[0]}/coordinates']=np.asarray(args[0]).copy()
                    arrays[prefix+f'/step{step[0]}/gradient']=result[1].copy()
                step[0]+=1
                return result
            def capture(frame,event,arg):
                if event=='return' and frame.f_code.co_filename=='<original-unit1-regressions>' and frame.f_code.co_name=='reduction_plot':
                    arrays[prefix+'/input']=np.asarray(frame.f_locals['features_data']).copy()
                    arrays[prefix+'/projection']=frame.f_locals['reduced_data'].copy()
            try:
                _t_sne._joint_probabilities_nn=observed_probability
                _t_sne._kl_divergence_bh=observed_objective
                sys.setprofile(capture)
                plt.close(scope['reduction_plot']())
            finally:
                _t_sne._joint_probabilities_nn=probability
                _t_sne._kl_divergence_bh=objective
                sys.setprofile(previous)
    return arrays
