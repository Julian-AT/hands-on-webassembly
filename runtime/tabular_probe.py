"""Execute original Unit 1 calculation bodies and observe t-SNE internals."""
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets as skds,datasets
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import _t_sne


def run_tabular(definitions):
    scope=dict(globals())
    class SeabornData:
        heatmap=staticmethod(sns.heatmap)
        @staticmethod
        def load_dataset(name):
            assert name=='penguins'
            return pd.read_csv('penguins.csv')
    scope['sns']=SeabornData
    arrays={};current=['']
    class TracedTSNE(TSNE):
        def _tsne(self,P,degrees_of_freedom,n_samples,X_embedded,neighbors=None,skip_num_points=0):
            prefix=current[0]
            arrays[prefix+'/initialization']=X_embedded.copy()
            arrays[prefix+'/probabilities']=P.toarray() if hasattr(P,'toarray') else P.copy()
            objective=_t_sne._kl_divergence_bh
            iteration=[0]
            def observed(*args,**kwargs):
                result=objective(*args,**kwargs)
                # Observe every optimizer objective call without changing its inputs.
                step=iteration[0]; iteration[0]+=1
                if step < 250:
                    arrays[f'{prefix}/iterations/{step}/coordinates']=np.asarray(args[0]).copy()
                    arrays[f'{prefix}/iterations/{step}/gradient']=result[1].copy()
                key=prefix+'/first-gradient'
                if key not in arrays:arrays[key]=result[1].copy()
                return result
            _t_sne._kl_divergence_bh=observed
            try:return super()._tsne(P,degrees_of_freedom,n_samples,X_embedded,neighbors,skip_num_points)
            finally:_t_sne._kl_divergence_bh=objective
    scope['TSNE']=TracedTSNE
    exec(compile(definitions,'<original-unit1-calculations>','exec'),scope)
    for name in ('wine','penguins','iris','breast'):
        scope['input']=SimpleNamespace(dataset=lambda:name,standardize=lambda:True)
        frame=scope['get_data']();target=scope['get_target_column']()
        arrays[name+'/raw']=frame.to_numpy()
        arrays[name+'/summary']=scope['data_summary']().drop(columns=['stat']).to_numpy()
        features=[c for c in frame.columns if c!=target]
        scope['input'].analysis_features=lambda:features
        captured={}
        def capture(frame,event,arg):
            if event=='return' and frame.f_code.co_filename=='<original-unit1-calculations>':
                if frame.f_code.co_name=='reduction_plot' and 'reduced_data' in frame.f_locals:
                    captured['scaled']=np.asarray(frame.f_locals['features_data']).copy()
                    captured['projection']=frame.f_locals['reduced_data'].copy()
                    reducer=frame.f_locals['reducer']
                    if hasattr(reducer,'explained_variance_ratio_'):captured['variance']=reducer.explained_variance_ratio_.copy()
                if frame.f_code.co_name=='analysis_plot' and 'corr_matrix' in frame.f_locals:captured['correlation']=frame.f_locals['corr_matrix'].to_numpy()
        previous=sys.getprofile()
        try:
            sys.setprofile(capture)
            plt.close(scope['analysis_plot']())
            arrays[name+'/correlation']=captured['correlation']
            for dims in (2,3):
                for method in ('pca','tsne'):
                    scope['input'].n_components=lambda:dims
                    scope['input'].reduction_method=lambda:method
                    scope['input'].perplexity=lambda:30
                    current[0]=f'{name}/{method}{dims}'
                    plt.close(scope['reduction_plot']())
                    arrays[name+'/scaled']=captured['scaled']
                    arrays[current[0]]=captured['projection']
                    if method=='pca':arrays[f'{name}/variance{dims}']=captured['variance']
        finally:sys.setprofile(previous)
    return arrays
