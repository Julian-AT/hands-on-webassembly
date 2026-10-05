"""Execute verbatim Unit 4 calculation bodies, without Shiny scheduling.

Only decorators are removed from the extracted functions. Inputs/Values mimic
session boundaries; split/model algorithms are never reimplemented here.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


class Value:
    def __init__(self, value=None): self.value = value
    def __call__(self): return self.value
    def set(self, value): self.value = value


def run_supervised(definitions):
    scope = dict(globals())
    scope['reactive'] = SimpleNamespace(isolate=nullcontext)
    exec(definitions, scope)
    arrays = {}
    for dataset in scope['DATASETS']:
        for standardize in (False, True):
            for seed in (42, 17):
                values = dict(load=1, dataset=dataset, use_all_features=True, feature_select=[],
                              split=70, val_split=15, data_seed=seed, standardize=standardize,
                              classifier_seed=seed, knn_k=3, dt_depth=5, rf_trees=50, rf_depth=5)
                scope['input'] = SimpleNamespace(**{key:Value(value) for key,value in values.items()})
                for key in ('Xtr','Xte','Xva','ytr','yte','yva','model'):
                    scope[key] = Value()
                scope['model_rev'] = Value(-1)
                scope['data_rev'] = Value(0)
                prefix = f'{dataset}/scaled{int(standardize)}/seed{seed}'
                frame = scope['df_loaded']()
                arrays[f'{prefix}/raw'] = frame.to_numpy()
                split = scope['compute_splits']()
                for part in ('train','val','test'):
                    arrays[f'{prefix}/{part}/indices'] = split['X_'+part].index.to_numpy(dtype=np.int64)
                    arrays[f'{prefix}/{part}/X'] = split['X_'+part].to_numpy()
                    arrays[f'{prefix}/{part}/y'] = split['y_'+part].to_numpy(dtype=np.int64)
                for classifier in ('k-NN','Decision Tree','Random Forest'):
                    scope['input'].classifier = Value(classifier)
                    scope['train_model']()
                    model = scope['model']()
                    for part in ('train','val','test'):
                        X, y = split['X_'+part], split['y_'+part]
                        prediction = model.predict(X)
                        arrays[f'{prefix}/{classifier}/{part}/predictions'] = prediction.astype(np.int64)
                        arrays[f'{prefix}/{classifier}/{part}/probabilities'] = model.predict_proba(X)
                        arrays[f'{prefix}/{classifier}/{part}/confusion'] = confusion_matrix(y,prediction)
                    if dataset == 'Digits' and seed == 42 and classifier == 'Random Forest':
                        for index, estimator in enumerate(model.estimators_):
                            tree = estimator.tree_
                            key = f'{prefix}/forest-trace/tree{index}'
                            arrays[key+'/seed'] = np.asarray(estimator.random_state, dtype=np.int64)
                            for field in ('children_left','children_right','feature','threshold','impurity','value','n_node_samples','weighted_n_node_samples'):
                                arrays[key+'/'+field] = getattr(tree,field).copy()
                    if hasattr(model,'feature_importances_'):
                        arrays[f'{prefix}/{classifier}/importances'] = model.feature_importances_
    for function in ('Noisy sine','Mystery function'):
        for seed in (123,17):
            for key,value in zip(('x','y','true'),scope['_gen_data'](40,0.2,seed,function)):
                arrays[f'{function}/{seed}/{key}'] = value
    return arrays
