"""Checksum-verified compact datasets; convert only the requested image."""
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

_cache={}


def asset_bytes(name):
    if sys.platform == 'emscripten':
        raise RuntimeError('Dataset is not ready. Load the complete dataset and retry.')
    local=Path(__file__).parent/'image-assets'/name
    if local.exists(): return local.read_bytes()
    raise FileNotFoundError(f'Dataset asset missing: {name}')


def load(name,split):
    key=f'{name}/{split}'
    if key not in _cache:
        if sys.platform == 'emscripten':
            raise RuntimeError('Dataset is not ready. Load the complete dataset and retry.')
        # Retain only the selected dataset's train/test pair.
        for previous in list(_cache):
            if not previous.startswith(name+'/'): del _cache[previous]
        manifest=json.loads(asset_bytes('manifest.json'))
        if key not in manifest['datasets']:
            raise RuntimeError(f'The complete {key} dataset has not been packaged')
        entry=manifest['datasets'][key]
        payload=asset_bytes(entry['file'])
        if hashlib.sha256(payload).hexdigest()!=entry['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {key}. Retry loading.')
        with np.load(io.BytesIO(payload),allow_pickle=False) as packed:
            images,labels=packed['images'],packed['labels']
        if images.dtype!=np.uint8 or list(images.shape)!=entry['image_shape'] or len(labels)!=entry['count']:
            raise ValueError(f'Dataset shape mismatch: {key}')
        _cache[key]=(images,labels,entry['layout'])
    return _cache[key]


def dataset_classes(torch):
    def make(name):
        class Dataset:
            def __init__(self,root=None,train=True,transform=None,target_transform=None,download=False,split=None):
                self.data,labels,self.layout=load(name,split or ('train' if train else 'test'))
                self.targets=torch.tensor(labels,dtype=torch.int64)
                self.labels=labels
                self.transform,self.target_transform=transform,target_transform
            def __len__(self):return len(self.labels)
            def __getitem__(self,index):
                raw=self.data[index]
                if self.layout=='NCHW':raw=np.moveaxis(raw,0,-1)
                image=Image.fromarray(raw)
                label=int(self.labels[index])
                if self.transform is not None:image=self.transform(image)
                if self.target_transform is not None:label=self.target_transform(label)
                return image,label
        Dataset.__name__=name
        return Dataset
    return SimpleNamespace(**{name:make(name) for name in ('MNIST','FashionMNIST','CIFAR10','SVHN','USPS')})


def prepare_mnist(dataset,invert,mean=0.1307,std=0.3081,augmentations=None):
    # Preserve Unit 6's normalization (including its FashionMNIST constants)
    # and order of operations, while avoiding a full float32 image copy.
    from browser_torch import torch
    class Prepared:
        tensors=(None,dataset.targets)
        def __len__(self):return len(dataset)
        def __getitem__(self,index):
            image=dataset.data[index]
            if invert:image=image ^ 255
            x=torch.tensor(image).float().div_(255).sub_(mean).div_(std).unsqueeze(0)
            if augmentations:x=augmentations(x)
            return x,dataset.targets[index]
    return Prepared()
