"""Package COMPLETE torchvision datasets with their native ordering and labels.

Runs only at build time. Upstream torchvision verifies published download MD5s;
our versioned asset lock binds the compact uint8 arrays to the frozen loader.
"""
import argparse
import inspect
import json
import time
import numpy as np
import torchvision
from common import PROOF, sha, write_json

DATASETS=('MNIST','FashionMNIST','CIFAR10','SVHN','USPS')
COUNTS={'MNIST':(60000,10000),'FashionMNIST':(60000,10000),'CIFAR10':(50000,10000),'SVHN':(73257,26032),'USPS':(7291,2007)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=DATASETS,action='append')
    args=parser.parse_args()
    root=PROOF/'cache/torchvision'
    assets=PROOF/'assets/v1/images'
    assets.mkdir(parents=True,exist_ok=True)
    manifest_path=assets/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'format':1,'torchvision':torchvision.__version__,'datasets':{}}
    if manifest['torchvision']!=torchvision.__version__: raise ValueError('Native torchvision version changed')
    failures={}
    for name in args.dataset or DATASETS:
        cls=getattr(torchvision.datasets,name)
        for index,split in enumerate(('train','test')):
            key=f'{name}/{split}'
            try:
                kwargs={'split':split} if name=='SVHN' else {'train':split=='train'}
                # Fetch through the selected reference's dataset parser.
                for attempt in range(3):
                    try:
                        ds=cls(root=str(root),download=True,**kwargs)
                        break
                    except Exception:
                        if attempt==2: raise
                        time.sleep(1+attempt)
                raw=np.asarray(ds.data)
                labels=np.asarray(ds.labels if name=='SVHN' else ds.targets,dtype=np.int64)
                assert raw.dtype==np.uint8 and len(raw)==len(labels)==COUNTS[name][index]
                file=assets/f'{name}-{split}.npz'
                np.savez_compressed(file,images=raw,labels=labels)
                with np.load(file) as packed:
                    np.testing.assert_array_equal(packed['images'],raw)
                    np.testing.assert_array_equal(packed['labels'],labels)
                entry={'file':file.name,'sha256':sha(file),'bytes':file.stat().st_size,'count':len(raw),
                       'image_shape':list(raw.shape),'image_dtype':str(raw.dtype),'label_dtype':str(labels.dtype),
                       'layout':'NCHW' if name=='SVHN' else 'NHWC' if name=='CIFAR10' else 'NHW',
                       'native_loader_sha256':sha(__import__('pathlib').Path(inspect.getfile(cls)))}
                if key in manifest['datasets'] and entry!=manifest['datasets'][key]: raise ValueError(f'Locked image asset changed: {key}')
                manifest['datasets'][key]=entry
                write_json(manifest_path,manifest)
                print(key,len(raw),file.stat().st_size,'bytes',flush=True)
                del ds,raw,labels
            except Exception as e:
                failures[key]=f'{type(e).__name__}: {e}'
                print(key,'FAILED',failures[key],flush=True)
    write_json(PROOF/'evidence/image-packaging.json',{'status':'fail' if failures else 'pass','failures':failures,'manifest_sha256':sha(manifest_path) if manifest_path.exists() else None,
               'qualification':'Complete compact source data only; browser training and preprocessing require separate tests.'})
    if failures: raise SystemExit(1)


if __name__=='__main__':main()
