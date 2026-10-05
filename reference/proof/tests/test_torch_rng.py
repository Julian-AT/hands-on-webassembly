"""Compare native torch and the adapter across independent/repeated streams."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import borch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from torch_rng import install
install(borch)
torch.set_num_threads(1)


class RNGTests(unittest.TestCase):
    def test_uniform_independent_generators(self):
        for seed in (0,1,42,2**32+17,-1):
            a=torch.Generator().manual_seed(seed);b=borch.Generator().manual_seed(seed)
            for n in (1,17,10000,3):
                np.testing.assert_array_equal(torch.rand(n,generator=a).numpy(),borch.rand(n,generator=b).numpy())
    def test_initialization(self):
        for seed in (0,42,17):
            torch.manual_seed(seed);borch.manual_seed(seed)
            for a,b in [(torch.nn.Linear(19,7),borch.nn.Linear(19,7)),(torch.nn.Conv2d(3,4,3),borch.nn.Conv2d(3,4,3))]:
                for p,q in zip(a.parameters(),b.parameters()): np.testing.assert_allclose(p.detach().numpy(),q.detach().numpy(),rtol=1e-4,atol=1e-5)
    def test_dropout_and_advancement(self):
        for seed in (0,42,17):
            torch.manual_seed(seed);borch.manual_seed(seed)
            for p,n in ((.5,10),(.1,511),(.9,1),(.0,10),(.3,10000)):
                np.testing.assert_array_equal(torch.nn.Dropout(p)(torch.ones(n)).numpy(),borch.nn.Dropout(p)(borch.ones(n)).numpy())
            np.testing.assert_array_equal(torch.rand(20).numpy(),borch.rand(20).numpy())
    def test_splits(self):
        for seed in (0,42,17):
            for n in (20,535,60000):
                a=torch.utils.data.random_split(list(range(n)),[n-3,3],generator=torch.Generator().manual_seed(seed))
                b=borch.utils.data.random_split(list(range(n)),[n-3,3],generator=borch.Generator().manual_seed(seed))
                for x,y in zip(a,b):self.assertEqual(x.indices,y.indices)
    def test_loaders_repeated_epochs(self):
        for seed in (0,42,17):
            for independent in (False,True):
                for shuffle in (False,True):
                    torch.manual_seed(seed);borch.manual_seed(seed)
                    loaders=[]
                    for lib in (torch,borch):
                        ds=lib.utils.data.TensorDataset(lib.arange(23))
                        loaders.append(lib.utils.data.DataLoader(ds,batch_size=7,shuffle=shuffle,generator=lib.Generator().manual_seed(seed) if independent else None))
                    for _ in range(3):
                        a,b=[np.concatenate([x[0].numpy() for x in loader]) for loader in loaders]
                        np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(torch.rand(10).numpy(),borch.rand(10).numpy())
    def test_preview_iterator_consumption(self):
        torch.manual_seed(42);borch.manual_seed(42)
        for lib in (torch,borch):
            ds=lib.utils.data.TensorDataset(lib.arange(100))
            loader=lib.utils.data.DataLoader(ds,batch_size=7,shuffle=True)
            next(iter(loader))
        np.testing.assert_array_equal(torch.rand(10).numpy(),borch.rand(10).numpy())


if __name__=='__main__':unittest.main()
