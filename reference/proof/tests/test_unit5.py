"""Exercise the original helper paths against the frozen native runtime."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torchvision
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'proof/runtime'))
import browser_torch as browser


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Unit5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = helper('native_u5', ROOT/'assignments/5/u5_utils.py')
        cls.candidate = helper('browser_u5', ROOT/'proof/build/unit5/u5_utils.py')
        torch.set_num_threads(1)

    def test_original_archive_bytes(self):
        import zipfile
        with zipfile.ZipFile(ROOT/'assignments/Material-20261003/app.zip') as archive:
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT/'assignments/5'/name).read_bytes())

    def test_original_logistic_training_and_random_advancement(self):
        for seed in (0, 42, 17):
            results = []
            for lib, util in ((torch, self.native), (browser.torch, self.candidate)):
                util.set_seed(seed)
                df = util.get_dataset_logistic(200)
                coeffs = util.minimize_ce(df, 20, .1, .9, False)
                results.append((np.asarray(coeffs), util.predict_logistic(df[['x']], coeffs), lib.rand(10).numpy()))
            np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-4, atol=1e-5)
            for i in (1, 2):
                np.testing.assert_array_equal(results[0][i], results[1][i])

    def test_keyword_squeeze_and_cross_entropy_gradients(self):
        for shape, dim in (((4, 2), 1), ((4, 1, 2), 1), ((1, 2, 1), (0, 2)), ((), 0)):
            a = torch.ones(shape, requires_grad=True)
            b = browser.torch.ones(shape, requires_grad=True)
            x, y = a.squeeze(dim=dim), b.squeeze(dim=dim)
            self.assertEqual(tuple(x.shape), tuple(y.shape))
            x.sum().backward(); y.sum().backward()
            np.testing.assert_array_equal(a.grad.numpy(), b.grad.numpy())
        results = []
        for lib in (torch, browser.torch):
            x = lib.tensor([[.1, -.2], [.5, 1.2]], requires_grad=True)
            loss = lib.nn.CrossEntropyLoss()(input=x, target=lib.tensor([0, 1]))
            loss.backward()
            results.append((loss.item(), x.grad.numpy()))
        for i in (0, 1):
            np.testing.assert_allclose(results[0][i], results[1][i], rtol=1e-4, atol=1e-5)

    def test_original_image_transform_order_and_loader_consumption(self):
        raw = np.random.RandomState(3).randint(0, 256, (23, 28, 28), dtype=np.uint8)
        labels = np.arange(23) % 10
        from PIL import Image
        class Dataset:
            def __init__(self, *args, transform=None, **kwargs): self.transform = transform
            def __len__(self): return len(raw)
            def __getitem__(self, i): return self.transform(Image.fromarray(raw[i])), int(labels[i])
        for dataset in ('mnist', 'fashionmnist'):
            for invert, hflip, vflip in ((False, 0, 0), (True, 1, 0), (True, .3, .8)):
                outputs = []
                for lib, vision, util in ((torch, torchvision, self.native), (browser.torch, browser.torchvision, self.candidate)):
                    name = 'MNIST' if dataset == 'mnist' else 'FashionMNIST'
                    with patch.object(vision.datasets, name, Dataset):
                        loader, _ = getattr(util, 'get_dataset_'+dataset)(7, hflip, vflip, invert, seed=42)
                        batches = [(x.numpy(), y.numpy()) for x, y in loader]
                        outputs.append((batches, lib.rand(10).numpy()))
                for a, b in zip(outputs[0][0], outputs[1][0]):
                    np.testing.assert_allclose(a[0], b[0], rtol=1e-4, atol=1e-5)
                    np.testing.assert_array_equal(a[1], b[1])
                np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


if __name__ == '__main__': unittest.main()
