import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
import browser_torch as browser


class TransformTests(unittest.TestCase):
    def test_flips_and_rng_advancement(self):
        for name in ("RandomHorizontalFlip", "RandomVerticalFlip"):
            for seed in (0, 17, 42):
                for probability in (0.0, 0.1, 0.5, 1.0):
                    torch.manual_seed(seed)
                    browser.torch.manual_seed(seed)
                    native = getattr(torchvision.transforms, name)(probability)
                    port = getattr(browser.T, name)(probability)
                    pixels = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
                    for _ in range(12):
                        a = native(torch.tensor(pixels)).numpy()
                        b = port(browser.torch.tensor(pixels)).numpy()
                        np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(
                        torch.rand(10).numpy(), browser.torch.rand(10).numpy()
                    )

    def test_original_cnn_transform_order(self):
        for channels in (1, 3):
            raw = np.arange(5 * 7 * channels, dtype=np.uint8).reshape(
                (5, 7) if channels == 1 else (5, 7, channels)
            )
            image = Image.fromarray(raw)
            for seed in (0, 42):
                results = []
                for lib, transforms in (
                    (torch, torchvision.transforms),
                    (browser.torch, browser.T),
                ):
                    lib.manual_seed(seed)
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Lambda(lambda value: 1 - value),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.3),
                            transforms.Normalize((0.5,) * channels, (0.25,) * channels),
                        ]
                    )
                    results.append([transform(image).numpy() for _ in range(8)])
                np.testing.assert_array_equal(results[0], results[1])
