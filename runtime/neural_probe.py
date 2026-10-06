"""Runs the ORIGINAL builders and presets against an injected torch backend.

Small, deterministic diagnostic fixtures. Passing these does not pass the
full-dataset, original-interface or stochastic-training acceptance gates.
"""
import copy
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def run_probe(torch, definitions):
    arrays, checks = {}, {}

    def record(name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            checks[name] = {"status": "pass", "detail": detail, "seconds": time.perf_counter() - start}
        except Exception as e:
            checks[name] = {"status": "fail", "error": f"{type(e).__name__}: {e}", "seconds": time.perf_counter() - start}

    def save(name, tensor):
        a = tensor.detach().cpu().numpy().copy()
        if not np.isfinite(a).all():
            raise AssertionError(f"Nonfinite result: {name}")
        arrays[name] = a

    def train(name, module, arch, shape):
        torch.manual_seed(42)
        model = module.build_model(module.parse_architecture(json.dumps(arch)), shape)
        for i, parameter in enumerate(model.parameters()):
            save(f"{name}/seeded_initialization/{i}", parameter)
        rng = np.random.default_rng(101)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.tensor(rng.normal(0, .05, size=tuple(parameter.shape)).astype(np.float32)))
        x = torch.tensor(rng.normal(size=(3, *shape)).astype(np.float32))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        for step in range(2):
            optimizer.zero_grad()
            y = model(x)
            if y.shape[-1] == 1:
                target = torch.tensor([[.2], [-.3], [.1]], dtype=torch.float32)
                loss = torch.nn.MSELoss()(y, target)
            else:
                target = torch.tensor([0, 1, 0], dtype=torch.long)
                loss = torch.nn.CrossEntropyLoss()(y, target)
            save(f"{name}/{step}/output", y)
            save(f"{name}/{step}/loss", loss)
            loss.backward()
            for i, p in enumerate(model.parameters()):
                save(f"{name}/{step}/gradient/{i}", p.grad)
            optimizer.step()
            for i, p in enumerate(model.parameters()):
                save(f"{name}/{step}/parameter/{i}", p)
        state = copy.deepcopy(model.state_dict())
        model.load_state_dict(state)
        model.eval()
        save(f"{name}/restored", model(x))
        # Same intermediate inspection used by Unit 7.
        with torch.no_grad():
            activation = x
            for i, layer in enumerate(model.children()):
                activation = layer(activation)
                save(f"{name}/activation/{i}", activation)
        return {"parameters": sum(p.numel() for p in model.parameters()), "steps": 2}

    for unit, source in definitions.items():
        module = types.ModuleType(f"unit{unit}_definitions")
        module.__dict__.update(globals(), torch=torch, nn=torch.nn)
        module.__name__ = f"unit{unit}_definitions"
        sys.modules[module.__name__] = module
        exec(source, module.__dict__)
        for name, arch in module.PRESETS.items():
            shape = (3, 32, 32) if "CIFAR" in name else (1, 28, 28) if "MNIST" in name else (1,)
            key = f"unit{unit}/{name}"
            record(key, lambda key=key, arch=arch, shape=shape: train(key, module, arch, shape))
        # Exercise every accepted layer; stochastic dropout is checked separately.
        if int(unit) == 6:
            layers = [{"type": "linear", "out_features": 8}, {"type": "batchnorm1d"}, {"type": "leakyrelu"},
                      {"type": "sigmoid"}, {"type": "gelu"}, {"type": "dropout", "p": 0}, {"type": "linear", "out_features": 2}]
            shape = (4,)
        else:
            layers = [{"type": "conv2d", "out_channels": 2, "kernel_size": 3, "padding": 1},
                      {"type": "batchnorm2d"}, {"type": "leakyrelu"}, {"type": "avgpool2d", "kernel_size": 2},
                      {"type": "sigmoid"}, {"type": "tanh"}, {"type": "dropout", "p": 0}, {"type": "linear", "out_features": 2}]
            shape = (1, 8, 8)
        key = f"unit{unit}/additional_layers"
        record(key, lambda: train(key, module, {"layers": layers}, shape))

    data = torch.utils.data
    def loader(with_worker):
        dataset = data.TensorDataset(torch.arange(12).reshape(6, 2), torch.arange(6))
        dl = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                             worker_init_fn=(lambda _: None) if with_worker else None,
                             generator=torch.Generator().manual_seed(42), persistent_workers=False)
        batches = list(dl)
        save("loader/order", torch.cat([y for _, y in batches]))
        return {"batches": len(batches)}
    record("original_loader_arguments", lambda: loader(True))
    record("single_worker_adapter", lambda: loader(False))
    record("set_num_threads", lambda: torch.set_num_threads(1))
    def adapter():
        from borch_compat import single_process_loader, set_single_thread
        calls = []
        ds = data.TensorDataset(torch.arange(6))
        dl = single_process_loader(torch)(ds, batch_size=2, num_workers=0, worker_init_fn=lambda _: calls.append(1))
        assert len(list(dl)) == 3 and not calls
        set_single_thread(1)
        for call in [lambda: set_single_thread(2), lambda: single_process_loader(torch)(ds, num_workers=1)]:
            try:
                call()
            except ValueError:
                continue
            raise AssertionError("Adapter silently accepted an unsupported worker/thread count")
    record("process_adapter", adapter)
    def split():
        parts = data.random_split(list(range(20)), [16, 4], generator=torch.Generator().manual_seed(42))
        arrays["split/indices"] = np.asarray(parts[0].indices)
    record("validation_split", split)
    def dropout():
        torch.manual_seed(42)
        save("dropout/seeded", torch.nn.Dropout(.5)(torch.ones(4, 8)))
    record("seeded_dropout", dropout)
    def augmentation():
        if torch.__name__ == 'borch':
            import borchvision as vision
            from image_transforms import install
            install(torch, vision)
        else:
            import torchvision as vision
        pixels = torch.arange(3*5*7, dtype=torch.float32).reshape(3,5,7)
        for name in ('RandomHorizontalFlip','RandomVerticalFlip'):
            for seed in (0,17,42):
                for probability in (0.,.1,.5,1.):
                    torch.manual_seed(seed)
                    transform = getattr(vision.transforms,name)(probability)
                    key = f'augmentation/{name}/{seed}/{probability}'
                    save(key, torch.stack([transform(pixels) for _ in range(5)]))
                    save(key+'/next-random-draw',torch.rand(10))
    record('original_tensor_augmentation',augmentation)
    return checks, arrays
