"""Explicit source-checked neural runtime patches. Original models/UI stay intact."""

from repository import runtime_asset, RUNTIME
import ast
import shutil
import zipfile
from common import PROOF, sha


def stage_neural(unit, stage, source, replace_once):
    wheel = runtime_asset("pyborch-1.14.1-py3-none-any.whl")
    import json

    expected = json.loads((PROOF / "assets.lock.json").read_text())[wheel.name]
    if sha(wheel) != expected["sha256"] or wheel.stat().st_size != expected["bytes"]:
        raise ValueError("Pinned neural wheel changed")
    with zipfile.ZipFile(wheel) as archive:
        # Include only CPU implementation and its complete torchvision transforms.
        for name in archive.namelist():
            if name.startswith("borch/") or name == "borchvision.py" or "/licenses/" in name:
                archive.extract(name, stage)
    for name in (
        "browser_torch.py",
        "neural_compat.py",
        "cnn_compat.py",
        "torch_rng.py",
        "borch_compat.py",
        "image_data.py",
        "module_hooks.py",
        "image_transforms.py",
    ):
        shutil.copy2(RUNTIME / name, stage / name)
    if unit in (5, 6):
        shutil.copy2(RUNTIME / "inspection_rng.py", stage / "inspection_rng.py")
    if unit == 6:
        source = replace_once(
            source,
            "import json, math, numpy as np, pandas as pd, torch",
            "import json, math, numpy as np, pandas as pd\nfrom browser_torch import torch\nimport asyncio",
        )
    elif unit == 7:
        source = replace_once(source, "import torch\n", "from browser_torch import torch\n")
        source = replace_once(
            source,
            "from torch.utils.data import ConcatDataset, Subset, DataLoader",
            "from browser_torch import ConcatDataset, Subset, DataLoader",
        )
        source = replace_once(
            source, "import torchvision\n", "from browser_torch import torchvision\n"
        )
        source = replace_once(
            source, "import torchvision.transforms as T", "from browser_torch import T"
        )
    if unit in (5, 6, 7):
        from reference_corrections import training_source
        from training_bundle import browser_build_id
        import json
        import cloudpickle
        from pathlib import Path

        shutil.copy2(RUNTIME / "isolated_training.py", stage / "isolated_training.py")
        body = training_source(unit)
        (stage / "course-training.json").write_text(
            json.dumps({"source": body, "build_id": browser_build_id(body)}, sort_keys=True)
        )
        shutil.copytree(
            Path(cloudpickle.__file__).parent,
            stage / "cloudpickle",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    if unit == 5:
        source = replace_once(
            source, "import torch\n", "from browser_torch import torch\nimport asyncio\n"
        )
        helper = stage / "u5_utils.py"
        text = helper.read_text()
        for old, new in [
            ("import torch\n", "from browser_torch import torch\n"),
            ("import torchvision\n", "from browser_torch import torchvision\n"),
            ("from torch.utils.data import DataLoader", "from browser_torch import DataLoader"),
        ]:
            text = replace_once(text, old, new)
        text = replace_once(
            text,
            "from distutils.version import LooseVersion",
            "from setuptools._distutils.version import LooseVersion",
        )
        helper.write_text(text)
    source = replace_once(source, "import torch.nn as nn", "from browser_torch import nn")
    if unit == 6:
        helper = stage / "u6_utils.py"
        text = helper.read_text()
        text = replace_once(text, "import torch\n", "from browser_torch import torch\n")
        text = replace_once(text, "import torchvision\n", "from browser_torch import torchvision\n")
        text = replace_once(
            text, "from torch.utils.data import DataLoader", "from browser_torch import DataLoader"
        )
        tree = ast.parse(text)
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_dataset_mnist"
        )
        prepare = next(
            n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_dataset"
        )
        lines = text.splitlines(keepends=True)
        old = "".join(lines[prepare.lineno - 1 : prepare.end_lineno])
        replacement = "    from image_data import prepare_mnist as prepare_dataset\n"
        text = replace_once(text, old, replacement)
        helper.write_text(text)
        from stage_image_loading import unit6_loading, prepared_only

        shutil.copy2(RUNTIME / "image_preload.py", stage / "image_preload.py")
        data = stage / "image_data.py"
        data.write_text(prepared_only(data.read_text()))
        source = unit6_loading(source, replace_once)
    if unit in (5, 7):
        from stage_image_loading import unit7_loading, prepared_only
        from stage_unit5_loading import unit5_loading

        shutil.copy2(RUNTIME / "image_preload.py", stage / "image_preload.py")
        data = stage / "image_data.py"
        data.write_text(prepared_only(data.read_text()))
        source = {5: unit5_loading, 7: unit7_loading}[unit](source, replace_once)
    return source
