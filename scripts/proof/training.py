"""Reference and Borch execution of the original neural training functions."""

from repository import EVIDENCE, RUNTIME, SITE
import ast
import asyncio
import json
from pathlib import Path
import shutil
import sys
import textwrap
import numpy as np
from common import ROOT, PROOF, extract, write_json, sha


def nested(unit, name):
    source = (ROOT / f"assignments/{unit}/app.py").read_text()
    server = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "server"
    )
    node = next(
        n
        for n in server.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    return textwrap.dedent("\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])) + "\n"


def main():
    target = SITE / "probes"
    training = {
        "6": nested(6, "_train"),
        "7": extract(7, ["get_first_conv_weights"]) + "\n" + nested(7, "run_training_sync"),
    }
    write_json(target / "training-definitions.json", training)
    for name in (
        "training_probe.py",
        "browser_torch.py",
        "neural_compat.py",
        "cnn_compat.py",
        "module_hooks.py",
        "image_transforms.py",
        "image_data.py",
        "torch_rng.py",
        "borch_compat.py",
    ):
        shutil.copy2(RUNTIME / name, target / name)
    sys.path.insert(0, str(RUNTIME))
    from training_probe import run_training

    definitions = json.loads((target / "definitions.json").read_text())
    import torch

    torch.set_num_threads(1)
    arrays = asyncio.run(run_training(torch, definitions, training))
    np.savez_compressed(target / "training-native.npz", **arrays)
    import browser_torch

    candidate = asyncio.run(run_training(browser_torch.torch, definitions, training))
    from neural import compare

    comparison = compare(candidate, np.load(target / "training-native.npz"))
    write_json(
        EVIDENCE / "training-native.json",
        {
            "sources": {str(unit): sha(ROOT / f"assignments/{unit}/app.py") for unit in (6, 7)},
            "fixtures": len(arrays),
            "seeds": [0, 42],
            "compute_threads": 1,
            "epochs": 4,
            "method": "Verbatim original training function bodies, all presets; deterministic diagnostic inputs, not full image datasets.",
        },
    )
    write_json(EVIDENCE / "training-borch-native.json", comparison)
    print(
        "training arrays",
        len(arrays),
        "failures",
        sum(v["status"] != "pass" for v in comparison.values()),
    )


if __name__ == "__main__":
    main()
