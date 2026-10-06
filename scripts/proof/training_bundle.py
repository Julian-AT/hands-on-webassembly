"""Identity of the browser compute bundle, including compiled arithmetic."""

from repository import runtime_asset, RUNTIME
import hashlib
import json
from pathlib import Path
import cloudpickle
from common import PROOF, sha

MODULES = (
    "isolated_training.py",
    "browser_torch.py",
    "neural_compat.py",
    "cnn_compat.py",
    "torch_rng.py",
    "borch_compat.py",
    "image_data.py",
    "module_hooks.py",
    "image_transforms.py",
)


def browser_build_id(source, *, module_paths=None, worker_path=None):
    paths = {f"runtime/{name}": RUNTIME / name for name in MODULES}
    for name, path in (module_paths or {}).items():
        if name not in MODULES:
            raise ValueError(f"Unknown training dependency: {name}")
        paths[f"runtime/{name}"] = Path(path)
    paths["runtime/course-training-worker.js"] = (
        Path(worker_path) if worker_path else RUNTIME / "course-training-worker.js"
    )
    for path in sorted(Path(cloudpickle.__file__).parent.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            paths["cloudpickle/" + str(path.relative_to(Path(cloudpickle.__file__).parent))] = path
    for name in (
        "pyborch-1.14.1-py3-none-any.whl",
        "runtime-packages/manifest.json",
        "runtime-packages/neural-manifest.json",
        "runtime-packages/cnn-manifest.json",
        "pyodide-0.27.7.js",
    ):
        paths["assets/" + name] = runtime_asset(name)
    record = {
        "source": hashlib.sha256(source.encode()).hexdigest(),
        "dependencies": {name: sha(path) for name, path in sorted(paths.items())},
        "protocol": 2,
        "python": "3.12",
        "pyodide": "0.27.7",
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
