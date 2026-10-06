"""Produce native numerical fixtures and the browser's exact-source probes."""

from repository import EVIDENCE, HARNESS, RUNTIME, SITE
import json
import shutil
import urllib.request
import sys
import numpy as np
from common import PROOF, extract, write_json


def main():
    target = SITE / "probes"
    target.mkdir(parents=True, exist_ok=True)
    names = ["parse_architecture", "build_model", "ACTIVATIONS", "PRESETS"]
    definitions = {
        "6": extract(6, names),
        "7": extract(7, names + ["_yamlish_to_json", "LayerSpec", "_tuple_or_int"]),
    }
    write_json(target / "definitions.json", definitions)
    shutil.copy2(RUNTIME / "neural_probe.py", target / "neural_probe.py")
    shutil.copy2(RUNTIME / "borch_compat.py", target / "borch_compat.py")
    sys.path.insert(0, str(RUNTIME))
    from neural_probe import run_probe
    import torch

    torch.set_num_threads(1)
    checks, arrays = run_probe(torch, definitions)
    write_json(EVIDENCE / "neural-native.json", {"torch": torch.__version__, "checks": checks})
    np.savez_compressed(target / "neural-native.npz", **arrays)
    import borch
    from torch_rng import install

    install(borch)
    checks, arrays = run_probe(borch, definitions)
    reference = np.load(target / "neural-native.npz")
    comparison = compare(arrays, reference)
    write_json(EVIDENCE / "neural-borch-native.json", {"checks": checks, "comparisons": comparison})
    # Resolve a pinned pure Python wheel at build time; browsers use only this local file.
    from prepare import download

    runtime = download(
        "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js", "pyodide-0.27.7.js"
    )
    shutil.copy2(runtime, SITE / "shinylive/pyodide/pyodide.js")
    filename = "pyborch-1.14.1-py3-none-any.whl"
    locked = json.loads((PROOF / "assets.lock.json").read_text()).get(filename)
    if locked:
        path = download(locked["url"], filename)
        expected = locked["sha256"]
    else:
        metadata = json.load(
            urllib.request.urlopen("https://pypi.org/pypi/pyborch/1.14.1/json", timeout=120)
        )
        wheel = next(f for f in metadata["urls"] if f["filename"] == filename)
        path = download(wheel["url"], filename)
        expected = wheel["digests"]["sha256"]
    if __import__("hashlib").sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Wheel checksum does not match PyPI")
    shutil.copy2(path, target / path.name)
    for path in (HARNESS).iterdir():
        shutil.copy2(path, target / path.name)
    shutil.copy2(RUNTIME / "torch_rng.py", target / "torch_rng.py")
    shutil.copy2(RUNTIME / "image_transforms.py", target / "image_transforms.py")


def compare(arrays, reference):
    result = {}
    for key in reference.files:
        if key not in arrays:
            result[key] = {"status": "missing"}
            continue
        a, b = arrays[key], reference[key]
        match = a.shape == b.shape and (
            np.array_equal(a, b)
            if b.dtype.kind in "biu"
            else np.allclose(a, b, rtol=1e-4, atol=1e-5)
        )
        result[key] = {
            "status": "pass" if match else "fail",
            "max_abs_error": float(np.max(np.abs(a - b))) if a.shape == b.shape else None,
        }
    return result


if __name__ == "__main__":
    main()
