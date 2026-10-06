"""Install checksum-locked pure Python wheels into the static Pyodide registry."""

from repository import ASSETS, EVIDENCE, SITE
import json
import shutil
from common import PROOF, sha, write_json


def pin_runtime_packages():
    source = ASSETS / "v1/runtime-packages"
    entries = json.loads((source / "manifest.json").read_text())
    forest = json.loads((source / "forest-manifest.json").read_text())
    entries[forest["name"]] = forest
    matplotlib = json.loads((source / "matplotlib-manifest.json").read_text())
    entries[matplotlib["name"]] = matplotlib
    neural = json.loads((source / "neural-manifest.json").read_text())
    entries[neural["name"]] = neural
    cnn = json.loads((source / "cnn-manifest.json").read_text())
    entries[cnn["name"]] = cnn
    tsne = json.loads((source / "tsne-manifest.json").read_text())
    entries[tsne["name"]] = tsne
    destination = SITE / "shinylive/pyodide"
    lock = destination / "pyodide-lock.json"
    registry = json.loads(lock.read_text())
    if registry["packages"]["plotly"]["version"] != "5.23.0":
        raise ValueError("Upstream Plotly registry contract changed")
    if registry["packages"]["numpy"]["version"] != "2.0.2":
        raise ValueError("Upstream NumPy registry contract changed")
    if registry["packages"]["matplotlib"]["version"] != "3.8.4":
        raise ValueError("Upstream Matplotlib registry contract changed")
    if registry["packages"]["scikit-learn"]["version"] != "1.6.1":
        raise ValueError("Pinned t-SNE estimator contract changed")
    for name, entry in entries.items():
        wheel = source / entry["file_name"]
        if sha(wheel) != entry["sha256"]:
            raise ValueError("Runtime wheel changed: " + name)
        shutil.copy2(wheel, destination / wheel.name)
        registry["packages"][name] = entry
    write_json(lock, registry)
    write_json(EVIDENCE / "runtime-package-pins.json", entries)
