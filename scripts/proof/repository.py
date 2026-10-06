"""Canonical repository paths. No path depends on the current working directory."""

from pathlib import Path
import json
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proof"
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / ".cache/science"
SCRIPTS = ROOT / "scripts/proof"
TESTS = ROOT / "tests/python"
RUNTIME = ROOT / "runtime"
PACKAGES = ROOT / "wasm/packages"
PATCHES = ROOT / "wasm/reference-patches"
HARNESS = PROOF / "harness/browser"
REQUIREMENTS = ROOT / "requirements"
EVIDENCE = ARTIFACTS / "proof/evidence"
ASSETS = ARTIFACTS / "assets"
SITE = ARTIFACTS / "site"
BUILD = ARTIFACTS / "build"
NATIVE = ARTIFACTS / "native"
REFERENCE = ARTIFACTS / "reference"
FIXTURES = ARTIFACTS / "fixtures"
BASELINE = ROOT / ".cache/history-restored/proof/continuation-baseline"


def original_path(name):
    """Large original resources are restored in artifacts under their original key."""
    path = ROOT / name
    return path if path.is_file() else ARTIFACTS / "originals" / name


def runtime_asset(name):
    """Resolve scientific inputs or their exact immutable release counterparts."""
    scientific = ASSETS / "v1" / name
    if scientific.is_file():
        return scientific
    records = json.loads((PROOF / "locks/runtime-inputs.json").read_text())
    if name in records:
        record = records[name]
        path = ROOT / record["path"]
        expected = record["sha256"]
    else:
        released = {
            "pyborch-1.14.1-py3-none-any.whl": ASSETS
            / "v1/neural-runtime/pyborch-1.14.1-py3-none-any.whl",
            "pyodide-0.27.7.js": ROOT / "web/public/shinylive/pyodide/pyodide.js",
        }
        path = released[name]
        expected = json.loads((PROOF / "assets.lock.json").read_text())[name]["sha256"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Immutable runtime input changed: {name}")
    return path


def copy_original_resources(unit, destination):
    """Rehydrate checksum-bound resources into either scientific staging tree."""
    prefix = f"assignments/{unit}/"
    for key in json.loads((PROOF / "archives/large-originals.json").read_text()):
        if key.startswith(prefix):
            target = destination / key.removeprefix(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_path(key), target)


def supplied_files(unit):
    """Yield logical original names with their current physical byte locations."""
    prefix = f"assignments/{unit}/"
    files = {
        str(path.relative_to(ROOT)): path
        for path in (ROOT / f"assignments/{unit}").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith(".")
    }
    for key in json.loads((PROOF / "archives/large-originals.json").read_text()):
        if key.startswith(prefix):
            files[key] = original_path(key)
    return sorted(files.items())


def evidence_locations(proof):
    """Resolve the current layout, while accepting standalone historical fixtures."""
    proof = proof.resolve()
    if proof in (PROOF.resolve(), EVIDENCE.parent.resolve()):
        return ROOT.resolve(), EVIDENCE.resolve()
    return proof.parent, proof / "evidence"
