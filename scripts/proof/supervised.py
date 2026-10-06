"""Generate Unit 4 numerical fixtures from original computation source paths."""

from repository import ASSETS, EVIDENCE, REQUIREMENTS, RUNTIME, SITE
import ast
import shutil
import sys
import textwrap
import numpy as np
from common import ROOT, PROOF, extract, sha, write_json


def definitions():
    source = (ROOT / "assignments/4/app.py").read_text()
    tree = ast.parse(source)
    server = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "server")
    names = {
        "df_loaded",
        "compute_splits",
        "train_model",
        "_gen_noisy_sine",
        "_gen_mystery_function",
        "_gen_data",
    }
    functions = []
    for node in server.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            functions.append(
                textwrap.dedent("\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno]))
            )
    if len(functions) != len(names):
        raise ValueError("Unit 4 source contract changed")
    return (
        extract(
            4,
            [
                "load_wine",
                "load_breast_cancer",
                "load_digits",
                "load_pima",
                "load_iris",
                "load_banknote",
                "DATASETS",
            ],
        )
        + "\n"
        + "\n\n".join(functions)
        + "\n"
    )


def main():
    import os
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(RUNTIME))
    from supervised_probe import run_supervised

    original = definitions()
    target = SITE / "probes"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNTIME / "supervised_probe.py", target / "supervised_probe.py")
    shutil.copy2(RUNTIME / "forest_runtime.py", target / "forest_runtime.py")
    from build import replace_once

    browser = original
    for url, name in {
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv": "pima.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt": "banknote.csv",
    }.items():
        browser = replace_once(browser, f'url = "{url}"', f'url = "{name}"')
        shutil.copy2(ASSETS / "v1" / name, target / name)
    # Native original loaders download their original URLs. Cache whole frames to
    # avoid 24 identical requests, without changing returned values or ordering.
    native_read = __import__("pandas").read_csv
    cache = {}

    def read_csv(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("https:"):
            key = (path, repr(args), repr(kwargs))
            if key not in cache:
                cache[key] = native_read(path, *args, **kwargs)
            return cache[key].copy(deep=True)
        return native_read(path, *args, **kwargs)

    __import__("pandas").read_csv = read_csv
    try:
        with threadpool_limits(limits=1):
            arrays = run_supervised(original)
    finally:
        __import__("pandas").read_csv = native_read
    np.savez_compressed(target / "supervised-native.npz", **arrays)
    (target / "supervised-definitions.py").write_text(browser)
    write_json(
        EVIDENCE / "supervised-native.json",
        {
            "source_sha256": sha(ROOT / "assignments/4/app.py"),
            "reference_lock_sha256": sha(REQUIREMENTS / "native.lock"),
            "definitions_sha256": sha(target / "supervised-definitions.py"),
            "fixture_sha256": sha(target / "supervised-native.npz"),
            "method": "Verbatim original nested calculation bodies, decorators removed; test input/value containers replace reactive scheduling.",
            "seeds": [42, 17, 123],
            "cpu_threads": 1,
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
        },
    )
    print(f"Generated {len(arrays)} original-path reference arrays")


if __name__ == "__main__":
    main()
