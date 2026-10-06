"""Verify the packaging substitutions against full native data and source UI."""

from repository import original_path, ASSETS, BUILD, EVIDENCE
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.datasets import fetch_openml
from common import PROOF, ROOT, UNITS, sha, write_json
from build import interface
from reference_corrections import corrected_source


def main():
    checks = {}
    for unit in (1, 2, 3, 4, 5, 6, 7):
        original = corrected_source(unit)
        staged = (BUILD / f"unit{unit}/app.py").read_text()
        assert interface(original) == interface(staged)
        checks[f"unit{unit}/ui_ast"] = {"status": "pass", "basis": "shared corrected reference"}
    spec = importlib.util.spec_from_file_location("local_data", BUILD / "unit3/local_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("seeds", "ionosphere"):
        native = fetch_openml(name, version=1, as_frame=True)
        local = module.fetch_openml(name, version=1, as_frame=True)
        # JSON numbers preserve values; explicit feature dtype conversion avoids
        # pandas' integer-vs-float inference changing the assertion scope.
        pd.testing.assert_frame_equal(native.data.astype(float), local.data.astype(float))
        pd.testing.assert_series_equal(native.target, local.target)
        pd.testing.assert_frame_equal(
            native.frame.astype(object), local.frame.astype(object), check_dtype=False
        )
        checks[name] = {
            "status": "pass",
            "rows": len(local.data),
            "features": len(local.feature_names),
        }
    for unit in (1, 3):
        assert sha(ASSETS / "v1/penguins.csv") == sha(BUILD / f"unit{unit}/resources/penguins.csv")
    original = json.loads((EVIDENCE / "source-inventory.json").read_text())
    for unit in original.values():
        for name, record in unit["files"].items():
            assert sha(original_path(name)) == record["sha256"], f"Original changed: {name}"
    checks["originals_unchanged"] = "pass"
    write_json(EVIDENCE / "packaging-checks.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
