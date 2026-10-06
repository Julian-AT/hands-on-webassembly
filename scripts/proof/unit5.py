"""Generate original-source Assignment 5 reference fixtures (optionally full data)."""

from repository import EVIDENCE

from repository import ASSETS, BUILD, HARNESS, NATIVE, RUNTIME, SITE
import argparse
import asyncio
import importlib.util
import os
import shutil
import sys
import numpy as np
from common import ROOT, PROOF, sha, write_json
from training import nested
from provenance import stamp


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    target = SITE / "probes"
    target.mkdir(parents=True, exist_ok=True)
    definitions = "\n".join(
        nested(5, name)
        for name in (
            "_pr_make",
            "_pr_fit",
            "_lc_make",
            "_lc_train",
            "_lc_make_2d",
            "_lc_train_2d",
            "_get_loaders",
            "_mn_train",
            "_mn_eval",
        )
    )
    (target / "unit5-definitions.py").write_text(definitions)
    for name in (
        "unit5_probe.py",
        "browser_torch.py",
        "neural_compat.py",
        "cnn_compat.py",
        "torch_rng.py",
        "borch_compat.py",
        "image_data.py",
        "image_preload.py",
        "image_worker_preload.py",
        "module_hooks.py",
        "image_transforms.py",
    ):
        shutil.copy2(RUNTIME / name, target / name)
    shutil.copy2(BUILD / "unit5/u5_utils.py", target / "u5_utils.py")
    shutil.copy2(ROOT / "assignments/Material-20261003/DataSet_LR_a.csv", target / "unit5-data.csv")
    shutil.copy2(HARNESS / "unit5-worker.js", target / "unit5-worker.js")
    shutil.copy2(HARNESS / "index.html", target / "index.html")
    sys.path.insert(0, str(RUNTIME))
    from unit5_probe import run_unit5, compare
    import torch

    torch.set_num_threads(1)
    native = load("u5_native", ROOT / "assignments/5/u5_utils.py")
    os.chdir(NATIVE / "unit5")
    arrays = asyncio.run(
        run_unit5(torch, native, definitions, str(target / "unit5-data.csv"), args.full)
    )
    fixture = "unit5-full-native.npz" if args.full else "unit5-native.npz"
    np.savez_compressed(target / fixture, **arrays)
    from stage_image_workers import write_probe_config

    write_probe_config(target)
    import browser_torch

    candidate = load("u5_browser", BUILD / "unit5/u5_utils.py")
    # Bind the same complete static arrays for local adapter execution.
    import image_data

    image_data.asset_bytes = lambda name: (ASSETS / "v1/images" / name).read_bytes()
    actual = asyncio.run(
        run_unit5(
            browser_torch.torch, candidate, definitions, str(target / "unit5-data.csv"), args.full
        )
    )
    comparisons = compare(actual, np.load(target / fixture))
    suffix = "-full" if args.full else ""
    write_json(
        EVIDENCE / f"unit5{suffix}-native.json",
        {
            "status": "pass"
            if all(v["status"] == "pass" for v in comparisons.values())
            else "fail",
            "comparisons": comparisons,
            "full_data": args.full,
            "provenance": stamp(),
            "fixture_sha256": sha(target / fixture),
            "original_app_sha256": sha(ROOT / "assignments/5/app.py"),
            "original_helper_sha256": sha(ROOT / "assignments/5/u5_utils.py"),
            "method": "Original functions and training/evaluation bodies; per-batch unrounded losses, parameters, sample order, predictions and evaluation observations.",
        },
    )
    print(
        len(arrays),
        "arrays;",
        sum(v["status"] != "pass" for v in comparisons.values()),
        "failures",
        flush=True,
    )


if __name__ == "__main__":
    main()
