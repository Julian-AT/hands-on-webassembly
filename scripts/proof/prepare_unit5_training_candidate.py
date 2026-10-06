"""Export Unit 5 isolation without mutating the integrated artifact."""

from repository import EVIDENCE

from repository import BUILD, CACHE, RUNTIME, SITE
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import cloudpickle
from common import PROOF, sha, write_json
from build import replace_once, interface
from reference_corrections import corrected_source, training_source
from stage_unit5_loading import unit5_loading
from stage_image_loading import write_loading_config
from training_bundle import browser_build_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    target = CACHE / f"unit5-training-{args.label}"
    if target.exists():
        raise FileExistsError("Choose a fresh candidate label")
    stage = target / "stage"
    shutil.copytree(BUILD / "unit5", stage)
    original = corrected_source(5)
    source = replace_once(original, "import torch\n", "from browser_torch import torch\n")
    source = replace_once(source, "import torch.nn as nn", "from browser_torch import nn")
    source = source.replace(
        "from shiny import App,", "from pathlib import Path\nfrom shiny import App,"
    )
    source = unit5_loading(source, replace_once)
    source = re.sub(
        r'"(?:\./)?resources/([^"\n]+)"', r'str(Path(__file__).parent / "resources/\1")', source
    )
    source = replace_once(
        source,
        "app = App(app_ui, server)",
        'app = App(app_ui, server, static_assets=Path(__file__).parent / "www")',
    )
    if interface(source) != interface(original):
        raise ValueError("Unit 5 UI changed")
    for name in ("isolated_training.py", "inspection_rng.py"):
        shutil.copy2(RUNTIME / name, stage / name)
    shutil.copytree(
        Path(cloudpickle.__file__).parent,
        stage / "cloudpickle",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    body = training_source(5)
    build_id = browser_build_id(body)
    (stage / "course-training.json").write_text(
        json.dumps(dict(source=body, build_id=build_id), sort_keys=True)
    )
    write_loading_config(stage, source)
    (stage / "app.py").write_text(source)
    exported = target / "export"
    subprocess.run(
        [
            str(Path(sys.executable).parent / "shinylive"),
            "export",
            str(stage),
            str(exported),
            "--subdir",
            "unit5",
        ],
        check=True,
    )
    site = target / "site"
    shutil.copytree(SITE, site, copy_function=os.link)
    bundle = site / "unit5/app.json"
    bundle.unlink()
    entries = json.loads((exported / "unit5/app.json").read_text())
    entries.sort(key=lambda entry: (entry["name"] != "app.py", entry["name"]))
    bundle.write_text(json.dumps(entries, sort_keys=True, separators=(",", ":")))
    worker = site / "shinylive/course-training-worker.js"
    worker.unlink()
    shutil.copy2(RUNTIME / "course-training-worker.js", worker)
    neural = site / "assets/v1/neural-runtime"
    shutil.rmtree(neural)
    shutil.copytree(SITE / "assets/v1/neural-runtime", neural)
    shutil.copy2(RUNTIME / "isolated_training.py", neural / "isolated_training.py")
    manifest = json.loads((neural / "manifest.json").read_text())
    manifest["build_ids"]["5"] = build_id
    manifest["sources"]["5"] = hashlib.sha256(body.encode()).hexdigest()
    manifest["files"] = {
        str(p.relative_to(neural)): sha(p)
        for p in sorted(neural.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (neural / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    write_json(
        EVIDENCE / f"unit5-training-{args.label}-inputs.json",
        dict(
            status="candidate",
            recipe_sha256=sha(Path(__file__)),
            app_sha256=sha(stage / "app.py"),
            bundle_sha256=sha(bundle),
            build_id=build_id,
            scope="Unit 5 candidate only. Integrated deployment and retained artifacts remain unchanged.",
        ),
    )
    print(site)


if __name__ == "__main__":
    main()
