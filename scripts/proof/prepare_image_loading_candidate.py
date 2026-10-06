"""Export a detached neural loading candidate while preserving the live artifact."""

from repository import BUILD, EVIDENCE

from repository import CACHE, RUNTIME, SCRIPTS, SITE
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import PROOF, sha, write_json
from build import replace_once
from stage_image_loading import unit6_loading, unit7_loading, prepared_only, write_loading_config
from stage_unit5_loading import unit5_loading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--unit", type=int, choices=(5, 6, 7), default=7)
    args = parser.parse_args()
    target = CACHE / f"unit{args.unit}-loading-{args.label}"
    if target.exists():
        raise FileExistsError("Retain previous candidates; choose a fresh label")
    target.mkdir()
    stage = target / "stage"
    shutil.copytree(BUILD / f"unit{args.unit}", stage, copy_function=os.link)
    app = stage / "app.py"
    transform = {5: unit5_loading, 6: unit6_loading, 7: unit7_loading}[args.unit]
    source = transform(app.read_text(), replace_once)
    app.unlink()
    app.write_text(source)
    for name in ("image_data.py", "image_preload.py"):
        destination = stage / name
        destination.unlink(missing_ok=True)
        shutil.copy2(RUNTIME / name, destination)
    # The candidate constructors fail recoverably on a missing prepared array;
    # there is no synchronous browser transport left in this candidate.
    data = stage / "image_data.py"
    data.write_text(prepared_only(data.read_text()))
    write_loading_config(stage, source)
    exported = target / "export"
    subprocess.run(
        [
            str(Path(sys.executable).parent / "shinylive"),
            "export",
            str(stage),
            str(exported),
            "--subdir",
            f"unit{args.unit}",
        ],
        check=True,
    )
    site = target / "site"
    shutil.copytree(SITE, site, copy_function=os.link)
    bundle = site / f"unit{args.unit}/app.json"
    bundle.unlink()
    entries = json.loads((exported / f"unit{args.unit}/app.json").read_text())
    entries.sort(key=lambda entry: (entry["name"] != "app.py", entry["name"]))
    bundle.write_text(json.dumps(entries, sort_keys=True, separators=(",", ":")))
    write_json(
        EVIDENCE / f"unit{args.unit}-loading-{args.label}-inputs.json",
        dict(
            status="candidate",
            recipe_sha256=sha(Path(__file__)),
            transform_sha256=sha(SCRIPTS / "stage_image_loading.py"),
            original_app_sha256=sha(BUILD / f"unit{args.unit}/app.py"),
            candidate_app_sha256=sha(app),
            bundle_sha256=sha(bundle),
            config=json.loads((stage / "course-loading.json").read_text()),
            scope=f"Detached Unit {args.unit} app export. Shared deployment is unchanged; original data calculation body and compact synchronous constructors preserved. Full calculation ownership remains a separate obligation.",
        ),
    )
    print(site)


if __name__ == "__main__":
    main()
