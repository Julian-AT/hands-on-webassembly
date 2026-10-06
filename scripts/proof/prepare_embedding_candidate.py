"""Stage abortable Unit 2 preparation without modifying the accepting artifact."""

from repository import EVIDENCE

from repository import ROOT, BUILD, CACHE, RUNTIME, SITE
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import PROOF, sha, write_json
from storage_budget import require_space
from detached_artifact import detach_shared_artifact


from stage_embedding_loading import adapter, application, write_embedding_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = CACHE / f"embedding-preparation-{args.label}"
    if root.exists():
        raise FileExistsError("Choose a fresh label; preserve failed candidates")
    require_space(512 * 1024**2)
    site, stage = root / "site", root / "unit2"
    shutil.copytree(SITE, site, copy_function=os.link)
    # Export writes runtime files too; detach every mutable shared inode first.
    detach_shared_artifact(site)
    shutil.copytree(BUILD / "unit2", stage)
    source = application((stage / "app.py").read_text())
    (stage / "app.py").write_text(source)
    (stage / "embedding_runtime.py").write_text(
        adapter((stage / "embedding_runtime.py").read_text())
    )
    shutil.copy2(RUNTIME / "embedding_preload.py", stage / "embedding_preload.py")
    inputs, build_id, source_id = write_embedding_config(stage, source)
    shutil.rmtree(site / "unit2")
    subprocess.run(
        [
            str(Path(sys.executable).parent / "shinylive"),
            "export",
            str(stage),
            str(site),
            "--subdir",
            "unit2",
        ],
        check=True,
    )
    # Preserve the verified outer bridge/bootstrap. Only this child's application
    # source identity changes; the production site remains byte-for-byte intact.
    from patch_runtime import session_startup_script

    original = (SITE / "unit2/index.html").read_text()
    old_source = sha(SITE / "unit2/app.json")
    (site / "unit2/index.html").write_text(
        original.replace(old_source, sha(site / "unit2/app.json"))
    )
    shutil.copy2(
        RUNTIME / "course-embedding-worker.js", site / "shinylive/course-embedding-worker.js"
    )
    write_json(
        EVIDENCE / f"embedding-preparation-candidate-{args.label}.json",
        dict(
            status="candidate",
            root=str(root.relative_to(ROOT)),
            inputs=inputs,
            build_id=build_id,
            source_id=source_id,
            scope="Isolated abortable Unit 2 preparation candidate; no promotion or acceptance claim.",
        ),
    )
    print(site, flush=True)


if __name__ == "__main__":
    main()
