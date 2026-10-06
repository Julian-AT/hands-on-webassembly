"""Retain each generated reference revision before regeneration changes it."""

from repository import EVIDENCE

from repository import ROOT, CACHE, PATCHES, REFERENCE
import hashlib
import json
import os
from pathlib import Path
import shutil
from common import PROOF, sha, write_json


def preserve():
    directories = [REFERENCE, PATCHES]
    files = {}
    links = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            name = str(path.relative_to(ROOT))
            if path.is_symlink():
                links[name] = os.readlink(path)
            elif path.is_file():
                files[name] = sha(path)
    if not files:
        return None
    inputs = dict(files=files, symlinks=links)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    retained = CACHE / f"native-reference-revision-{identity}"
    if not retained.exists():
        temporary = retained.with_name(retained.name + ".preserving")
        if temporary.exists():
            raise FileExistsError("An incomplete reference preservation needs investigation")
        temporary.mkdir()
        for directory in directories:
            if directory.exists():
                shutil.copytree(
                    directory, temporary / directory.name, symlinks=True, copy_function=shutil.copy2
                )
        temporary.rename(retained)
    observed_files = {}
    observed_links = {}
    for path in sorted(retained.rglob("*")):
        name = str(path.relative_to(retained))
        if path.is_symlink():
            observed_links[name] = os.readlink(path)
        elif path.is_file():
            observed_files[name] = sha(path)
    if observed_files != files or observed_links != links:
        raise ValueError("Retained native revision differs from the current reference")
    report = dict(
        status="pass",
        revision=identity,
        retained_directory=str(retained.relative_to(ROOT)),
        executor="scripts/proof/preserve_reference_revision.py",
        executor_sha256=sha(Path(__file__)),
        inputs=inputs,
        scope="Exact generated-reference and patch bytes and symlink metadata retained before regeneration. Unchanged dataset directories remain shared through their existing links.",
    )
    write_json(EVIDENCE / f"native-reference-revision-{identity}.json", report)
    return report


if __name__ == "__main__":
    report = preserve()
    print("preserved", report["revision"] if report else "no generated references")
