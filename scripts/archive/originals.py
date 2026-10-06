"""Restore larger course originals from the checksum-locked supplemental archive."""

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest = json.loads((ROOT / "proof/archives/large-originals.json").read_text())
    if all(
        (ROOT / "artifacts/originals" / name).is_file()
        and sha(ROOT / "artifacts/originals" / name) == entry["sha256"]
        for name, entry in manifest.items()
    ):
        return
    lock = json.loads((ROOT / "proof/archives/original-source-release.json").read_text())
    archive = ROOT / ".cache/assets" / lock["name"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        local = os.environ.get("COURSE_SOURCE_ARCHIVE")
        if local:
            shutil.copyfile(local, archive)
        else:
            token = os.environ.get("COURSE_ASSET_TOKEN")
            if not token:
                raise ValueError(
                    "Set COURSE_ASSET_TOKEN or COURSE_SOURCE_ARCHIVE to restore larger originals"
                )
            request = urllib.request.Request(
                lock["api_url"],
                headers={
                    "Accept": "application/octet-stream",
                    "Authorization": "Bearer " + token,
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urllib.request.urlopen(request) as response, archive.open("xb") as output:
                shutil.copyfileobj(response, output)
    if archive.stat().st_size != lock["bytes"] or sha(archive) != lock["sha256"]:
        raise ValueError("Original source archive checksum mismatch")
    seen = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for entry in bundle:
            if entry.name not in manifest:
                continue
            expected = manifest[entry.name]
            if not entry.isfile() or entry.name in seen or entry.size != expected["bytes"]:
                raise ValueError("Invalid original source member")
            target = ROOT / "artifacts/originals" / entry.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if any(p.is_symlink() for p in [target, *target.parents]):
                raise ValueError("Original restoration must not follow links")
            with bundle.extractfile(entry) as data, target.open("wb") as output:
                shutil.copyfileobj(data, output)
            if sha(target) != expected["sha256"]:
                raise ValueError("Original source member checksum mismatch")
            seen.add(entry.name)
    if seen != set(manifest):
        raise ValueError("Original source archive is incomplete")


if __name__ == "__main__":
    main()
