"""Preserve and checksum the supplied Assignment 5 archive and materials."""

import json
import zipfile
from common import ROOT, PROOF, sha, write_json
from repository import original_path


def main():
    materials = ROOT / "assignments/Material-20261003"
    destination = ROOT / "assignments/5"
    archive = materials / "app.zip"
    lock = PROOF / "assignment5-materials.lock.json"
    names = (
        json.loads(lock.read_text())["materials"]
        if lock.exists()
        else {
            str(p.relative_to(ROOT)): None
            for p in materials.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        }
    )
    supplied = {name: sha(original_path(name)) for name in sorted(names)}
    if lock.exists():
        previous = json.loads(lock.read_text())
        if supplied != previous["materials"]:
            raise ValueError("Assignment 5 supplied materials changed")
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != {"app.py", "u5_utils.py"}:
            raise ValueError("Unexpected Assignment 5 archive members")
        for name in bundle.namelist():
            data = bundle.read(name)
            path = destination / name
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Extracted original changed: {path}")
            if not path.exists():
                path.write_bytes(data)
    sources = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (destination / "app.py", destination / "u5_utils.py")
    }
    record = {"archive": str(archive.relative_to(ROOT)), "materials": supplied, "sources": sources}
    if lock.exists() and record != previous:
        raise ValueError("Assignment 5 source lock changed")
    write_json(lock, record)
    print(f"Preserved {len(supplied)} material files and {len(sources)} extracted originals")


if __name__ == "__main__":
    main()
