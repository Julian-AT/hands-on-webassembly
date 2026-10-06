"""Verify and restore immutable history objects without extracting tar paths."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(index_path, directory):
    record = json.loads(index_path.read_text())
    entries = [record["inventory"], *record["shards"]]
    for entry in entries:
        name = entry["name"]
        if PurePosixPath(name).name != name or name in (".", ".."):
            raise ValueError("Unsafe archive filename")
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha(path) != entry["sha256"]
        ):
            raise ValueError(f"Missing or corrupt archive: {name}")
    with gzip.open(directory / record["inventory"]["name"], "rt") as stream:
        inventory = json.load(stream)
    if [{k: v for k, v in s.items() if k != "objects"} for s in inventory["shards"]] != record[
        "shards"
    ]:
        raise ValueError("Inventory shard mismatch")
    for name, entry in inventory["files"].items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or str(parts) != name or ".." in parts.parts or "\\" in name:
            raise ValueError("Unsafe historical path")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0 or len(entry["sha256"]) != 64:
            raise ValueError("Invalid historical file record")
    return record, inventory


def verify(index_path, directory, destination=None, prefixes=(), baseline=False):
    if destination is not None:
        if destination.is_symlink():
            raise ValueError("Restoration path contains a symlink")
        destination = destination.resolve()
    record, inventory = load(index_path, directory)
    files = inventory["files"]
    wanted = {
        name: entry
        for name, entry in files.items()
        if not prefixes or any(name.startswith(p) for p in prefixes)
    }
    if baseline:
        # Read the immutable manifest object first, then restore its complete binding closure.
        verify(index_path, directory, destination, ("proof/continuation-baseline/",))
        manifest = json.loads(
            (destination / "proof/continuation-baseline/manifest.corrected.json").read_text()
        )
        names = {entry["binding"] for entry in manifest["files"].values()}
        names |= {name for name in files if name.startswith("proof/continuation-baseline/")}
        names |= {
            "proof/scripts/verify_continuation_baseline.py",
            "proof/scripts/common.py",
            ".cache/restructure/old-history.bundle",
        }
        if names - files.keys():
            raise ValueError(f"Missing baseline dependencies: {sorted(names - files.keys())[:5]}")
        wanted = {name: files[name] for name in names}
    by_digest = {}
    for name, entry in wanted.items():
        by_digest.setdefault(entry["sha256"], []).append((name, entry))
    expected = {entry["sha256"]: entry["bytes"] for entry in files.values()}
    seen = set()
    for shard in inventory["shards"]:
        local = set()
        with tarfile.open(directory / shard["name"], "r|gz") as archive:
            for member in archive:
                digest = member.name.removeprefix("objects/")
                if (
                    member.name != "objects/" + digest
                    or not member.isfile()
                    or digest not in expected
                    or digest in seen
                    or member.size != expected[digest]
                ):
                    raise ValueError("Unexpected, duplicate or invalid history object")
                data = archive.extractfile(member)
                hasher = hashlib.sha256()
                targets = by_digest.get(digest, []) if destination else []
                output = None
                if targets:
                    target = destination / targets[0][0]
                    # Existing links are never followed during restoration.
                    if any(p.is_symlink() for p in [target, *target.parents]):
                        raise ValueError("Restoration path contains a symlink")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = target.open("xb") if not target.exists() else None
                    if output is None and sha(target) != digest:
                        raise ValueError(f"Conflicting restored file: {target}")
                try:
                    while chunk := data.read(1024 * 1024):
                        hasher.update(chunk)
                        if output:
                            output.write(chunk)
                finally:
                    if output:
                        output.close()
                if hasher.hexdigest() != digest:
                    raise ValueError("History object checksum mismatch")
                if targets:
                    target.chmod(targets[0][1].get("mode", 0o644))
                    for name, entry in targets[1:]:
                        other = destination / name
                        if any(p.is_symlink() for p in [other, *other.parents]):
                            raise ValueError("Restoration path contains a symlink")
                        other.parent.mkdir(parents=True, exist_ok=True)
                        if other.exists():
                            if sha(other) != digest:
                                raise ValueError("Conflicting restored file")
                        else:
                            shutil.copyfile(target, other)
                        other.chmod(entry.get("mode", 0o644))
                local.add(digest)
                seen.add(digest)
        if local != set(shard["objects"]):
            raise ValueError("Incomplete shard")
    if seen != set(expected):
        raise ValueError("Missing archive dependencies")
    if destination:
        for name, entry in wanted.items():
            if sha(destination / name) != entry["sha256"]:
                raise ValueError("Restored checksum mismatch")
    return {
        "status": "pass",
        "files": len(files),
        "objects": len(seen),
        "restored": len(wanted) if destination else 0,
        "old_main": record["old_main"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "verify", "restore"])
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "proof/archives/history.json",
    )
    parser.add_argument("--directory", type=Path, default=Path(".cache/history"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        record = json.loads(args.index.read_text())
        args.directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                record["tag"],
                "--repo",
                record["repository"],
                "--dir",
                str(args.directory),
                "--skip-existing",
            ],
            check=True,
        )
    if args.command == "restore" and args.destination is None:
        parser.error("restore requires --destination")
    result = verify(
        args.index,
        args.directory,
        args.destination if args.command == "restore" else None,
        tuple(args.prefix),
        args.baseline,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
