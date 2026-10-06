"""Conserve disk using transparent macOS compression; retain all logical bytes."""

from repository import ROOT, EVIDENCE
import argparse
from pathlib import Path
import shutil
import subprocess
from common import PROOF, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = EVIDENCE / f"history-compression-{args.label}.json"
    if output.exists():
        raise FileExistsError("Retain previous compression evidence")
    result = dict(
        status="running",
        files=[],
        executor_sha256=sha(Path(__file__)),
        scope="Transparent APFS compression of completed historical JSON files only. Every path and logical byte is retained. Allocation is st_blocks times 512; extended-attribute and filesystem accounting can differ from actual volume reclamation.",
    )
    result["volume_free_before"] = shutil.disk_usage(PROOF).free
    for path in sorted((EVIDENCE / "history").rglob("*.json")):
        before = path.stat()
        if before.st_size < 1024**2 or before.st_blocks * 512 < before.st_size * 0.8:
            continue
        expected = sha(path)
        temporary = path.with_name(path.name + ".apfs-compressing")
        if temporary.exists():
            raise FileExistsError("Incomplete compression must be investigated")
        subprocess.run(["ditto", "--hfsCompression", str(path), str(temporary)], check=True)
        if sha(temporary) != expected:
            raise ValueError("Transparent compression changed historical bytes")
        allocated = temporary.stat().st_blocks * 512
        if allocated >= before.st_blocks * 512:
            temporary.unlink()
            continue
        temporary.replace(path)
        if sha(path) != expected:
            raise ValueError("Retained history differs after replacement")
        result["files"].append(
            dict(
                path=str(path.relative_to(ROOT)),
                sha256=expected,
                logical_bytes=before.st_size,
                allocated_bytes_before=before.st_blocks * 512,
                allocated_bytes_after=allocated,
            )
        )
    result.update(status="pass", volume_free_after=shutil.disk_usage(PROOF).free)
    write_json(output, result)
    print("preserved and compressed", len(result["files"]), "historical files", flush=True)


if __name__ == "__main__":
    main()
