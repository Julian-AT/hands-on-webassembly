"""Inventory supplied files and run imports in isolated native processes."""

from repository import EVIDENCE, supplied_files
import ast
import json
import os
import platform
import subprocess
import sys
from common import ROOT, PROOF, UNITS, sha, write_json


def main():
    inventory, imports = {}, {}
    for unit, folder in UNITS.items():
        path = ROOT / folder / "app.py"
        source = path.read_text()
        tree = ast.parse(source)
        ui, dependencies = [], set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                dependencies.update(
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names]
                )
            if isinstance(node, ast.Call):
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else ""
                )
                if name.startswith(("input_", "output_", "nav_")) or name in (
                    "download",
                    "download_button",
                    "download_link",
                ):
                    ui.append({"call": name, "line": node.lineno, "args": ast.unparse(node)})
        inventory[str(unit)] = {
            "folder": folder,
            "files": {
                name: {"sha256": sha(p), "bytes": p.stat().st_size}
                for name, p in supplied_files(unit)
            },
            "imports": sorted(dependencies),
            "interface": sorted(ui, key=lambda x: x["line"]),
            "declarations": {
                target.id: ast.unparse(node.value)
                for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(target, ast.Name)
                and target.id in ("DATASETS", "PRESETS", "ACTIVATIONS", "DEFAULT_DS")
            },
        }
        result = subprocess.run(
            [sys.executable, "-c", "import app; print(type(app.app).__name__)"],
            cwd=ROOT / folder,
            capture_output=True,
            text=True,
            timeout=120,
        )
        imports[str(unit)] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    hashes = {
        name: info["sha256"] for unit in inventory.values() for name, info in unit["files"].items()
    }
    lock = PROOF / "source.lock.json"
    if lock.exists():
        previous = json.loads(lock.read_text())
        if hashes != previous:
            raise ValueError(
                "Supplied files differ from source.lock.json; review the source contract before updating the lock"
            )
    else:
        write_json(lock, hashes)
    write_json(EVIDENCE / "source-inventory.json", inventory)
    write_json(EVIDENCE / "native-imports.json", imports)
    write_json(
        EVIDENCE / "machine.json",
        {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "logical_cpus": os.cpu_count(),
            "native_environment": "frozen local environment selected as numerical reference; see reference-environment.json",
            "hardware": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
                capture_output=True,
                text=True,
            ).stdout,
        },
    )


if __name__ == "__main__":
    main()
