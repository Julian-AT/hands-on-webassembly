"""Build the small numerical compatibility extension with a pinned toolchain."""

from repository import ASSETS, CACHE, PACKAGES
import json
import os
from pathlib import Path
import shutil
import subprocess
from common import ROOT, PROOF, sha, write_json


def main():
    source = PACKAGES / "native_forest"
    stage = CACHE / "forest-build"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(
        source,
        stage,
        ignore=shutil.ignore_patterns("build", "*.c", "*.so", "*.egg-info", "__pycache__"),
    )
    toolchain = CACHE / "emsdk"
    python_env = CACHE / "wasm-build-env"
    cross = ROOT / ".cache/pyodide-xbuildenv-0.29.3/xbuildenv/xbuildenv/pyodide-root"
    if not cross.exists() or not (toolchain / "emsdk_env.sh").exists():
        raise RuntimeError(
            "Install the pinned cross-build environment; see wasm/packages/native_forest/README.md"
        )
    env = dict(
        os.environ, SOURCE_DATE_EPOCH="1740787200", PYTHONHASHSEED="0", PYODIDE_ROOT=str(cross)
    )
    env["PATH"] = str(python_env / "bin") + os.pathsep + env["PATH"]
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1; exec "$2" build "$3" --outdir "$4"',
            "forest-build",
            str(toolchain / "emsdk_env.sh"),
            str(python_env / "bin/pyodide"),
            str(stage),
            str(CACHE / "forest-wheels"),
        ],
        check=True,
        env=env,
        cwd=ROOT,
    )
    wheel = (
        CACHE / "forest-wheels/course_forest_criterion-1.0.0-cp312-cp312-pyodide_2024_0_wasm32.whl"
    )
    destination = ASSETS / "v1/runtime-packages" / wheel.name
    if destination.exists() and sha(destination) != sha(wheel):
        raise ValueError("Compiled wheel differs from its lock; inspect before replacing it")
    shutil.copy2(wheel, destination)
    files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    record = {
        "name": "course-forest-criterion",
        "version": "1.0.0",
        "file_name": wheel.name,
        "sha256": sha(wheel),
        "depends": ["numpy", "scikit-learn"],
        "imports": ["forest_criterion"],
        "install_dir": "site",
        "package_type": "package",
    }
    write_json(ASSETS / "v1/runtime-packages/forest-manifest.json", record)
    write_json(
        ASSETS / "v1/runtime-packages/forest-build.json",
        {
            "source_files": files,
            "scikit_learn": "1.6.1",
            "pyodide": "0.27.7",
            "pyodide_build": "0.29.3",
            "emscripten": "3.1.58",
            "emsdk_commit": subprocess.check_output(
                ["git", "-C", str(toolchain), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_date_epoch": 1740787200,
            "pythonhashseed": 0,
            "flags": ["-ffp-contract=off"],
            "wheel": record,
        },
    )
    print("Built and locked", wheel.name, sha(wheel))


if __name__ == "__main__":
    main()
