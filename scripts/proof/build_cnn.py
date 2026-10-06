"""Build the pinned CNN arithmetic package without altering the native env."""

from repository import ASSETS, CACHE, PACKAGES, SCRIPTS
import json
import os
import shutil
import subprocess
from common import ROOT, PROOF, sha, write_json


def main():
    source = PACKAGES / "native_cnn"
    upstream = json.loads((source / "upstream.json").read_text())
    for name, digest in upstream["vendor_files"].items():
        if sha(source / name) != digest:
            raise ValueError("Pinned CNN upstream source changed: " + name)
    stage = CACHE / "native-cnn-build"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(
        source, stage, ignore=shutil.ignore_patterns("build", "*.so", "*.egg-info", "__pycache__")
    )
    tools = CACHE / "wasm-build-env"
    env = dict(
        os.environ,
        SOURCE_DATE_EPOCH="1740787200",
        PYTHONHASHSEED="0",
        PYODIDE_ROOT=str(ROOT / ".cache/pyodide-xbuildenv-0.29.3/xbuildenv/xbuildenv/pyodide-root"),
    )
    env["PATH"] = str(tools / "bin") + os.pathsep + env["PATH"]

    def emcc(arguments):
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1" >/dev/null 2>&1; shift; exec emcc "$@"',
                "cnn-kernel-build",
                str(CACHE / "emsdk/emsdk_env.sh"),
                *arguments,
            ],
            cwd=stage,
            env=env,
            check=True,
        )

    emcc(
        [
            "-O0",
            "-fPIC",
            "-ffp-contract=on",
            "-Ivendor",
            "-Ivendor/include",
            "-Ivendor/src",
            "-S",
            "-emit-llvm",
            "vendor/src/psimd/2d-fourier-16x16.c",
            "-o",
            "fft-original.ll",
        ]
    )
    ir = (stage / "fft-original.ll").read_text()
    if ir.count("llvm.fmuladd") != 44:
        raise ValueError("Pinned FFT contraction sites changed")
    (stage / "fft.ll").write_text(ir.replace("llvm.fmuladd", "llvm.fma").replace(" optnone", ""))
    emcc(["-O3", "-fPIC", "-ffp-contract=off", "-c", "fft.ll", "-o", "fft.o"])
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1; exec "$2" build "$3" --outdir "$4"',
            "native-cnn-build",
            str(CACHE / "emsdk/emsdk_env.sh"),
            str(tools / "bin/pyodide"),
            str(stage),
            str(CACHE / "native-cnn-wheels"),
        ],
        check=True,
        env=env,
        cwd=ROOT,
    )
    wheel = (
        CACHE / "native-cnn-wheels/course_native_cnn-1.0.0-cp312-cp312-pyodide_2024_0_wasm32.whl"
    )
    destination = ASSETS / "v1/runtime-packages" / wheel.name
    if destination.exists() and sha(destination) != sha(wheel):
        raise ValueError("Compiled CNN wheel differs from its lock")
    shutil.copy2(wheel, destination)
    record = dict(
        name="course-native-cnn",
        version="1.0.0",
        file_name=wheel.name,
        sha256=sha(wheel),
        depends=["numpy"],
        imports=["native_cnn"],
        install_dir="site",
        package_type="package",
    )
    write_json(destination.parent / "cnn-manifest.json", record)
    write_json(
        destination.parent / "cnn-build.json",
        dict(
            wheel=record,
            upstream=upstream,
            source_files={
                str(path.relative_to(source)): sha(path)
                for path in sorted(source.rglob("*"))
                if path.is_file()
            },
            recipe_sha256=sha(SCRIPTS / "build_cnn.py"),
            emscripten="3.1.58",
            pyodide="0.27.7",
            python="3.12",
            abi="2024_0",
            source_date_epoch=1740787200,
            contraction_references=44,
            contraction_rule="Require fma at pinned expression-contraction sites; contract off elsewhere.",
        ),
    )
    print(wheel.name, sha(wheel), flush=True)


if __name__ == "__main__":
    main()
