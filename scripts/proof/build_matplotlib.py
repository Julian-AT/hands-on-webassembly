"""Build reference Matplotlib for the pinned browser ABI, without native changes."""

from repository import CACHE, PACKAGES
import json
import configparser
import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
from common import ROOT, PROOF, sha, write_json
from prepare import download


def main():
    url = "https://pypi.org/pypi/matplotlib/3.10.8/json"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    source = next(x for x in metadata["urls"] if x["packagetype"] == "sdist")
    archive = download(source["url"], "runtime-sources/" + source["filename"])
    if sha(archive) != "2299372c19d56bcd35cf05a2738308758d32b9eaed2371898d8f5bd33f084aa3":
        raise ValueError("Matplotlib source checksum mismatch")
    destination = CACHE / "matplotlib-source"
    stage = destination / "matplotlib-3.10.8"
    if stage.exists():
        shutil.rmtree(stage)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    # The timer only emits a font-cache warning; browser workers have no threads.
    font_manager = stage / "lib/matplotlib/font_manager.py"
    original = font_manager.read_text()
    before = "        timer.start()"
    after = '        if sys.platform != "emscripten":\n            timer.start()'
    if original.count(before) != 1:
        raise ValueError("Matplotlib font-warning timer source changed")
    font_manager.write_text(original.replace(before, after))
    # Keep the reference's FreeType 2.6.1, correcting callback signatures as in
    # emscripten-ports/FreeType commit 40a760c963bc2b76575918ff91a515c8474db4e0.
    # Native CPUs tolerate these casts; WebAssembly checks indirect-call types.
    wrap = configparser.ConfigParser()
    wrap.read(stage / "subprojects/freetype-2.6.1.wrap")
    spec = wrap["wrap-file"]
    freetype_archive = CACHE / "runtime-sources" / spec["source_filename"]
    freetype_archive.parent.mkdir(parents=True, exist_ok=True)
    if not freetype_archive.exists():
        for url in (spec["source_url"], spec["source_fallback_url"]):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != spec["source_hash"]:
                    raise ValueError("FreeType source checksum mismatch")
                freetype_archive.write_bytes(data)
                break
            except OSError:
                if url == spec["source_fallback_url"]:
                    raise
    if sha(freetype_archive) != spec["source_hash"]:
        raise ValueError("FreeType source checksum mismatch")
    with tarfile.open(freetype_archive) as bundle:
        bundle.extractall(stage / "subprojects", filter="data")
    freetype = stage / "subprojects/freetype-2.6.1"
    shutil.copytree(
        stage / "subprojects/packagefiles" / spec["patch_directory"], freetype, dirs_exist_ok=True
    )
    callback_patch = PACKAGES / "matplotlib/freetype-wasm-callbacks.patch"
    subprocess.run(
        ["patch", "-p1", "--fuzz=0", "-i", str(callback_patch)], cwd=freetype, check=True
    )
    env = dict(
        os.environ,
        SOURCE_DATE_EPOCH="1740787200",
        PYTHONHASHSEED="0",
        PYODIDE_ROOT=str(ROOT / ".cache/pyodide-xbuildenv-0.29.3/xbuildenv/xbuildenv/pyodide-root"),
    )
    tools = CACHE / "wasm-build-env"
    env["PIP_CONSTRAINT"] = str(PACKAGES / "matplotlib/build-constraints.txt")
    env["PATH"] = str(tools / "bin") + os.pathsep + env["PATH"]
    flags = [
        "-Csetup-args=-Dcpp_args=['-Wno-c++11-narrowing','-fexceptions']",
        "-Csetup-args=-Dcpp_link_args=['-fexceptions']",
        "-Csetup-args=-Db_lto=false",
    ]
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1; shift; exec "$@"',
            "matplotlib-build",
            str(CACHE / "emsdk/emsdk_env.sh"),
            str(tools / "bin/pyodide"),
            "build",
            str(stage),
            "--outdir",
            str(CACHE / "matplotlib-wheels"),
            *flags,
        ],
        check=True,
        env=env,
        cwd=ROOT,
    )
    wheel = CACHE / "matplotlib-wheels/matplotlib-3.10.8-cp312-cp312-pyodide_2024_0_wasm32.whl"
    entry = {
        "name": "matplotlib",
        "version": "3.10.8",
        "file_name": wheel.name,
        "sha256": sha(wheel),
        "depends": [
            "numpy",
            "contourpy",
            "cycler",
            "fonttools",
            "kiwisolver",
            "packaging",
            "pillow",
            "pyparsing",
            "python-dateutil",
        ],
        "imports": ["matplotlib"],
        "install_dir": "site",
        "package_type": "package",
    }
    # A successful compilation is not rendering evidence. Keep candidates out
    # of deployed packages until the original application paths pass.
    write_json(CACHE / "matplotlib-wheels/matplotlib-manifest.json", entry)
    write_json(
        CACHE / "matplotlib-wheels/matplotlib-build.json",
        {
            "source_url": source["url"],
            "source_sha256": sha(archive),
            "pyodide": "0.27.7",
            "emscripten": "3.1.58",
            "pyodide_build": "0.29.3",
            "source_date_epoch": 1740787200,
            "flags": flags,
            "patch": {
                "file": "lib/matplotlib/font_manager.py",
                "before": before,
                "after": after,
                "patched_sha256": sha(font_manager),
            },
            "freetype": {
                "version": "2.6.1",
                "source_url": spec["source_url"],
                "source_sha256": sha(freetype_archive),
                "callback_patch_sha256": sha(callback_patch),
                "upstream_fix": "https://github.com/emscripten-ports/FreeType/commit/40a760c963bc2b76575918ff91a515c8474db4e0",
            },
            "build_constraints_sha256": sha(PACKAGES / "matplotlib/build-constraints.txt"),
            "vendored_library_configs": {
                p.name: sha(p) for p in (stage / "subprojects").glob("*.wrap")
            },
            "wheel": entry,
        },
    )
    print("Built candidate Matplotlib (not deployed)", sha(wheel))


if __name__ == "__main__":
    main()
