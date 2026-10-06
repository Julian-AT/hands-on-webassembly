"""Stage disposable preparation without changing dataset constructors or RNG."""

from repository import ASSETS, REQUIREMENTS, RUNTIME
import hashlib
import json
import shutil
from common import PROOF, sha

WORKERS = ("course-image-worker.js", "course-image-decode-worker.js")


def application(source, unit):
    if unit not in (5, 6, 7):
        raise ValueError("Image preparation is limited to assignments 5–7")
    count = 2 if unit == 5 else 1
    if source.count("import image_preload\n") != 1 or source.count("image_preload.Owner(") != count:
        raise ValueError("Application image ownership contract changed")
    return source.replace(
        "import image_preload\n", "import image_worker_preload as image_preload\n", 1
    ).replace("image_preload.Owner(", f"image_preload.Owner({unit},")


def write_config(stage, source, unit):
    shutil.copy2(RUNTIME / "image_worker_preload.py", stage / "image_worker_preload.py")
    inputs = {
        name: sha(stage / name)
        for name in ("image_preload.py", "image_worker_preload.py", "image_data.py")
    }
    inputs.update({name: sha(RUNTIME / name) for name in WORKERS})
    source_id = hashlib.sha256(source.encode()).hexdigest()
    manifest = sha(ASSETS / "v1/images/manifest.json")
    identity = dict(
        unit=unit,
        source_id=source_id,
        manifest_sha256=manifest,
        dependencies=inputs,
        environment=sha(REQUIREMENTS / "native.lock"),
    )
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config = dict(build_id=build_id, source_id=source_id, manifest_sha256=manifest)
    (stage / "course-loading.json").write_text(json.dumps(config, sort_keys=True))
    return config, inputs


def write_probe_config(stage, *, worker="unit5-worker.js"):
    """Bind the full-data probe's preparation to its actual numerical inputs."""
    names = (
        "image_data.py",
        "image_preload.py",
        "image_worker_preload.py",
        "unit5_probe.py",
        "unit5-definitions.py",
        "u5_utils.py",
        worker,
    )
    inputs = {name: sha(stage / name) for name in names}
    inputs.update(
        {
            name: sha(stage / name)
            for name in ("unit5-native.npz", "unit5-full-native.npz")
            if (stage / name).exists()
        }
    )
    inputs.update({name: sha(RUNTIME / name) for name in WORKERS})
    source_id = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    manifest = sha(ASSETS / "v1/images/manifest.json")
    build_id = hashlib.sha256(
        json.dumps(
            dict(
                unit=5,
                source_id=source_id,
                manifest_sha256=manifest,
                environment=sha(REQUIREMENTS / "native.lock"),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    config = dict(build_id=build_id, source_id=source_id, manifest_sha256=manifest)
    (stage / "course-probe-image-loading.json").write_text(json.dumps(config, sort_keys=True))
    return config, inputs
