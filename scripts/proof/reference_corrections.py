"""Authorized application fixes, shared by native reference and browser staging.

Supplied files and historical fixtures are never edited. Each substitution is
checked against the supplied source; a source change requires explicit review.
"""

from repository import (
    original_path,
    ASSETS,
    BASELINE,
    CACHE,
    EVIDENCE,
    PATCHES,
    REFERENCE,
    REQUIREMENTS,
    RUNTIME,
)
import difflib
import ast
import json
import shutil
from common import ROOT, PROOF, UNITS, sha, write_json


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"Reference correction source changed: {old!r}")
    return source.replace(old, new)


def corrected_source(unit, filename="app.py"):
    source = (ROOT / UNITS[unit] / filename).read_text()
    if unit == 2 and filename == "u2_utils.py":
        # First occurrence order agrees with the old indexing for unique words.
        source = replace(
            source, "words_vocab = np.unique(vocab)", "words_vocab = list(dict.fromkeys(vocab))"
        )
        source = replace(
            source, "one_hot[i, vocab.index(word)] = 1", "one_hot[i, words_vocab.index(word)] = 1"
        )
    if filename != "app.py":
        return source
    # The macOS canvas applies the physical Retina scale before tight_layout,
    # even when the web capture uses scale one. Shiny then saves at logical DPI,
    # retaining that earlier, different layout. Use the same Agg canvas as WASM
    # before any figure is created. Original calculation bodies stay intact.
    source = replace(
        source,
        "import matplotlib.pyplot as plt",
        'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt',
    )
    if unit in (5, 6):
        source = replace(
            source,
            "import torch\n" if unit == 5 else "import torch.nn as nn\n",
            ("import torch\n" if unit == 5 else "import torch.nn as nn\n")
            + "import inspection_rng\n",
        )
        if unit == 5:
            for name in ("mnist_samples", "fashionmnist_samples"):
                source = replace(
                    source,
                    f"    def {name}():",
                    f"    @inspection_rng.protected(torch)\n    def {name}():",
                )
        else:
            source = replace(
                source,
                "    def data_info():",
                "    @inspection_rng.protected(torch, prepare=data_bundle)\n    def data_info():",
            )
            source = replace(
                source,
                "    def model_summary():",
                "    @inspection_rng.protected(torch, prepare=lambda: data_bundle() if arch_state().strip() else None)\n    def model_summary():",
            )
    if unit == 2:
        source = replace(
            source,
            "pd.DataFrame(one_hot.to_numpy(),\n                     index=get_words(),",
            "pd.DataFrame(one_hot.to_numpy(),\n                     index=one_hot.index,",
        )
        # A blocking head script is ready before reactive fragments execute.
        source = replace(
            source,
            "    ui.tags.head(\n",
            '    ui.tags.head(\n        ui.tags.script(src="vendor/plotly/plotly.min.js"),\n',
        )
        source = replace(source, 'include_plotlyjs="cdn"', "include_plotlyjs=False", count=5)
        for element in ("div", "d"):
            source = replace(
                source,
                f"window.Plotly.Plots.resize({element});",
                f"if ({element}.isConnected && {element}.offsetParent !== null && {element}.clientWidth > 0) "
                f"window.Plotly.Plots.resize({element}).catch(() => undefined);",
                count=2,
            )
    if unit in (6, 7):
        quote = '"' if unit == 6 else "'"
        old = f"layers = data.get({quote}layers{quote}, data if isinstance(data, list) else [])"
        source = replace(
            source,
            old,
            f"layers = data if isinstance(data, list) else data.get({quote}layers{quote}, []) if isinstance(data, dict) else None",
        )
    if unit == 5:
        from stage_unit5_training import unit5_training

        source = unit5_training(source, replace)
    if unit == 6:
        from stage_unit6_training import unit6_training

        source = unit6_training(source, replace, native=True)
    if unit == 7:
        source = replace(source, "import torch\n", "import torch\nimport isolated_training\n")
        source = replace(
            source,
            "    def run_training_sync(params: dict) -> dict:",
            "    _training_context = isolated_training.Executor(7)\n"
            "    session.on_ended(_training_context.cancel)\n\n"
            "    def run_training_sync(params: dict) -> dict:",
        )
        source = replace(
            source,
            "        return await asyncio.to_thread(run_training_sync, params)",
            "        result = await _training_context.run(params, _q_put)\n"
            '        result["model"] = isolated_training.restore_model(result.pop("model_record"), build_model, parse_architecture(params["arch_text"]))\n'
            "        return result",
        )
        for name in ("_reset_model", "_start_train_task"):
            source = replace(
                source,
                f"    def {name}():\n",
                f"    def {name}():\n        _training_context.cancel()\n        train_task.cancel()\n"
                "        while not _train_q.empty():\n            _train_q.get_nowait()\n",
            )
        source = replace(source, "def _dim(Lin, k, s, p):", "def _dim(height, width, k, s, p):")
        source = replace(
            source,
            "return math.floor((Lin + 2 * px - kx) / sx + 1), math.floor((Lin + 2 * py - ky) / sy + 1)",
            "return math.floor((height + 2 * px - kx) / sx + 1), math.floor((width + 2 * py - ky) / sy + 1)",
        )
        source = replace(source, "h, w = _dim(h, k, s, p)", "h, w = _dim(h, w, k, s, p)")
        source = replace(source, "def _dim(Lin, k, s):", "def _dim(height, width, k, s):")
        source = replace(
            source,
            "return math.floor((Lin - kx) / sx + 1), math.floor((Lin - ky) / sy + 1)",
            "return math.floor((height - kx) / sx + 1), math.floor((width - ky) / sy + 1)",
        )
        source = replace(source, "h, w = _dim(h, k, s)", "h, w = _dim(h, w, k, s)")
    return source


def training_source(unit):
    if unit == 5:
        from unit5_training_source import training_source as unit5_source

        return unit5_source()
    if unit == 6:
        from unit6_training_source import training_source as unit6_source

        return unit6_source()
    if unit != 7:
        raise ValueError("Unsupported training context")
    source = corrected_source(unit)
    tree = ast.parse(source)
    names = {
        "parse_architecture",
        "build_model",
        "ACTIVATIONS",
        "PRESETS",
        "_yamlish_to_json",
        "LayerSpec",
        "_tuple_or_int",
        "get_first_conv_weights",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            selected.append(node)
    lines = source.splitlines()
    pieces = []
    for node in selected:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        pieces.append("\n".join(lines[start - 1 : node.end_lineno]))
    server = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "server"
    )
    body = next(
        node
        for node in server.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_training_sync"
    )
    import textwrap

    pieces.append(textwrap.dedent("\n".join(lines[body.lineno - 1 : body.end_lineno])))
    return "\n\n".join(pieces) + "\n"


def main():
    from plotly.offline import get_plotlyjs
    from preserve_reference_revision import preserve

    preserve()
    if (BASELINE / "manifest.corrected.json").exists():
        from preserve_reference_baseline import main as preserve_references

        preserve_references()
    for name, expected in json.loads((PROOF / "source.lock.json").read_text()).items():
        if sha(original_path(name)) != expected:
            raise ValueError(f"Supplied source changed: {name}")
    records = {}
    for unit, folder in UNITS.items():
        original = ROOT / folder
        target = REFERENCE / f"unit{unit}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        from repository import copy_original_resources

        copy_original_resources(unit, target)
        for path in original.glob("*.py"):
            corrected = corrected_source(unit, path.name)
            (target / path.name).write_text(corrected)
            diff = "".join(
                difflib.unified_diff(
                    path.read_text().splitlines(True),
                    corrected.splitlines(True),
                    fromfile=f"supplied/unit{unit}/{path.name}",
                    tofile=f"corrected/unit{unit}/{path.name}",
                )
            )
            if diff:
                patch = PATCHES / f"unit{unit}-{path.stem}.patch"
                patch.parent.mkdir(exist_ok=True)
                patch.write_text(diff)
                records[f"unit{unit}/{path.name}"] = {
                    "original_sha256": sha(path),
                    "corrected_sha256": sha(target / path.name),
                    "patch_sha256": sha(patch),
                }
        if unit == 2:
            static = target / "www/vendor/plotly"
            static.mkdir(parents=True, exist_ok=True)
            (static / "plotly.min.js").write_text(get_plotlyjs())
        # Native and browser captures use the same immutable font/CSS bytes.
        # Google Fonts serves a different WOFF2 subset than our pinned TTFs.
        theme = target / "www/vendor/zephyr"
        shutil.copytree(ASSETS / "v1/www/vendor/zephyr", theme, dirs_exist_ok=True)
        css = theme / "bootstrap.min.css"
        css.write_text(
            css.read_text().replace(
                "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap",
                "inter.css",
            )
        )
        app = target / "app.py"
        source = replace(
            app.read_text(),
            "https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/zephyr/bootstrap.min.css",
            "vendor/zephyr/bootstrap.min.css",
        )
        source = replace(
            source,
            "app = App(app_ui, server)",
            'app = App(app_ui, server, static_assets=Path(__file__).parent / "www")',
        )
        if "from pathlib import Path" not in source:
            source = "from pathlib import Path\n" + source
        app.write_text(source)
        if unit in (5, 6, 7):
            # Use complete cached datasets without touching supplied materials.
            resources = target / "resources"
            if not resources.exists():
                resources.symlink_to(CACHE / "torchvision", target_is_directory=True)
        if unit in (5, 6):
            shutil.copy2(RUNTIME / "inspection_rng.py", target / "inspection_rng.py")
        if unit in (5, 6, 7):
            shutil.copy2(RUNTIME / "isolated_training.py", target / "isolated_training.py")
            import hashlib

            body = training_source(unit)
            identity = hashlib.sha256(
                (
                    body + sha(target / "isolated_training.py") + sha(REQUIREMENTS / "native.lock")
                ).encode()
            ).hexdigest()
            (target / "course-training.json").write_text(
                json.dumps({"source": body, "build_id": identity}, sort_keys=True)
            )
        # Include final native delivery substitutions in the retained patches.
        for path in original.glob("*.py"):
            corrected = (target / path.name).read_text()
            diff = "".join(
                difflib.unified_diff(
                    path.read_text().splitlines(True),
                    corrected.splitlines(True),
                    fromfile=f"supplied/unit{unit}/{path.name}",
                    tofile=f"corrected/unit{unit}/{path.name}",
                )
            )
            if diff:
                patch = PATCHES / f"unit{unit}-{path.stem}.patch"
                patch.parent.mkdir(exist_ok=True)
                patch.write_text(diff)
                records[f"unit{unit}/{path.name}"] = dict(
                    original_sha256=sha(path),
                    corrected_sha256=sha(target / path.name),
                    patch_sha256=sha(patch),
                )
    write_json(
        EVIDENCE / "reference-corrections.json",
        {
            "files": records,
            "native_theme_assets": {
                str(path.relative_to(REFERENCE)): sha(path)
                for path in sorted((REFERENCE).glob("unit*/www/vendor/zephyr/*"))
                if path.is_file()
            },
            "runnable_apps": {
                str(unit): sha(REFERENCE / f"unit{unit}" / "app.py") for unit in UNITS
            },
            "scope": "Deterministic Agg canvas in all seven applications: Retina 200-DPI pre-layout versus browser 100-DPI mismatch reproduced in correlation-render-control-ssl-oct05.json, eliminated with identical original data/body in correlation-render-control-agg-oct05.json, and exact decoded native/browser captures in unit1-matched-fonts-agg-native-candidate-oct05.json; one-hot row labels and duplicate vocabulary; Plotly readiness/hidden resize; architecture lists; rectangular CNN dimensions; isolated CNN training RNG; Units 5/6 inspection RNG isolation reproduced in native-inspection-rng-audit.json; Unit 6 disposable training and Reset responsiveness reproduced in native-unit6-reset-audit-acknowledged-inputs-oct04.json; Unit 5 dataset-replacement defect reproduced in native-unit5-training-replacement-oct05-original-full-deadline.json and corrected in native-unit5-training-replacement-oct05-corrected-full-deadline.json. Earlier one-millisecond audit observations are retained harness faults, not acceptance evidence.",
            "historical_fixtures": "retained; no numerical fixture regenerated by this command",
        },
    )


if __name__ == "__main__":
    main()
