"""Stage original apps with narrowly checked runtime-only substitutions."""
import ast
import json
import shutil
import subprocess
import sys
import re
import os
from pathlib import Path

from common import ROOT, PROOF, UNITS, sha, write_json
from reference_corrections import corrected_source


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Source contract changed: expected one {old!r}")
    return source.replace(old, new)


def interface(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "app_ui" for t in n.targets))
    # Only static URL paths differ. Everything else in the UI must be identical.
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            for url, path in URLS.items():
                if n.value in (url, path):
                    n.value = path
    return ast.dump(node, include_attributes=False)


URLS = {
    "vendor/plotly/plotly.min.js": "../../assets/v1/plotly/plotly.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.3/ace.js": "../../assets/v1/ace/ace.js",
    "https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/zephyr/bootstrap.min.css": "vendor/zephyr/bootstrap.min.css",
    "https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css": "vendor/katex/katex.min.css",
    "https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js": "vendor/katex/katex.min.js",
    "https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js": "vendor/katex/auto-render.min.js",
}


def main():
    from storage_budget import build_budget
    print('Storage preflight:', build_budget(), flush=True)
    for name, expected in json.loads((PROOF / "source.lock.json").read_text()).items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Original source contract changed: {name}")
    assets = PROOF / "assets/v1"
    from detached_artifact import detach_shared_artifact
    detach_shared_artifact(PROOF/'site')
    # Shinylive intentionally retains differing existing runtime files. Our
    # checked patches therefore require a fresh runtime on every build.
    runtime = PROOF / "site/shinylive"
    if runtime.exists():
        shutil.rmtree(runtime)
    (PROOF / "site/shinylive-sw.js").unlink(missing_ok=True)
    evidence = {}
    for unit in (1, 2, 3, 4, 5, 6, 7):
        src = ROOT / UNITS[unit]
        stage = PROOF / "build" / f"unit{unit}"
        if stage.exists():
            shutil.rmtree(stage)
        shutil.copytree(src, stage, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        shutil.copytree(assets / "www", stage / "www", dirs_exist_ok=True)
        # Ace is served once as a physical asset; it need not also inflate each
        # assignment's virtual filesystem (four apps do not use it at all).
        shutil.rmtree(stage / "www/vendor/ace")
        css = stage / "www/vendor/zephyr/bootstrap.min.css"
        css.write_text(css.read_text().replace("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap", "inter.css"))
        (stage / "resources").mkdir(exist_ok=True)
        shutil.copy2(assets / "penguins.csv", stage / "resources/penguins.csv")
        original = corrected_source(unit)
        for helper in src.glob("*.py"):
            if helper.name != "app.py":
                (stage / helper.name).write_text(corrected_source(unit, helper.name))
        source = original.replace('sns.load_dataset("penguins")', 'pd.read_csv("resources/penguins.csv")')
        source = source.replace('from shiny import App,', 'from pathlib import Path\nfrom shiny import App,')
        if unit in (5,6,7):
            from stage_neural import stage_neural
            source = stage_neural(unit,stage,source,replace_once)
            from stage_image_workers import application as image_worker_application
            source = image_worker_application(source,unit)
        if unit == 1:
            source=replace_once(source,'from sklearn.manifold import TSNE','import tsne_runtime\nfrom sklearn.manifold import TSNE')
            shutil.copy2(PROOF/'runtime/tsne_runtime.py',stage/'tsne_runtime.py')
        if unit == 2:
            source = replace_once(source,'import spacy\n','import embedding_runtime as spacy\n')
            source = replace_once(source,'import pkg_resources\n','')
            node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='install_spacy_model')
            old = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
            source = replace_once(source,old,'')
            source = replace_once(source,'install_spacy_model("en_core_web_md")','')
            helper = stage / 'u2_utils.py'
            helper.write_text(replace_once(helper.read_text(),'import spacy\n','import embedding_runtime as spacy\n'))
            shutil.copy2(PROOF/'runtime/embedding_runtime.py',stage/'embedding_runtime.py')
            from stage_embedding_loading import adapter,application
            (stage/'embedding_runtime.py').write_text(adapter((stage/'embedding_runtime.py').read_text()))
            shutil.copy2(PROOF/'runtime/embedding_preload.py',stage/'embedding_preload.py')
            source=application(source)
            (stage/'embedding-assets').mkdir()
            shutil.copy2(assets/'embedding-runtime/manifest.json',stage/'embedding-assets/manifest.json')
            embedding_target=PROOF/'site/assets/v1/embedding-runtime'
            if embedding_target.exists():shutil.rmtree(embedding_target)
            shutil.copytree(assets/'embedding-runtime',embedding_target,copy_function=os.link)
            from plotly.offline import get_plotlyjs
            plotly = PROOF/'site/assets/v1/plotly'
            plotly.mkdir(parents=True,exist_ok=True)
            (plotly/'plotly.min.js').write_text(get_plotlyjs())
            source = source.replace('vendor/plotly/plotly.min.js', '../../assets/v1/plotly/plotly.min.js')
        if unit == 3:
            source = replace_once(source, "from tkinter.font import names\n", "")
            source = replace_once(source, "from sklearn.datasets import fetch_openml,", "from local_data import fetch_openml\nfrom sklearn.datasets import")
            shutil.copy2(PROOF / "runtime/local_data.py", stage / "local_data.py")
            for name in ("seeds", "ionosphere"):
                shutil.copy2(assets / f"{name}.json", stage / "resources" / f"{name}.json")
        if unit == 4:
            source=replace_once(source,"import numpy as np", "import forest_runtime\nimport numpy as np")
            shutil.copy2(PROOF/"runtime/forest_runtime.py",stage/"forest_runtime.py")
            for url, name in {
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv": "pima.csv",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt": "banknote.csv",
            }.items():
                source = replace_once(source, f'url = "{url}"', f'url = "resources/{name}"')
                shutil.copy2(assets / name, stage / "resources" / name)
        source = re.sub(r'"(?:\./)?resources/([^"\n]+)"', r'str(Path(__file__).parent / "resources/\1")', source)
        for url, path in URLS.items():
            source = source.replace(url, path)
        source = replace_once(source, "app = App(app_ui, server)", 'app = App(app_ui, server, static_assets=Path(__file__).parent / "www")')
        if interface(original) != interface(source):
            raise ValueError(f"Unit {unit}: UI contract changed")
        if unit in (5,6,7):
            from stage_image_workers import write_config as write_image_worker_config
            write_image_worker_config(stage,source,unit)
        if unit == 2:
            from stage_embedding_loading import write_embedding_config
            write_embedding_config(stage,source)
        (stage / "app.py").write_text(source)
        exported = PROOF / "site" / f"unit{unit}"
        if exported.exists():
            shutil.rmtree(exported)
        subprocess.run([str(Path(sys.executable).parent / "shinylive"), "export", str(stage), str(PROOF / "site"), "--subdir", f"unit{unit}"], check=True)
        # The exporter walks directories in filesystem order. Preserve app.py
        # as the entry file and canonicalize the rest for reproducible bytes.
        bundle = exported / "app.json"
        entries = json.loads(bundle.read_text())
        entries.sort(key=lambda entry: (entry['name'] != 'app.py', entry['name']))
        bundle.write_text(json.dumps(entries, sort_keys=True, separators=(',', ':')))
        evidence[str(unit)] = {"source_sha256": sha(src / "app.py"), "staged_sha256": sha(stage / "app.py"), "ui_ast_preserved": True}
    from patch_runtime import patch_runtime
    patch_runtime()
    # The isolated compute context imports only its pinned numerical modules.
    # This physical bundle is requested when training starts, not at page load.
    import cloudpickle, hashlib
    neural = PROOF/'site/assets/v1/neural-runtime'
    if neural.exists(): shutil.rmtree(neural)
    neural.mkdir(parents=True)
    for name in ('isolated_training.py','browser_torch.py','neural_compat.py','cnn_compat.py','torch_rng.py','borch_compat.py',
                 'image_data.py','module_hooks.py','image_transforms.py'):
        shutil.copy2(PROOF/'runtime'/name,neural/name)
    shutil.copytree(Path(cloudpickle.__file__).parent,neural/'cloudpickle',ignore=shutil.ignore_patterns('__pycache__'))
    borch='pyborch-1.14.1-py3-none-any.whl'
    shutil.copy2(assets/borch,neural/borch)
    shutil.copy2(PROOF/'build/unit7/course-training.json',neural/'course-training.json')
    from reference_corrections import training_source
    from training_bundle import browser_build_id
    manifest={'borch':borch,'build_id':browser_build_id(training_source(7)),
              'build_ids':{str(unit):browser_build_id(training_source(unit)) for unit in (5,6,7)},
              'sources':{str(unit):hashlib.sha256(training_source(unit).encode()).hexdigest() for unit in (5,6,7)},
              'files':{str(path.relative_to(neural)):sha(path) for path in sorted(neural.rglob('*')) if path.is_file()}}
    (neural/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':')))
    shutil.copy2(PROOF/'runtime/course-training-worker.js',PROOF/'site/shinylive/course-training-worker.js')
    shutil.copy2(PROOF/'runtime/course-embedding-worker.js',PROOF/'site/shinylive/course-embedding-worker.js')
    for name in ('course-image-worker.js','course-image-decode-worker.js'):
        shutil.copy2(PROOF/'runtime'/name,PROOF/'site/shinylive'/name)
    write_json(PROOF / "evidence/build.json", evidence)
    from pin_runtime_packages import pin_runtime_packages
    pin_runtime_packages()
    lock = PROOF / "site/shinylive/pyodide/pyodide-lock.json"
    shutil.copy2(lock, PROOF / "evidence/pyodide-lock.json")
    shutil.copy2(assets/'pyodide-0.27.7.js',PROOF/'site/shinylive/pyodide/pyodide.js')
    shutil.copytree(assets/'www/vendor/ace',PROOF/'site/assets/v1/ace',dirs_exist_ok=True)
    supplied_csv = ROOT / "assignments/Material-20261003/DataSet_LR_a.csv"
    csv_target = PROOF / "site/assets/v1/assignment5"
    csv_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(supplied_csv, csv_target / supplied_csv.name)
    images = assets / "images"
    if images.exists():
        target = PROOF / "site/assets/v1/images"
        if target.exists(): shutil.rmtree(target)
        shutil.copytree(images,target,copy_function=os.link)
    from patch_runtime import version_runtime
    version_runtime()
    # Keep diagnostic workers in sync with their checked-in sources too;
    # reference arrays are generated separately by the native fixture scripts.
    probes = PROOF/'site/probes'
    probes.mkdir(exist_ok=True)
    for name in ('browser_torch.py','neural_compat.py','cnn_compat.py','tsne_runtime.py','tabular_regressions.py'):
        shutil.copy2(PROOF/'runtime'/name,probes/name)
    for path in (PROOF/'web').iterdir():
        if path.is_file():
            shutil.copy2(path, probes/path.name)
    shutil.copy2(PROOF/'runtime/course-embedding-worker.js',probes/'course-embedding-worker.js')
    for name in ('image_data.py','image_preload.py','image_worker_preload.py'):
        shutil.copy2(PROOF/'runtime'/name,probes/name)
    from stage_image_workers import write_probe_config
    write_probe_config(probes)


if __name__ == "__main__":
    main()
