"""Prepare full tabular assets and audit the exact original embedding model.

Network access is a maintainer/build operation. Downloads are checksum locked;
reruns fail if an upstream file differs from the recorded asset.
"""
import json
import re
import urllib.request
from pathlib import Path

from common import PROOF, write_json, sha

ASSETS = PROOF / "assets/v1"
LOCK = PROOF / "assets.lock.json"
manifest = json.loads(LOCK.read_text()) if LOCK.exists() else {}


def download(url, relative):
    path = ASSETS / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        path.write_bytes(data)
    entry = {"url": url, "sha256": sha(path), "bytes": path.stat().st_size}
    if relative in manifest and entry != manifest[relative]:
        raise ValueError(f"Asset changed: {relative}; review before updating the lock")
    manifest[relative] = entry
    write_json(LOCK, manifest)
    return path


def main():
    download("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv", "penguins.csv")
    download("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", "pima.csv")
    download("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", "banknote.csv")
    for name in ('ace.js','mode-json.js','theme-textmate.js','theme-github.js','worker-json.js'):
        download('https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.3/'+name,'www/vendor/ace/'+name)
    # Preserve the original CSS and KaTeX; include referenced font files as well.
    urls = [
        ("https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/zephyr/bootstrap.min.css", "www/vendor/zephyr/bootstrap.min.css"),
        ("https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css", "www/vendor/katex/katex.min.css"),
        ("https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js", "www/vendor/katex/katex.min.js"),
        ("https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js", "www/vendor/katex/auto-render.min.js"),
    ]
    for url, dest in urls:
        p = download(url, dest)
        if "katex.min.css" in dest:
            for font in sorted(set(re.findall(r"url\((fonts/[^)]+)\)", p.read_text()))):
                download("https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/" + font, "www/vendor/katex/" + font)
    fonts_url = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap"
    css = download(fonts_url, "inter-upstream.css").read_text()
    for url in sorted(set(re.findall(r"url\((https://[^)]+)\)", css))):
        relative = "www/vendor/zephyr/" + url.rsplit("/", 1)[-1]
        download(url, relative)
        css = css.replace(url, url.rsplit("/", 1)[-1])
    (ASSETS / "www/vendor/zephyr/inter.css").write_text(css)

    from sklearn.datasets import fetch_openml
    tabular = {}
    for name in ("seeds", "ionosphere"):
        ds = fetch_openml(name, version=1, as_frame=True)
        path = ASSETS / f"{name}.json"
        payload = {"columns": list(ds.frame.columns), "target": ds.target.name,
                   "categories": list(ds.target.cat.categories),
                   "rows": ds.frame.astype(object).values.tolist(), "details": ds.details}
        write_json(path, payload)
        tabular[name] = {"shape": list(ds.data.shape), "sha256": sha(path), "details": ds.details}
    write_json(PROOF / "evidence/tabular-assets.json", tabular)

    # Export ALL vector rows and ALL keys. These are research assets, not a
    # replacement tokenizer or a declaration that Unit 2 works in the browser.
    import numpy as np
    import spacy
    nlp = spacy.load("en_core_web_md")
    if nlp.meta["version"] != "3.7.1":
        raise ValueError("The course requires en_core_web_md 3.7.1")
    model = ASSETS / "en_core_web_md-3.7.1"
    model.mkdir(exist_ok=True)
    np.save(model / "vectors.npy", nlp.vocab.vectors.data, allow_pickle=False)
    keys = nlp.vocab.vectors.key2row
    np.save(model / "keys.npy", np.asarray(list(keys), dtype=np.uint64), allow_pickle=False)
    np.save(model / "rows.npy", np.asarray(list(keys.values()), dtype=np.int32), allow_pickle=False)
    nlp.vocab.strings.to_disk(model / "strings.json")
    nlp.tokenizer.to_disk(model / "tokenizer.bin")
    cases = ["king", "King", "KING", "New York", "can't", "U.S.A.", "ice-cream", "", "   ", "zzqvnotaword", "cat zzqvnotaword", "Hello, world!", "café", "👋"]
    fixtures = [{"text": s, "tokens": [t.text for t in nlp(s)], "vector": nlp(s).vector.tolist(),
                 "vector_norm": float(nlp(s).vector_norm)} for s in cases]
    write_json(PROOF / "evidence/embedding-reference.json", {
        "model": nlp.meta, "spacy": spacy.__version__, "vector_shape": list(nlp.vocab.vectors.data.shape),
        "key_count": len(keys), "string_count": len(nlp.vocab.strings), "fixtures": fixtures,
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in model.iterdir()},
        "browser_parity": "NOT_PROVEN: tokenizer.bin requires spaCy; vector lookup alone cannot preserve arbitrary multiword inputs",
    })


if __name__ == "__main__":
    main()
