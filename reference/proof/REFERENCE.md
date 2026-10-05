# Numerical reference

Frozen working local Python environment explicitly selected by user; course YAML is provenance, not the numerical target.

`iml_env.yaml` is preserved byte for byte. Shiny is unpinned there; PyTorch and torchvision are absent. The course setup separately identifies `en_core_web_md` 3.7.1.

| Package | Course YAML | Frozen reference |
|---|---|---|
| python | 3.12 | 3.12.13 |
| jupyter | unpinned | absent |
| ipywidgets | unpinned | 8.1.9 |
| numpy | 1.26.4 | 1.26.4 |
| pandas | 2.2.2 | 2.2.3 |
| matplotlib | 3.8.4 | 3.10.8 |
| seaborn | 0.13.2 | 0.13.2 |
| scipy | 1.13.1 | 1.14.1 |
| scikit-learn | 1.5.0 | 1.6.1 |
| scikit-image | 0.23.2 | absent |
| tqdm | 4.66.4 | 4.70.1 |
| plotly | 6.3.0 | 7.1.0 |
| opencv | unpinned | absent |
| spacy | 3.7.2 | 3.7.5 |
| shiny | unpinned | 1.8.0 |
| rdkit | 2024.09.1 | absent |
| gymnasium | 0.28.1 | absent |
| pillow | 11.0.0 | 12.3.0 |
| imutils | 0.5.4 | absent |
| shinywidgets | 0.7.0 | 0.8.1 |

The complete installed-package snapshot and file checksums are in `evidence/reference-environment.json`. No environment upgrade was performed. Browser package differences remain subject to numerical verification.

Native numerical fixtures use CPU and one compute thread, as recorded in `evidence/reference-environment.json`. Match that retained execution policy when launching a reference for numerical/appearance comparison; do not silently compare it with a default multithreaded process. For example:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/shiny run --host 127.0.0.1 --port 8111 proof/reference/unit1/app.py
```

This sets process execution conditions, preserving supplied sources, dependency versions, algorithms, parameters, and historical fixtures. Broad native/browser appearance equivalence still requires its own evidence.
