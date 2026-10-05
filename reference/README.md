# Hands-on AI assignments in the browser

Assignments **1–7** have static Shinylive exports at `/unit1/` through `/unit7/`. Their Python code executes in browser workers. Hosting serves files only.

The supplied applications are preserved. Assignment 5 was extracted byte-for-byte from `assignments/Material-20261003/app.zip`; its source, CSV, and supplied Fashion-MNIST files are checksum-locked. Confirmed application corrections are shared by the [runnable Python reference and browser staging](proof/REFERENCE-CORRECTIONS.md). Browser-only changes live in separate runtime modules.

**Complete 1:1 parity is not certified.** The verified t-SNE matrices, direct image classifier, and small MNIST CNN now pass numerical comparisons. Other training presets/settings, exhaustive interface and appearance coverage, startup intermittency, and browser/hardware acceptance remain blockers. The Next.js shell remains deferred until application parity passes. See [the report](proof/REPORT.md) and [coverage gate](proof/evidence/gate.json).

Units 5–7 now train in disposable browser workers. The [October 5 implementation record](proof/IMPLEMENTATION-OCT05.md) links the Unit 5 native controls, cancellation/recovery checks and preserved evidence. The release gate retains the original 133 suites and 1,824 behavior IDs and adds seven delivery suites.

Preview the built files with Node 18+ (no dependencies or Python server):

```sh
node tools/preview.mjs
```

| Assignment | Preview |
|---|---|
| 1 — Tabular Data | http://127.0.0.1:8008/unit1/ |
| 2 — Data Types | http://127.0.0.1:8008/unit2/ |
| 3 — Unsupervised Learning | http://127.0.0.1:8008/unit3/ |
| 4 — Supervised Learning | http://127.0.0.1:8008/unit4/ |
| 5 — Logistic Regression | http://127.0.0.1:8008/unit5/ |
| 6 — Neural Networks | http://127.0.0.1:8008/unit6/ |
| 7 — CNNs | http://127.0.0.1:8008/unit7/ |

Production hosting requires HTTPS. Upload the complete `proof/site/` directory, retaining paths and MIME types. No inference service or application server is needed. These exports remain development previews until the gate passes.

[Build and verification instructions](proof/README.md) describe assets and tests. The frozen native environment remains the numerical reference; [`iml_env.yaml`](iml_env.yaml) is unchanged and [its differences are documented](proof/REFERENCE.md).
