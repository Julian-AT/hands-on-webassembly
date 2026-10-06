# Hands-on WebAssembly

[Open the assignments](https://h1.julianschmidt.cv).

Seven machine learning assignments run entirely in the browser, using preserved course sources and frozen scientific dependencies. A Next.js static selector opens one assignment at a time in a same-origin iframe. 

| Area            | Responsibility                                               |
| --------------- | ------------------------------------------------------------ |
| `assignments/`  | Original course sources and small supplied resources         |
| `web/`          | Next.js shell and static export configuration                |
| `runtime/`      | Browser adapters, workers and session bridge                 |
| `wasm/`         | Native arithmetic sources, patches, recipes and licenses     |
| `scripts/`      | Asset preparation, build, preview and verification utilities |
| `tests/`        | Python and JavaScript regressions and small fixtures         |
| `proof/`        | Active contracts, immutable locks and archive indexes        |
| `docs/`         | Development, architecture, provenance and hosting runbooks   |
| `requirements/` | Separate native, browser and development locks               |

With Python 3.12 and a build-only asset credential configured:

```sh
make setup
make check
make test
make build
make preview
```

The wrapper verifies Node 24.21.0 and npm 11.19.0. Next 16.3.8 and React 19.3.0 remain pinned. Downloads, environments and tool caches live in ignored `.cache/`; generated science and proof runs live in ignored `artifacts/`.

Read [development](docs/development.md), [architecture](docs/architecture.md), [provenance and archives](docs/provenance.md), [verification](docs/verification.md), and [static hosting](docs/hosting.md). [History reconstruction](docs/history.md) records the original authorship, dates and preservation process. The workspace structure and focused documentation follow [miRacle](https://github.com/o-hayat/miRacle).
