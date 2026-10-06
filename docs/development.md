# Development

Use Python 3.12 (`python3.12` on PATH, or set `PYTHON` to its executable), Git and a bootstrap Node executable. The wrapper downloads checksum-locked Node 24.21.0 and npm 11.19.0 for macOS ARM64 and Linux x64/ARM64. It enforces the existing 15 GiB storage reserve.

Create a fresh checkout and run commands from its root:

```sh
git clone https://github.com/Julian-AT/hands-on-webassembly.git
cd hands-on-webassembly
make setup
make check
make test
make build
make preview
```

Set `COURSE_ASSET_TOKEN` in your environment to a build-only GitHub credential with read access to this private repository's immutable release. It is used only to download archives, and is never embedded in static output. For offline setup, set `COURSE_ASSET_ARCHIVE` to the verified application archive and `COURSE_SOURCE_ARCHIVE` to `original-reference-sources.tar.gz`. Both are verified before use.

`make setup` uses the single root npm lockfile, installs the frozen native environment in `.cache/env/native`, installs Ruff separately in `.cache/env/tools`, prepares verified application payloads, restores larger originals in `artifacts/originals`, and generates the native training fixture from canonical source. It installs Husky hooks. No scientific dependency is upgraded to add developer tools.

Pre-commit rejects generated files and runs lint-staged on editable source. Python is checked and formatted with Ruff; JavaScript and documents use Prettier, and JavaScript uses ESLint. Original course files, vendored/native sources, runtime payloads, contracts, fixtures and immutable manifests are excluded from formatting. Commit messages use Conventional Commits. Hook behavior has an isolated Git regression test.

Use `node scripts/toolchain.mjs format` to format owned source and `node scripts/toolchain.mjs commit-check` to check the complete reconstructed history. `make check` runs formatting, linting and repository hygiene. `make test` runs all regression tests. Generated Next.js directories remain `web/public`, `web/.next` and `web/out`.

Browser automation has its own frozen lock. Install it separately when running installed-browser certification:

```sh
python3.12 -m venv .cache/env/browser
.cache/env/browser/bin/python -m pip install -r requirements/browser.lock
```

The native lock and `iml_env.yaml` are retained byte-for-byte. [Course setup](course-setup.md) describes the supplied environment. Host-specific scientific rebuilds are documented in [verification](verification.md).
