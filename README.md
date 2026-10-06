# Hands-on WebAssembly

Seven browser-only machine learning assignments delivered as a Next.js static export. The root selector uses one same-origin iframe and the existing verified readiness/disposal bridge. `/` defaults to Assignment 1. `/?unit=1` through `/?unit=7` select assignments; `/unit1/` through `/unit7/` remain direct entry points.

This repository is a deployment candidate, not the completed application certification. The retained application gate is 9/140 current suites, with `nextjs_authorized: false`. The user explicitly authorized deployment before completing that gate. No pending numerical, pixel, workload or four-browser requirement is waived or recorded as passing.

## Reproducible build

The wrapper selects verified Node 24.21.0 and npm 11.19.0 on macOS ARM64 or Linux x64/ARM64. Next 16.3.8, React and React DOM 19.3.0 are pinned with the npm lockfile.

```sh
node scripts/toolchain.mjs install
node scripts/toolchain.mjs test
node scripts/toolchain.mjs build
```

Set `COURSE_ASSET_TOKEN` to a build-only GitHub credential able to read this private repository's release assets. For local offline builds, `COURSE_ASSET_ARCHIVE` can instead point to the verified application archive. Neither value is emitted to the browser. The download is verified against its locked archive checksum and all 218 path/size/hash records before publication. Traversal, links, duplicate or unexpected entries, truncation and checksum failures stop the build. Failed staging directories remain available for inspection.

The build uses `output: 'export'`, `trailingSlash: true` and `next build --webpack`, emits `out/`, verifies the original assets and the seven locked startup UI changes, and writes a complete `release-manifest.json`. Build identity is derived from locked shell sources, configuration, toolchain and asset inputs.

## Hosting

Use Vercel Hobby project `hands-on-webassembly`, Other framework preset, the explicit install/build commands in `vercel.json`, and output directory `out`. This configuration defines no functions. The small source upload excludes application assets, tool caches, build outputs and native environments. Large datasets are downloaded during the build. Mutable entries, service workers and bootstrap files revalidate; content-addressed Next chunks cache immutably. All production computation and resources belong to the static application.

Vercel CLI needs a normal interactive login. `--global-config` may select a writable configuration directory, and `NO_UPDATE_NOTIFIER=1` disables the unrelated CLI updater. Credentials and `.vercel` state must never be committed.

## Preserved references

`reference/` retains original assignment sources, corrected runnable native sources, patches, locks, executable proof scripts, tests and requirement inventories. `manifests/reference-source-files.json` binds the full reference snapshot. Original files larger than 5 MiB and the complete snapshot are retained in the supplemental GitHub release archive; historical evidence, datasets and native environments also remain in the authoritative Desktop proof workspace. Its report identifies the actual 24 GB M4 Pro MacBook.

The local release checkout is staged inside the authorized Desktop workspace. The intended `~/Documents/GitHub/hands-on-webassembly` location is outside the enforced writable roots in this session.

## Release and rollback

Deploy only the source repository, with the locked asset release and build-only asset credential configured. Verify all deployed hashes, MIME types, ranges, GET/HEAD, cache headers and direct entry points against `out/release-manifest.json`. Record installed-browser tests separately from source tests. Production certification remains pending until every retained and added obligation passes in its final context.

For rollback, select the previous verified deployment in Vercel or run `vercel rollback <previous-deployment-url>`. A first candidate has no previously verified Vercel release, so rollback cannot yet be certified. Preserve the current source commit, pinned asset release and export manifest for future rollback testing.

## Loading animation

The original Shinylive animation is visible during loading. The custom recovery panel is hidden until an actual startup error, preserving Retry and the existing connected/initialized readiness checks. The immutable application archive stays unchanged; `manifests/startup-ui.json` locks the seven approved entry-page changes and the corresponding source-generator correction. All original assets are verified before applying the presentation change in staging, and the export is checked against the resulting hashes.
