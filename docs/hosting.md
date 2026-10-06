# Static hosting

Vercel configuration stays at the repository root. Use Other framework, `node scripts/toolchain.mjs install` as the install command, `node scripts/toolchain.mjs build` as the build command, and `web/out` as output. Configure the build-only `COURSE_ASSET_TOKEN` to read the immutable application asset release. Deployment uploads exclude `.cache`, `artifacts`, dependencies and generated output.

The repository defines no runtime functions. Direct `/unit1/` through `/unit7/` routes, query-based selection, WebAssembly/JavaScript MIME types, same-origin isolation and cache policy are retained. Mutable entry pages, service workers and bootstrap resources revalidate; content-addressed Next.js chunks cache immutably.

Use `make preview` for the local export at `http://127.0.0.1:8008`. The dependency-free preview server supports GET, HEAD, ranges and trailing-slash redirects. The browser computes and owns all resources.

Verify deployment against `web/out/release-manifest.json` with `scripts/proof/verify_vercel_delivery.py`, and record installed-browser navigation, readiness, disposal, Retry and startup-animation behavior separately. Source restructuring does not grant a new certification pass or change the deployed application automatically.

Rollback selects the previous verified Vercel deployment, or uses `vercel rollback <previous-deployment-url>`. Preserve each release manifest and its pinned application tag. A rollback is certified only after the deployed hashes and browser behavior are verified.
