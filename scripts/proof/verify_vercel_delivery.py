"""Verify a static HTTPS deployment against the locally frozen file manifest."""

import argparse
import hashlib
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path


def request(url, method="GET", headers=None):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, method=method, headers={"Accept-Encoding": "identity", **(headers or {})}
        ),
        timeout=120,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    origin = args.origin.rstrip("/")
    if not origin.startswith("https://"):
        raise SystemExit("An HTTPS candidate is required")
    manifest = json.loads(args.manifest.read_text())
    checks = []
    failures = []

    def add(name, observed, expected):
        passed = observed == expected
        checks.append(
            dict(
                name=name, observed=observed, expected=expected, status="pass" if passed else "fail"
            )
        )
        if not passed:
            failures.append(name)

    with request(origin + "/release-manifest.json") as response:
        deployed = json.load(response)
        add("locked-deployed-manifest", deployed, manifest)
    for index, (name, info) in enumerate(manifest["files"].items(), 1):
        url = origin + "/" + urllib.parse.quote(name, safe="/")
        with request(url, "HEAD") as response:
            add(name + ":HEAD-status", response.status, 200)
            length = response.headers.get("Content-Length")
            if length is not None:
                add(name + ":HEAD-bytes", int(length), info["bytes"])
            content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
            if name.endswith(".wasm"):
                add(name + ":WASM-MIME", content_type, "application/wasm")
            if name.endswith((".js", ".mjs")):
                add(
                    name + ":JS-MIME",
                    content_type in ("application/javascript", "text/javascript"),
                    True,
                )
            cache = response.headers.get("Cache-Control", "").lower()
            if name.startswith("_next/static/"):
                add(name + ":immutable-cache", "immutable" in cache, True)
            else:
                add(
                    name + ":mutable-revalidation",
                    "max-age=0" in cache and ("must-revalidate" in cache or "no-cache" in cache),
                    True,
                )
        with request(url) as response:
            add(name + ":GET-status", response.status, 200)
            digest = hashlib.sha256()
            size = 0
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            add(name + ":GET-bytes", size, info["bytes"])
            add(name + ":GET-SHA256", digest.hexdigest(), info["sha256"])
        if index % 25 == 0:
            print(f"Verified {index}/{len(manifest['files'])} files", flush=True)
    with request(origin + "/") as response:
        add("root:status", response.status, 200)
        add("root:COOP", response.headers.get("Cross-Origin-Opener-Policy"), "same-origin")
        add("root:COEP", response.headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
    for unit in range(1, 8):
        for path in (f"/unit{unit}/", f"/?unit={unit}"):
            with request(origin + path) as response:
                add(path + ":entry-status", response.status, 200)
        with request(origin + f"/unit{unit}") as response:
            add(
                f"unit{unit}:slash-redirect",
                urllib.parse.urlparse(response.url).path,
                f"/unit{unit}/",
            )
    for name in ("shinylive/pyodide/pyodide.asm.wasm", "assets/v1/images/SVHN-train.npz"):
        if name not in manifest["files"]:
            continue
        with request(origin + "/" + name, headers={"Range": "bytes=0-1023"}) as response:
            add(name + ":range-status", response.status, 206)
            add(name + ":range-size", len(response.read()), 1024)
            add(
                name + ":content-range",
                response.headers.get("Content-Range"),
                f"bytes 0-1023/{manifest['files'][name]['bytes']}",
            )
    report = dict(
        status="fail" if failures else "pass",
        origin=origin,
        build_id=manifest["build_id"],
        checks=checks,
        failures=failures,
        scope="HTTPS GET/HEAD exact file bytes, MIME, ranges, cache headers, redirects and direct entry points. This does not certify browser workloads or numerical/pixel parity.",
        timestamp=time.time(),
    )
    if args.report.exists():
        raise SystemExit("Preserve existing evidence; choose a new report path")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], len(checks), "assertions")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
