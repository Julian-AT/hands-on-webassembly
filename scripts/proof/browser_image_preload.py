"""Real abort/corruption/retry checks for the isolated async image owner."""

from repository import CACHE, EVIDENCE, RUNTIME, SITE
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from playwright.sync_api import sync_playwright
from common import ROOT, PROOF, sha, write_json
from browser_html_proxy import site_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["chrome", "edge"], required=True)
    parser.add_argument("--port", type=int, default=8063)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    site = CACHE / f"image-preload-{args.browser}-{args.label}"
    if site.exists():
        raise FileExistsError("Retain prior candidates; choose a fresh label")
    shutil.copytree(SITE, site, copy_function=os.link)
    for name in ("image_data.py", "image_preload.py", "image-preload-probe.js"):
        destination = site / "shinylive" / name
        destination.unlink(missing_ok=True)
        shutil.copy2(RUNTIME / name, destination)
    inputs = site_manifest(site)
    result = dict(
        status="running",
        browser=args.browser,
        distribution="installed",
        executor_sha256=sha(Path(__file__)),
        inputs=inputs,
        provenance=dict(
            scope="isolated-artifact",
            fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        ),
        scope="Isolated owner prototype under real installed-browser Pyodide and fetch. Cancellation, checksums, complete compact data and repeated retry; application integration is not asserted.",
        cases={},
    )
    path = EVIDENCE / f"image-preload-{args.browser}-{args.label}.json"
    write_json(path, result)
    server = subprocess.Popen(
        ["node", str(ROOT / "scripts/web/preview.mjs"), str(site)],
        env=dict(os.environ, PORT=str(args.port)),
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("Isolated server failed to start")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/probes/", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            result["version"] = browser.version
            context = browser.new_context()
            failed = []
            context.on(
                "requestfailed",
                lambda request: failed.append(dict(url=request.url, failure=request.failure)),
            )

            def delayed(route):
                time.sleep(1)
                try:
                    route.fulfill(
                        status=200,
                        body=(site / "assets/v1/images/manifest.json").read_bytes(),
                        content_type="application/json",
                    )
                except Exception:
                    pass  # Cancellation has already closed the routed response.

            context.route("**/images/manifest.json?loading-probe=cancel", delayed)
            context.route(
                "**/images/MNIST-test.npz?loading-probe=corrupt",
                lambda route: route.fulfill(status=200, body=b"corrupted test split"),
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{args.port}/probes/", wait_until="domcontentloaded")
            config = dict(
                files={
                    name: sha(site / "shinylive" / name)
                    for name in ("image_data.py", "image_preload.py")
                },
                manifest_sha256=sha(site / "assets/v1/images/manifest.json"),
                build_id=json.loads((EVIDENCE / "runtime-version.json").read_text())["version"],
            )
            page.evaluate(
                """config=>{
              window.preloadResult=null;
              const worker=new Worker('../shinylive/image-preload-probe.js');
              worker.onmessage=({data})=>{window.preloadResult=data;worker.terminate();};
              worker.onerror=event=>{window.preloadResult={error:event.message};worker.terminate();};
              worker.postMessage(config);
            }""",
                config,
            )
            page.wait_for_function("()=>window.preloadResult", timeout=180000)
            observed = page.evaluate("window.preloadResult")
            if observed.get("error"):
                raise RuntimeError(observed["error"])
            result["cases"] = observed["cases"]
            deadline = time.perf_counter() + 2
            while page.workers and time.perf_counter() < deadline:
                page.wait_for_timeout(20)
            if page.workers:
                raise AssertionError("Abandoned preload worker survived disposal deadline")
            cancelled = [r for r in failed if "loading-probe=cancel" in r["url"]]
            if not cancelled:
                raise AssertionError("Cancellation did not abort the actual network request")
            result["cases"]["network-abort"] = dict(status="pass", requests=cancelled)
            result["cases"]["worker-disposal"] = dict(
                status="pass", workers_remaining=len(page.workers)
            )
            unexpected = [r for r in failed if r not in cancelled]
            if unexpected:
                raise AssertionError(unexpected)
            result["status"] = (
                "pass" if all(c["status"] == "pass" for c in result["cases"].values()) else "fail"
            )
            browser.close()
    except Exception as error:
        result.update(status="fail", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        write_json(path, result)
    print(args.browser, result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
