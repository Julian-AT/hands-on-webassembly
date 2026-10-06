"""Consume real application responses in installed Chrome/Edge on an isolated origin."""

from repository import EVIDENCE

from repository import ROOT, CACHE, SCRIPTS
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from playwright.sync_api import sync_playwright
from common import ROOT, PROOF, sha, write_json
from browser_startup import application_ready


def site_manifest(site):
    return {str(p.relative_to(site)): sha(p) for p in sorted(site.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=CACHE / "startup-html-site")
    parser.add_argument("--browser", choices=["chrome", "edge"], required=True)
    parser.add_argument("--port", type=int, default=8048)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()
    inputs = site_manifest(args.site)
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    output = EVIDENCE / f"html-proxy-{args.browser}-{args.label}.json"
    result = dict(
        status="running",
        browser=args.browser,
        distribution="installed",
        artifact=str(args.site.resolve().relative_to(ROOT)),
        executor="scripts/proof/browser_html_proxy.py",
        dependencies={
            "scripts/proof/browser_startup.py": sha(SCRIPTS / "browser_startup.py"),
            "scripts/proof/common.py": sha(SCRIPTS / "common.py"),
        },
        provenance=dict(fingerprint=fingerprint, scope="isolated-artifact"),
        inputs=inputs,
        executor_sha256=sha(Path(__file__)),
        cases={},
        scope="All seven connected applications and complete real proxy response consumption; no upgrade reliability claim.",
    )
    server = subprocess.Popen(
        ["node", str(ROOT / "scripts/web/preview.mjs"), str(args.site)],
        env=dict(os.environ, PORT=str(args.port)),
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("Isolated static server failed to start")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/unit1/", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            result["version"] = browser.version
            try:
                for unit in range(1, 8):
                    context = browser.new_context()
                    page = context.new_page()
                    failures = []
                    page.on(
                        "requestfailed",
                        lambda request: failures.append(
                            dict(url=request.url, failure=request.failure)
                        ),
                    )
                    try:
                        page.goto(
                            f"http://127.0.0.1:{args.port}/unit{unit}/",
                            wait_until="domcontentloaded",
                        )
                        state = application_ready(page)
                        observed = page.evaluate("""async()=>{
                          const url=document.querySelector('#root iframe').contentWindow.location.href;
                          const results=await Promise.all(Array.from({length:3},async()=>{
                            const response=await fetch(url,{cache:'no-store'});
                            const bytes=new Uint8Array(await response.arrayBuffer());
                            const body=new TextDecoder('utf-8',{fatal:true}).decode(bytes);
                            return {status:response.status,headers:Object.fromEntries(response.headers),
                              delivered_bytes:bytes.length,body,sha256:Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),
                                byte=>byte.toString(16).padStart(2,'0')).join('')};
                          }));
                          return {url,responses:results};
                        }""")
                        for response in observed["responses"]:
                            assert response["status"] == 200
                            assert response["delivered_bytes"] > 0
                            for header in (
                                "content-length",
                                "content-range",
                                "accept-ranges",
                                "etag",
                                "content-md5",
                                "digest",
                                "content-digest",
                                "repr-digest",
                            ):
                                assert header not in response["headers"], header
                            assert response["body"].count("shinylive-inject-socket.js") == 1
                            assert "</head>" in response["body"]
                            del response["body"]
                        assert len({r["sha256"] for r in observed["responses"]}) == 1
                        assert not failures, failures
                        result["cases"][f"unit{unit}"] = dict(
                            status="pass",
                            application=state,
                            assertion=dict(
                                kind="workflow",
                                expected=dict(
                                    connected=True,
                                    completed_responses=3,
                                    invalid_headers=False,
                                    insertion_count=1,
                                    truncated_requests=0,
                                ),
                                observed=dict(
                                    connected=state["connected"],
                                    completed_responses=len(observed["responses"]),
                                    invalid_headers=False,
                                    insertion_count=1,
                                    truncated_requests=len(failures),
                                ),
                                matched=True,
                            ),
                            response=observed,
                        )
                    except Exception as error:
                        result["cases"][f"unit{unit}"] = dict(
                            status="fail", error=str(error), requests_failed=failures
                        )
                    finally:
                        context.close()
                    write_json(output, result)
                    print(
                        args.browser,
                        f"unit{unit}",
                        result["cases"][f"unit{unit}"]["status"],
                        flush=True,
                    )
            finally:
                browser.close()
        result["status"] = (
            "pass"
            if len(result["cases"]) == 7
            and all(c["status"] == "pass" for c in result["cases"].values())
            else "fail"
        )
        if inputs != site_manifest(args.site):
            result.update(observed_status=result["status"], status="stale")
    except Exception as error:
        result.update(status="error", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        write_json(output, result)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
