"""Uninstrumented, fail-fast consecutive startup/reload/update sequence."""

from repository import EVIDENCE

from repository import SITE
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from playwright.sync_api import sync_playwright
from common import ROOT, PROOF, sha, write_json
from browser_startup import application_ready
from browser_html_proxy import site_manifest
import hashlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["chrome", "edge"], required=True)
    parser.add_argument("--unit", type=int, choices=range(1, 8), required=True)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--site", type=Path, default=SITE)
    parser.add_argument("--port", type=int, default=8058)
    parser.add_argument("--label", default="html-integrated")
    parser.add_argument("--query-token", default="sequence-update")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("Positive cycle count required")
    inputs = site_manifest(args.site)
    stamp = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    result = dict(
        status="running",
        browser=args.browser,
        distribution="installed",
        provenance={"scope": "isolated-artifact", "fingerprint": stamp},
        inputs=inputs,
        executor_sha256=sha(Path(__file__)),
        cycle_evidence=[],
        cases={},
        scope="Uninstrumented consecutive startup/reload/query-update sequence. Stops on first failure; genuine content upgrades and recovery remain separate obligations.",
    )
    path = EVIDENCE / f"lifecycle-{args.browser}-unit{args.unit}-{args.label}.json"
    server = subprocess.Popen(
        ["node", str(ROOT / "scripts/web/preview.mjs"), str(args.site)],
        env=dict(os.environ, PORT=str(args.port)),
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("Isolated server failed to start")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/unit{args.unit}/", timeout=1
                ).close()
                break
            except OSError:
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            result["version"] = browser.version
            context = browser.new_context()
            try:
                for index in range(1, args.cycles + 1):
                    page = context.new_page()
                    record = {"index": index}
                    operation = "startup"
                    try:
                        page.goto(
                            f"http://127.0.0.1:{args.port}/unit{args.unit}/",
                            wait_until="domcontentloaded",
                        )
                        record["startup"] = dict(status="pass", application=application_ready(page))
                        operation = "reload"
                        page.reload(wait_until="domcontentloaded")
                        record["reload"] = dict(status="pass", application=application_ready(page))
                        operation = "service-worker-update"
                        page.evaluate(
                            """async ({token,index})=>{
                          const target=new URL('../shinylive-sw.js?'+token+'='+index,location.href).href;
                          await navigator.serviceWorker.register(target,{type:'module',updateViaCache:'none'});
                          const deadline=performance.now()+30000;
                          while(navigator.serviceWorker.controller?.scriptURL!==target){
                            if(performance.now()>deadline)throw new Error('Replacement did not activate');
                            await new Promise(resolve=>setTimeout(resolve,25));
                          }
                        }""",
                            dict(
                                token=args.query_token,
                                index=1 if args.query_token == "regression-update" else index,
                            ),
                        )
                        page.reload(wait_until="domcontentloaded")
                        record["service-worker-update"] = dict(
                            status="pass", application=application_ready(page)
                        )
                    except Exception as error:
                        record[operation] = dict(status="fail", error=str(error))
                        # Diagnostics are added only after failure, never during an accepting sequence.
                        try:
                            record["failure_state"] = page.evaluate("""async()=>({
                              controller:navigator.serviceWorker.controller?.scriptURL,
                              workers:(await navigator.serviceWorker.getRegistrations()).map(r=>({
                                active:r.active?.scriptURL,active_state:r.active?.state,
                                waiting:r.waiting?.scriptURL,waiting_state:r.waiting?.state,
                                installing:r.installing?.scriptURL,installing_state:r.installing?.state})),
                              text:document.body.innerText})""")
                        except Exception as diagnostic_error:
                            record["diagnostic_error"] = str(diagnostic_error)
                    result["cycle_evidence"].append(record)
                    for operation in ("startup", "reload", "service-worker-update"):
                        if operation in record:
                            result["cases"][f"cycle-{index}/{operation}"] = record[operation]
                    write_json(path, result)
                    page.close()
                    passed = all(
                        record.get(op, {}).get("status") == "pass"
                        for op in ("startup", "reload", "service-worker-update")
                    )
                    print(args.browser, args.unit, index, "pass" if passed else "fail", flush=True)
                    if not passed:
                        break
                result["status"] = (
                    "pass"
                    if len(result["cycle_evidence"]) == args.cycles
                    and all(c["status"] == "pass" for c in result["cases"].values())
                    else "fail"
                )
            finally:
                browser.close()
        if inputs != site_manifest(args.site):
            result.update(observed_status=result["status"], status="stale")
    except Exception as error:
        result.update(status="error", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        write_json(path, result)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
