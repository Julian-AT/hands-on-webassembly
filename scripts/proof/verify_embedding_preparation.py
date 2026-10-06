"""Installed-browser candidate checks: real arrays, corruption, Retry and abort."""

from repository import EVIDENCE

from repository import ROOT, CACHE, SCRIPTS
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from playwright.sync_api import sync_playwright, expect
from common import ROOT, PROOF, sha, write_json
from owned_preview import start
from host_conditions import snapshot, apply
from browser_generation_sequence import site_manifest
from storage_budget import require_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = EVIDENCE / f"embedding-preparation-{args.browser}-{args.label}.json"
    if output.exists():
        raise FileExistsError("Use a fresh label")
    site = CACHE / f"embedding-check-{args.browser}-{args.label}"
    if site.exists():
        raise FileExistsError("Use a fresh label; preserve tested snapshots")
    require_space(64 * 1024**2)
    shutil.copytree(args.site, site, copy_function=os.link)
    before = snapshot()
    server, identity = start(site, 0)
    inputs = site_manifest(site)
    report = dict(
        status="running",
        browser=args.browser,
        distribution="installed",
        preview_identity=identity,
        executor=str(Path(__file__).relative_to(ROOT)),
        executor_sha256=sha(Path(__file__)),
        inputs=inputs,
        artifact=str(site.resolve().relative_to(ROOT)),
        provenance=dict(
            scope="isolated-artifact",
            fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        ),
        dependencies={
            str((SCRIPTS / n).relative_to(ROOT)): sha(SCRIPTS / n)
            for n in (
                "common.py",
                "owned_preview.py",
                "host_conditions.py",
                "audit_host_sleep.py",
                "browser_generation_sequence.py",
            )
        },
        cases={},
        scope="Isolated Unit 2 abortable preparation and recovery; no promotion or exhaustive assignment acceptance.",
    )
    write_json(output, report)

    def retain(name, expected, observed, **details):
        if expected != observed:
            raise AssertionError((name, expected, observed))
        report["cases"][name] = dict(
            status="pass",
            assertion=dict(kind="workflow", expected=expected, observed=observed, matched=True),
            **details,
        )
        write_json(output, report)
        print(args.browser, name, "pass", flush=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            report["version"] = browser.version
            context = browser.new_context()
            page = context.new_page()
            requests = []
            report["console"] = []
            page.on(
                "console",
                lambda message: (
                    report["console"].append(message.text) if message.type == "error" else None
                ),
            )
            page.on("request", lambda request: requests.append(request.url))
            page.goto(f"http://127.0.0.1:{identity['port']}/unit2/", wait_until="domcontentloaded")
            app = page.frame_locator("iframe")
            expect(app.locator("#run_embed.shiny-bound-input")).to_be_attached(timeout=120000)
            app.get_by_role("tab", name="Text Processing", exact=True).click()
            retain(
                "deferred-assets",
                dict(model_fetches=0),
                dict(model_fetches=sum("/embedding-runtime/" in u for u in requests)),
            )
            pattern = "**/assets/v1/embedding-runtime/vectors.npy"
            context.route(pattern, lambda route: route.fulfill(status=200, body=b"corrupted"))
            app.locator("#run_embed").click()
            app.get_by_role("tab", name="Word Embeddings", exact=True).click()
            expect(app.locator("#embed_df")).to_contain_text("checksum mismatch", timeout=60000)
            retain(
                "checksum-rejection",
                dict(rejected=True),
                dict(rejected="checksum mismatch" in app.locator("#embed_df").inner_text()),
            )
            context.unroute(pattern)
            app.locator("#run_embed").click()
            expect(app.locator("#embed_df")).to_contain_text("dog", timeout=60000)
            rows_present = "dog" in app.locator("#embed_df").inner_text()
            app.get_by_role("tab", name="Individual Words", exact=True).click()
            app.locator("#word_select").select_option("dog")
            expect(app.locator("#word_vector_size")).to_contain_text(
                "300 dimensions", timeout=30000
            )
            retain(
                "retry-complete-model",
                dict(dimensions=300, rows=True),
                dict(
                    dimensions=300
                    if "300 dimensions" in app.locator("#word_vector_size").inner_text()
                    else 0,
                    rows=rows_present,
                ),
            )
            # A fresh application context has no prepared arrays. Hold transport
            # so replacement must abort an actual in-flight model download.
            context.close()
            context = browser.new_context()
            page = context.new_page()
            pending = []
            failed = []
            context.route(pattern, lambda route: pending.append(route))
            page.on(
                "requestfailed",
                lambda request: failed.append(dict(url=request.url, failure=request.failure)),
            )
            page.goto(f"http://127.0.0.1:{identity['port']}/unit2/", wait_until="domcontentloaded")
            app = page.frame_locator("iframe")
            expect(app.locator("#run_embed.shiny-bound-input")).to_be_attached(timeout=120000)
            app.get_by_role("tab", name="Text Processing", exact=True).click()
            app.locator("#run_embed").click()
            deadline = time.monotonic() + 30
            while not pending and time.monotonic() < deadline:
                page.wait_for_timeout(25)
            if not pending:
                raise AssertionError("No actual embedding request was held")
            cdp = context.new_cdp_session(page)

            def workers():
                return [
                    t
                    for t in cdp.send("Target.getTargets")["targetInfos"]
                    if t["type"] == "worker" and "course-embedding-worker.js" in t["url"]
                ]

            owned = workers()
            if len(owned) != 1:
                raise AssertionError("Expected one owned embedding preparation worker")
            started = time.monotonic()
            app.locator("#word_list").fill("king,queen")
            app.locator("#word_list").press("Tab")
            while workers() and time.monotonic() - started < 2:
                page.wait_for_timeout(25)
            elapsed = time.monotonic() - started
            retain(
                "replacement-worker-disappearance",
                dict(workers=0, deadline_met=True),
                dict(workers=len(workers()), deadline_met=elapsed < 2),
                seconds=elapsed,
            )
            for route in pending:
                try:
                    route.abort("failed")
                except Exception:
                    pass
            context.unroute(pattern)
            app.locator("#run_embed").click()
            app.get_by_role("tab", name="Word Embeddings", exact=True).click()
            expect(app.locator("#embed_df")).to_contain_text("queen", timeout=60000)
            retain(
                "replacement-retry",
                dict(current_vocabulary=True),
                dict(current_vocabulary="queen" in app.locator("#embed_df").inner_text()),
                failed_requests=failed,
            )
            context.close()
            browser.close()
            report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        apply(report, before, snapshot())
        if inputs != site_manifest(site):
            report.update(observed_status=report["status"], status="stale")
        write_json(output, report)
    print(report["status"], report.get("error", ""), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
