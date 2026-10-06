"""Trace the actual Unit 7 cancellation message and browser request events."""

from repository import EVIDENCE

from repository import ROOT, CACHE
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import time
from playwright.async_api import async_playwright, expect
from common import PROOF, sha, write_json
from owned_preview import start
from storage_budget import require_space


async def run(site, identity, result, action):
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        result["version"] = browser.version
        try:
            context = await browser.new_context()
            page = await context.new_page()
            console = []
            failed = []
            result["console"] = console
            result["failed_requests"] = failed
            context.on(
                "console",
                lambda m: console.append(
                    dict(text=m.text, time=time.time(), worker=m.worker.url if m.worker else None)
                ),
            )
            context.on(
                "requestfailed",
                lambda r: failed.append(dict(url=r.url, time=time.time(), failure=r.failure)),
            )
            await page.goto(
                f"http://127.0.0.1:{identity['port']}/unit7/", wait_until="domcontentloaded"
            )
            app = page.frame_locator("iframe")
            await expect(app.locator("#variant.shiny-bound-input")).to_be_attached(timeout=120000)
            await app.get_by_role("tab", name="CNN: Data", exact=True).click()
            started, release = asyncio.Event(), asyncio.Event()

            async def hold(route):
                started.set()
                await release.wait()
                try:
                    await route.continue_()
                except Exception as error:
                    result["route_completion"] = str(error)

            await context.route("**/images/MNIST-train.npz", hold)
            await app.locator("#load_data").click()
            await asyncio.wait_for(started.wait(), 30)
            cdp = await context.new_cdp_session(page)
            result["workers_before"] = (await cdp.send("Target.getTargets"))["targetInfos"]
            if action == "reset":
                await app.get_by_role("tab", name="CNN: Training", exact=True).click()
            result["selection_started"] = time.time()
            if action == "reset":
                await app.locator("#reset_model").click()
            else:
                await app.locator("#variant").select_option("FashionMNIST")
            await asyncio.sleep(2)
            result["workers_after"] = (await cdp.send("Target.getTargets"))["targetInfos"]
            result["inputs"] = await app.locator("body").evaluate(
                "()=>window.Shiny.shinyapp.$inputValues"
            )
            result["selection_finished"] = time.time()
            release.set()
            await asyncio.sleep(0.25)
            result["status"] = "diagnostic"
            await context.close()
        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--action", choices=["selection", "reset"], default="selection")
    args = parser.parse_args()
    site = CACHE / f"image-cancellation-diagnostic-{args.label}"
    if site.exists():
        raise FileExistsError("Preserve attempts")
    require_space(64 * 1024**2)
    shutil.copytree(args.site, site, copy_function=os.link)
    worker = site / "shinylive/course-image-worker.js"
    source = worker.read_text()
    worker.unlink()
    source = source.replace(
        "self.postMessage({kind:'cancelled',tags:owner});self.close();",
        "console.log('course-image-diagnostic:'+JSON.stringify({kind:'cancelled',tags:owner,time:Date.now(),aborted:abort.signal.aborted}));self.postMessage({kind:'cancelled',tags:owner});self.close();",
    )
    source = source.replace(
        "owner=tags;abort=new AbortController();",
        "owner=tags;abort=new AbortController();console.log('course-image-diagnostic:'+JSON.stringify({kind:'started',tags,time:Date.now()}));",
    )
    worker.write_text(source)
    server, identity = start(site, 0)
    result = dict(
        status="running",
        browser="chrome",
        distribution="installed",
        preview_identity=identity,
        artifact=str(site.relative_to(ROOT)),
        executor_sha256=sha(Path(__file__)),
        worker_sha256=sha(worker),
        scope="Instrumented Unit 7 cancellation diagnosis; no acceptance or timing benchmark.",
    )
    output = EVIDENCE / f"image-cancellation-diagnostic-{args.label}.json"
    try:
        asyncio.run(run(site, identity, result, args.action))
    except Exception as error:
        result.update(status="fail", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        write_json(output, result)
    print(result["status"], result.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
