"""Installed-browser cancellation, worker disposal and owned-error recovery."""

from repository import EVIDENCE

from repository import ROOT, SCRIPTS
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright, expect
from common import PROOF, ROOT, sha, write_json
from browser_html_proxy import site_manifest
from owned_preview import start as start_preview
from host_conditions import snapshot as power_snapshot, apply as apply_power_conditions


async def run(args, result, output):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome" if args.browser == "chrome" else "msedge"
        )
        result["version"] = browser.version
        context = await browser.new_context()
        url = f"http://127.0.0.1:{args.port}/unit5/"

        def workers(page):
            return [w for w in page.workers if "course-training-worker.js" in w.url]

        async def disposed(page):
            deadline = time.perf_counter() + 2
            while workers(page) and time.perf_counter() < deadline:
                await asyncio.sleep(0.01)
            if workers(page):
                raise AssertionError("Training worker survived two seconds")

        def retain(name, observed, expected=None, **details):
            expected = expected if expected is not None else {key: True for key in observed}
            if observed != expected:
                raise AssertionError((name, expected, observed))
            result["cases"][name] = dict(
                status="pass",
                assertion=dict(kind="workflow", expected=expected, observed=observed, matched=True),
                **details,
            )
            write_json(output, result)

        try:
            for seed in (0, 42, 123):
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded")
                app = page.frame_locator("iframe")
                await expect(app.locator("#mn_ds.shiny-bound-input")).to_be_attached(timeout=120000)
                await app.get_by_role("tab", name="Linear Classifier on MNIST", exact=True).click()

                async def acknowledged(name, value):
                    await app.locator("body").evaluate(
                        """(el,[name,value])=>new Promise((resolve,reject)=>{
                      const end=performance.now()+10000;
                      const poll=()=>{const values=window.Shiny.shinyapp.$inputValues;
                        const key=Object.keys(values).find(key=>key===name||key.startsWith(name+':'));
                        if(values[key]===value)return resolve();
                        if(performance.now()>end)return reject(new Error('Input '+name+' not acknowledged'));
                        setTimeout(poll,20)};poll();
                    })""",
                        [name, value],
                    )

                async def number(name, value):
                    await app.locator("#" + name).fill(str(value))
                    await app.locator("#" + name).press("Tab")
                    await acknowledged(name, value)

                async def choose(value):
                    await app.locator("#mn_ds").select_option(value)
                    await acknowledged("mn_ds", value)

                await number("mn_seed", seed)
                await number("mn_epochs", 100)
                await app.locator("#mn_train").click()
                await expect(app.locator("#mn_progress")).to_contain_text("Epoch", timeout=120000)
                owning = workers(page)
                if len(owning) != 1:
                    raise AssertionError("No single training worker")
                closed = []
                owning[0].on("close", lambda worker: closed.append(time.perf_counter()))
                start = time.perf_counter()
                await choose("fashion")
                await expect(app.locator("#mn_progress .progress-bar")).to_have_attribute(
                    "aria-valuenow", "0", timeout=1000
                )
                await expect(app.locator("#mn_progress")).not_to_contain_text("Epoch", timeout=1000)
                seconds = time.perf_counter() - start
                while not closed and time.perf_counter() - start < 2:
                    await asyncio.sleep(0.01)
                if not closed:
                    raise AssertionError("Abandoned worker survived replacement")
                await asyncio.sleep(0.1)
                progress = await app.locator("#mn_progress").inner_text()
                await app.locator("#mn_eval").click()
                await expect(app.locator("#mn_acc")).not_to_contain_text(
                    "Test accuracy", timeout=1000
                )
                retain(
                    f"seed-{seed}/replacement",
                    dict(
                        acknowledged_within_one_second=seconds <= 1,
                        worker_disposed=closed[0] - start <= 2,
                        obsolete_progress_absent="Epoch" not in progress,
                        obsolete_completion_absent="Training complete" not in progress,
                    ),
                    acknowledged_seconds=seconds,
                    disposal_seconds=closed[0] - start,
                )
                await number("mn_epochs", 1)
                await app.locator("#mn_train").click()
                await expect(app.locator("#mn_progress")).to_contain_text(
                    "Training complete", timeout=180000
                )
                await disposed(page)
                await app.locator("#mn_eval").click()
                await expect(app.locator("#mn_acc")).to_contain_text(
                    "Test accuracy on Fashion-MNIST", timeout=120000
                )
                retain(
                    f"seed-{seed}/retry",
                    dict(completed=True, evaluation_matches_dataset=True, worker_disposed=True),
                    metrics=await app.locator("#mn_acc").inner_text(),
                )
                if seed == 123:
                    pattern = "**/assets/v1/neural-runtime/manifest.json"

                    async def fail(route):
                        await route.fulfill(status=503, body=b"Injected unavailable runtime")

                    await context.route(pattern, fail)
                    await app.locator("#mn_train").click()
                    await expect(app.locator("#mn_progress")).to_contain_text(
                        "Training runtime could not load", timeout=60000
                    )
                    await disposed(page)
                    retain(
                        "owned-error",
                        dict(error_visible=True, worker_disposed=True),
                        message=await app.locator("#mn_progress").inner_text(),
                    )
                    await context.unroute(pattern, fail)
                    started, release = asyncio.Event(), asyncio.Event()

                    async def hold(route):
                        started.set()
                        await release.wait()
                        try:
                            await route.fulfill(status=503, body=b"Abandoned runtime failure")
                        except Exception:
                            pass

                    await context.route(pattern, hold)
                    await app.locator("#mn_train").click()
                    await asyncio.wait_for(started.wait(), 30)
                    old = workers(page)
                    if len(old) != 1:
                        raise AssertionError("No worker before stale-error test")
                    await choose("mnist")
                    release.set()
                    await context.unroute(pattern, hold)
                    await disposed(page)
                    await expect(app.locator("#mn_progress .progress-bar")).to_have_attribute(
                        "aria-valuenow", "0", timeout=1000
                    )
                    text = await app.locator("#mn_progress").inner_text()
                    retain(
                        "abandoned-error",
                        dict(
                            obsolete_error_absent="Error:" not in text,
                            obsolete_completion_absent="Training complete" not in text,
                            worker_disposed=True,
                        ),
                    )
                    await app.locator("#mn_train").click()
                    await expect(app.locator("#mn_progress")).to_contain_text(
                        "Training complete", timeout=180000
                    )
                    await disposed(page)
                    retain("error-retry", dict(completed=True, worker_disposed=True))
                    await number("mn_epochs", 100)
                    await app.locator("#mn_train").click()
                    await expect(app.locator("#mn_progress")).to_contain_text(
                        "Epoch", timeout=120000
                    )
                    old = list(page.workers)
                    closed = []
                    for worker in old:
                        worker.on("close", lambda worker: closed.append(time.perf_counter()))
                    start = time.perf_counter()
                    await page.goto(
                        url.replace("/unit5/", "/probes/"), wait_until="domcontentloaded"
                    )
                    while len(closed) < len(old) and time.perf_counter() - start < 2:
                        await asyncio.sleep(0.01)
                    retain(
                        "navigation",
                        dict(
                            all_workers_disposed=bool(old) and len(closed) == len(old),
                            disposal_deadline_met=bool(closed) and max(closed) - start <= 2,
                        ),
                        disposal_seconds=max(closed) - start if closed else None,
                    )
                await page.close()
                print(args.browser, seed, "pass", flush=True)
            result["status"] = "pass"
        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--browser", choices=("chrome", "edge"), required=True)
    parser.add_argument("--port", type=int, default=8126)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = EVIDENCE / f"unit5-owned-training-ui-{args.browser}-{args.label}.json"
    if output.exists():
        raise FileExistsError("Choose a fresh report label")
    inputs = site_manifest(args.site)
    helpers = (
        "browser_html_proxy.py",
        "common.py",
        "owned_preview.py",
        "host_conditions.py",
        "audit_host_sleep.py",
    )
    result = dict(
        status="running",
        cases={},
        browser=args.browser,
        distribution="installed",
        executor=str(Path(__file__).relative_to(ROOT)),
        executor_sha256=sha(Path(__file__)),
        dependencies={f"scripts/proof/{name}": sha(SCRIPTS / name) for name in helpers},
        artifact=str(args.site.resolve().relative_to(ROOT)),
        inputs=inputs,
        provenance=dict(
            scope="isolated-artifact",
            fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        ),
        scope="Bounded Unit 5 installed-browser worker ownership, cancellation, stale-error rejection and retry. No diagnostic application hooks. Full training trajectories and comprehensive release acceptance remain separate.",
    )
    server = None
    before = power_snapshot()
    try:
        server, result["preview_identity"] = start_preview(args.site, args.port)
        args.port = result["preview_identity"]["port"]
        asyncio.run(run(args, result, output))
    except Exception as error:
        result.update(status="fail", error=str(error))
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=10)
            (args.site / result["preview_identity"]["marker"]).unlink()
        apply_power_conditions(result, before, power_snapshot())
        if inputs != site_manifest(args.site):
            result.update(observed_status=result["status"], status="stale")
        write_json(output, result)
    print(args.browser, result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
