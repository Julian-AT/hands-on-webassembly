"""Exercise Unit 5 prepared previews and training through its original controls."""

from repository import EVIDENCE

from repository import ROOT, SCRIPTS
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright, expect
from browser_html_proxy import site_manifest
from browser_unit5 import unit5
from common import ROOT, PROOF, sha, write_json
from owned_preview import start as start_preview
from host_conditions import snapshot as power_snapshot, apply as apply_power_conditions
from image_cancellation import workers as image_workers, acknowledged as cancellation_acknowledged


async def sequences(args, result, retain):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome" if args.browser == "chrome" else "msedge"
        )
        context = await browser.new_context()
        page = await context.new_page()
        console = []
        result["console"] = console
        page.on("console", lambda message: console.append(message.text))
        url = f"http://127.0.0.1:{args.port}/unit5/"
        await page.goto(url, wait_until="domcontentloaded")
        app = page.frame_locator("iframe")
        await expect(app.locator("#mn_ds.shiny-bound-input")).to_be_attached(timeout=120000)
        await app.get_by_role("tab", name="Linear Classifier on MNIST", exact=True).click()

        async def acknowledged(name, value):
            await app.locator("body").evaluate(
                """(el,[name,value])=>new Promise((resolve,reject)=>{
                const deadline=performance.now()+10000;
                const poll=()=>{
                  const values=window.Shiny.shinyapp.$inputValues;
                  const key=Object.keys(values).find(key=>key===name||key.startsWith(name+':'));
                  if(values[key]===value)return resolve();
                  if(performance.now()>deadline)return reject(new Error('Input '+name+' did not reach '+value));
                  setTimeout(poll,20);
                };poll();
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

        started, release = asyncio.Event(), asyncio.Event()

        async def hold(route):
            started.set()
            await release.wait()
            try:
                await route.continue_()
            except Exception:
                pass

        async def hold_train():
            started.clear()
            release.clear()
            await context.route("**/images/MNIST-train.npz", hold)

        async def release_train():
            release.set()
            await context.unroute("**/images/MNIST-train.npz", hold)

        await number("mn_epochs", 1)
        await hold_train()
        await app.locator("#mn_train").click()
        await asyncio.wait_for(started.wait(), 30)
        failed = []
        context.on(
            "requestfailed",
            lambda request: failed.append((request.url, time.perf_counter(), request.failure)),
        )
        owners = await image_workers(context, page)
        start = time.perf_counter()
        await choose("fashion")
        cancellation = await cancellation_acknowledged(
            context, page, owners, failed, "MNIST", start
        )
        retain("loading/cancel-on-selection", dict(status="pass", **cancellation))
        await release_train()
        await expect(app.locator("#mn_progress .progress-bar")).to_have_attribute(
            "aria-valuenow", "0", timeout=30000
        )
        await app.locator("#mn_train").click()
        await expect(app.locator("#mn_progress")).to_contain_text(
            "Training complete", timeout=180000
        )
        await app.locator("#mn_eval").click()
        await expect(app.locator("#mn_acc")).to_contain_text(
            "Test accuracy on Fashion-MNIST", timeout=120000
        )
        metrics = await app.locator("#mn_acc").inner_text()
        retain(
            "loading/retry-after-cancel",
            dict(
                status="pass",
                metrics=metrics,
                assertion=dict(
                    kind="workflow",
                    expected=True,
                    observed="Test accuracy on Fashion-MNIST" in metrics,
                    matched=True,
                ),
            ),
        )

        async def corrupt(route):
            await route.fulfill(status=200, body=b"corrupt test split")

        await choose("mnist")
        await context.route("**/images/MNIST-test.npz", corrupt)
        await app.locator("#mn_train").click()
        await expect(app.locator("#mn_progress")).to_contain_text("checksum", timeout=120000)
        error = await app.locator("#mn_progress").inner_text()
        retain(
            "loading/corrupt-pair",
            dict(
                status="pass",
                message=error,
                assertion=dict(
                    kind="recoverable-error",
                    expected=True,
                    observed="checksum" in error,
                    matched=True,
                ),
            ),
        )
        await context.unroute("**/images/MNIST-test.npz", corrupt)
        # Observe every delivered progress DOM update while the epoch input
        # changes during preparation. This is a task snapshot case, not a timing
        # benchmark or an uninstrumented lifecycle reliability sequence.
        await app.locator("#mn_progress").evaluate(r"""el=>{
            window.courseEpochs=[];
            new MutationObserver(()=>{const match=el.innerText.match(/Epoch (\d+)\/(\d+)/);if(match)window.courseEpochs.push(match.slice(1).map(Number));})
              .observe(el,{childList:true,subtree:true,characterData:true});
        }""")
        await hold_train()
        await app.locator("#mn_train").click()
        await asyncio.wait_for(started.wait(), 30)
        await number("mn_epochs", 2)
        await release_train()
        await expect(app.locator("#mn_progress")).to_contain_text(
            "Training complete", timeout=180000
        )
        epochs = await app.locator("body").evaluate("()=>window.courseEpochs")
        if not epochs or any(total != 1 or epoch != 1 for epoch, total in epochs):
            raise AssertionError(epochs)
        retain(
            "loading/training-input-snapshot",
            dict(
                status="pass",
                assertion=dict(
                    kind="workflow",
                    expected=[[1, 1]],
                    observed=sorted({tuple(item) for item in epochs}),
                    matched=True,
                ),
            ),
        )
        await app.locator("#mn_eval").click()
        await expect(app.locator("#mn_acc")).to_contain_text(
            "Test accuracy on MNIST", timeout=120000
        )
        metrics = await app.locator("#mn_acc").inner_text()
        retain(
            "loading/retry-after-corruption",
            dict(
                status="pass",
                metrics=metrics,
                assertion=dict(
                    kind="workflow",
                    expected=True,
                    observed="Test accuracy on MNIST" in metrics,
                    matched=True,
                ),
            ),
        )
        await hold_train()
        await app.locator("#mn_train").click()
        await asyncio.wait_for(started.wait(), 30)
        workers = list(page.workers)
        disposed = []
        if not workers:
            raise AssertionError("No owning application worker observed")
        for worker in workers:
            worker.on("close", lambda closed: disposed.append(time.perf_counter()))
        start = time.perf_counter()
        await page.goto(url.replace("/unit5/", "/probes/"), wait_until="domcontentloaded")
        deadline = start + 2
        while len(disposed) < len(workers) and time.perf_counter() < deadline:
            await asyncio.sleep(0.01)
        release.set()
        if len(disposed) != len(workers):
            raise AssertionError("Abandoned workers survived two seconds")
        seconds = max(disposed) - start
        retain(
            "loading/navigation-disposal",
            dict(
                status="pass",
                seconds=seconds,
                workers=len(workers),
                assertion=dict(
                    kind="workflow",
                    expected=True,
                    observed=seconds <= 2 and len(disposed) == len(workers),
                    matched=True,
                ),
            ),
        )
        await browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--browser", choices=("chrome", "edge"), required=True)
    parser.add_argument("--port", type=int, default=8072)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    inputs = site_manifest(args.site)
    output = EVIDENCE / f"unit5-loading-ui-{args.browser}-{args.label}.json"
    if output.exists():
        raise FileExistsError("Retain previous evidence; choose a fresh label")
    from hardware_contract import collect

    result = dict(
        status="running",
        browser=args.browser,
        distribution="installed",
        inputs=inputs,
        cases={},
        hardware=collect(),
        executor_sha256=sha(Path(__file__)),
        artifact=str(args.site.resolve().relative_to(ROOT)),
        executor="scripts/proof/browser_unit5_loading_ui.py",
        dependencies={
            f"scripts/proof/{name}": sha(SCRIPTS / name)
            for name in (
                "browser_unit5.py",
                "browser_html_proxy.py",
                "common.py",
                "owned_preview.py",
                "host_conditions.py",
                "audit_host_sleep.py",
                "image_cancellation.py",
                "hardware_contract.py",
            )
        },
        provenance=dict(
            scope="isolated-artifact",
            fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        ),
        scope="Unit 5 original bounded workflows including both previews and full training/evaluation; loading cancellation, corruption, retry, training input snapshot and navigation disposal. Disposable calculation workers and complete release parity remain separate obligations.",
    )

    def retain(name, case):
        result["cases"][name] = case
        write_json(output, result)

    server = None
    before = power_snapshot()
    try:
        server, result["preview_identity"] = start_preview(args.site, args.port)
        args.port = result["preview_identity"]["port"]
        url = f"http://127.0.0.1:{args.port}/unit5/"
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            result["version"] = browser.version
            original = unit5(
                browser.new_page(), url, f"candidate-{args.browser}-unit5-{args.label}"
            )
            result["cases"].update(original["cases"])
            write_json(output, result)
            browser.close()
            if any(case["status"] != "pass" for case in original["cases"].values()):
                raise AssertionError("Original bounded Unit 5 workflow failed")
        asyncio.run(sequences(args, result, retain))
        result["status"] = "pass"
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
