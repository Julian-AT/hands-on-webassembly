"""Bounded installed-browser Reset/worker-ownership verification for Unit 6."""

from repository import EVIDENCE

from repository import ROOT, SCRIPTS
import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
import urllib.request
from playwright.async_api import async_playwright, expect
from common import ROOT, PROOF, sha, write_json
from browser_html_proxy import site_manifest


async def run(args, result, output):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome" if args.browser == "chrome" else "msedge"
        )
        result["version"] = browser.version
        try:
            for seed in (0, 42, 123):
                page = await browser.new_page()
                console = []
                page.on(
                    "console",
                    lambda message: console.append(dict(type=message.type, text=message.text)),
                )
                await page.goto(
                    f"http://127.0.0.1:{args.port}/unit6/", wait_until="domcontentloaded"
                )
                app = page.frame_locator("iframe")
                await expect(app.locator("#dataset.shiny-bound-input")).to_be_attached(
                    timeout=120000
                )

                async def acknowledged(name, value):
                    await app.locator("body").evaluate(
                        """(el,[name,value])=>new Promise((resolve,reject)=>{
                      const end=performance.now()+10000;
                      const poll=()=>{const values=window.Shiny.shinyapp.$inputValues;
                        const key=Object.keys(values).find(key=>key===name||key.startsWith(name+':'));
                        if(values[key]===value)return resolve();
                        if(performance.now()>end)return reject(new Error('Input '+name+' was not acknowledged'));
                        setTimeout(poll,20)};poll();
                    })""",
                        [name, value],
                    )

                async def slider(name, value):
                    await app.locator("#" + name).evaluate(
                        """(el,value)=>{
                        window.jQuery(el).data('ionRangeSlider').update({from:value});
                        window.jQuery(el).trigger('change');
                    }""",
                        value,
                    )
                    await acknowledged(name, value)

                await app.get_by_role("tab", name="FNN: Data", exact=True).click()
                await app.locator("#dataset").select_option("toy_reg")
                await acknowledged("dataset", "toy_reg")
                await slider("n_pairs", 2000)
                await app.locator("#load").click()
                await expect(app.locator("#data_info")).to_contain_text("N=2000", timeout=30000)
                await app.get_by_role("tab", name="FNN:Architecture", exact=True).click()
                await app.locator("#preset").select_option("Toy Regression – 2×Hidden (ReLU)")
                await app.locator("#load_preset").click()
                await expect(app.locator("#arch_text")).to_have_value(
                    re.compile('"type": "relu"'), timeout=30000
                )
                architecture = await app.locator("#arch_text").input_value()
                await acknowledged("arch_text", architecture)
                await app.locator("#apply_arch").click()
                await expect(app.locator("#model_summary")).to_contain_text(
                    "Total parameters", timeout=30000
                )
                await app.get_by_role("tab", name="FNN: Training", exact=True).click()
                await app.locator("#train_seed").fill(str(seed))
                await app.locator("#train_seed").press("Tab")
                await acknowledged("train_seed", seed)
                await slider("epochs", 100)
                await app.locator("#train").click()
                await expect(app.locator("#train_progress")).to_contain_text("Epoch", timeout=60000)
                before = await app.locator("#train_progress").inner_text()
                workers = [w for w in page.workers if "course-training-worker.js" in w.url]
                if len(workers) != 1:
                    raise AssertionError(
                        f"Expected one owning training worker, observed {len(workers)}"
                    )
                disposed = []
                workers[0].on("close", lambda worker: disposed.append(time.perf_counter()))
                start = time.perf_counter()
                await app.locator("#reset").click()
                await expect(app.locator("#train_progress .progress-bar")).to_have_attribute(
                    "aria-valuenow", "0", timeout=1000
                )
                await expect(app.locator("#train_progress")).not_to_contain_text(
                    "Epoch", timeout=1000
                )
                acknowledged_seconds = time.perf_counter() - start
                if acknowledged_seconds > 1:
                    raise AssertionError(f"Reset acknowledgement took {acknowledged_seconds}s")
                while not disposed and time.perf_counter() - start < 2:
                    await asyncio.sleep(0.01)
                if not disposed:
                    raise AssertionError("Owning training worker survived two seconds")
                await asyncio.sleep(0.1)
                after = await app.locator("#train_progress").inner_text()
                info = await app.locator("#best_model_info").inner_text()
                observed = dict(
                    reset_acknowledged=acknowledged_seconds <= 1,
                    worker_disposed=disposed[0] - start <= 2,
                    obsolete_progress_absent="Epoch" not in after
                    and "Training complete" not in after,
                    obsolete_model_absent="not set" in info,
                )
                expected = {key: True for key in observed}
                if observed != expected:
                    raise AssertionError(observed)
                result["cases"][f"seed-{seed}/reset"] = dict(
                    status="pass",
                    before=before,
                    after=after,
                    acknowledged_seconds=acknowledged_seconds,
                    disposal_seconds=disposed[0] - start,
                    assertion=dict(
                        kind="workflow", expected=expected, observed=observed, matched=True
                    ),
                    console=console,
                )
                write_json(output, result)
                await slider("epochs", 2)
                await app.locator("#train").click()
                await expect(app.locator("#train_progress")).to_contain_text(
                    "Training complete", timeout=60000
                )
                await expect(app.locator("#best_model_info")).to_contain_text(
                    "Using last epoch model", timeout=30000
                )
                # The old 100-epoch context must stay gone after replacement completes.
                current = [w for w in page.workers if "course-training-worker.js" in w.url]
                deadline = time.perf_counter() + 2
                while current and time.perf_counter() < deadline:
                    await asyncio.sleep(0.01)
                    current = [w for w in page.workers if "course-training-worker.js" in w.url]
                if current:
                    raise AssertionError("Completed task retained its worker")
                result["cases"][f"seed-{seed}/retry"] = dict(
                    status="pass",
                    assertion=dict(
                        kind="workflow",
                        expected=dict(complete=True, owning_workers=0),
                        observed=dict(complete=True, owning_workers=len(current)),
                        matched=True,
                    ),
                )
                write_json(output, result)
                await page.close()
                print(args.browser, seed, "reset and retry pass", flush=True)
            result["status"] = "pass"
        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--browser", choices=("chrome", "edge"), required=True)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = EVIDENCE / f"unit6-owned-training-ui-{args.browser}-{args.label}.json"
    if output.exists():
        raise FileExistsError("Retain previous evidence; choose another label")
    inputs = site_manifest(args.site)
    result = dict(
        status="running",
        cases={},
        browser=args.browser,
        distribution="installed",
        inputs=inputs,
        artifact=str(args.site.resolve().relative_to(ROOT)),
        executor="scripts/proof/verify_unit6_owned_training_ui.py",
        executor_sha256=sha(Path(__file__)),
        dependencies={
            "scripts/proof/browser_html_proxy.py": sha(SCRIPTS / "browser_html_proxy.py"),
            "scripts/proof/common.py": sha(SCRIPTS / "common.py"),
        },
        provenance=dict(
            scope="isolated-artifact",
            fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        ),
        scope="Bounded Unit 6 worker Reset and subsequent training at seeds 0/42/123. Full numerical, image-training and comprehensive control matrices remain required.",
    )
    server = subprocess.Popen(
        ["node", str(ROOT / "scripts/web/preview.mjs"), str(args.site)],
        env=dict(os.environ, PORT=str(args.port)),
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("Candidate server exited")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/unit6/", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        asyncio.run(run(args, result, output))
    except Exception as error:
        result.update(status="fail", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        if inputs != site_manifest(args.site):
            result.update(observed_status=result["status"], status="stale")
        write_json(output, result)
    print(args.browser, result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
