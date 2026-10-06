"""Reproduce native dataset-replacement delay in the original Unit 5 trainer."""

from repository import EVIDENCE

from repository import ROOT, REFERENCE, REQUIREMENTS
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright, expect
from common import PROOF, sha, write_json
from host_conditions import snapshot as power_snapshot, apply as apply_power_conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=8118)
    parser.add_argument("--reference", type=Path, default=REFERENCE / "unit5")
    args = parser.parse_args()
    reference = args.reference.resolve()
    output = EVIDENCE / f"native-unit5-training-replacement-{args.label}.json"
    if output.exists():
        raise FileExistsError("Retain prior audits; choose a fresh label")
    result = dict(
        status="running",
        cases={},
        browser="chrome",
        distribution="installed",
        executor="scripts/proof/audit_unit5_training_replacement.py",
        executor_sha256=sha(Path(__file__)),
        inputs={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                reference / "app.py",
                reference / "u5_utils.py",
                reference / "inspection_rng.py",
                REQUIREMENTS / "native.lock",
            )
        },
        scope="Native original regular asynchronous Unit 5 training: dataset replacement acknowledgement and obsolete progress at seeds 0/42/123. Defect audit only; no arithmetic or release acceptance claim.",
    )
    before = power_snapshot()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            result["version"] = browser.version
            for seed in (0, 42, 123):
                log = EVIDENCE / f"native-unit5-training-replacement-{args.label}-seed{seed}.log"
                if log.exists():
                    raise FileExistsError("Retain audit logs")
                with log.open("w") as stream:
                    server = subprocess.Popen(
                        [
                            str(Path(sys.executable).parent / "shiny"),
                            "run",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(args.port),
                            "app.py",
                        ],
                        cwd=reference,
                        env=dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1"),
                        stdout=stream,
                        stderr=stream,
                    )
                    try:
                        for _ in range(120):
                            if server.poll() is not None:
                                raise RuntimeError("Native control server exited")
                            try:
                                urllib.request.urlopen(
                                    f"http://127.0.0.1:{args.port}/", timeout=1
                                ).close()
                                break
                            except OSError:
                                time.sleep(0.1)
                        page = browser.new_page()
                        page.goto(f"http://127.0.0.1:{args.port}/", wait_until="domcontentloaded")
                        expect(page.locator("#mn_ds.shiny-bound-input")).to_be_attached(
                            timeout=120000
                        )
                        page.get_by_role(
                            "tab", name="Linear Classifier on MNIST", exact=True
                        ).click()

                        def acknowledged(name, value):
                            page.locator("body").evaluate(
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

                        for name, value in [("mn_seed", seed), ("mn_epochs", 100)]:
                            page.locator("#" + name).fill(str(value))
                            page.locator("#" + name).press("Tab")
                            acknowledged(name, value)
                        page.locator("#mn_ds").select_option("mnist")
                        acknowledged("mn_ds", "mnist")
                        page.locator("#mn_train").click()
                        expect(page.locator("#mn_progress")).to_contain_text("Epoch", timeout=60000)
                        progress = page.locator("#mn_progress").inner_text()
                        start = time.perf_counter()
                        page.locator("#mn_ds").select_option("fashion")
                        acknowledged("mn_ds", "fashion")
                        reset = False
                        error = None
                        try:
                            # Both conditions must settle before one shared
                            # deadline. 0% can also be the first training batch.
                            deadline = start + 1
                            expect(page.locator("#mn_progress")).not_to_contain_text(
                                "Epoch",
                                timeout=max(1, int((deadline - time.perf_counter()) * 1000)),
                            )
                            expect(page.locator("#mn_progress .progress-bar")).to_have_attribute(
                                "aria-valuenow",
                                "0",
                                timeout=max(1, int((deadline - time.perf_counter()) * 1000)),
                            )
                            reset = time.perf_counter() <= deadline
                        except Exception as failure:
                            error = str(failure)
                        seconds = time.perf_counter() - start
                        after = page.locator("#mn_progress").inner_text()
                        defect = not reset and ("Epoch" in after or "Training complete" in after)
                        result["cases"][f"seed-{seed}"] = dict(
                            status="pass",
                            defect_reproduced=defect,
                            replacement_acknowledged_within_one_second=reset and seconds <= 1,
                            seconds=seconds,
                            before=progress,
                            after=after,
                            observation_error=error,
                        )
                        print(
                            seed,
                            "defect reproduced" if defect else "replacement acknowledged",
                            flush=True,
                        )
                        page.close()
                    finally:
                        server.terminate()
                        try:
                            server.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait()
                result["cases"][f"seed-{seed}"]["log_sha256"] = sha(log)
                write_json(output, result)
            browser.close()
        result["status"] = "pass"
        result["defect_reproduced"] = all(c["defect_reproduced"] for c in result["cases"].values())
    except Exception as error:
        result.update(status="fail", error=str(error))
    finally:
        apply_power_conditions(result, before, power_snapshot())
        write_json(output, result)
    print(result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
