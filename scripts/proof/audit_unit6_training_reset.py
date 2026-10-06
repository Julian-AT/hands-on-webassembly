"""Observe native Reset acknowledgement during an original training handler."""

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
from common import PROOF, sha, write_json, archive_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=8116)
    args = parser.parse_args()
    reference = REFERENCE / "unit6"
    output = EVIDENCE / f"native-unit6-reset-audit-{args.label}.json"
    if output.exists():
        raise FileExistsError("Retain previous diagnostic; choose another label")
    log = EVIDENCE / f"native-unit6-reset-audit-{args.label}.log"
    archive_file(log)
    result = dict(
        status="running",
        cases={},
        browser="chrome",
        distribution="installed",
        executor="scripts/proof/audit_unit6_training_reset.py",
        executor_sha256=sha(Path(__file__)),
        inputs={
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                reference / "app.py",
                reference / "u6_utils.py",
                REQUIREMENTS / "native.lock",
            )
        },
        scope="Native Unit 6 original regular asynchronous training effect; Reset latency and obsolete training continuation. Defect audit only, not release parity.",
    )
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
            stdout=stream,
            stderr=stream,
            env=dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1"),
        )
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError("Native process exited")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{args.port}/", timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.25)
            with sync_playwright() as p:
                browser = p.chromium.launch(channel="chrome")
                result["version"] = browser.version
                for seed in (0, 42, 123):
                    page = browser.new_page()
                    page.goto(f"http://127.0.0.1:{args.port}/")
                    expect(page.locator("#dataset.shiny-bound-input")).to_be_attached(
                        timeout=120000
                    )

                    def acknowledged(name, value):
                        page.locator("body").evaluate(
                            """(el,[name,value])=>new Promise((resolve,reject)=>{
                            const deadline=performance.now()+10000;
                            const poll=()=>{const v=window.Shiny.shinyapp.$inputValues;
                              const key=Object.keys(v).find(key=>key===name||key.startsWith(name+':'));
                              if(v[key]===value)return resolve();
                              if(performance.now()>deadline)return reject(new Error('Input '+name+' did not reach '+value));
                              setTimeout(poll,20)};poll();
                        })""",
                            [name, value],
                        )

                    page.get_by_role("tab", name="FNN: Data", exact=True).click()
                    page.locator("#dataset").select_option("toy_reg")
                    acknowledged("dataset", "toy_reg")
                    page.locator("#n_pairs").evaluate(
                        "(el)=>{window.jQuery(el).data('ionRangeSlider').update({from:2000});window.jQuery(el).trigger('change');}"
                    )
                    acknowledged("n_pairs", 2000)
                    page.locator("#load").click()
                    expect(page.locator("#data_info")).to_contain_text("N=2000", timeout=30000)
                    page.get_by_role("tab", name="FNN:Architecture", exact=True).click()
                    page.locator("#preset").select_option("Toy Regression – 2×Hidden (ReLU)")
                    page.locator("#load_preset").click()
                    expect(page.locator("#model_summary")).to_contain_text(
                        "Total parameters", timeout=30000
                    )
                    page.locator("#apply_arch").click()
                    page.get_by_role("tab", name="FNN: Training", exact=True).click()
                    page.locator("#train_seed").fill(str(seed))
                    page.locator("#train_seed").press("Tab")
                    page.locator("#epochs").evaluate(
                        "(el)=>{window.jQuery(el).data('ionRangeSlider').update({from:100});window.jQuery(el).trigger('change');}"
                    )
                    acknowledged("train_seed", seed)
                    acknowledged("epochs", 100)
                    page.locator("#train").click()
                    expect(page.locator("#train_progress")).to_contain_text("Epoch", timeout=30000)
                    before = page.locator("#train_progress").inner_text()
                    start = time.perf_counter()
                    page.locator("#reset").click()
                    acknowledged = False
                    failure = None
                    try:
                        expect(page.locator("#train_progress .progress-bar")).to_have_attribute(
                            "aria-valuenow", "0", timeout=1000
                        )
                        expect(page.locator("#train_progress")).not_to_contain_text(
                            "Epoch", timeout=1
                        )
                        acknowledged = True
                    except Exception as error:
                        failure = str(error)
                    seconds = time.perf_counter() - start
                    after = page.locator("#train_progress").inner_text()
                    result["cases"][f"seed-{seed}"] = dict(
                        status="pass",
                        reset_acknowledged_within_one_second=acknowledged,
                        seconds=seconds,
                        before=before,
                        after=after,
                        observation_error=failure,
                        defect_reproduced=not acknowledged
                        and ("Epoch" in after or "Training complete" in after),
                    )
                    write_json(output, result)
                    print(
                        seed,
                        "reset acknowledged" if acknowledged else "reset delayed",
                        seconds,
                        flush=True,
                    )
                    page.close()
                browser.close()
            result["defect_reproduced"] = all(
                case["defect_reproduced"] for case in result["cases"].values()
            )
            result["status"] = "pass"
        except Exception as error:
            result.update(status="fail", error=str(error))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    result["log_sha256"] = sha(log)
    write_json(output, result)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
