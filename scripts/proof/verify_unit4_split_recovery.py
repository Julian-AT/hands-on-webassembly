"""Independent installed-browser reproduction of Unit 4's sticky slider maximum."""

from repository import EVIDENCE

from repository import CACHE
import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
import urllib.request
from playwright.sync_api import sync_playwright
from common import ROOT, PROOF, sha, write_json
from storage_budget import require_space

OLD = """        max_val = 100 - input.split() - 5
        if input.val_split() > max_val:
            ui.update_slider("val_split", max=max_val, value=max_val)"""
NEW = """        max_val = min(40, 100 - input.split() - 5)
        ui.update_slider("val_split", max=max_val, value=min(input.val_split(), max_val))"""


def observe(browser, candidate, log_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                str(Path(sys.executable).parent / "shiny"),
                "run",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "app.py",
            ],
            cwd=candidate,
            stdout=log,
            stderr=log,
        )
        page = browser.new_page()
        try:
            url = f"http://127.0.0.1:{port}/"
            for _ in range(200):
                if process.poll() is not None:
                    raise RuntimeError("Native process exited")
                try:
                    urllib.request.urlopen(url, timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.05)
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("()=>window.Shiny?.shinyapp?.isConnected()", timeout=90000)
            page.get_by_role("tab", name="Data Settings & PCA", exact=True).click()

            def set_train(value):
                page.locator("#split").evaluate(
                    """(el,value)=>{
                  window.jQuery(el).data('ionRangeSlider').update({from:value});
                  window.jQuery(el).trigger('change');
                }""",
                    value,
                )
                page.wait_for_function(
                    '(value)=>document.getElementById("split_overview").textContent.includes(`Train: ${value.toFixed(1)}%`)',
                    arg=value,
                )

            set_train(90)
            page.wait_for_function(
                '()=>window.jQuery("#val_split").data("ionRangeSlider").options.max===5'
            )
            at_90 = page.evaluate(
                '()=>({max:window.jQuery("#val_split").data("ionRangeSlider").options.max,value:window.Shiny.shinyapp.$inputValues.val_split})'
            )
            set_train(70)
            # Wait for a full reactive round trip, then record the actual bound.
            page.wait_for_function('()=>!document.documentElement.classList.contains("shiny-busy")')
            page.wait_for_timeout(300)
            at_70 = page.evaluate(
                '()=>({max:window.jQuery("#val_split").data("ionRangeSlider").options.max,value:window.Shiny.shinyapp.$inputValues.val_split})'
            )
            page.locator("#val_split").evaluate("""el=>{
              window.jQuery(el).data('ionRangeSlider').update({from:15});
              window.jQuery(el).trigger('change');
            }""")
            page.wait_for_timeout(300)
            attempted = page.evaluate("()=>window.Shiny.shinyapp.$inputValues.val_split")
            return dict(
                at_90=at_90, at_70=at_70, requested_validation=15, observed_validation=attempted
            )
        finally:
            page.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = CACHE / f"unit4-split-recovery-{args.label}"
    if root.exists():
        raise FileExistsError("Choose a fresh label")
    require_space(128 * 1024**2)
    original = ROOT / "assignments/4"
    before, after = root / "original", root / "candidate"
    shutil.copytree(original, before)
    shutil.copytree(original, after)
    path = after / "app.py"
    source = path.read_text()
    if source.count(OLD) != 1:
        raise ValueError("Original split contract changed")
    path.write_text(source.replace(OLD, NEW))
    output = EVIDENCE / f"unit4-split-recovery-{args.label}.json"
    report = dict(
        status="running",
        cases={},
        browser="chrome",
        distribution="installed",
        executor=str(Path(__file__).relative_to(ROOT)),
        executor_sha256=sha(Path(__file__)),
        inputs={
            str(p.relative_to(ROOT)): sha(p)
            for p in (original / "app.py", before / "app.py", after / "app.py")
        },
        scope="Original native defect and isolated correction; no browser deployment or comprehensive Unit 4 certification.",
    )
    write_json(output, report)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            report["version"] = browser.version
            try:
                for name, candidate in [("original", before), ("candidate", after)]:
                    actual = observe(
                        browser, candidate, EVIDENCE / f"unit4-split-{name}-{args.label}.log"
                    )
                    expected = dict(
                        at_90=dict(max=5, value=5),
                        at_70=dict(max=5 if name == "original" else 25, value=5),
                        requested_validation=15,
                        observed_validation=5 if name == "original" else 15,
                    )
                    matched = actual == expected
                    report["cases"][name] = dict(
                        status="pass" if matched else "fail",
                        assertion=dict(
                            kind="workflow", expected=expected, observed=actual, matched=matched
                        ),
                    )
                    write_json(output, report)
                    print(name, actual, flush=True)
            finally:
                browser.close()
        report["status"] = (
            "pass" if all(c["status"] == "pass" for c in report["cases"].values()) else "fail"
        )
    except Exception as error:
        report.update(status="fail", error=str(error))
    write_json(output, report)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
