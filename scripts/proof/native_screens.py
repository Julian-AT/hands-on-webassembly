"""Capture original app landing screens and control defaults, with no edits."""

from repository import EVIDENCE
import os
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright
from common import ROOT, PROOF, UNITS, write_json


def main():
    results = {}
    screenshots = EVIDENCE / "screenshots"
    screenshots.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        for unit, folder in UNITS.items():
            port = 8110 + unit
            log = (EVIDENCE / f"native-unit{unit}.log").open("w")
            process = subprocess.Popen(
                [
                    str(__import__("pathlib").Path(sys.executable).parent / "shiny"),
                    "run",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "app.py",
                ],
                cwd=ROOT / folder,
                stdout=log,
                stderr=log,
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                url = f"http://127.0.0.1:{port}/"
                for _ in range(120):
                    try:
                        urllib.request.urlopen(url, timeout=1).close()
                        break
                    except OSError:
                        if process.poll() is not None:
                            raise RuntimeError("Native process exited; see log")
                        time.sleep(0.25)
                page.goto(url)
                page.locator("h2").first.wait_for(timeout=30000)
                page.screenshot(
                    path=str(screenshots / f"native-unit{unit}-landing.png"), full_page=True
                )
                results[str(unit)] = {
                    "status": "pass",
                    "title": page.locator("h2").first.inner_text(),
                    "tabs": page.get_by_role("tab").all_text_contents(),
                    "defaults": page.locator("input[id],select[id],textarea[id]").evaluate_all(
                        "els => els.map(e => ({id:e.id, type:e.type, value:e.value, checked:e.checked, choices:e.tagName === 'SELECT' ? [...e.options].map(o=>({value:o.value,text:o.text})) : undefined}))"
                    ),
                    "scope": "landing screen only; not every exercise workflow",
                }
            except Exception as e:
                results[str(unit)] = {"status": "fail", "error": str(e)}
            finally:
                page.close()
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            print("native landing", unit, results[str(unit)]["status"], flush=True)
        browser.close()
    write_json(EVIDENCE / "native-screens.json", results)


if __name__ == "__main__":
    main()
