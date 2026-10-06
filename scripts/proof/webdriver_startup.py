"""Installed Firefox/Safari lifecycle checks; never substitute bundled browsers.

Use the separate requirements-browser.lock environment. This bounded report is
additional evidence, not the exhaustive release lifecycle report.
"""

from repository import ROOT, EVIDENCE, REQUIREMENTS, SITE
import argparse
import hashlib
import json
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import selenium
from common import PROOF, sha, write_json

READY = """
const frame=document.querySelector('#root iframe'), doc=frame?.contentDocument;
const app=frame?.contentWindow?.Shiny?.shinyapp;
const loading=document.getElementById('course-startup');
return !!(doc?.querySelector('h2')?.getBoundingClientRect().width &&
 doc.querySelector('.shiny-bound-input') && app?.isConnected() && (!loading || loading.hidden));
"""
STATE = """
const frame=document.querySelector('#root iframe');
return {title:frame.contentDocument.querySelector('h2').textContent,
 document_url:frame.contentWindow.location.href,
 bound_inputs:frame.contentDocument.querySelectorAll('.shiny-bound-input').length,
 connected:frame.contentWindow.Shiny.shinyapp.isConnected(),
 controller:navigator.serviceWorker.controller?.scriptURL,
 transfers:performance.getEntriesByType('resource').map(r=>({url:r.name,bytes:r.transferSize}))};
"""


def inputs():
    paths = [
        Path(__file__),
        REQUIREMENTS / "browser.lock",
        PROOF / "source.lock.json",
        REQUIREMENTS / "native.lock",
        EVIDENCE / "reference-corrections.json",
    ]
    paths += [p for p in (SITE).rglob("*") if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["firefox", "safari"], required=True)
    parser.add_argument(
        "--units", type=int, nargs="+", choices=range(1, 8), default=list(range(1, 8))
    )
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--label", default="installed")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("Positive cycle count required")
    path = EVIDENCE / f"startup-actual-{args.browser}-{args.label}.json"
    result = dict(
        browser=args.browser,
        distribution="installed",
        status="running",
        cycles_requested=args.cycles,
        units=args.units,
        cases={},
        traces={},
        selenium_version=selenium.__version__,
        input_fingerprint=inputs(),
        scope="Sequential startup/reload/service-worker replacement; additional bounded lifecycle evidence.",
    )
    write_json(path, result)
    driver = None
    try:
        if args.browser == "firefox":
            options = webdriver.FirefoxOptions()
            options.binary_location = "/Applications/Firefox.app/Contents/MacOS/firefox"
            options.add_argument("-headless")
            driver = webdriver.Firefox(options=options)
        else:
            driver = webdriver.Safari()
        result["capabilities"] = driver.capabilities
        result["version"] = driver.capabilities["browserVersion"]
        driver.set_window_size(1440, 1000)
        driver.set_page_load_timeout(120)
        driver.set_script_timeout(45)
        wait = WebDriverWait(driver, 120, poll_frequency=0.1)
        for unit in args.units:
            for cycle in range(1, args.cycles + 1):
                key = f"unit{unit}/cycle-{cycle:02}"
                operation = "startup"
                try:
                    start = time.perf_counter()
                    # The initial cycle retains the cold profile. Subsequent
                    # cycles retain caches but replace the document/workers.
                    driver.get(f"http://127.0.0.1:8008/unit{unit}/")
                    wait.until(lambda d: d.execute_script(READY))
                    result["cases"][key + "/startup"] = dict(
                        status="pass",
                        seconds=time.perf_counter() - start,
                        application=driver.execute_script(STATE),
                    )
                    for operation in ("reload", "service-worker-update"):
                        if operation == "service-worker-update":
                            outcome = driver.execute_async_script("""
const done=arguments[arguments.length-1];
(async()=>{
 const desired=new URL('../shinylive-sw.js?actual-update=1',location.href).href;
 await navigator.serviceWorker.register(desired,{type:'module'});
 const deadline=Date.now()+30000;
 while(navigator.serviceWorker.controller?.scriptURL!==desired){
  if(Date.now()>deadline)throw new Error('Updated worker did not take control');
  await new Promise(resolve=>setTimeout(resolve,30));
 }
 return desired;
})().then(value=>done({value}),error=>done({error:String(error)}));
""")
                            if outcome.get("error"):
                                raise RuntimeError(outcome["error"])
                        driver.refresh()
                        wait.until(lambda d: d.execute_script(READY))
                        result["cases"][key + "/" + operation] = dict(
                            status="pass", application=driver.execute_script(STATE)
                        )
                    driver.get("about:blank")
                except Exception as error:
                    result["cases"][key + "/lifecycle"] = dict(
                        status="fail", operation=operation, error=str(error)
                    )
                    try:
                        result["traces"][key] = dict(
                            url=driver.current_url,
                            page=driver.page_source,
                            state=driver.execute_script(
                                "return {controller:navigator.serviceWorker?.controller?.scriptURL}"
                            ),
                        )
                        driver.save_screenshot(
                            str(
                                EVIDENCE
                                / f"screenshots/actual-{args.browser}-unit{unit}-cycle{cycle}.png"
                            )
                        )
                    except Exception as trace_error:
                        result["traces"][key] = {"error": str(trace_error)}
                    # A failed cycle cannot be erased by an isolated retry.
                    write_json(path, result)
                    print(args.browser, key, "fail", operation, flush=True)
                    break
                write_json(path, result)
                print(args.browser, key, "pass", flush=True)
        expected = len(args.units) * args.cycles * 3
        result["status"] = (
            "pass"
            if len(result["cases"]) == expected
            and all(c["status"] == "pass" for c in result["cases"].values())
            else "fail"
        )
    except Exception as error:
        result.update(status="unavailable" if driver is None else "error", error=str(error))
    finally:
        if driver:
            driver.quit()
        if result["input_fingerprint"] != inputs():
            result["observed_status"] = result["status"]
            result["status"] = "stale"
        write_json(path, result)
    print(result["status"], flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
