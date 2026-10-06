"""Run Unit 7's unchanged default small preset on every MNIST training row."""

from repository import EVIDENCE
import time
import argparse
import re
from playwright.sync_api import sync_playwright, expect
from common import PROOF, write_json
from provenance import stamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-url")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Positive epochs required")
    result = {
        "browser": "chrome",
        "distribution": "installed",
        "provenance": stamp(),
        "cases": {},
        "status": "error",
        "configuration": f"Unit 7 MNIST small preset, {args.epochs} epochs; complete data.",
        "default_configuration": args.epochs == 5,
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        result["version"] = browser.version
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        app = None
        try:
            page.goto(
                args.native_url or "http://127.0.0.1:8008/unit7/",
                wait_until="domcontentloaded",
                timeout=120000,
            )
            app = page if args.native_url else page.frame_locator("iframe")
            app.locator("h2").wait_for(timeout=120000)

            def tab(name):
                app.get_by_role("tab", name=name, exact=True).click()

            tab("CNN: Data")
            app.locator("#load_data").click()
            expect(app.locator("#data_info")).to_contain_text("54000", timeout=120000)
            result["data_info"] = app.locator("#data_info").inner_text()
            result["cases"]["full-data"] = {"status": "pass"}
            tab("CNN: Architecture")
            app.locator("#load_preset").click()
            expect(app.locator("#arch_text")).not_to_have_value("", timeout=30000)
            app.locator("#apply_arch").click()
            expect(app.locator("#model_summary")).to_contain_text("Conv2d", timeout=30000)
            result["architecture"] = app.locator("#arch_text").input_value()
            tab("CNN: Training")
            app.locator("#epochs").evaluate(
                '(el,value)=>{window.jQuery(el).data("ionRangeSlider").update({from:value});window.Shiny.setInputValue("epochs",value,{priority:"event"})}',
                args.epochs,
            )
            app.locator("body").evaluate(
                """(el,expected)=>new Promise((resolve,reject)=>{
              const end=Date.now()+30000;function poll(){if(window.Shiny.shinyapp.$inputValues.epochs===expected)return resolve();if(Date.now()>end)return reject(new Error('Epoch setting not accepted'));setTimeout(poll,25)}poll();
            })""",
                args.epochs,
            )
            start = time.perf_counter()
            app.locator("#start_train").click()
            previous = ""
            while time.perf_counter() - start < 1800:
                message = app.locator("#train_progress").inner_text()
                if message != previous:
                    print(message.replace("\n", " "), flush=True)
                    previous = message
                if "Error:" in message:
                    raise AssertionError(message)
                if "Training complete" in message:
                    break
                page.wait_for_timeout(2000)
            else:
                raise TimeoutError("Full default training exceeded 30 minutes")
            result["cases"][
                "full-default-training" if args.epochs == 5 else "configured-training"
            ] = {
                "status": "pass",
                "seconds": time.perf_counter() - start,
                "best_info": app.locator("#early_stop_info").inner_text(),
            }
            tab("CNN: Inspect")
            app.locator("#filter_epoch.shiny-bound-input").wait_for(state="attached", timeout=30000)
            app.locator("body").evaluate("""()=>new Promise((resolve,reject)=>{
              const end=Date.now()+30000;
              function poll(){if(window.Shiny.shinyapp.$inputValues.filter_epoch!==undefined)return resolve();if(Date.now()>end)return reject(new Error('Epoch selector not initialized'));setTimeout(poll,25)}poll();
            })""")
            app.locator("#do_inspect").click()
            expect(app.locator("#inspect_plot img")).to_have_attribute(
                "src", re.compile("^data:image/"), timeout=60000
            )
            result["cases"]["filter-inspection"] = {"status": "pass"}
            tab("CNN: Predict")
            app.locator("#sample_preds").click()
            expect(app.locator("#pred_plot img")).to_have_attribute(
                "src", re.compile("^data:image/"), timeout=60000
            )
            result["cases"]["predictions"] = {"status": "pass"}
            tab("CNN: Analysis")
            app.locator("#compute_analysis").click()
            expect(app.locator("#metrics_text")).not_to_be_empty(timeout=120000)
            errors_visible = app.locator(".shiny-output-error:visible").all_text_contents()
            if errors_visible:
                raise AssertionError(errors_visible)
            result["cases"]["evaluation"] = {
                "status": "pass",
                "metrics": app.locator("#metrics_text").inner_text(),
            }
            page.screenshot(path=str(EVIDENCE / "screenshots/unit7-full-mnist.png"))
            result["status"] = "pass"
        except Exception as e:
            result["error"] = str(e)
            if app is not None:
                result["visible_errors"] = app.locator(
                    ".shiny-output-error:visible"
                ).all_text_contents()
                inspection = app.locator("#inspect_plot")
                result["inspection_text"] = inspection.inner_text() if inspection.count() else None
            page.screenshot(path=str(EVIDENCE / "screenshots/unit7-full-mnist-failure.png"))
        finally:
            browser.close()
    result["page_errors"] = errors
    if result["provenance"]["fingerprint"] != stamp()["fingerprint"]:
        result["original_status"] = result["status"]
        result["status"] = "stale"
    prefix = "native-unit7-full-mnist" if args.native_url else "unit7-full-mnist"
    suffix = "-" + args.label if args.label else ""
    write_json(EVIDENCE / f"{prefix}{suffix}.json", result)
    print(result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
