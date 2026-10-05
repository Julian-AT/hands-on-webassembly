"""Real UI checks; every outcome is persisted, including unavailable browsers.

WebKit is labelled WebKit, never Safari. No skipped check counts as a pass.
"""
import argparse
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from common import ROOT, PROOF, UNITS, write_json
from provenance import browser_stamp, stamp


def unit1(page, url, prefix, full=True):
    errors, external, cases = [], set(), {}
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda req: external.add(req.url) if req.url.startswith("http") and "127.0.0.1" not in req.url else None)
    start = time.perf_counter()
    page.goto(url, wait_until="domcontentloaded")
    app = page.frame_locator("iframe") if "/unit1/" in url else page
    app.locator("h2").wait_for(timeout=120000)
    # A rendered UI is not evidence that the reactive server has connected.
    app.locator("#viz_features option").first.wait_for(state="attached", timeout=120000)
    startup = time.perf_counter() - start
    screenshots = PROOF / "evidence/screenshots"
    screenshots.mkdir(exist_ok=True)
    page.screenshot(path=str(screenshots / f"{prefix}-intro.png"), full_page=True)

    def check(name, fn):
        t = time.perf_counter()
        try:
            detail = fn()
            visible_errors = app.locator(".shiny-output-error:visible").all_text_contents()
            if visible_errors:
                raise AssertionError(visible_errors)
            cases[name] = {"status": "pass", "seconds": time.perf_counter() - t, "detail": detail}
        except Exception as e:
            cases[name] = {"status": "fail", "error": str(e), "seconds": time.perf_counter() - t}
        print(prefix, name, cases[name]["status"], flush=True)

    def image(button, output):
        el = app.locator(f"#{output}")
        old = el.locator("img").get_attribute("src") if el.locator("img").count() else ""
        app.locator(f"#{button}").click()
        expect(el.locator("img")).to_have_attribute("src", __import__("re").compile(r"^data:image/"), timeout=90000)
        # A new request can produce identical bytes; wait on Shiny's progress too.
        if old:
            expect(el.locator("img")).not_to_have_attribute("src", old, timeout=90000)
        expect(el).not_to_have_class(__import__("re").compile("recalculating"), timeout=90000)
        return {"image_bytes_base64": len(el.locator("img").get_attribute("src"))}

    for dataset, rows in [("wine",178),("penguins",333),("iris",150),("breast",569)]:
        def overview():
            app.get_by_role("tab", name="Dataset Overview", exact=True).click()
            app.locator("#dataset").select_option(dataset)
            app.locator("#load_data").click()
            expect(app.locator("#dataset_info")).to_contain_text(f"Samples: {rows}", timeout=30000)
            expect(app.locator("#data_table")).to_contain_text(f"of {rows}", timeout=30000)
            return {"info": app.locator("#dataset_info").inner_text(), "summary": app.locator("#data_summary").inner_text()}
        check(f"{dataset}/overview", overview)
        app.get_by_role("tab", name="Data Visualization", exact=True).click()
        for kind in (["pairplot","boxplot","histogram","scatter","violinplot"] if full else ["scatter"]):
            def visualize():
                app.locator("#viz_type").select_option(kind)
                return image("generate_viz", "visualization_plot")
            check(f"{dataset}/{kind}", visualize)
        app.get_by_role("tab", name="Dimensionality Reduction", exact=True).click()
        for method, dimensions in ([("pca",2),("pca",3),("tsne",2),("tsne",3)] if full else [("pca",2)]):
            def reduce():
                app.locator("#reduction_method").select_option(method)
                app.locator("#n_components").fill(str(dimensions))
                app.locator("#n_components").press("Tab")
                result = image("apply_reduction", "reduction_plot")
                if method == "pca":
                    expect(app.locator("#explained_variance")).to_contain_text("Explained Variance Ratio", timeout=30000)
                    result["variance"] = app.locator("#explained_variance").inner_text()
                return result
            check(f"{dataset}/{method}/{dimensions}", reduce)
        app.get_by_role("tab", name="Correlation Analysis", exact=True).click()
        def correlation():
            result = image("run_analysis", "analysis_plot")
            expect(app.locator("#analysis_results")).to_contain_text("Strongest Correlations", timeout=30000)
            result["text"] = app.locator("#analysis_results").inner_text()
            return result
        check(f"{dataset}/correlation", correlation)
    page.screenshot(path=str(screenshots / f"{prefix}-correlation.png"), full_page=True)
    # Resetting the page tests a fresh worker and service-worker interaction.
    page.reload(wait_until="domcontentloaded")
    app.locator("h2").wait_for(timeout=120000)
    app.get_by_role("tab", name="Dataset Overview", exact=True).click()
    app.locator("#load_data").click()
    expect(app.locator("#dataset_info")).to_contain_text("Samples: 178", timeout=90000)
    cases["reload"] = {"status":"pass"}
    return {"startup_seconds":startup, "cases": cases, "page_errors":errors, "external_requests": sorted(external)}


def unit3(page, url, prefix):
    page.goto(url, wait_until="domcontentloaded")
    app = page.frame_locator("iframe") if "/unit3/" in url else page
    app.locator("h2").wait_for(timeout=120000)
    app.locator("body").evaluate("""() => {
      window.proofValues = {};
      window.jQuery(document).on('shiny:value.proof shiny:error.proof', e => {
        window.proofValues[e.name] = (window.proofValues[e.name] || 0) + 1;
      });
    }""")
    def trigger(button, output):
        before = app.locator("body").evaluate("(el, id) => window.proofValues[id] || 0", output)
        app.locator(f"#{button}").click()
        app.locator("body").evaluate("""(el, {output, before}) => new Promise((resolve, reject) => {
          const deadline = Date.now() + 60000;
          const poll = () => {
            if ((window.proofValues[output] || 0) > before) return resolve();
            if (Date.now() > deadline) return reject(new Error('No fresh Shiny value: ' + output));
            setTimeout(poll, 25);
          };
          poll();
        })""", {"output":output,"before":before})
    cases = {}
    app.get_by_role("tab", name="Data Selection", exact=True).click()
    for name in ["blobs","gaussian","gaussian2","moons","circles","uniform","iris","wine","breast_cancer","ionosphere","seeds","tweets","penguins","mall","spotify"]:
        start = time.perf_counter()
        try:
            app.locator("#data_kind").select_option(name)
            app.get_by_role("tab", name="Data Summary", exact=True).click()
            trigger("generate_data", "data_summary")
            rows = {"blobs":300,"gaussian":300,"gaussian2":600,"moons":300,"circles":300,"uniform":300,"iris":150,"wine":178,"breast_cancer":569,"ionosphere":351,"seeds":210,"tweets":1000,"penguins":333,"mall":200,"spotify":600}[name]
            expect(app.locator("#data_summary")).to_contain_text(f"Number of samples: {rows}", timeout=60000)
            if app.locator(".shiny-output-error:visible").count():
                raise AssertionError(app.locator(".shiny-output-error:visible").all_text_contents())
            cases[name] = {"status":"pass", "summary":app.locator("#data_summary").inner_text(), "seconds":time.perf_counter()-start}
        except Exception as e:
            cases[name] = {"status":"fail", "error":str(e)}
        print(prefix, name, cases[name]["status"], flush=True)
    # Exercise Plotly rendering and all four clustering choices on the default dataset.
    app.locator("#data_kind").select_option("blobs")
    app.locator("#generate_data").click()
    app.get_by_role("tab", name="Data Plot (True Groups)", exact=True).click()
    expect(app.locator("#data_plot .js-plotly-plot")).to_be_visible(timeout=60000)
    app.get_by_role("tab", name="Cluster Analysis", exact=True).click()
    for method in ["km","hc","db","gm"]:
        try:
            app.get_by_role("tab", name="Cluster Results Plot", exact=True).click()
            app.locator("#clustering_method").select_option(method)
            if method in ("hc", "db"):
                suffix = "ag" if method == "hc" else "db"
                expect(app.locator(f"#show_regions_{suffix}.shiny-bound-input")).to_be_visible(timeout=30000)
            output = "clustering_plot_gmm" if method == "gm" else "clustering_plot"
            app.get_by_role("tab", name="Cluster Validation", exact=True).click()
            trigger("run_clustering", "clustering_info")
            expect(app.locator("#clustering_info")).not_to_be_empty(timeout=60000)
            if app.locator(".shiny-output-error:visible").count():
                raise AssertionError(app.locator(".shiny-output-error:visible").all_text_contents())
            cases[f"cluster/{method}"] = {"status":"pass", "info":app.locator("#clustering_info").inner_text()}
            app.get_by_role("tab", name="Cluster Results Plot", exact=True).click()
            expect(app.locator(f"#{output} .js-plotly-plot")).to_be_visible(timeout=60000)
        except Exception as e:
            cases[f"cluster/{method}"] = {"status":"fail", "error":str(e)}
        print(prefix, method, cases[f"cluster/{method}"]["status"], flush=True)
    page.screenshot(path=str(PROOF / "evidence/screenshots" / f"{prefix}-unit3.png"), full_page=True)
    return {"cases":cases}


def uploads(page, url, prefix):
    page.goto(url,wait_until="domcontentloaded")
    app = page.frame_locator("iframe") if "/unit3/" in url else page
    app.locator("h2").wait_for(timeout=120000)
    cases = {}
    try:
        app.get_by_role("tab",name="Data Selection",exact=True).click()
        app.locator("#data_kind").select_option("csv")
        app.locator("#csv_file").set_input_files(str(ROOT / UNITS[3] / "resources/mall_customers.csv"))
        # Shiny acknowledges transfer before the data is available server-side.
        expect(app.locator("#csv_file_progress")).to_contain_text("Upload complete",timeout=30000)
        app.locator("#generate_data").click()
        app.get_by_role("tab",name="Data Summary",exact=True).click()
        expect(app.locator("#data_summary")).to_contain_text("Number of samples: 200",timeout=30000)
        cases["csv_upload"] = {"status":"pass","summary":app.locator("#data_summary").inner_text()}
    except Exception as e:
        cases["csv_upload"] = {"status":"fail","error":str(e)}
    try:
        app.get_by_role("tab",name="Application: Image Segmentation",exact=True).click()
        app.locator("#image_file").set_input_files(str(ROOT / UNITS[2] / "resources/charlie_tiny.jpg"))
        expect(app.locator("#image_file_progress")).to_contain_text("Upload complete",timeout=30000)
        app.locator("#run_segmentation").click()
        expect(app.locator("#segmentation_plot img")).to_have_attribute("src",__import__("re").compile("^data:image/"),timeout=60000)
        errors = app.locator(".shiny-output-error:visible").all_text_contents()
        if errors:
            raise AssertionError(errors)
        page.screenshot(path=str(PROOF / "evidence/screenshots" / f"{prefix}.png"))
        cases["jpeg_segmentation"] = {"status":"pass","fixture":f"{UNITS[2]}/resources/charlie_tiny.jpg","segments":5}
    except Exception as e:
        cases["jpeg_segmentation"] = {"status":"fail","error":str(e)}
    return {"cases":cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["chrome","edge","firefox","webkit"], default="chrome")
    parser.add_argument("--native-url")
    parser.add_argument("--corrected-reference", action="store_true",
                        help="Bind a native URL launched from proof/reference/unitN to its native dependencies")
    parser.add_argument("--unit", choices=["1","2","3","4","5","6","7","uploads","neural","tabular","supervised","embedding","training","unit5probe","unit5full","runtime","unit5life"], default="1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.corrected_reference and (not args.native_url or not args.unit.isdigit()):
        parser.error('--corrected-reference requires --native-url and an assignment number')
    if args.corrected_reference:
        from provenance import native_stamp
        record_stamp = lambda: native_stamp(int(args.unit))
    elif args.native_url:
        record_stamp = stamp
    else:
        record_stamp = lambda: browser_stamp(args.unit)
    prefix = f"{'native' if args.native_url else 'browser'}-{args.browser}-unit{args.unit}"
    result = {"browser":args.browser, "distribution":"installed" if args.browser in ("chrome","edge") else "playwright", "safari":False, "scope":"smoke" if args.smoke else "listed workflows"}
    result["provenance"] = record_stamp()
    with sync_playwright() as p:
        browser = None
        try:
            if args.browser in ("chrome","edge"):
                browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            else:
                browser = getattr(p,args.browser).launch()
            result["version"] = browser.version
            context = browser.new_context(viewport={"width":1440,"height":1000})
            transfers = []
            def finished(request):
                try:
                    transfers.append({"url":request.url, **request.sizes()})
                except Exception:
                    pass
            context.on("requestfinished", finished)
            page = context.new_page()
            page.set_default_timeout(15000)
            page.set_default_navigation_timeout(120000)
            if args.unit in ("neural","tabular","supervised","embedding","training","unit5probe","unit5full","runtime"):
                page.goto("http://127.0.0.1:8008/probes/")
                result.update(page.evaluate("kind => window.runProof(kind)", args.unit))
            elif args.unit == "1":
                result.update(unit1(page, args.native_url or "http://127.0.0.1:8008/unit1/", prefix, not args.smoke))
            elif args.unit in ("6","7"):
                from browser_neural_ui import neural_ui
                result.update(neural_ui(page, args.native_url or f"http://127.0.0.1:8008/unit{args.unit}/", prefix, int(args.unit)))
            elif args.unit == "2":
                from browser_unit2 import unit2
                result.update(unit2(page, args.native_url or "http://127.0.0.1:8008/unit2/", prefix))
            elif args.unit == "unit5life":
                from browser_unit5 import unit5_lifecycle
                result.update(unit5_lifecycle(page,"http://127.0.0.1:8008/unit5/"))
            elif args.unit == "5":
                from browser_unit5 import unit5
                result.update(unit5(page, args.native_url or "http://127.0.0.1:8008/unit5/", prefix))
            elif args.unit == "4":
                from browser_unit4 import unit4
                result.update(unit4(page, args.native_url or "http://127.0.0.1:8008/unit4/", prefix))
            elif args.unit == "3":
                result.update(unit3(page, args.native_url or "http://127.0.0.1:8008/unit3/", prefix))
            else:
                result.update(uploads(page, args.native_url or "http://127.0.0.1:8008/unit3/", prefix))
            result["status"] = "fail" if result.get("page_errors") or any(v["status"] != "pass" for key in ["cases","checks","comparisons"] for v in result.get(key,{}).values()) else "pass"
            result["response_body_bytes"] = sum(r["responseBodySize"] for r in transfers)
            result["network"] = transfers
            if args.browser in ("chrome","edge"):
                result["page_js_heap"] = context.new_cdp_session(page).send("Runtime.getHeapUsage")
        except Exception as e:
            result.update(status="unavailable" if browser is None else "error", error=str(e))
        finally:
            if browser:
                browser.close()
            end_stamp = record_stamp()
            if end_stamp["fingerprint"] != result["provenance"]["fingerprint"]:
                result["original_status"] = result["status"]
                result["status"] = "stale"
                result["stale_reason"] = "Inputs changed during the test run"
            write_json(PROOF / "evidence" / f"{prefix}.json", result)
    print(prefix, result["status"], flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
