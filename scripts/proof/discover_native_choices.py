"""Execute explicit parent states against corrected native apps; retain unresolved matrices."""

from repository import REFERENCE

from repository import EVIDENCE
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright
from common import PROOF, sha, write_json, archive_file
from provenance import native_stamp
from dynamic_choices import choice_cases


def plans(unit):
    if unit == 1:
        return [
            dict(
                parent_state={"dataset": dataset, "loaded": True},
                inputs={"dataset": dataset},
                tabs=["Dataset Overview"],
                action="load_data",
                controls=["viz_features", "analysis_features"],
            )
            for dataset in ("iris", "wine", "breast", "penguins")
        ]
    if unit == 2:
        return [
            dict(
                parent_state={"word_list": words},
                inputs={"word_list": words},
                tabs=["Text Processing"],
                action="run_embed",
                controls=["word_select", "word_select1", "word_select2", "word_select_3"],
            )
            for words in (
                "dog,cat,tiger,lion,car,bike,apple,banana,jeans,dress,man,woman,king,queen",
                "",
                "king,king,queen",
                "New York,ice-cream,cat",
                "unseenwordzzzz,cat",
                "Grüße,中文,cat",
            )
        ]
    if unit in (6, 7):
        return [dict(parent_state={"initial": True}, inputs={}, controls=["preset"])]
    if unit == 4:
        datasets = ("Wine", "Breast Cancer", "Digits", "Pima Diabetes", "Iris", "Banknotes")
        baseline = [
            dict(
                parent_state={"dataset": dataset, "use_all_features": True, "load": True},
                inputs={"dataset": dataset},
                tabs=["Data Settings & PCA"],
                action="load",
                controls=[
                    "dataset",
                    "feature_select",
                    "pairplot_features_all_samples",
                    "pairplot_features",
                    "db_features",
                ],
            )
            for dataset in datasets
        ]
        trained = [
            dict(
                parent_state=dict(
                    dataset=dataset,
                    use_all_features=True,
                    classifier=classifier,
                    split=split,
                    val_split=validation,
                    trained=True,
                ),
                inputs=dict(
                    dataset=dataset,
                    use_all_features=True,
                    classifier=classifier,
                    split=split,
                    val_split=validation,
                ),
                tabs=["Data Settings & PCA"],
                action="load",
                training=True,
                controls=["curve_class", "db_features", "pairplot_features"],
            )
            for dataset in datasets
            for classifier in ("k-NN", "Decision Tree", "Random Forest")
            for split, validation in ((70, 15), (50, 5), (90, 5))
        ]
        return baseline + trained
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--units", type=int, nargs="+", choices=[1, 2, 4, 6, 7], default=[1, 2, 4, 6, 7]
    )
    args = parser.parse_args()
    output = EVIDENCE / "native-choice-discovery.json"
    previous = json.loads(output.read_text()) if output.exists() else {}
    retained = {
        name: record
        for name, record in previous.get("controls", {}).items()
        if record.get("unit") not in args.units
    }
    report = dict(
        status="inventory",
        browser="chrome",
        distribution="installed",
        controls=retained,
        unresolved=[
            "Unit 4 selected-feature subsets, trained curve_class, split/settings and validation parent states need expansion.",
            "Unit 6/7 preset dataset compatibility states need behavior execution.",
            "Observed choices are requirements; choice-selection effects have not been executed.",
        ],
        scope="Corrected native runtime discovery for explicit parent states. Incomplete matrices block exhaustive acceptance.",
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        report["version"] = browser.version
        try:
            for unit in args.units:
                unit_plans = plans(unit)
                # Declare obligations before any execution, including failures.
                for plan in unit_plans:
                    for identifier in plan["controls"]:
                        record = report["controls"].setdefault(
                            f"unit{unit}/{identifier}",
                            dict(
                                unit=unit,
                                identifier=identifier,
                                expected_parent_states=[],
                                observations=[],
                            ),
                        )
                        record["expected_parent_states"].append(plan["parent_state"])
                port = 8150 + unit
                log_path = EVIDENCE / f"native-choice-unit{unit}.log"
                archive_file(log_path)
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
                        cwd=REFERENCE / f"unit{unit}",
                        stdout=log,
                        stderr=log,
                    )
                    page = browser.new_page()
                    try:
                        url = f"http://127.0.0.1:{port}/"
                        for _ in range(120):
                            if process.poll() is not None:
                                raise RuntimeError("Native app process exited; see log")
                            try:
                                urllib.request.urlopen(url, timeout=1).close()
                                break
                            except OSError:
                                time.sleep(0.25)
                        page.goto(url, wait_until="domcontentloaded")
                        page.wait_for_function(
                            "()=>window.Shiny?.shinyapp?.isConnected()", timeout=90000
                        )
                        stamp = native_stamp(unit)
                        for plan in unit_plans:
                            success = False
                            try:
                                for tab in plan.get("tabs", []):
                                    page.get_by_role("tab", name=tab, exact=True).click()
                                for identifier, value in plan["inputs"].items():
                                    page.locator("#" + identifier).evaluate(
                                        """(element,value)=>{
                                      if(element.selectize)element.selectize.setValue(value);
                                      else if(element.type==='checkbox'){element.checked=value;window.jQuery(element).trigger('change');}
                                      else if(window.jQuery(element).data('ionRangeSlider')){window.jQuery(element).data('ionRangeSlider').update({from:value});window.jQuery(element).trigger('change');}
                                      else if(element.tagName==='SELECT'){element.value=value;window.jQuery(element).trigger('change');}
                                      else{element.value=value;window.jQuery(element).trigger('input').trigger('change');}
                                    }""",
                                        value,
                                    )
                                    if identifier == "split":
                                        # Parent-dependent bounds arrive after the
                                        # server flush. Sending val_split sooner
                                        # clamps it against the preceding state.
                                        page.wait_for_function(
                                            """maximum=>
                                          window.jQuery('#val_split').data('ionRangeSlider').options.max===maximum
                                        """,
                                            arg=100 - value - 5,
                                            timeout=15000,
                                        )
                                page.wait_for_function(
                                    """expected=>{
                                  const values=window.Shiny.shinyapp.$inputValues;
                                  return Object.entries(expected).every(([name,value])=>{
                                    const key=Object.keys(values).find(k=>k===name||k.startsWith(name+':'));
                                    return JSON.stringify(values[key])===JSON.stringify(value);
                                  });
                                }""",
                                    arg=plan["inputs"],
                                    timeout=15000,
                                )
                                if plan.get("action"):
                                    page.locator("#" + plan["action"]).click()
                                if plan.get("training"):
                                    page.get_by_role(
                                        "tab", name="Classification Models", exact=True
                                    ).click()
                                    page.get_by_role(
                                        "tab", name="Classification Report", exact=True
                                    ).click()
                                    page.evaluate("""()=>{
                                      window.courseDiscoveryReportRevision=0;
                                      window.jQuery(document).off('shiny:value.courseDiscovery').on('shiny:value.courseDiscovery',
                                        e=>{if(e.name==='class_report')window.courseDiscoveryReportRevision++});
                                    }""")
                                    page.locator("#apply_classifier").click()
                                    page.wait_for_function(
                                        """()=>{
                                      const report=document.getElementById('class_report');
                                      return window.courseDiscoveryReportRevision>0 && report?.textContent.includes('precision') &&
                                        !report.classList.contains('recalculating');
                                    }""",
                                        timeout=90000,
                                    )
                                # Synchronize with Shiny's idle notification, then record all options.
                                page.wait_for_function(
                                    '()=>!document.documentElement.classList.contains("shiny-busy")',
                                    timeout=90000,
                                )
                                # Choice messages may arrive after input transmission. Wait for options to stabilize.
                                observed = page.evaluate(
                                    """async identifiers=>{
                                  function read(){return Object.fromEntries(identifiers.map(id=>{
                                    const element=document.getElementById(id);
                                    if(!element)throw new Error('Missing native control '+id);
                                    return [id,element.selectize?Object.keys(element.selectize.options):Array.from(element.options||[],o=>o.value)];
                                  }));}
                                  let previous=null,stable=0;const deadline=performance.now()+15000;
                                  while(performance.now()<deadline){
                                    await new Promise(resolve=>setTimeout(resolve,100));
                                    const current=read(),text=JSON.stringify(current);
                                    stable=text===previous?stable+1:0;previous=text;
                                    if(stable>=5)return current;
                                  }throw new Error('Native choices did not stabilize');
                                }""",
                                    plan["controls"],
                                )
                                for identifier, choices in observed.items():
                                    record = report["controls"].setdefault(
                                        f"unit{unit}/{identifier}",
                                        dict(
                                            unit=unit,
                                            identifier=identifier,
                                            expected_parent_states=[],
                                            observations=[],
                                        ),
                                    )
                                    record["observations"].append(
                                        dict(
                                            status="pass",
                                            parent_state=plan["parent_state"],
                                            choices=choices,
                                            acknowledged_inputs=plan["inputs"],
                                            reference_scope=stamp["scope"],
                                            reference_fingerprint=stamp["fingerprint"],
                                            executor={
                                                "path": str(
                                                    Path(__file__).relative_to(PROOF.parent)
                                                ),
                                                "sha256": sha(Path(__file__)),
                                            },
                                            dependencies={
                                                str(
                                                    (REFERENCE / f"unit{unit}/app.py").relative_to(
                                                        PROOF.parent
                                                    )
                                                ): sha(REFERENCE / f"unit{unit}/app.py")
                                            },
                                            cases=choice_cases(
                                                unit, identifier, plan["parent_state"], choices
                                            ),
                                        )
                                    )
                                success = True
                            except Exception as error:
                                report["unresolved"].append(
                                    dict(
                                        unit=unit,
                                        parent_state=plan["parent_state"],
                                        error=str(error),
                                    )
                                )
                            write_json(output, report)
                            print(
                                unit,
                                plan["parent_state"],
                                "observed" if success else "failed",
                                flush=True,
                            )
                    except Exception as error:
                        report["unresolved"].append(dict(unit=unit, error=str(error)))
                    finally:
                        page.close()
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                write_json(output, report)
        finally:
            browser.close()
    print("Inventory retained;", len(report["unresolved"]), "unresolved obligations")


if __name__ == "__main__":
    main()
