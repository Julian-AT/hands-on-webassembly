"""Complete image bytes, supervised nested cancellation, atomic failure and Retry."""

from repository import EVIDENCE

from repository import ROOT, CACHE, RUNTIME, SCRIPTS, SITE
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from playwright.sync_api import sync_playwright
from common import ROOT, PROOF, sha, write_json
from owned_preview import start
from host_conditions import snapshot, apply
from browser_generation_sequence import site_manifest
from storage_budget import require_space
from hardware_contract import collect as collect_hardware


PROBE = """importScripts('./pyodide/pyodide.js');
self.onmessage=async ({data})=>{
  try{
    const py=await loadPyodide({indexURL:'./pyodide/'});await py.loadPackage(['numpy','pillow']);
    for(const [name,digest] of Object.entries(data.files)){
      const raw=new Uint8Array(await (await fetch('./'+name)).arrayBuffer());
      const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),x=>x.toString(16).padStart(2,'0')).join('');
      if(actual!==digest)throw new Error('Prototype source checksum changed');py.FS.writeFile(name,raw);
    }
    py.globals.set('configuration',JSON.stringify(data));
    py.globals.set('case_source',data.case_source);
    const result=await py.runPythonAsync(`
import asyncio,json,hashlib,numpy as np
import image_data
from image_worker_preload import Owner
config=json.loads(configuration)
owner=Owner(7,config['build_id'],config['files']['image_worker_preload.py'],'image-worker-proof')
cases={}
exec(case_source)
result=await run_case()
json.dumps(dict(done=True,**result))
`);
    self.postMessage(JSON.parse(result));
  }catch(error){self.postMessage({error:String(error)});}
};"""


COMPLETE = """async def run_case():
    observations={}
    for name in config['datasets']:
        image_data._cache.clear()
        receipt=await owner.preload(name,manifest_sha256=config['manifest_sha256'])
        for split in ('train','test'):
            images,labels,layout=image_data.load(name,split)
            observations[f'{name}/{split}']=dict(images_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),shape=list(images.shape),
                label_dtype=str(labels.dtype),layout=layout)
    return dict(observations=observations)
"""


CANCEL = """async def run_case():
    task=asyncio.create_task(owner.preload('MNIST',manifest_sha256=config['manifest_sha256']))
    while owner.active is None:await asyncio.sleep(.01)
    await asyncio.sleep(config['cancel_delay'])
    owner.cancel()
    try:
        await task
        raise AssertionError('Obsolete preparation committed a result')
    except asyncio.CancelledError:pass
    assert not image_data._cache
    return dict(disposal=owner.last_disposal,cache_entries=len(image_data._cache))
"""


CORRUPT = """async def run_case():
    try:
        await owner.preload('MNIST',manifest_sha256=config['manifest_sha256'])
        raise AssertionError('Corrupt test split was accepted')
    except RuntimeError as error:
        assert 'checksum' in str(error).lower() and not image_data._cache
        return dict(rejection_message=str(error),cache_entries=0)
"""


PACKAGE_FAILURE = """async def run_case():
    try:
        await owner.preload('MNIST',manifest_sha256=config['manifest_sha256'])
        raise AssertionError('Missing numerical package was accepted')
    except RuntimeError as error:
        assert 'NumPy did not load' in str(error) and not image_data._cache
        return dict(rejection_message=str(error),cache_entries=0)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["chrome", "edge"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["MNIST", "FashionMNIST", "USPS", "CIFAR10", "SVHN"],
        default=["MNIST", "FashionMNIST", "USPS", "CIFAR10", "SVHN"],
    )
    args = parser.parse_args()
    root = CACHE / f"image-worker-{args.browser}-{args.label}"
    output = EVIDENCE / f"image-worker-{args.browser}-{args.label}.json"
    if root.exists() or output.exists():
        raise FileExistsError("Retain attempts; choose a fresh label")
    require_space(128 * 1024**2)
    shutil.copytree(SITE, root, copy_function=os.link)
    names = [
        "image_data.py",
        "image_preload.py",
        "image_worker_preload.py",
        "course-image-worker.js",
        "course-image-decode-worker.js",
    ]
    for name in names:
        path = root / "shinylive" / name
        path.unlink(missing_ok=True)
        shutil.copy2(RUNTIME / name, path)
    (root / "shinylive/image-worker-verification.js").write_text(PROBE)
    manifest = json.loads((root / "assets/v1/images/manifest.json").read_text())
    controls = {}
    for name in args.datasets:
        for split in ("train", "test"):
            key = f"{name}/{split}"
            entry = manifest["datasets"][key]
            with np.load(root / "assets/v1/images" / entry["file"], allow_pickle=False) as packed:
                images, labels = packed["images"], packed["labels"]
            controls[key] = dict(
                images_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                shape=list(images.shape),
                label_dtype=str(labels.dtype),
                layout=entry["layout"],
            )
    before = snapshot()
    server, identity = start(root, 0)
    inputs = site_manifest(root)
    config = dict(
        files={name: sha(root / "shinylive" / name) for name in names[:3]},
        build_id=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        manifest_sha256=sha(root / "assets/v1/images/manifest.json"),
        datasets=args.datasets,
    )
    report = dict(
        status="running",
        browser=args.browser,
        distribution="installed",
        artifact=str(root.relative_to(ROOT)),
        hardware=collect_hardware(),
        dependencies={
            str(path.relative_to(ROOT)): sha(path)
            for path in [
                PROOF / "hardware-contract.json",
                *[
                    SCRIPTS / name
                    for name in (
                        "common.py",
                        "owned_preview.py",
                        "host_conditions.py",
                        "hardware_contract.py",
                        "browser_generation_sequence.py",
                        "storage_budget.py",
                    )
                ],
            ]
        },
        provenance=dict(
            scope="isolated-artifact",
            fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        ),
        executor=str(Path(__file__).relative_to(ROOT)),
        executor_sha256=sha(Path(__file__)),
        inputs=inputs,
        preview_identity=identity,
        cases={},
        controls=controls,
        scope="Isolated supervisor and decoder candidate. Complete packaged array equality, cancellation and failure recovery; application integration remains required.",
    )
    write_json(output, report)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            report["version"] = browser.version
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(300000)
            page.goto(f"http://127.0.0.1:{identity['port']}/probes/", wait_until="domcontentloaded")

            def run(source, **extra):
                return page.evaluate(
                    """configuration=>new Promise((resolve,reject)=>{
                  const worker=new Worker('../shinylive/image-worker-verification.js');
                  const timer=setTimeout(()=>{worker.terminate();reject(new Error('Image proof timeout'))},300000);
                  worker.onmessage=({data})=>{clearTimeout(timer);worker.terminate();data.error?reject(new Error(data.error)):resolve(data)};
                  worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(new Error(e.message))};
                  worker.postMessage(configuration);
                })""",
                    dict(config, case_source=source, **extra),
                )

            def retain(name, expected, observed, **detail):
                if expected != observed:
                    raise AssertionError((name, expected, observed))
                report["cases"][name] = dict(
                    status="pass",
                    assertion=dict(
                        kind="computation", expected=expected, observed=observed, matched=True
                    ),
                    **detail,
                )
                write_json(output, report)
                print(args.browser, name, "pass", flush=True)

            held = []
            pattern = "**/images/MNIST-test.npz"
            context.route(pattern, lambda route: held.append(route))
            result = run(CANCEL, cancel_delay=1)
            if not held:
                raise AssertionError("No actual download was blocked")
            retain(
                "blocked-download-cancellation",
                dict(acknowledged=True, deadline_met=True, cache_entries=0),
                dict(
                    acknowledged=result["disposal"]["acknowledged"],
                    deadline_met=result["disposal"]["seconds"] < 1,
                    cache_entries=result["cache_entries"],
                ),
                disposal=result["disposal"],
            )
            for route in held:
                try:
                    route.abort("failed")
                except Exception:
                    pass
            context.unroute(pattern)
            context.route(
                pattern, lambda route: route.fulfill(status=200, body=b"corrupt test split")
            )
            result = run(CORRUPT)
            retain(
                "complete-set-atomic-rejection",
                0,
                result["cache_entries"],
                rejection_message=result["rejection_message"],
            )
            context.unroute(pattern)
            # Delay the numerical child runtime, after both original archives
            # have arrived. Cancelling must remove supervisor and nested worker.
            nested = []
            wasm_requests = []

            def hold_nested(route):
                wasm_requests.append(route.request.url)
                if len(wasm_requests) == 1:
                    route.continue_()
                else:
                    nested.append(route)

            context.route("**/shinylive/pyodide/pyodide.asm.wasm", hold_nested)
            result = run(CANCEL, cancel_delay=1)
            if not nested:
                raise AssertionError("No nested numerical worker was blocked")
            cdp = context.new_cdp_session(page)
            started = time.monotonic()

            def remaining():
                return [
                    t
                    for t in cdp.send("Target.getTargets")["targetInfos"]
                    if t["type"] == "worker" and "course-image-" in t["url"]
                ]

            while remaining() and time.monotonic() - started < 2:
                page.wait_for_timeout(25)
            retain(
                "nested-worker-cancellation",
                dict(acknowledged=True, deadline_met=True, workers=0),
                dict(
                    acknowledged=result["disposal"]["acknowledged"],
                    deadline_met=result["disposal"]["seconds"] < 1,
                    workers=len(remaining()),
                ),
                disposal=result["disposal"],
                worker_cleanup_seconds=time.monotonic() - started,
            )
            for route in nested:
                try:
                    route.abort("failed")
                except Exception:
                    pass
            context.unroute("**/shinylive/pyodide/pyodide.asm.wasm")
            package_requests = []

            def reject_numerical_child(route):
                package_requests.append(route.request.url)
                # The first runtime belongs to the observing Python worker.
                # Fail only the nested decoder's independent package download.
                if len(package_requests) == 1:
                    route.continue_()
                else:
                    route.fulfill(status=503, body=b"Unavailable numerical package")

            context.route("**/shinylive/pyodide/numpy-*.whl", reject_numerical_child)
            result = run(PACKAGE_FAILURE)
            if len(package_requests) < 2:
                raise AssertionError("No actual decoder package request failed")
            retain(
                "numerical-package-failure",
                0,
                result["cache_entries"],
                rejection_message=result["rejection_message"],
            )
            context.unroute("**/shinylive/pyodide/numpy-*.whl")
            result = run(COMPLETE)
            for key, expected in controls.items():
                retain("complete-array/" + key, expected, result["observations"].get(key))
            started = time.monotonic()
            while remaining() and time.monotonic() - started < 2:
                page.wait_for_timeout(25)
            retain(
                "full-workload-worker-disposal",
                0,
                len(remaining()),
                seconds=time.monotonic() - started,
            )
            context.close()
            browser.close()
            report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        apply(report, before, snapshot())
        if inputs != site_manifest(root):
            report.update(observed_status=report["status"], status="stale")
        write_json(output, report)
    print(report["status"], report.get("error", ""), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
