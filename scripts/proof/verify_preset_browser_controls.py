"""Every retained preset/dataset/seed control through the deployed arithmetic."""

from repository import ROOT, CACHE, EVIDENCE, RUNTIME, SCRIPTS, SITE
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from playwright.sync_api import sync_playwright
from common import PROOF, ROOT, extract, sha, write_json
from discover_preset_compatibility import STATES
from browser_generation_sequence import site_manifest
from hardware_contract import collect
from host_conditions import snapshot, apply
from owned_preview import start
from storage_budget import require_space

WORKER = r"""importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage=async({data})=>{
 try{
  const progress=name=>self.postMessage({kind:'progress',name});
  progress('Pyodide initialization');
  const py=await loadPyodide({indexURL:'../shinylive/pyodide/'});
  progress('production package loading');
  await py.loadPackage(['numpy','pandas','scipy','pillow','course-native-neural','course-native-cnn']);
  const bytes=async name=>{
   const response=await fetch(name);
   if(!response.ok)throw new Error(name+': HTTP '+response.status);
   return new Uint8Array(await response.arrayBuffer());
  };
  progress('Borch wheel installation');
  py.unpackArchive(await bytes('pyborch-1.14.1-py3-none-any.whl'),'zip',{extractDir:'/home/pyodide'});
  for(const name of data.files){progress('source staging: '+name);py.FS.writeFile(name,await bytes(name));}
  py.FS.mkdirTree(py.FS.cwd()+'/controls');
  for(const record of data.cases){
   progress('control staging: '+record.name);
   const raw=await bytes(record.control);
   const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),x=>x.toString(16).padStart(2,'0')).join('');
   if(digest!==record.control_sha256)throw new Error('Retained native control changed');
   py.FS.writeFile(record.control,raw);
  }
  py.globals.set('configuration',JSON.stringify(data));
  py.globals.set('course_progress',name=>self.postMessage({kind:'progress',name}));
  const result=await py.runPythonAsync(`
import json,hashlib,numpy as np
course_progress('runtime import')
from browser_torch import torch
from preset_trajectory_probe import trajectory
config=json.loads(configuration)
definitions=json.load(open('definitions.json'))
cases={};artifacts={}
for record in config['cases']:
    course_progress(record['name'])
    expected=record['expected']
    observed,arrays=trajectory(torch,definitions,record)
    reference=np.load(record['control'],allow_pickle=False)
    shape_keys=('loss_executable','task_output_shape_matches','output_shape')
    flags={key:observed[key] for key in shape_keys}
    expected_flags={key:expected[key] for key in shape_keys}
    comparisons={};first=None
    for key in list(reference.files)+sorted(set(arrays)-set(reference.files)):
        a,b=arrays.get(key),reference[key] if key in reference.files else None
        exact=b is not None and b.dtype.kind in 'biu'
        matched=a is not None and b is not None and a.shape==b.shape and (
            np.array_equal(a,b) if exact else np.allclose(a,b,rtol=1e-4,atol=1e-5))
        comparisons[key]=dict(status='pass' if matched else 'fail',exact=exact,
            expected_shape=list(b.shape) if b is not None else None,
            observed_shape=list(a.shape) if a is not None else None,
            expected_sha256=hashlib.sha256(b.tobytes()).hexdigest() if b is not None else None,
            observed_sha256=hashlib.sha256(a.tobytes()).hexdigest() if a is not None else None,
            max_abs_error=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
                if a is not None and b is not None and a.shape==b.shape and a.size else None,
            assertion=dict(kind='computation',expected=True,observed=bool(matched),matched=bool(matched)))
        if not matched and first is None:first=key
    cases[record['name']]=dict(status='pass' if first is None and flags==expected_flags else 'fail',
        parent_state=record['parent_state'],expected=expected,observed=observed,
        first_divergent_operation=first,comparisons=comparisons,
        assertion=dict(kind='computation',expected=expected_flags,observed=flags,matched=flags==expected_flags))
    artifacts.update({record['id']+'/'+key:value for key,value in arrays.items()})
np.savez_compressed('observed.npz',**artifacts)
json.dumps(cases)
`);
  const raw=py.FS.readFile('observed.npz');
  self.postMessage({done:true,cases:JSON.parse(result),artifact:raw},[raw.buffer]);
 }catch(error){self.postMessage({error:String(error)+' '+(error?.message||'')+' '+(error?.stack||'')});}
 finally{self.close();}
};"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=("chrome", "edge"), required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    native = json.loads(args.controls.read_text())
    if native.get("status") != "pass" or len(native.get("cases", {})) != 213:
        raise ValueError("All 213 original/corrected native controls are required")
    root = CACHE / f"preset-browser-{args.browser}-{args.label}"
    output = EVIDENCE / f"preset-browser-{args.browser}-{args.label}.json"
    if root.exists() or output.exists():
        raise FileExistsError("Retain attempts; use a fresh label")
    require_space(256 * 1024**2)
    shutil.copytree(SITE, root, copy_function=os.link)
    probes = root / "probes"
    records = []
    (probes / "controls").mkdir()
    names = [
        "preset_trajectory_probe.py",
        "browser_torch.py",
        "neural_compat.py",
        "cnn_compat.py",
        "torch_rng.py",
        "borch_compat.py",
        "image_data.py",
        "module_hooks.py",
        "image_transforms.py",
    ]
    for name in names:
        target = probes / name
        target.unlink(missing_ok=True)
        source = (
            RUNTIME / name
            if name == "preset_trajectory_probe.py"
            else SITE / "assets/v1/neural-runtime" / name
        )
        shutil.copy2(source, target)
    definitions = {
        str(unit): extract(
            unit,
            ["ACTIVATIONS", "PRESETS", "parse_architecture", "build_model"]
            + (["_yamlish_to_json", "LayerSpec", "_tuple_or_int"] if unit == 7 else []),
        )
        for unit in (6, 7)
    }
    target = probes / "definitions.json"
    target.unlink()
    target.write_text(json.dumps(definitions, sort_keys=True))
    dependencies = {str(args.controls.resolve().relative_to(ROOT)): sha(args.controls)}
    for name, case in native["cases"].items():
        path = ROOT / case["trajectory"]["path"]
        if sha(path) != case["trajectory"]["sha256"]:
            raise ValueError("Retained native fixture changed")
        ident = path.stem
        shutil.copy2(path, probes / "controls" / path.name)
        parent = case["parent_state"]
        records.append(
            dict(
                name=name,
                id=ident,
                parent_state=parent,
                state=STATES[parent["unit"]][parent["dataset"]],
                expected=case["assertion"]["expected"],
                control="controls/" + path.name,
                control_sha256=sha(path),
            )
        )
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    for name in (
        "common.py",
        "discover_preset_compatibility.py",
        "browser_generation_sequence.py",
        "hardware_contract.py",
        "host_conditions.py",
        "audit_host_sleep.py",
        "owned_preview.py",
        "storage_budget.py",
    ):
        dependencies[f"scripts/proof/{name}"] = sha(SCRIPTS / name)
    dependencies["proof/hardware-contract.json"] = sha(PROOF / "hardware-contract.json")
    worker = probes / "preset-control-worker.js"
    worker.write_text(WORKER)
    before = snapshot()
    server, identity = start(root, 0)
    inputs = site_manifest(root)
    report = dict(
        status="running",
        browser=args.browser,
        distribution="installed",
        hardware=collect(),
        cases={},
        artifact=str(root.relative_to(ROOT)),
        inputs=inputs,
        preview_identity=identity,
        executor=str(Path(__file__).relative_to(ROOT)),
        executor_sha256=sha(Path(__file__)),
        dependencies=dependencies,
        provenance=dict(
            scope="isolated-artifact",
            fingerprint=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        ),
        scope="All 213 supplied preset/dataset shapes at seeds 0,42,123, synthetic two-sample initialization, forward, loss, gradients, SGD/momentum and exact uniform RNG state. Full-data trajectories, warnings, validation and restoration remain separate requirements.",
    )
    write_json(output, report)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome" if args.browser == "chrome" else "msedge")
            report["version"] = browser.version
            context = browser.new_context()
            page = context.new_page()
            report["console"] = []
            report["failed_requests"] = []
            context.on(
                "console",
                lambda message: report["console"].append(
                    dict(
                        text=message.text,
                        type=message.type,
                        worker=message.worker.url if message.worker else None,
                        location=message.location,
                    )
                ),
            )
            context.on(
                "requestfailed",
                lambda request: report["failed_requests"].append(
                    dict(url=request.url, failure=request.failure)
                ),
            )
            page.goto(f"http://127.0.0.1:{identity['port']}/probes/", wait_until="domcontentloaded")
            result = page.evaluate(
                r"""config=>new Promise((resolve,reject)=>{
              const worker=new Worker('preset-control-worker.js');
              let current='runtime initialization';
              const timer=setTimeout(()=>{worker.terminate();reject(Error('Preset control timeout'))},600000);
              worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message))};
              worker.onmessage=async({data})=>{
                if(data.kind==='progress'){current=data.name;console.log('Preset diagnostic: '+current);return;}
                clearTimeout(timer);worker.terminate();if(data.error)return reject(Error(current+': '+data.error));
                const reader=new FileReader();reader.onload=()=>resolve({cases:data.cases,artifact:reader.result.split(',')[1]});
                reader.readAsDataURL(new Blob([data.artifact]));
              };worker.postMessage(config);
            })""",
                dict(files=names + ["definitions.json"], cases=records),
            )
            report["cases"] = result["cases"]
            path = EVIDENCE / f"preset-browser-{args.browser}-{args.label}.npz"
            path.write_bytes(base64.b64decode(result["artifact"]))
            report["observed_arrays"] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
            report["status"] = (
                "pass" if all(c["status"] == "pass" for c in report["cases"].values()) else "fail"
            )
            browser.close()
    except Exception as error:
        report.update(status="fail", error=str(error))
    finally:
        server.terminate()
        server.wait(timeout=10)
        apply(report, before, snapshot())
        if inputs != site_manifest(root):
            report.update(observed_status=report["status"], status="stale")
        write_json(output, report)
    failures = [
        (name, case.get("first_divergent_operation"))
        for name, case in report["cases"].items()
        if case["status"] != "pass"
    ]
    print(
        report["status"],
        len(report["cases"]),
        "cases; first divergences:",
        failures[:8],
        report.get("error", ""),
        flush=True,
    )
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
