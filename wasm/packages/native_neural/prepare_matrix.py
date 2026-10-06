"""Execute untouched Unit 5 training bodies for additional full-data settings.

Only the existing observation harness's case enumeration changes. Historical
fixtures and supplied application functions are never replaced.
"""
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np

PROOF=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROOF/'scripts'))
from common import ROOT,sha,write_json


def main():
    source=(PROOF/'runtime/unit5_probe.py').read_text()
    start=source.index('    for seed in (0,42,17):')
    end=source.index('    if full:',start)
    source=source[:start]+source[end:]
    cases=[(seed,256,False,False) for seed in (0,42,123)]+[(42,257,False,False),(42,256,True,True)]
    source=source.replace("        for dataset in ('mnist','fashion'):",
        "        for dataset, seed, batch, flip, invert in [(ds,*case) for case in "+repr(cases)+" for ds in ('mnist','fashion')]:")
    source=source.replace("key=f'full/{dataset}/seed42'", "key=f'full/{dataset}/seed{seed}/batch{batch}/flip{int(flip)}/invert{int(invert)}'")
    source=source.replace('inputs(mn_seed=42,mn_batch=256,', 'inputs(mn_seed=seed,mn_batch=batch,')
    source=source.replace('mn_hflip=False,mn_invert=False)', 'mn_hflip=flip,mn_invert=invert)')
    target=PROOF/'evidence/neural-training-matrix-probe.py'
    target.write_text(source)
    scope={'__name__':'native_training_matrix'}
    exec(compile(source,str(target),'exec'),scope)
    sys.path.insert(0,str(PROOF/'runtime'))
    import torch
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('u5_original_matrix',ROOT/'assignments/5/u5_utils.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    definitions=(PROOF/'site/probes/unit5-definitions.py').read_text()
    os.chdir(PROOF/'native/unit5')
    arrays=asyncio.run(scope['run_unit5'](torch,helper,definitions,str(PROOF/'site/probes/unit5-data.csv'),True))
    historical=np.load(PROOF/'site/probes/unit5-full-native.npz')
    for ds in ('mnist','fashion'):
        for key in historical.files:
            prefix=f'full/{ds}/seed42/'
            if key.startswith(prefix):
                current=key.replace(prefix,f'full/{ds}/seed42/batch256/flip0/invert0/')
                assert np.array_equal(arrays[current],historical[key]),key
    fixture=PROOF/'evidence/neural-training-matrix-native.npz'
    np.savez_compressed(fixture,**arrays)
    write_json(PROOF/'evidence/neural-training-matrix-inputs.json',dict(
        method='Original full training/evaluation bodies; complete datasets, three epochs; original seed42 results verified bitwise unchanged.',
        cases=[dict(dataset=ds,seed=seed,batch=batch,flip=flip,invert=invert) for seed,batch,flip,invert in cases for ds in ('mnist','fashion')],
        source_sha256=sha(ROOT/'assignments/5/app.py'),helper_sha256=sha(ROOT/'assignments/5/u5_utils.py'),
        harness_sha256=sha(target),fixture_sha256=sha(fixture),environment_sha256=sha(PROOF/'requirements-native.lock'),
        arrays=len(arrays),historical_fixture_sha256=sha(PROOF/'site/probes/unit5-full-native.npz')))
    print(len(arrays),'native arrays; historical seed42 arrays unchanged.',flush=True)


if __name__=='__main__':main()
