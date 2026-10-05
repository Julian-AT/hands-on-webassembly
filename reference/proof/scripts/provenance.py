"""Content fingerprints invalidate evidence after source/runtime/test changes."""
import hashlib
import inspect
import json
from datetime import datetime, timezone
from common import ROOT, PROOF, UNITS, sha


def fingerprint():
    paths = [ROOT/'iml_env.yaml', PROOF/'requirements-native.lock', PROOF/'source.lock.json', PROOF/'assets.lock.json', PROOF/'assignment5-materials.lock.json', PROOF/'hardware-contract.json']
    for folder in [*(ROOT/folder for folder in UNITS.values()),PROOF/'runtime',PROOF/'scripts',PROOF/'tests',PROOF/'web',PROOF/'packages',ROOT/'tools',PROOF/'assets',PROOF/'site']:
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.') and p.suffix != '.log')
    manifest = {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.exists()}
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def stamp():
    return {'fingerprint':fingerprint(),'recorded_at':datetime.now(timezone.utc).isoformat()}


def browser_fingerprint(kind):
    """Actual deployment/reference inputs plus the selected executing harness.

Unrelated validators and unit-test files do not change an executed browser
workflow. Release requirements are checked separately by the coverage gate.
"""
    harnesses = {
        'startup':['browser_startup.py'], 'recovery':['browser_recovery.py','browser_startup.py'],
        'upgrade':['browser_upgrade_fault.py','browser_startup.py'],
        **{str(unit):['browser.py'] for unit in UNITS},
        **{name:['browser.py'] for name in ('uploads','neural','tabular','supervised','embedding',
                                           'training','unit5probe','unit5full','runtime','unit5life')},
    }
    for unit, helper in {'2':'browser_unit2.py','4':'browser_unit4.py','5':'browser_unit5.py',
                         '6':'browser_neural_ui.py','7':'browser_neural_ui.py','unit5life':'browser_unit5.py'}.items():
        harnesses[unit].append(helper)
    if kind not in harnesses:
        raise ValueError(f'Unknown browser harness: {kind}')
    paths = [ROOT/'iml_env.yaml', PROOF/'requirements-native.lock', PROOF/'source.lock.json',
             PROOF/'assets.lock.json', PROOF/'assignment5-materials.lock.json']
    paths += [PROOF/'scripts'/name for name in ['common.py','provenance.py','reference_corrections.py',
        'build.py','stage_neural.py','training_bundle.py','patch_runtime.py','pin_runtime_packages.py',*harnesses[kind]]]
    for folder in [*(ROOT/folder for folder in UNITS.values()), PROOF/'runtime', PROOF/'web',
                   PROOF/'packages', PROOF/'assets', PROOF/'site', ROOT/'tools']:
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and not p.name.startswith('.') and p.suffix != '.log')
    manifest = {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.exists()}
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def browser_stamp(kind):
    return {'scope':'browser', 'harness':kind, 'fingerprint':browser_fingerprint(kind),
            'recorded_at':datetime.now(timezone.utc).isoformat()}


def native_fingerprint(unit):
    """Bind corrected native UI evidence independently of browser deployment.

Test and shared-correction changes still invalidate it. The actual browser
version/hardware remain recorded by the executing harness.
"""
    if unit not in UNITS:
        raise ValueError('Unknown native assignment')
    reference = PROOF / 'reference' / f'unit{unit}'
    if not (reference / 'app.py').exists():
        raise ValueError('Generate the corrected runnable reference first')
    paths = [ROOT/'iml_env.yaml', PROOF/'requirements-native.lock', PROOF/'source.lock.json',
             PROOF/'assignment5-materials.lock.json', PROOF/'scripts/reference_corrections.py',
             PROOF/'scripts/common.py']
    paths.append(PROOF/'scripts/browser.py')
    test_module = {2:'browser_unit2.py', 4:'browser_unit4.py', 5:'browser_unit5.py',
                   6:'browser_neural_ui.py', 7:'browser_neural_ui.py'}.get(unit)
    if test_module:
        paths.append(PROOF/'scripts'/test_module)
    paths += list((PROOF/'tests').glob('test_reference*.py'))
    folders = [reference, ROOT/UNITS[unit]]
    if unit in (5, 6, 7):
        folders.append(PROOF/'cache/torchvision')
    for folder in folders:
        paths += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and not p.name.startswith('.') and not any('.chunks' in part for part in p.parts)]
    manifest = {str(path.relative_to(ROOT)):sha(path) for path in sorted(set(paths))}
    # Changes to browser evidence binding do not change a native UI execution.
    # Still bind the native fingerprint/stamp algorithms themselves.
    manifest['$native-evidence-algorithm']=hashlib.sha256(
        (inspect.getsource(native_fingerprint)+inspect.getsource(native_stamp)).encode()).hexdigest()
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def native_stamp(unit):
    return {'scope':'corrected-native', 'unit':unit, 'fingerprint':native_fingerprint(unit),
            'recorded_at':datetime.now(timezone.utc).isoformat()}
