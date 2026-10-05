"""Coverage-based, content-bound release gate. Missing/stale checks fail closed."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from common import ROOT, PROOF, sha, write_json
from coverage_contract import requirements
from provenance import fingerprint, native_fingerprint, browser_fingerprint
from evidence_tree import validate_tree
from dynamic_choices import validate_matrix
from assertions import assertion_matches
from host_conditions import condition_errors


def bound_case_errors(check, *, appearance=False):
    """Validate case-level bindings equally at every depth of an aggregate."""
    errors = []
    root = ROOT.resolve()
    executor = check.get('executor', {})
    dependencies = check.get('dependencies', {})
    allowed = {'pixel-equality'} if appearance else {
        'value', 'content', 'computation', 'workflow', 'recoverable-error', 'download-roundtrip'}
    assertion = check.get('assertion', {})
    if (not isinstance(executor, dict) or not isinstance(executor.get('path'), str)
            or not executor.get('path') or not executor.get('sha256')):
        errors.append('Executor binding missing')
    else:
        path = (root / executor['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != executor['sha256']:
            errors.append('Executor binding stale or outside workspace')
    if not isinstance(dependencies, dict) or not dependencies:
        errors.append('Dependency bindings missing')
    else:
        for dependency, digest in dependencies.items():
            if not isinstance(dependency, str):
                errors.append('Dependency path invalid')
                continue
            path = (root / dependency).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
                errors.append(f'Dependency stale or outside workspace: {dependency}')
    if (not isinstance(assertion, dict) or assertion.get('kind') not in allowed
            or not assertion_matches(assertion)):
        errors.append('Behavior assertion missing or contradictory')
    if not isinstance(assertion, dict): assertion = {}
    if appearance and (assertion.get('different_pixels') != 0 or assertion.get('masks') != []
            or assertion.get('tolerance') != 0 or not assertion.get('matched_state')
            or not assertion.get('viewport') or not assertion.get('display_scale')
            or not assertion.get('native_capture_sha256') or not assertion.get('browser_capture_sha256')):
        errors.append('Exact matched-state pixel comparison missing')
    if appearance:
        from PIL import Image, ImageChops
        captures = []
        for kind in ('native', 'browser'):
            name = assertion.get(f'{kind}_capture_path')
            digest = assertion.get(f'{kind}_capture_sha256')
            if not isinstance(name, str) or not name:
                errors.append(f'{kind} capture path missing')
                continue
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
                errors.append(f'{kind} capture missing, outside workspace or checksum mismatched')
                continue
            try:
                with Image.open(path) as image:
                    captures.append(image.convert('RGBA'))
            except (OSError, ValueError) as error:
                errors.append(f'{kind} capture cannot be decoded: {error}')
        if len(captures) == 2 and (captures[0].size != captures[1].size or
                ImageChops.difference(captures[0], captures[1]).getbbox(alpha_only=False) is not None):
            errors.append('Capture pixels differ despite the passing assertion')
    return errors


def evaluate(requirement, report, current_fingerprint, proof=PROOF):
    if report is None: return {'status':'missing','reason':'Required report has not been produced'}
    if report.get('provenance',{}).get('fingerprint') != current_fingerprint:
        return {'status':'stale','reason':'Evidence does not match current source, assets, runtime and tests'}
    if report.get('status') != 'pass':
        return {'status':'fail','reason':f"Report status: {report.get('status', 'missing')}"}
    def child_fingerprint(child, name):
        stamp = child.get('provenance', {})
        scope = stamp.get('scope')
        if scope == 'browser':
            harness = stamp.get('harness')
            if not Path(name).name.endswith(f'-unit{harness}.json'):
                raise ValueError('Browser harness does not match child report')
            return browser_fingerprint(harness)
        if scope == 'corrected-native':
            unit = stamp.get('unit')
            if not Path(name).name.startswith('native-') or not Path(name).name.endswith(f'-unit{unit}.json'):
                raise ValueError('Native scope does not match child report')
            return native_fingerprint(unit)
        if scope in ('isolated-artifact','two-isolated-artifacts'):
            from artifact_provenance import fingerprint as artifact_fingerprint
            return artifact_fingerprint(child,proof)
        if scope is not None:
            raise ValueError('Unknown child dependency scope')
        return current_fingerprint
    child_errors = validate_tree(report, proof, current_fingerprint,
        browser=requirement.get('browser'), version=report.get('version'),
        hardware=report.get('hardware') if requirement.get('hardware') else None,
        fingerprint_resolver=child_fingerprint,
        require_awake_host=bool(requirement.get('minimum_consecutive_cycles') or requirement.get('hardware') or requirement.get('comprehensive')),
        case_validator=(lambda check: bound_case_errors(check, appearance=requirement.get('appearance', False)))
            if requirement.get('comprehensive') else None)
    if child_errors:
        return {'status':'fail','reason':'Aggregate dependency validation failed','child_errors':child_errors}
    if 'browser' in requirement:
        if report.get('browser') != requirement['browser'] or not report.get('version'):
            return {'status':'fail','reason':'Actual browser identity/version missing or incorrect'}
        if report.get('distribution') != 'installed':
            return {'status':'fail','reason':'Installed production browser execution has not been established'}
    if requirement.get('hardware'):
        from hardware_contract import errors as hardware_errors
        failures = hardware_errors(report.get('hardware'))
        if failures:
            return {'status':'fail','reason':'Active certification device not established',
                    'hardware_errors':failures}
    checks = report.get(requirement['section'],{})
    if not isinstance(checks,dict) or not checks:
        return {'status':'fail','reason':'No executed checks'}
    missing = sorted(set(requirement['required_cases'])-set(checks))
    failed = [name for name,check in checks.items() if not isinstance(check,dict) or check.get('status')!='pass']
    if missing or failed:
        return {'status':'fail','missing_cases':missing,'failed_cases':failed}
    contradictory = [name for name, check in checks.items()
                     if 'assertion' in check and not assertion_matches(check['assertion'])]
    if contradictory:
        return {'status': 'fail', 'reason': 'Expected/observed assertions contradict passing checks',
                'contradictory_assertions': contradictory}
    kinds={'value','content','computation','workflow','recoverable-error','download-roundtrip'}
    insufficient=[]
    for name in requirement.get('behavior_assertions',[]):
        assertion=checks[name].get('assertion',{})
        if (not isinstance(assertion,dict) or assertion.get('kind') not in kinds or
            not assertion_matches(assertion)):
            insufficient.append(name)
    if insufficient:
        return {'status':'fail','reason':'Behavior evidence is missing; DOM presence alone cannot pass',
                'insufficient_assertions':insufficient}
    if requirement.get('comprehensive'):
        incomplete = []
        allowed = {'pixel-equality'} if requirement.get('appearance') else kinds
        for name, check in checks.items():
            if bound_case_errors(check, appearance=requirement.get('appearance', False)):
                incomplete.append(name)
        if incomplete:
            return {'status':'fail', 'reason':'Comprehensive cases require executable, dependency-bound behavior evidence',
                    'incomplete_cases':sorted(set(incomplete))}
        expansion_errors = validate_matrix(report.get('dynamic_choice_matrix', {}),
            requirement.get('unit'), requirement.get('dynamic_controls', []), checks,
            assertion_kinds=allowed)
        matrix = report.get('dynamic_choice_matrix', {})
        matrix_controls = matrix.get('controls', {}) if isinstance(matrix, dict) else {}
        unit = requirement.get('unit')
        native_current = None
        for identifier in requirement.get('dynamic_controls', []):
            record = matrix_controls.get(identifier, {}) if isinstance(matrix_controls, dict) else {}
            for observation in record.get('observations', []) if isinstance(record, dict) else []:
                if not isinstance(observation, dict):continue
                if native_current is None:native_current = native_fingerprint(unit)
                if observation.get('reference_fingerprint') != native_current:
                    expansion_errors.append(f'{identifier}: native discovery is stale')
                executor = observation.get('executor', {})
                if not isinstance(executor, dict):
                    expansion_errors.append(f'{identifier}: native discovery executor is unbound')
                    continue
                bindings = dict(observation.get('dependencies', {}))
                if executor.get('path'):bindings[executor['path']] = executor.get('sha256')
                else:expansion_errors.append(f'{identifier}: native discovery executor is missing')
                for name, digest in bindings.items():
                    path = (ROOT / name).resolve()
                    if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != digest:
                        expansion_errors.append(f'{identifier}: native discovery dependency is stale ({name})')
        if expansion_errors:
            return {'status':'fail','reason':'Dynamic parent-state choice matrix is unresolved',
                    'matrix_errors':expansion_errors}
    minimum=requirement.get('minimum_consecutive_cycles')
    if minimum:
        cycles=report.get('cycle_evidence',[])
        operations=('startup','reload','service-worker-update','build-generation-update')
        if (not isinstance(cycles,list) or len(cycles)<minimum or
            any(not isinstance(cycle,dict) or cycle.get('index')!=index+1 or
                any(not isinstance(cycle.get(operation),dict) or cycle[operation].get('status')!='pass'
                    or cycle[operation].get('assertion',{}).get('kind')!='workflow'
                    or not assertion_matches(cycle[operation].get('assertion',{})) for operation in operations)
                for index,cycle in enumerate(cycles))):
            return {'status':'fail','reason':f'{minimum} consecutive asserted startup/reload/query-update/build-update cycles not established'}
        if any(not cycle.get('before_generation') or not cycle.get('after_generation')
                or cycle['before_generation']==cycle['after_generation']
                or (index and cycles[index-1]['after_generation']!=cycle['before_generation'])
                or cycle['build-generation-update']['assertion']['observed']!=cycle['after_generation']
                for index,cycle in enumerate(cycles)):
            return {'status':'fail','reason':'A continuous sequence of genuine build-generation changes is missing'}
        conditions=report.get('host_conditions',{})
        host_errors=condition_errors(conditions)
        if host_errors:
            return {'status':'fail','reason':'Uninterrupted native FullWake conditions are not established',
                    'host_condition_errors':host_errors}
        owner=report.get('preview_identity',{})
        if (not owner.get('identity') or not owner.get('marker') or not owner.get('pid')
                or not isinstance(owner.get('port'),int) or not 0<owner['port']<65536):
            return {'status':'fail','reason':'Preview origin ownership is not established'}
        if report.get('diagnostic_instrumentation') is not False:
            return {'status':'fail','reason':'An uninstrumented lifecycle sequence is required'}
        insufficient=[name for name in requirement['required_cases'] if
            checks[name].get('assertion',{}).get('kind') not in kinds
            or not assertion_matches(checks[name].get('assertion',{}))]
        if insufficient:
            return {'status':'fail','reason':'Lifecycle behavior assertions are missing or contradictory',
                    'insufficient_assertions':insufficient}
    return {'status':'pass','executed_cases':len(checks)}


def readiness(rows, results, source_unchanged):
    application_failures = [row['id'] for row in rows if row['acceptance_scope']=='application' and results[row['id']]['status']!='pass']
    release_passed = source_unchanged and all(result['status']=='pass' for result in results.values())
    return source_unchanged and not application_failures, release_passed, application_failures


def main():
    evidence = PROOF/'evidence'
    inventory = json.loads((evidence/'source-inventory.json').read_text())
    current = fingerprint()
    rows = requirements(inventory)
    results = {}
    scoped_fingerprints = {}
    for requirement in rows:
        path = evidence/requirement['report']
        try:
            report = json.loads(path.read_text()) if path.exists() else None
            expected = current
            if report and report.get('provenance',{}).get('scope') == 'corrected-native':
                unit = report['provenance'].get('unit')
                if requirement['report'].startswith(f'native-') and requirement['id'].startswith(f'unit{unit}/'):
                    expected = native_fingerprint(unit)
                else:
                    raise ValueError('Native dependency scope does not match the required report')
            elif report and report.get('provenance',{}).get('scope') == 'browser':
                harness = report['provenance'].get('harness')
                if requirement['report'].startswith('browser-') and requirement['report'].endswith(f'-unit{harness}.json'):
                    if harness not in scoped_fingerprints:
                        scoped_fingerprints[harness] = browser_fingerprint(harness)
                    expected = scoped_fingerprints[harness]
                else:
                    raise ValueError('Browser dependency scope does not match the required report')
            elif report and report.get('provenance',{}).get('scope') is not None:
                raise ValueError('Unknown evidence dependency scope')
            result = evaluate(requirement,report,expected)
        except (ValueError,TypeError,AttributeError) as e:
            result = {'status':'fail','reason':f'Invalid evidence: {e}'}
        results[requirement['id']] = dict(result,report=requirement['report'])
    locked = json.loads((PROOF/'source.lock.json').read_text())
    source_unchanged = all((ROOT/name).is_file() and sha(ROOT/name)==expected for name,expected in locked.items())
    materials = json.loads((PROOF/'assignment5-materials.lock.json').read_text())
    source_unchanged = source_unchanged and all((ROOT/name).is_file() and sha(ROOT/name)==expected for name,expected in materials['materials'].items())
    failures = {id:result for id,result in results.items() if result['status']!='pass'}
    application_passed, passed, application_failures = readiness(rows, results, source_unchanged)
    report = {'generated_at':datetime.now(timezone.utc).isoformat(),
              'status':'PASS' if passed else 'BLOCKED','nextjs_authorized':application_passed,
              'application_parity':'PASS' if application_passed else 'BLOCKED',
              'application_blockers':application_failures,
              'fingerprint':current,'source_unchanged':source_unchanged,
              'required':len(rows),'passed':len(rows)-len(failures),
              'coverage':results,'blockers':[f"{id}: {result['status']} ({result['report']})" for id,result in failures.items()]}
    if not source_unchanged: report['blockers'].append('Original source contract changed')
    write_json(evidence/'coverage-contract.json',rows)
    write_json(evidence/'gate.json',report)
    site = PROOF/'site'
    write_json(evidence/'static-assets.json',{str(p.relative_to(site)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(site.rglob('*')) if p.is_file()})
    print(f"{report['status']}: {report['passed']}/{report['required']} required suites current and passing")
    print('See proof/evidence/gate.json for every missing, stale or failed requirement.')
    return 0 if passed else 1


if __name__=='__main__': sys.exit(main())
