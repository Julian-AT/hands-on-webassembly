"""Validate every dependency in aggregate evidence; never trust its pass summary."""
import json
from pathlib import Path
from common import sha
from assertions import assertion_matches
from host_conditions import condition_errors


def references(value):
    if isinstance(value, dict):
        if 'report' in value:
            yield value
        for key, child in value.items():
            if key != 'report':
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def validate_tree(report, proof, fingerprint, *, browser=None, version=None,
                  hardware=None, ancestors=(), fingerprint_resolver=None, case_validator=None,
                  require_awake_host=False):
    errors = []
    if require_awake_host or 'cycle_evidence' in report:
        errors.extend(condition_errors(report.get('host_conditions')))
    for reference in references(report):
        name = reference.get('report')
        if not isinstance(name, str):
            errors.append('Child report path is invalid')
            continue
        path = (proof / name).resolve()
        evidence = (proof / 'evidence').resolve()
        if not path.is_relative_to(evidence) or not path.is_file():
            errors.append(f'{name}: child report missing or outside evidence')
            continue
        if path in ancestors:
            errors.append(f'{name}: cyclic evidence dependency')
            continue
        expected_hash = reference.get('report_sha256')
        if not expected_hash or sha(path) != expected_hash:
            errors.append(f'{name}: child checksum missing or mismatched')
            continue
        try:
            child = json.loads(path.read_text())
            if not isinstance(child, dict):
                raise ValueError('Expected an object')
        except (ValueError, OSError) as error:
            errors.append(f'{name}: invalid child report ({error})')
            continue
        if reference.get('status') != 'pass' or child.get('status') != 'pass':
            errors.append(f'{name}: child is not passing')
        stamp = child.get('provenance', {})
        actual = stamp.get('fingerprint', child.get('input_fingerprint'))
        bound = reference.get('input_fingerprint', reference.get('fingerprint'))
        try:
            current_child = fingerprint_resolver(child, name) if fingerprint_resolver else fingerprint
        except (ValueError, TypeError, KeyError) as error:
            current_child = None
            errors.append(f'{name}: child dependency scope invalid ({error})')
        if not bound or actual != bound or actual != current_child:
            errors.append(f'{name}: child dependency fingerprint missing or stale')
        expected_browser = browser or report.get('browser') or reference.get('browser') or child.get('browser')
        expected_version = version or report.get('version') or reference.get('version') or child.get('version')
        if expected_browser and (child.get('browser') != expected_browser or
                child.get('distribution') != 'installed' or not child.get('version') or
                child.get('version') != expected_version):
            errors.append(f'{name}: child installed browser identity/version mismatched')
        if hardware is not None and child.get('hardware') != hardware:
            errors.append(f'{name}: child physical hardware differs from aggregate claim')
        for section in ('cases', 'comparisons', 'checks', 'suites'):
            checks = child.get(section)
            if checks is not None and not isinstance(checks, dict):
                errors.append(f'{name}: child {section} has an invalid check map')
            if isinstance(checks, dict) and any(not isinstance(c, dict) or c.get('status') != 'pass'
                                               for c in checks.values()):
                errors.append(f'{name}: child contains nonpassing checks')
            if isinstance(checks, dict):
                for case_name, check in checks.items():
                    if not isinstance(check, dict): continue
                    if 'assertion' in check and not assertion_matches(check['assertion']):
                        errors.append(f'{name}/{case_name}: child assertion missing values or contradictory')
                    if case_validator:
                        errors.extend(f'{name}/{case_name}: {error}' for error in case_validator(check))
        if not any(isinstance(child.get(section), dict) and child[section]
                   for section in ('cases', 'comparisons', 'checks', 'suites')):
            errors.append(f'{name}: child has no executed checks or bounded children')
        errors.extend(validate_tree(child, proof, fingerprint, browser=expected_browser,
            version=expected_version, hardware=hardware, ancestors=(*ancestors, path),
            fingerprint_resolver=fingerprint_resolver, case_validator=case_validator,
            require_awake_host=require_awake_host))
    return errors
