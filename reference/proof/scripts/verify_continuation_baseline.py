"""Repair bindings from preserved bytes without rewriting the original manifest."""
import json
from pathlib import Path
from common import ROOT, PROOF, sha, write_json


def repair(root, baseline):
    original_path = baseline / 'manifest.json'
    original = json.loads(original_path.read_text())
    files, inconsistencies, failures = {}, [], []
    for name, expected in original['files'].items():
        source = root / name
        saved = baseline / name
        if saved.is_file():
            binding = saved
            actual = {'sha256': sha(saved), 'bytes': saved.stat().st_size}
            if actual != expected:
                inconsistencies.append(dict(path=name, kind='copied-file-binding',
                    original=expected, corrected=actual))
        else:
            binding = source
            actual = {'sha256': sha(binding), 'bytes': binding.stat().st_size} if binding.is_file() else None
            if actual != expected:
                # Mutable reports archive their previous bytes before replacement.
                candidates = []
                if name.startswith('proof/evidence/'):
                    relative = Path(name).relative_to('proof/evidence')
                    candidates = sorted((root / 'proof/evidence/history' / relative.parent).glob(
                        relative.stem + '-*' + relative.suffix))
                binding = next((p for p in candidates if p.stat().st_size == expected['bytes']
                                and sha(p) == expected['sha256']), None)
                if binding is None:
                    failures.append(dict(path=name, reason='Original checksum-bound bytes unavailable'))
                    continue
                actual = expected
                inconsistencies.append(dict(path=name, kind='archived-evidence',
                    resolved_path=str(binding.relative_to(root))))
        files[name] = dict(actual, binding=str(binding.relative_to(root)),
                           storage='preserved-copy' if saved.is_file() else 'checksum-bound')
    # Include every preserved file, even ones omitted by the first manifest.
    for saved in sorted(baseline.rglob('*')):
        if not saved.is_file() or '__pycache__' in saved.parts:
            continue
        name = str(saved.relative_to(baseline))
        if name.startswith('proof/') and name not in files:
            files[name] = dict(sha256=sha(saved), bytes=saved.stat().st_size,
                binding=str(saved.relative_to(root)), storage='preserved-copy')
            inconsistencies.append(dict(path=name, kind='omitted-preserved-file'))
    environment = baseline / 'native-environment.txt'
    if not environment.is_file() or sha(environment) != original['environment_sha256']:
        failures.append(dict(path='native-environment.txt', reason='Environment checksum mismatch'))
    return dict(status='pass' if not failures else 'fail',
        original_manifest_sha256=sha(original_path),
        environment_sha256=sha(environment) if environment.is_file() else None,
        files=files, inconsistencies=inconsistencies, failures=failures,
        scope='Preserved copies are authoritative; unchanged inputs retain original hashes. No original bytes or manifest modified.')


def resolve_binding(root, name, entry):
    path = (root / entry['binding']).resolve()
    def matches(candidate):
        candidate = candidate.resolve()
        return (candidate.is_relative_to(root.resolve()) and candidate.is_file()
                and candidate.stat().st_size == entry['bytes'] and sha(candidate) == entry['sha256'])
    if matches(path):return path
    if entry.get('storage') == 'preserved-copy':return None
    # Runnable corrected references and retained patches can legitimately be
    # extended after a native defect is reproduced. Their original bytes must
    # first be copied and verified against this immutable manifest. Supplied
    # sources, datasets and numerical fixtures never use this fallback.
    if name.startswith(('proof/reference/', 'proof/reference-patches/')):
        saved = root/'proof/continuation-baseline'/name
        if matches(saved):return saved
    if name.startswith('proof/evidence/'):
        saved = root/'proof/continuation-baseline'/name
        if matches(saved):return saved
        relative = Path(name).relative_to('proof/evidence')
        for archived in sorted((root/'proof/evidence/history'/relative.parent).glob(
                relative.stem+'-*'+relative.suffix)):
            if matches(archived):return archived
    return None


def verify(root, record):
    failures = []
    for name, entry in record['files'].items():
        if resolve_binding(root, name, entry) is None:
            failures.append(name)
    return failures


def main():
    baseline = PROOF / 'continuation-baseline'
    corrected = baseline / 'manifest.corrected.json'
    if corrected.exists():
        record = json.loads(corrected.read_text())
        failures = verify(ROOT, record)
        if sha(baseline/'manifest.json') != record['original_manifest_sha256']:
            failures.append('Original manifest changed')
        if sha(baseline/'native-environment.txt') != record['environment_sha256']:
            failures.append('Preserved environment changed')
    else:
        record = repair(ROOT, baseline)
        failures = record['failures'] + verify(ROOT, record)
        if not failures:
            corrected.write_text(json.dumps(record, indent=2) + '\n')
    report = dict(status='fail' if failures else 'pass', failures=failures,
        files_verified=len(record['files']), inconsistencies=record['inconsistencies'],
        corrected_manifest=str(corrected.relative_to(PROOF)),
        original_manifest_sha256=sha(baseline / 'manifest.json'),
        corrected_manifest_sha256=sha(corrected) if corrected.exists() else None)
    write_json(PROOF / 'evidence/continuation-baseline-verification.json', report)
    print(report['status'], report['files_verified'], 'files;', len(report['inconsistencies']), 'binding corrections')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
