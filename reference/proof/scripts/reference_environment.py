"""Record the selected numerical reference without changing its packages."""
import importlib.metadata
import json
import platform
import re
import sys
from datetime import datetime, timezone
from common import ROOT, PROOF, sha, write_json


def main():
    course = ROOT / 'iml_env.yaml'
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in importlib.metadata.distributions()}
    # The supplied YAML only needs name/version declarations, not YAML evaluation.
    declarations = []
    for line in course.read_text().splitlines():
        match = re.match(r'\s*-\s+([\w-]+)(?:={1,2}([^\s#]+))?\s*(?:#.*)?$', line)
        if not match or match[1] in ('conda-forge', 'defaults', 'pip'):
            continue
        name, requested = match.groups()
        actual = platform.python_version() if name == 'python' else packages.get({'opencv':'opencv-python', 'pillow':'pillow'}.get(name, name))
        matches = actual is not None and (requested is None or actual == requested or name == 'python' and actual.startswith(requested + '.'))
        declarations.append({'package':name, 'course_version':requested, 'reference_version':actual,
                             'comparison':'unpinned' if requested is None else 'match' if matches else 'different_or_absent'})
    data = {'generated_at':datetime.now(timezone.utc).isoformat(),
            'selection':'Frozen working local Python environment explicitly selected by user; course YAML is provenance, not the numerical target.',
            'python':sys.version, 'platform':platform.platform(), 'packages':dict(sorted(packages.items())),
            'course_yaml_sha256':sha(course), 'native_lock_sha256':sha(PROOF/'requirements-native.lock'),
            'course_comparison':declarations,
            'unspecified_by_course':['torch','torchvision','en-core-web-md'],
            'execution_policy':{'device':'cpu','compute_threads':1,'seeds':'Recorded per numerical fixture; original app defaults preserved.'}}
    write_json(PROOF/'evidence/reference-environment.json', data)
    lines = ['# Numerical reference', '', data['selection'], '',
             '`iml_env.yaml` is preserved byte for byte. Shiny is unpinned there; PyTorch and torchvision are absent. The course setup separately identifies `en_core_web_md` 3.7.1.', '',
             '| Package | Course YAML | Frozen reference |', '|---|---|---|']
    for item in declarations:
        lines.append(f"| {item['package']} | {item['course_version'] or 'unpinned'} | {item['reference_version'] or 'absent'} |")
    lines += ['', 'The complete installed-package snapshot and file checksums are in `evidence/reference-environment.json`. No environment upgrade was performed. Browser package differences remain subject to numerical verification.']
    (PROOF/'REFERENCE.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
