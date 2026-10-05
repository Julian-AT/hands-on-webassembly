"""Preserve the incumbent deployment and checksum continuation inputs once."""
import json
import importlib.metadata
import platform
import shutil
import sys
from datetime import datetime, timezone
from common import ROOT, PROOF, sha, write_json


def main():
    target = PROOF / 'continuation-baseline'
    if (target / 'manifest.json').exists():
        raise FileExistsError('The continuation baseline is immutable; use its retained manifest')
    target.mkdir(exist_ok=True)
    folders = ('assignments', 'proof/assets', 'proof/native', 'proof/reference',
               'proof/reference-patches', 'proof/runtime', 'proof/scripts',
               'proof/tests', 'proof/packages', 'proof/evidence', 'proof/site')
    paths = [ROOT / 'iml_env.yaml', *PROOF.glob('*.lock*')]
    for name in folders:
        paths.extend(p for p in (ROOT / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and not p.name.startswith('.'))
    manifest = {}
    preserved_folders = ('proof/site', 'proof/runtime', 'proof/scripts', 'proof/tests')
    for p in sorted(set(paths)):
        name = str(p.relative_to(ROOT))
        saved = target / name
        if any(name.startswith(folder + '/') for folder in preserved_folders) and (target / name.split('/')[0] / name.split('/')[1]).exists():
            if not saved.exists():
                continue  # Added after the incumbent snapshot; not a baseline input.
            p = saved
        manifest[name] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    # Evidence remains in place with write_json's archive-on-replacement policy.
    # Preserve files that staging/export rewrites, and the editable patch inputs.
    for name in ('proof/site', 'proof/runtime', 'proof/scripts', 'proof/tests'):
        if not (target / name).exists():
            shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns('__pycache__'))
    environment = '\n'.join(sorted(f"{d.metadata['Name']}=={d.version}"
                                     for d in importlib.metadata.distributions())) + '\n'
    (target / 'native-environment.txt').write_text(environment)
    record = dict(status='preserved', created_at=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(), executable=sys.executable,
                  environment_sha256=sha(target / 'native-environment.txt'), files=manifest,
                  scope='Incumbent static deployment and editable implementation copied; sources, datasets, native fixtures and historical evidence checksum-bound in place.')
    (target / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    write_json(PROOF / 'evidence/continuation-baseline.json',
               dict(status='preserved', manifest='continuation-baseline/manifest.json',
                    sha256=sha(target / 'manifest.json'), files=len(manifest)))
    print('Preserved', len(manifest), 'files; incumbent artifact copied to', target)


if __name__ == '__main__':
    main()
