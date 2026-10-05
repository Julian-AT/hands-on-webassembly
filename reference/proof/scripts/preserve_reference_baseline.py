"""Preserve every manifest-bound reference byte before extending corrections."""
import json
import os
import shutil
from common import ROOT,PROOF,sha,write_json
from verify_continuation_baseline import verify


def main():
    baseline=PROOF/'continuation-baseline'
    manifest=baseline/'manifest.corrected.json'
    record=json.loads(manifest.read_text())
    failures=verify(ROOT,record)
    if failures:raise ValueError(f'Preserved baseline must verify before reference changes: {failures}')
    files={}
    for name,entry in record['files'].items():
        if not name.startswith(('proof/reference/','proof/reference-patches/')):continue
        target=baseline/name
        source=ROOT/entry['binding']
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            # The reference generator removes its whole directory before
            # rewriting it. Patches are rewritten in place and need copies.
            if name.startswith('proof/reference/'):os.link(source,target)
            else:shutil.copy2(source,target)
        if sha(target)!=entry['sha256'] or target.stat().st_size!=entry['bytes']:
            raise ValueError(f'Preserved reference differs: {name}')
        files[name]=dict(sha256=sha(target),bytes=target.stat().st_size,path=str(target.relative_to(ROOT)))
    write_json(PROOF/'evidence/continuation-reference-preservation.json',dict(status='pass',
        corrected_manifest_sha256=sha(manifest),files=files,executor_sha256=sha(PROOF/'scripts/preserve_reference_baseline.py'),
        scope='Original corrected reference and patch bytes retained before authorized extension. Original manifests, supplied materials, datasets and numerical fixtures unchanged.'))
    print('Preserved',len(files),'reference bindings')


if __name__=='__main__':main()
