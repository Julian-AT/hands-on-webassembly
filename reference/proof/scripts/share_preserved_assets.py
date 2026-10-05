"""Share byte-identical pinned assets without dropping any candidate or report.

Only checksum-bound image and embedding data are eligible. Runtime scripts,
candidate bundles, native fixtures and historical evidence remain detached.
"""
import argparse
import filecmp
import json
import itertools
import os
from pathlib import Path
from common import PROOF, sha, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--include-site',action='store_true')
    args=parser.parse_args()
    baseline=json.loads((PROOF/'continuation-baseline/manifest.corrected.json').read_text())
    canonical={}
    for folder in ('images','embedding-runtime'):
        for path in (PROOF/f'assets/v1/{folder}').rglob('*'):
            if path.is_file():
                name='proof/'+str(path.relative_to(PROOF))
                expected=baseline['files'].get(name,{}).get('sha256')
                if expected is None:continue
                if sha(path)!=expected:raise ValueError(f'Pinned canonical asset changed: {path}')
                canonical[f'{folder}/{path.relative_to(PROOF/f"assets/v1/{folder}")}']=path
    records=[]
    trees=[(PROOF/'cache').rglob('*')]
    if args.include_site:trees.append((PROOF/'site').rglob('*'))
    for path in itertools.chain.from_iterable(trees):
        if not path.is_file() or path.is_symlink():continue
        relative=path.relative_to(PROOF)
        parts=relative.parts
        for index in range(len(parts)-3):
            if parts[index:index+2]!=('assets','v1'):continue
            key='/'.join(parts[index+2:])
            source=canonical.get(key)
            if source is None:break
            before=path.stat();base=source.stat()
            if before.st_ino==base.st_ino and before.st_dev==base.st_dev:break
            if before.st_size!=base.st_size or not filecmp.cmp(source,path,shallow=False):break
            digest=sha(source)
            if sha(path)!=digest:raise ValueError(f'Asset changed during comparison: {path}')
            record=dict(path=str(relative),source=str(source.relative_to(PROOF)),bytes=before.st_size,sha256=digest)
            if args.apply:
                temporary=path.with_name(path.name+'.course-share')
                if temporary.exists():raise FileExistsError(temporary)
                os.link(source,temporary)
                os.replace(temporary,path)
                if sha(path)!=digest:raise ValueError(f'Shared asset differs: {path}')
                record['status']='shared'
            else:record['status']='eligible'
            records.append(record)
            break
    report=dict(status='pass',applied=args.apply,files=records,eligible_bytes=sum(r['bytes'] for r in records),
        baseline_sha256=sha(PROOF/'continuation-baseline/manifest.corrected.json'),
        byte_count_definition='Sum of eligible path lengths; pre-existing hardlinks mean this is not reclaimed disk space.',
        scope='Byte-identical immutable dataset and embedding assets only; all candidate paths and evidence retained.',executor_sha256=sha(Path(__file__)))
    if args.apply:write_json(PROOF/'evidence/preserved-assets-sharing.json',report)
    print(report['status'],len(records),'files',report['eligible_bytes'],'eligible bytes')


if __name__=='__main__':main()
