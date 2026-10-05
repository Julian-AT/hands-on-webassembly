"""Prepare all three browser applications against the verified disposable image owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import PROOF,sha,write_json
from storage_budget import require_space
from stage_image_workers import application,write_config
import ast


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    root=PROOF/'cache'/f'image-worker-app-{args.label}'
    if root.exists():raise FileExistsError('Preserve candidates; use a fresh label')
    require_space(512*1024**2)
    site=root/'site';shutil.copytree(PROOF/'site',site,copy_function=os.link)
    records={}
    workers=('course-image-worker.js','course-image-decode-worker.js')
    for name in workers:
        target=site/'shinylive'/name;target.unlink(missing_ok=True)
        shutil.copy2(PROOF/'runtime'/name,target)
    for unit in (5,6,7):
        stage=root/f'unit{unit}';shutil.copytree(PROOF/f'build/unit{unit}',stage)
        source=(stage/'app.py').read_text()
        if 'import image_worker_preload as image_preload\n' in source:
            owners=[node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='image_preload' and node.func.attr=='Owner']
            if len(owners)!=(2 if unit==5 else 1) or any(
                    not node.args or not isinstance(node.args[0],ast.Constant) or node.args[0].value!=unit
                    for node in owners):raise ValueError('Integrated image ownership contract changed')
        else:
            source=application(source,unit)
        (stage/'app.py').write_text(source)
        config,inputs=write_config(stage,source,unit)
        exported=root/f'export{unit}'
        subprocess.run([str(Path(sys.executable).parent/'shinylive'),'export',str(stage),str(exported),
            '--subdir',f'unit{unit}'],check=True)
        destination=site/f'unit{unit}/app.json';old_id=sha(destination);destination.unlink()
        entries=json.loads((exported/f'unit{unit}/app.json').read_text())
        entries.sort(key=lambda entry:(entry['name']!='app.py',entry['name']))
        destination.write_text(json.dumps(entries,sort_keys=True,separators=(',',':')))
        index=site/f'unit{unit}/index.html';html=index.read_text();index.unlink();index.write_text(html.replace(old_id,sha(destination)))
        records[str(unit)]=dict(config=config,inputs=inputs,application_sha256=sha(stage/'app.py'),bundle_sha256=sha(destination))
    write_json(PROOF/f'evidence/image-worker-app-candidate-{args.label}.json',dict(status='candidate',
        artifact=str(site.relative_to(PROOF)),executor_sha256=sha(Path(__file__)),units=records,
        scope='Detached all-three application candidate. Installed-browser verification required before promotion.'))
    print(site,flush=True)


if __name__=='__main__':main()
