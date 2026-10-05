"""Reproducibly package the verified native-compatible Barnes-Hut kernel."""
import json
import os
import shutil
import subprocess
from common import ROOT,PROOF,sha,write_json


def main():
    source=PROOF/'packages/native_tsne'
    upstream=json.loads((source/'upstream.json').read_text())
    for name,digest in upstream['source_files'].items():
        if sha(source/name)!=digest:raise ValueError('Pinned t-SNE package source changed: '+name)
    stage=PROOF/'cache/native-tsne-build'
    if stage.exists():shutil.rmtree(stage)
    shutil.copytree(source,stage,ignore=shutil.ignore_patterns('build','*.so','*.egg-info','__pycache__'))
    tools=PROOF/'cache/wasm-build-env'
    env=dict(os.environ,SOURCE_DATE_EPOCH='1740787200',PYTHONHASHSEED='0',
        PYODIDE_ROOT=str(ROOT/'.pyodide-xbuildenv-0.29.3/xbuildenv/xbuildenv/pyodide-root'))
    env['PATH']=str(tools/'bin')+os.pathsep+env['PATH']
    subprocess.run(['bash','-c','source "$1" >/dev/null 2>&1; exec "$2" build "$3" --outdir "$4"',
        'native-tsne-build',str(PROOF/'cache/emsdk/emsdk_env.sh'),str(tools/'bin/pyodide'),
        str(stage),str(PROOF/'cache/native-tsne-wheels')],check=True,env=env,cwd=ROOT)
    wheel=PROOF/'cache/native-tsne-wheels/course_native_tsne-1.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl'
    target=PROOF/'assets/v1/runtime-packages'/wheel.name
    if target.exists() and sha(target)!=sha(wheel):raise ValueError('Compiled t-SNE wheel differs from its lock')
    shutil.copy2(wheel,target)
    record=dict(name='course-native-tsne',version='1.0.2',file_name=wheel.name,sha256=sha(wheel),
        depends=['numpy','scikit-learn'],imports=['native_barnes_hut_tsne','native_quad_tree','native_neighbors','neighbor_compat'],
        install_dir='site',package_type='package')
    write_json(target.parent/'tsne-manifest.json',record)
    write_json(target.parent/'tsne-build.json',dict(wheel=record,upstream=upstream,
        source_files={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()},
        recipe_sha256=sha(PROOF/'scripts/build_tsne.py'),emscripten='3.1.58',pyodide='0.27.7',python='3.12',
        abi='2024_0',source_date_epoch=1740787200,flags=['-ffp-contract=off']))
    print(wheel.name,sha(wheel),flush=True)


if __name__=='__main__':main()
