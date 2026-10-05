"""Build and lock the verified native-compatible linear training extension."""
import json
import os
import shutil
import subprocess
from common import ROOT,PROOF,sha,write_json


def main():
    source=PROOF/'packages/native_neural'
    upstream=json.loads((source/'upstream.json').read_text())
    for name,digest in upstream['files'].items():
        if sha(source/'vendor'/os.path.basename(name)) != digest:
            raise ValueError('Vendored SLEEF source changed: '+name)
    stage=PROOF/'cache/native-neural-build'
    if stage.exists():shutil.rmtree(stage)
    shutil.copytree(source,stage,ignore=shutil.ignore_patterns('build','*.so','*.egg-info','__pycache__'))
    tools=PROOF/'cache/wasm-build-env'
    env=dict(os.environ,SOURCE_DATE_EPOCH='1740787200',PYTHONHASHSEED='0',
        PYODIDE_ROOT=str(ROOT/'.pyodide-xbuildenv-0.29.3/xbuildenv/xbuildenv/pyodide-root'))
    env['PATH']=str(tools/'bin')+os.pathsep+env['PATH']
    subprocess.run(['bash','-c','source "$1" >/dev/null 2>&1; exec "$2" build "$3" --outdir "$4"',
        'native-neural-build',str(PROOF/'cache/emsdk/emsdk_env.sh'),str(tools/'bin/pyodide'),
        str(stage),str(PROOF/'cache/native-neural-wheels')],check=True,env=env,cwd=ROOT)
    wheel=PROOF/'cache/native-neural-wheels/course_native_neural-1.0.0-cp312-cp312-pyodide_2024_0_wasm32.whl'
    target=PROOF/'assets/v1/runtime-packages'/wheel.name
    if target.exists() and sha(target)!=sha(wheel):raise ValueError('Compiled neural wheel differs from its lock')
    shutil.copy2(wheel,target)
    record=dict(name='course-native-neural',version='1.0.0',file_name=wheel.name,
        sha256=sha(wheel),depends=['numpy'],imports=['native_linear'],install_dir='site',package_type='package')
    write_json(target.parent/'neural-manifest.json',record)
    write_json(target.parent/'neural-build.json',dict(wheel=record,source_files={str(path.relative_to(source)):sha(path)
        for path in sorted(source.rglob('*')) if path.is_file()},upstream=upstream,
        emscripten='3.1.58',pyodide='0.27.7',python='3.12',abi='2024_0',source_date_epoch=1740787200,
        flags=['-ffp-contract=off']))
    print(wheel.name,sha(wheel),flush=True)


if __name__=='__main__':main()
