"""Lock pure Python rendering wheels to the frozen reference versions."""
import json
import urllib.request
from common import PROOF, sha, write_json
from prepare import download

PACKAGES = {'plotly': ('7.1.0', ['narwhals','packaging']), 'narwhals': ('2.26.0', [])}


def main():
    entries={}
    for name,(version,depends) in PACKAGES.items():
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60) as response:
            metadata=json.load(response)
        wheel=next(item for item in metadata['urls'] if item['filename'].endswith('py3-none-any.whl'))
        path=download(wheel['url'],'runtime-packages/'+wheel['filename'])
        if sha(path)!=wheel['digests']['sha256']: raise ValueError('PyPI wheel checksum mismatch')
        entries[name]={'name':name,'version':version,'file_name':path.name,'sha256':sha(path),
                       'depends':depends,'imports':[name],'install_dir':'site','package_type':'package'}
    base='https://cdn.jsdelivr.net/pyodide/v0.26.4/full/'
    registry_path=download(base+'pyodide-lock.json','runtime-packages/pyodide-0.26.4-lock.json')
    registry=json.loads(registry_path.read_text())
    if registry['info']['abi_version']!='2024_0' or registry['info']['platform']!='emscripten_3_1_58':
        raise ValueError('NumPy binary build ABI changed')
    entry=registry['packages']['numpy']
    if entry['version']!='1.26.4': raise ValueError('NumPy reference version changed')
    numpy_wheel=download(base+entry['file_name'],'runtime-packages/'+entry['file_name'])
    if sha(numpy_wheel)!=entry['sha256']: raise ValueError('NumPy release checksum mismatch')
    download('https://raw.githubusercontent.com/pyodide/pyodide/0.26.4/packages/numpy/meta.yaml','runtime-packages/numpy-1.26.4-meta.yaml')
    entries['numpy']=entry
    path=PROOF/'assets/v1/runtime-packages/manifest.json'
    if path.exists() and any(entries.get(k)!=v for k,v in json.loads(path.read_text()).items()): raise ValueError('Runtime package lock changed')
    write_json(path,entries)


if __name__=='__main__':main()
