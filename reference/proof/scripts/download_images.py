"""Fetch large original archives in bounded ranges, verify published MD5s.

Downloads are resumable build inputs, not browser requests. A file is promoted
from .part only after its exact torchvision checksum is verified.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import urllib.request
import re
import time
import argparse
from common import PROOF


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'md5').hexdigest()


def fetch(url,filename,expected,*,chunk_bytes=4*1024*1024,workers=8):
    root=PROOF/'cache/torchvision';root.mkdir(parents=True,exist_ok=True)
    dest=root/filename
    if dest.exists() and digest(dest)==expected:return
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as response:
        size=int(response.headers['Content-Length']);url=response.url
    chunks=root/(filename+'.chunks');chunks.mkdir(exist_ok=True)
    if not 64*1024 <= chunk_bytes <= 4*1024*1024 or not 1 <= workers <= 32:
        raise ValueError('Use 64 KiB–4 MiB chunks and 1–32 download workers')
    step=chunk_bytes
    def part(start):
        end=min(size-1,start+step-1);target=chunks/str(start)
        if target.exists() and target.stat().st_size==end-start+1:return
        for attempt in range(3):
            try:
                temporary=target.with_suffix('.part')
                resume=temporary.stat().st_size if temporary.exists() else 0
                if resume>end-start+1:raise ValueError('Oversized partial byte range')
                if resume==end-start+1:
                    temporary.replace(target)
                    return
                first=start+resume
                req=urllib.request.Request(url,headers={'Range':f'bytes={first}-{end}'})
                with urllib.request.urlopen(req,timeout=120) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {first}-{end}/{size}':raise ValueError('Server did not honor exact byte range')
                    with temporary.open('ab') as stream:
                        while data:=response.read(256*1024):stream.write(data)
                    if temporary.stat().st_size!=end-start+1:raise ValueError('Incomplete byte range')
                    temporary.replace(target)
                return
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(part,start) for start in range(0,size,step)]
        for i,future in enumerate(as_completed(jobs),1):
            future.result();print(filename,i,'/',len(jobs),'ranges',flush=True)
    partial=dest.with_suffix(dest.suffix+'.part')
    with partial.open('wb') as stream:
        for start in range(0,size,step):
            with (chunks/str(start)).open('rb') as chunk:
                while data:=chunk.read(1024*1024):stream.write(data)
    if digest(partial)!=expected:raise ValueError(f'Checksum mismatch: {filename}')
    partial.replace(dest)
    print(filename,'verified',flush=True)


def main():
    import torchvision
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['CIFAR10','SVHN','USPS'],action='append')
    parser.add_argument('--chunk-kib',type=int,default=4096)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    selected=args.dataset or ['CIFAR10','SVHN','USPS']
    options=dict(chunk_bytes=args.chunk_kib*1024,workers=args.workers)
    if 'CIFAR10' in selected:
        c=torchvision.datasets.CIFAR10
        fetch(c.url,c.filename,c.tgz_md5,**options)
    for name in ('SVHN','USPS'):
        if name not in selected:continue
        cls=getattr(torchvision.datasets,name)
        for split in ('train','test'):
            url,filename,md5=cls.split_list[split]
            fetch(url,filename,md5,**options)


if __name__=='__main__':main()
