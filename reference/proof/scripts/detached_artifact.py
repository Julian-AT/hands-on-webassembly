"""Detach writable artifact files before rebuilding a hardlink-shared site."""
import os
import shutil


def detach_shared_artifact(site):
    retained=[]
    if not site.exists():return retained
    for path in sorted(site.rglob('*')):
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink<2:continue
        relative=path.relative_to(site)
        # Pinned datasets may be shared; builds never modify their bytes.
        if relative.parts[:3] in (('assets','v1','images'),('assets','v1','embedding-runtime')):continue
        temporary=path.with_name(path.name+'.course-detach')
        if temporary.exists():raise FileExistsError(temporary)
        shutil.copy2(path,temporary)
        os.replace(temporary,path)
        retained.append(str(relative))
    return retained
