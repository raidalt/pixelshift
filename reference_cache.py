"""Reuse unchanged decoded focus references with exclusive hard links.

The private index lives beside generated images, never in the app checkout.
"""
import hashlib,json,os,uuid
from pathlib import Path
import numpy as np
import capture_metadata


def key(path):
    try:signature=capture_metadata.signature(path)
    except OSError:return None
    return hashlib.sha256(json.dumps(signature,separators=(',',':')).encode()).hexdigest()

def find(paths,parent):
    index=Path(parent)/'.pixelshift-data'/'references.json'
    try:entries=json.loads(index.read_text())
    except (OSError,ValueError):return [None]*len(paths)
    if not isinstance(entries,dict):return [None]*len(paths)
    result=[]
    for path in paths:
        candidate=None;entry=entries.get(key(path))
        if isinstance(entry,dict):
            try:
                file=Path(entry['file']);stat=file.stat()
                preview=file.with_suffix('.preview.npy');preview_valid=not preview.exists() or (preview.stat().st_size,preview.stat().st_mtime_ns)==tuple(entry.get('preview_version',()))
                if preview_valid and (stat.st_size,stat.st_mtime_ns)==tuple(entry['version']) and stat.st_dev==Path(parent).stat().st_dev:
                    array=np.load(file,mmap_mode='r',allow_pickle=False)
                    if array.dtype==np.uint16 and array.ndim==3 and array.shape[2]==3:candidate=file
                    del array
            except (OSError,ValueError,KeyError,TypeError):pass
        result.append(candidate)
    first=next((p for p in result if p is not None),None)
    if first:
        probe=Path(parent)/('.pixelshift-link-'+uuid.uuid4().hex)
        try:os.link(first,probe)
        except OSError:return [None]*len(paths)
        finally:probe.unlink(missing_ok=True)
    return result

def register(paths,cache,versions=None):
    index=Path(cache).parent/'references.json'
    try:entries=json.loads(index.read_text())
    except (OSError,ValueError):entries={}
    if not isinstance(entries,dict):entries={}
    for i,path in enumerate(paths):
        identity=key(path);file=Path(cache)/f'frame-{i}.npy'
        if identity and (versions is None or versions[i]==identity) and file.is_file():
            stat=file.stat();entries.pop(identity,None);entries[identity]={'file':str(file),'version':[stat.st_size,stat.st_mtime_ns]}
            preview=file.with_suffix('.preview.npy')
            if preview.exists():
                stat=preview.stat();entries[identity]['preview_version']=[stat.st_size,stat.st_mtime_ns]
    entries=dict(list(entries.items())[-2000:]);temporary=index.with_name('references-'+uuid.uuid4().hex+'.tmp')
    try:temporary.write_text(json.dumps(entries));os.replace(temporary,index)
    finally:temporary.unlink(missing_ok=True)
