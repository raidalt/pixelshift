"""Shared ExifTool reads with a bounded, file-versioned in-memory cache."""
import json,shutil,subprocess,threading
from collections import OrderedDict
from pathlib import Path

TAGS=('Model','BodySerialNumber','DateTimeOriginal','SubSecTimeOriginal','SequenceNumber',
      'SequenceImageNumber','FocusPosition2','FocusDistance2','ExposureTime','FNumber','ISO',
      'PixelShiftInfo','ImageWidth','ImageHeight','ColorSpace','TransferCharacteristics',
      'ColorPrimaries','ProfileDescription')
CACHE=OrderedDict();LOCK=threading.RLock();LIMIT=4096

def signature(path):
    path=Path(path);stat=path.stat()
    return str(path.absolute()),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns

def read(paths):
    exe=shutil.which('exiftool')
    if not exe:raise ValueError('ExifTool is required to verify capture metadata. Install ExifTool, then restart the app.')
    requests=[];missing=[];existing={}
    with LOCK:
        for path in paths:
            try:key=signature(path)
            except OSError:key=None
            requests.append((str(path),key))
            if key not in CACHE:missing.append((str(path),key))
            elif key is not None:CACHE.move_to_end(key);existing[str(path)]=dict(CACHE[key])
    fresh={}
    for start in range(0,len(missing),256):
        batch=missing[start:start+256]
        run=subprocess.run([exe,'-j','-n',*['-'+tag for tag in TAGS],*[p for p,_ in batch]],capture_output=True,text=True)
        if run.returncode not in (0,1):raise ValueError('Could not read capture metadata: '+run.stderr[:200])
        found={row['SourceFile']:row for row in json.loads(run.stdout or '[]') if 'SourceFile' in row}
        with LOCK:
            for path,key in batch:
                row=found.get(path)
                if row is None:continue
                fresh[path]=row
                # Do not cache a result if the file changed during the external read.
                try:unchanged=key is not None and signature(path)==key
                except OSError:unchanged=False
                if unchanged:CACHE[key]=row
            while len(CACHE)>LIMIT:CACHE.popitem(last=False)
    resolved={**existing,**fresh}
    return [{**resolved[path],'SourceFile':path} for path,_ in requests if path in resolved]
