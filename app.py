"""Local-only experimental, image-domain pixel-shift fusion."""
import json, os, secrets, threading, webbrowser, shutil, traceback, subprocess, re
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np
import cv2
import rawpy
import tifffile
import raw_engine
from urllib.parse import parse_qs
import io

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path.home()/'Pictures'/'Pixel Shift'
TOKEN = secrets.token_urlsafe(32)
BASE_H, BASE_W = 6336, 9504
STATE = {'status':'idle', 'progress':0, 'message':'Ready for a verified RAW capture set.'}
COMPARE = {}
COMPARE_LOCK = threading.Lock()
LOCK = threading.Lock()
CANCEL = threading.Event()
CAPTURES = {}
SCAN_LOCK = threading.Lock()

def pick_folder():
    script = 'POSIX path of (choose folder with prompt "Choose your SD card, drive, or photo folder")'
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    if result.returncode:
        if '-128' in result.stderr: return None
        raise ValueError('Could not open the folder picker: '+result.stderr.strip())
    return result.stdout.strip()

def safe_size(path):
    try: return path.stat().st_size
    except OSError: return 0

def scan_captures(folder, count):
    requested=Path(folder).expanduser()
    if not requested.is_absolute() and len(requested.parts)==1 and (Path('/Volumes')/requested).is_dir():
        requested=Path('/Volumes')/requested
    root=requested.resolve()
    if count not in (4,16): raise ValueError('Choose 4 or 16 frames.')
    if not root.is_dir(): raise ValueError('The selected drive or folder is unavailable.')
    found=[]; skipped=0; visited=0; total=0
    def denied(error):
        nonlocal skipped
        skipped+=1
    captures={}; all_raws=[]
    # Explicit stack has no recursion-depth or folder-count limit. Include
    # hidden directories; track directory identities to avoid link cycles.
    pending=[root]; seen=set()
    while pending:
        directory=pending.pop()
        try:
            stat=directory.stat()
            identity=(stat.st_dev,stat.st_ino)
            if identity in seen: continue
            seen.add(identity)
            raws=[]
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        path=Path(entry.path)
                        if entry.is_dir(follow_symlinks=True):
                            # Follow directory links only inside the selected root.
                            if not entry.is_symlink() or path.resolve().is_relative_to(root):
                                pending.append(path)
                        elif entry.is_file() and entry.name.lower().endswith('.arw'):
                            raws.append(path)
                    except OSError as error: denied(error)
            raws.sort(key=lambda p:p.name.lower())
            visited+=1
        except OSError as error:
            denied(error)
            continue
        total+=len(raws)
        all_raws.extend(raws)
    groups,incomplete=raw_engine.group_rows(raw_engine.metadata(all_raws),count)
    for key,ordered in groups:
        batch=[Path(row['SourceFile']) for row in ordered]
        identity=secrets.token_urlsafe(12)
        captures[identity]=batch
        found.append({'id':identity,'folder':str(batch[0].parent),'first':batch[0].name,
            'last':batch[-1].name,'files':[p.name for p in batch],
            'bytes':sum(safe_size(p) for p in batch),'verified':True,'group':key[3]})
    with SCAN_LOCK:
        CAPTURES.update(captures)
        while len(CAPTURES)>1000:CAPTURES.pop(next(iter(CAPTURES)))
    return {'folder':str(root),'sets':found,'total':total,'skipped':skipped,
            'unassigned':total-len(found)*count,'folders':visited,'incomplete':incomplete}


def update(**values):
    with LOCK: STATE.update(values)

def check_cancel():
    if CANCEL.is_set(): raise InterruptedError('Merge cancelled. Temporary data removed.')

def output_path(value, paths, count):
    output=Path(value or DEFAULT_OUTPUT).expanduser().resolve()
    if output==DEFAULT_OUTPUT.resolve(): output.mkdir(parents=True,exist_ok=True)
    if output.is_dir():
        stem=f'PixelShift-{paths[0].stem}-{count}frames'
        output=output/(stem+'.tif')
        number=2
        while output.exists() or output.with_name(output.name+'.partial').exists():
            output=output.parent/f'{stem}-{number}.tif'
            number+=1
    if output.suffix.lower() not in ('.tif','.tiff'):
        raise ValueError('Choose an existing output folder or a filename ending in .tif or .tiff.')
    return output

def srgb_profile():
    candidates=[Path('/System/Library/ColorSync/Profiles/sRGB Profile.icc'),Path('/Library/ColorSync/Profiles/sRGB Profile.icc')]
    for path in candidates:
        if path.is_file():return path.read_bytes()
    return None

def prepare_compare(original,output,meta,scale,regions):
    sensor,source_meta=raw_engine.read_sensor(original)
    image=tifffile.memmap(output)
    left,top,width,height=source_meta['crop']
    previews={}
    positions={'center':(.5,.5),'upper-left':(.25,.25),'upper-right':(.75,.25),'lower-left':(.25,.75),'lower-right':(.75,.75)}
    if regions:
        best=regions[0]
        positions['detail']=((best['left']+best['size']/2-left)/width,(best['top']+best['size']/2-top)/height)
    for region,(fx,fy) in positions.items():
        size=512
        x=max(8,min(width-size-8,int(width*fx-size/2)//2*2));y=max(8,min(height-size-8,int(height*fy-size/2)//2*2))
        # Decode baseline from a padded RAW crop to avoid demosaicing edge artifacts.
        patch=sensor[top+y-8:top+y+size+8,left+x-8:left+x+size+8]
        before=raw_engine.render(raw_engine.demosaic(patch,source_meta),source_meta)[8:-8,8:-8]
        after=np.asarray(image[y*scale:(y+size)*scale,x*scale:(x+size)*scale]).copy()
        if scale==2:before=cv2.resize(before,(1024,1024),interpolation=cv2.INTER_NEAREST)
        for kind,pixels in [('before',before),('after',after)]:
            # PNG previews are 8-bit sRGB; TIFF output remains 16-bit.
            ok,encoded=cv2.imencode('.png',cv2.cvtColor((pixels/257).round().astype(np.uint8),cv2.COLOR_RGB2BGR))
            if not ok:raise ValueError('Could not encode comparison crop.')
            previews[(region,kind)]=encoded.tobytes()
    with COMPARE_LOCK:COMPARE.clear();COMPARE.update(previews)
    del image,sensor

def merge(config):
    scratch = None
    partial = None
    try:
        folder = Path(config['folder']).expanduser().resolve()
        count = int(config.get('count',16))
        if count not in (4,16): raise ValueError('Choose 4 or 16 frames.')
        if config.get('capture'):
            with SCAN_LOCK: paths=list(CAPTURES.get(config['capture'],[]))
            if not paths: raise ValueError('This capture selection expired. Scan the drive again.')
        else:
            paths = sorted([p for p in folder.iterdir() if p.suffix.lower()=='.arw'], key=lambda p:p.name.lower())
        if len(paths) != count: raise ValueError(f'The folder must contain exactly {count} ARW files; found {len(paths)}. Use one capture set per folder.')
        output=output_path(config.get('output',''),paths,count)
        if output.exists(): raise ValueError('Output already exists. Choose a new filename.')
        if not output.parent.is_dir(): raise ValueError('Output directory does not exist.')
        paths,group=raw_engine.verify(paths,count)
        scale=2 if count==16 else 1
        if count==4 and config.get('upscale'):
            raise ValueError('Four-frame interpolation has been removed. Use a verified 16-shot sequence for 240.8 MP.')
        h,w=BASE_H*scale,BASE_W*scale
        needed=count*6400*9600*2+h*w*12+2_000_000_000
        if shutil.disk_usage(output.parent).free<needed:raise ValueError(f'Allow {needed/1e9:.1f} GB free disk space.')
        import tempfile
        scratch=Path(tempfile.mkdtemp(prefix='.pixelshift-',dir=output.parent))
        shifts=[];metas=[];reference=None;alignment_warnings=[];regions=None
        for i,path in enumerate(paths):
            check_cancel()
            update(progress=round(i/count*35),message=f'Reading original sensor samples {i+1}/{count}: {path.name}')
            sensor,meta=raw_engine.read_sensor(path)
            if regions is None:regions=raw_engine.select_regions(sensor,meta)
            patches=raw_engine.gray_patches(sensor,meta,regions)
            if i==0:
                reference=patches
                shift={'x':0.,'y':0.,'confidence':1.,'patch_spread':0.}
            else:shift=raw_engine.alignment(reference,patches)
            shift['name']=path.name;shifts.append(shift);metas.append(meta)
            if shift.get('warning'):
                alignment_warnings.append(f'{path.name}: {shift["warning"]}')
                update(warning=' '.join(alignment_warnings))
            if i and any(meta[k]!=metas[0][k] for k in ('crop','pattern','black','white','wb','transform','flip')):
                raise ValueError('Sensor calibration or white balance changes within this capture set.')
            np.save(scratch/f'{i}.npy',sensor)
            del sensor,patches
        sensors=[np.load(scratch/f'{i}.npy',mmap_mode='r') for i in range(count)]
        candidate=output.with_name(output.name+'.partial')
        if candidate.exists():raise ValueError('Partial output already exists; choose a different filename.')
        partial=candidate
        icc=srgb_profile()
        result=tifffile.memmap(partial,shape=(h,w,3),dtype='uint16',bigtiff=True,photometric='rgb',
            extratags=[(34675,'B',len(icc),icc,False)] if icc else [],
            metadata={'axes':'YXS','Software':'PixelShift Lab RAW reconstruction v2',
                'Description':'sRGB encoded; reconstructed from original black-corrected CFA samples using measured translations.'})
        fallback=0
        for y in range(0,h,96):
            check_cancel();rows=min(96,h-y)
            rgb,missing=raw_engine.reconstruct_band(sensors,metas,shifts,scale,y,rows,w)
            fallback+=missing
            result[y:y+rows]=raw_engine.render(rgb,metas[0])
            update(progress=35+round((y+rows)/h*59),message=f'Reconstructing RAW color samples · {round((y+rows)/h*100)}%')
        result.flush();del result,sensors
        update(progress=95,message='Saving color-managed 16-bit TIFF…')
        check_cancel()
        # Exclusive creation works on APFS and exFAT, without overwriting files.
        created=False
        try:
            with output.open('xb') as destination:
                created=True
                with partial.open('rb') as source:
                    shutil.copyfileobj(source,destination,16*1024*1024)
        except Exception:
            if created: output.unlink(missing_ok=True)
            raise
        partial.unlink(); partial=None
        report=output.with_suffix(output.suffix+'.json')
        report_data={'registration_regions':regions,'engine':'raw-cfa-normalized-convolution-v2.1-texture-registration','frames':shifts,'width':w,'height':h,
            'alignment_warnings':alignment_warnings,'sony_group':group,'color':'sRGB transfer function, camera WB and matrix, embedded ICC when available',
            'fallback_channel_fraction':fallback/(h*w*3),
            'limitations':['Independent implementation, not Sony-certified','Global translation only; no local subject-motion correction',
                          '16-shot reconstruction tested on synthetic data and real-camera captures; not compared against Sony output',
                          'Irregularly sampled gaps use wider RAW-sample interpolation; reported fraction quantifies this']}
        # Report failure must not invalidate a successfully written image.
        try:
            with report.open('x') as f: json.dump(report_data,f,indent=2)
        except OSError: pass
        update(progress=98,message='Preparing matched before/after detail crops…')
        try:
            prepare_compare(paths[0],output,metas[0],scale,regions)
            compare_ready=True
        except Exception:
            traceback.print_exc();compare_ready=False
        update(status='done',progress=100,message=f'Saved {output}',output=str(output),compare=compare_ready,
               warning=' '.join(alignment_warnings+[f'RAW gap interpolation: {report_data["fallback_channel_fraction"]:.2%} of channel positions. No local subject-motion correction.']))
    except InterruptedError as e: update(status='cancelled',message=str(e))
    except Exception as e:
        update(status='error',message=str(e)); traceback.print_exc()
    finally:
        if partial and partial.exists():
            try: partial.unlink()
            except OSError: pass
        if scratch: shutil.rmtree(scratch,ignore_errors=True)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,data,status=200):
        body=json.dumps(data).encode(); self.send_response(status)
        self.send_header('Content-Type','application/json'); self.send_header('Cache-Control','no-store')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith('/api/compare?'):
            query=parse_qs(urlparse(self.path).query)
            key=(query.get('region',['center'])[0],query.get('kind',['after'])[0])
            with COMPARE_LOCK:body=COMPARE.get(key)
            if body is None:return self.send_error(404)
            self.send_response(200);self.send_header('Content-Type','image/png')
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body);return
        if self.path=='/api/status':
            with LOCK: snapshot=STATE.copy()
            return self.reply(snapshot)
        if self.path=='/':
            body=(ROOT/'static/index.html').read_text().replace('__TOKEN__',TOKEN).replace('__DEFAULT_OUTPUT__',json.dumps(str(DEFAULT_OUTPUT))).encode()
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(body)
        else: self.send_error(404)
    def do_POST(self):
        if self.headers.get('X-Local-Token')!=TOKEN: return self.reply({'error':'Unauthorized'},403)
        if int(self.headers.get('Content-Length','0'))>16384: return self.reply({'error':'Request too large'},413)
        try: config=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
        except Exception: return self.reply({'error':'Invalid request'},400)
        if self.path in ('/api/pick','/api/scan'):
            try:
                folder=pick_folder() if self.path=='/api/pick' else config.get('folder','')
                if folder is None: return self.reply({'cancelled':True})
                return self.reply(scan_captures(folder,int(config.get('count',16))))
            except Exception as e: return self.reply({'error':str(e)},400)
        if self.path=='/api/cancel': CANCEL.set(); return self.reply({'ok':True})
        if self.path!='/api/start': return self.reply({'error':'Not found'},404)
        with LOCK:
            if STATE['status']=='running': return self.reply({'error':'A merge is already running.'},409)
            STATE.update(status='running',progress=0,message='Verifying Sony RAW sequence metadata…',output=None,compare=False,warning=None)
        CANCEL.clear(); threading.Thread(target=merge,args=(config,),daemon=True).start()
        self.reply({'ok':True})

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',int(os.environ.get('PIXELSHIFT_PORT','0'))),Handler)
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'PixelShift Lab: {url}\nKeep this window open. Control-C to stop.')
    webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: CANCEL.set(); server.server_close()
