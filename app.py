"""Local RAW pixel-shift reconstruction, tuning, and full-image inspection."""
import json,os,secrets,threading,webbrowser,shutil,traceback,subprocess,time,hashlib,mimetypes
from datetime import datetime
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import numpy as np
import cv2,rawpy,tifffile
import raw_engine,photo_settings,image_tools,inspector,focus_engine

ROOT=Path(__file__).resolve().parent
DEFAULT_OUTPUT=Path.home()/'Pictures'/'Pixel Shift'
TOKEN=secrets.token_urlsafe(32)
BASE_H,BASE_W=6336,9504
STATE={'status':'idle','progress':0,'message':'Ready to build a composite.','phase':'idle','diagnostics':[]}
LOCK=threading.RLock();CANCEL=threading.Event();CAPTURES={};SCAN_LOCK=threading.Lock();RESULTS={};FOCUS_CAPTURES={}
RECONSTRUCTION_KEYS=('alignment_mode','disagreement','minimum_correlation','maximum_shift','region_mode','regions','kernel','support','fallback_support','focus_radius','focus_selectivity','focus_alignment')
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


def scan_focus(folder):
    root=Path(folder).expanduser()
    if not root.is_absolute() and len(root.parts)==1:root=Path('/Volumes')/root
    root=root.resolve()
    if not root.is_dir():raise ValueError('The selected drive or folder is unavailable.')
    pending=[root];seen=set();paths=[];skipped=0;folders=0
    while pending:
        directory=pending.pop()
        try:
            stat=directory.stat();key=(stat.st_dev,stat.st_ino)
            if key in seen:continue
            seen.add(key);folders+=1
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        path=Path(entry.path)
                        if entry.is_dir(follow_symlinks=True):
                            if not entry.is_symlink() or path.resolve().is_relative_to(root):pending.append(path)
                        elif entry.is_file() and path.suffix.lower() in focus_engine.EXTENSIONS:paths.append(path)
                    except OSError:skipped+=1
        except OSError:skipped+=1
    groups=focus_engine.group_rows(focus_engine.metadata(paths));sets=[]
    with SCAN_LOCK:
        for rows in groups:
            key=secrets.token_urlsafe(12);FOCUS_CAPTURES[key]=rows
            positions={row.get('FocusPosition2') for row in rows if row.get('FocusPosition2') is not None}
            sets.append({'id':key,'first':Path(rows[0]['SourceFile']).name,'last':Path(rows[-1]['SourceFile']).name,
                'files':[Path(r['SourceFile']).name for r in rows],'frames':[{'name':Path(r['SourceFile']).name,'focus':r.get('FocusDistance2'),'position':r.get('FocusPosition2')} for r in rows],
                'focus_varies':len(positions)>1,'verified':False,'width':rows[0].get('ImageWidth'),'height':rows[0].get('ImageHeight')})
        while len(FOCUS_CAPTURES)>1000:FOCUS_CAPTURES.pop(next(iter(FOCUS_CAPTURES)))
    return {'folder':str(root),'sets':sets,'folders':folders,'total':len(paths),'skipped':skipped,'unassigned':0}

def focus_overview(capture):
    with SCAN_LOCK:rows=FOCUS_CAPTURES.get(capture)
    if not rows:raise ValueError('Select a focus sequence first.')
    path=rows[0]['SourceFile']
    # Embedded previews make browsing a large card fast, without full HIF decoding.
    for tag in ('PreviewImage','ThumbnailImage'):
        r=subprocess.run(['exiftool','-b','-'+tag,path],capture_output=True)
        if r.returncode==0 and len(r.stdout)>100:
            image=cv2.imdecode(np.frombuffer(r.stdout,np.uint8),cv2.IMREAD_COLOR)
            if image is not None:return image_tools.png(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    return image_tools.png(raw_engine.srgb_encode(focus_engine.preview(focus_engine.decode(path),1200)))

def focus_merge(config):
    cache=None;success=False
    try:
        settings=photo_settings.validate(config.get('settings'))
        if settings['white_balance']!='camera':raise ValueError('Focus stacks use developed sRGB color. Choose As shot white balance.')
        with SCAN_LOCK:rows=list(FOCUS_CAPTURES.get(config.get('capture'),[]))
        selected=config.get('frames')
        if selected is not None:
            if not isinstance(selected,list) or any(type(i)!=int or i<0 or i>=len(rows) for i in selected) or len(set(selected))!=len(selected):raise ValueError('Invalid focus frame selection.')
            rows=[rows[i] for i in sorted(selected)]
        if not 2<=len(rows)<=100:raise ValueError('Select between 2 and 100 focus frames from one candidate sequence.')
        focus_engine.check_color(rows);paths=[Path(r['SourceFile']) for r in rows]
        output=output_path(config.get('output',''),paths,len(paths),settings)
        width=int(rows[0].get('ImageWidth',9504));height=int(rows[0].get('ImageHeight',6336))
        needed=width*height*(len(paths)*6+24)+1_000_000_000
        if shutil.disk_usage(output.parent).free<needed:raise ValueError(f'Allow {needed/1e9:.1f} GB free for this focus stack and its offline references.')
        import uuid
        cache=output.parent/'.pixelshift-data'/uuid.uuid4().hex;cache.mkdir(parents=True)
        previews=[];arrays=[];frames=[];warnings=[];shape=None
        for i,path in enumerate(paths):
            check_cancel();update(phase='decode',frame=i+1,total=len(paths),progress=round(i/len(paths)*25),message=f'Decoding focus frame {i+1}/{len(paths)} · {path.name}')
            rgb=focus_engine.decode(path)
            if shape is None:shape=rgb.shape
            if rgb.shape!=shape:raise ValueError('Focus frames must have matching dimensions.')
            previews.append(focus_engine.preview(rgb));dest=np.lib.format.open_memmap(cache/f'frame-{i}.npy',mode='w+',dtype='uint16',shape=rgb.shape)
            for y in range(0,rgb.shape[0],128):check_cancel();dest[y:y+128]=np.rint(np.clip(rgb[y:y+128],0,1)*65535).astype(np.uint16)
            dest.flush();del dest,rgb;arrays.append(np.load(cache/f'frame-{i}.npy',mmap_mode='r'))
        h,w=shape[:2];ratio=w/previews[0].shape[1]
        for i,preview in enumerate(previews):
            check_cancel();update(phase='alignment',progress=25+round(i/len(paths)*10),message=f'Aligning focus frame {i+1}/{len(paths)} · correcting scale and translation')
            if i==0 or settings['focus_alignment']=='none':matrix=np.eye(2,3,dtype=np.float32);corr=1.;scale=1.;inliers=0
            else:matrix,corr,scale,inliers=focus_engine.register(previews[0],preview)
            matrix[:,2]*=ratio
            warning=None
            if corr<.8:warning=f'{path.name}: low focus-frame correlation ({corr:.3f}); inspect registration and moving subjects.';warnings.append(warning)
            frames.append({'name':paths[i].name,'x':float(matrix[0,2]),'y':float(matrix[1,2]),'confidence':corr,'patch_spread':0.,'warning':warning,'matrix':matrix.tolist(),'scale':scale,'inliers':inliers})
            update(diagnostics=frames.copy(),warnings=warnings.copy())
        positions={r.get('FocusPosition2') for r in rows if r.get('FocusPosition2') is not None}
        if len(positions)<=1:warnings.append('No changing focus position was recorded. This is a candidate burst, not a verified focus sweep; stacking cannot restore detail that is out of focus in every frame.')
        linear=np.lib.format.open_memmap(cache/'linear.npy',mode='w+',dtype='float32',shape=(h,w,3))
        maps=np.lib.format.open_memmap(cache/'focus-maps.npy',mode='w+',dtype='uint8',shape=(h,w,2));contribution=np.zeros(len(paths),np.int64);confidence_sum=0.;invalid=0
        update(phase='reconstruction',phase_started=time.monotonic(),phase_fraction=0,warnings=warnings)
        # Each band begins on the same pyramid phase; a 96px halo exceeds all filters.
        for y in range(0,h,96):
            check_cancel();start=max(0,y-96);end=min(h,y+192);images=[];validity=[]
            for array,frame in zip(arrays,frames):
                rgb,valid=focus_engine.warp_band(array,np.asarray(frame['matrix'],np.float32),start,end-start,w);images.append(rgb);validity.append(valid)
            fused,winner,confidence,missing=focus_engine.fuse(images,validity,settings);a=y-start;b=min(y+96,h)-start
            linear[y:y+b-a]=fused[a:b];maps[y:y+b-a,:,0]=winner[a:b];maps[y:y+b-a,:,1]=np.rint(confidence[a:b]*255).astype(np.uint8)
            contribution+=np.bincount(winner[a:b].ravel(),minlength=len(paths));confidence_sum+=float(confidence[a:b].sum());invalid+=int(missing[a:b].sum())
            update(progress=35+round(min(y+96,h)/h*44),phase_fraction=min(y+96,h)/h,message=f'Blending focused detail · {round(min(y+96,h)/h*100)}%')
        linear.flush();maps.flush();del maps
        report={'kind':'focus','scale':1,'sources':[str(p) for p in paths],'frames':frames,'warnings':warnings,'registration_regions':[],
            'focus':{'method':'Modified-Laplacian sharpness with three-level Laplacian pyramid blending','candidate_only':True,'positions':sorted(positions),'frame_contributions':(contribution/(h*w)).tolist(),'mean_selection_margin':confidence_sum/(h*w),'uncovered_fraction':invalid/(h*w)},'reconstruction_settings':settings}
        meta={'space':'linear_srgb','crop':[0,0,w,h]}
        path=export_file(linear,meta,settings,output,report,cache);del linear
        info=remember(path);success=True;update(status='done',phase='done',progress=100,message=f'Saved focus stack · {output}',output=str(output),inspection=info,warnings=warnings,ended=time.monotonic())
    except InterruptedError as e:update(status='cancelled',phase='cancelled',message=str(e),ended=time.monotonic())
    except Exception as e:update(status='error',phase='error',message=str(e),ended=time.monotonic());traceback.print_exc()
    finally:
        if cache and not success:shutil.rmtree(cache,ignore_errors=True)


def update(**values):
    with LOCK:STATE.update(values)

def status():
    with LOCK:
        result=STATE.copy()
        if result.get('started'):
            result['elapsed']=round((time.monotonic() if result['status']=='running' else result.get('ended',time.monotonic()))-result['started'],1)
            fraction=result.get('phase_fraction',0)
            if result['status']=='running' and result['phase']=='reconstruction' and fraction>.02:
                result['eta']=round((time.monotonic()-result['phase_started'])*(1-fraction)/fraction)
    try:result['inspection']=inspector.describe()
    except ValueError:result['inspection']=None
    return result

def check_cancel():
    if CANCEL.is_set():raise InterruptedError('Processing cancelled. Original photos are untouched.')

def output_path(value,paths,count,settings=None):
    settings=photo_settings.validate(settings)
    output=Path(value or DEFAULT_OUTPUT).expanduser().resolve()
    if output==DEFAULT_OUTPUT.resolve():output.mkdir(parents=True,exist_ok=True)
    if output.is_dir():
        stem=settings['filename'].format(first=paths[0].stem,count=count,date=datetime.now().strftime('%Y%m%d'))
        stem=Path(stem).stem if stem.lower().endswith(('.tif','.tiff')) else stem
        output=output/(stem+'.tif');n=2
        while output.exists() or output.with_name(output.name+'.partial').exists():
            output=output.parent/f'{stem}-{n}.tif';n+=1
    if output.suffix.lower() not in ('.tif','.tiff'):raise ValueError('Choose an existing output folder or a .tif filename.')
    if output.exists():raise ValueError('Output exists. Choose a folder for automatic unique naming.')
    if not output.parent.is_dir():raise ValueError('Output folder does not exist.')
    return output

def library():
    if DEFAULT_OUTPUT.is_dir():
        for path in DEFAULT_OUTPUT.glob('*.tif*.json'):
            if path.is_file():RESULTS[hashlib.sha256(str(path).encode()).hexdigest()[:16]]=path
    found=[]
    for key,path in list(RESULTS.items()):
        try:
            r=json.loads(path.read_text());image=Path(str(path)[:-5])
            if not image.is_file():continue
            found.append({'id':key,'name':image.name,'modified':image.stat().st_mtime,'width':r.get('width'),'height':r.get('height'),'tunable':bool(r.get('cache'))})
        except (OSError,ValueError):continue
    return sorted(found,key=lambda x:x['modified'],reverse=True)

def remember(path):
    key=hashlib.sha256(str(path).encode()).hexdigest()[:16];RESULTS[key]=path
    return inspector.open_report(path)

def preset_file():return DEFAULT_OUTPUT/'.pixelshift-data'/'presets.json'

def presets():
    try:return json.loads(preset_file().read_text())
    except (OSError,ValueError):return {}

def save_preset(config):
    name=str(config.get('name','')).strip()
    if not name or len(name)>60:raise ValueError('Preset name must contain 1–60 characters.')
    data=presets()
    if config.get('delete'):data.pop(name,None)
    else:data[name]=photo_settings.validate(config.get('settings'))
    path=preset_file();path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)
    return data

def source_overview(capture):
    with SCAN_LOCK:paths=CAPTURES.get(capture)
    if not paths:raise ValueError('Select a capture set first.')
    with rawpy.imread(str(paths[0])) as raw:
        rgb=raw.postprocess(use_camera_wb=True,no_auto_bright=True,half_size=True,output_bps=8,user_flip=0)
        sizes=raw.sizes
        if (getattr(sizes,'crop_width',0),getattr(sizes,'crop_height',0))==(9504,6336):
            x=max(0,(sizes.crop_left_margin-sizes.left_margin)//2);y=max(0,(sizes.crop_top_margin-sizes.top_margin)//2)
            rgb=rgb[y:y+3168,x:x+4752]
    factor=min(1,1400/rgb.shape[1]);rgb=cv2.resize(rgb,(int(rgb.shape[1]*factor),int(rgb.shape[0]*factor)),interpolation=cv2.INTER_AREA)
    return image_tools.png(rgb)

def write_reference(sensor,meta,cache):
    left,top,w,h=meta['crop'];path=cache/'reference.npy'
    result=np.lib.format.open_memmap(path,mode='w+',shape=(h,w,3),dtype='float32')
    for y in range(0,h,128):
        check_cancel();start=max(0,y-8)//2*2;end=min(h,y+128+8)//2*2
        rgb=raw_engine.demosaic(sensor[top+start:top+end,left:left+w],meta)
        result[y:min(y+128,h)]=rgb[y-start:min(y+128,h)-start]
    result.flush();del result

def export_file(linear,meta,settings,output,report,cache):
    settings=photo_settings.validate(settings);h,w=linear.shape[:2];border=settings['border']
    if min(h,w)<=2*border:raise ValueError('Border crop removes the entire image.')
    partial=output.with_name(output.name+'.partial');created=False
    if partial.exists():raise ValueError('An unfinished output already exists at that name.')
    try:
        icc=image_tools.profile(settings['encoding']=='linear')
        result=tifffile.memmap(partial,shape=(h-2*border,w-2*border,3),dtype='uint16',bigtiff=True,photometric='rgb',
            extratags=[(34675,'B',len(icc),icc,False)],metadata={'axes':'YXS','Software':'Pixelshift 3',
                'Description':'Linear sRGB primaries, gamma 1.0' if settings['encoding']=='linear' else 'sRGB transfer curve and primaries'})
        high=low=pixels=0;hist=np.zeros((3,64),np.int64)
        for y in range(border,h-border,96):
            check_cancel();stop=min(y+96,h-border);start=max(0,y-32);end=min(h,stop+32)
            band=raw_engine.render_float(np.asarray(linear[start:end]).copy(),meta,settings)
            band=band[y-start:stop-start,border:w-border]
            stats=image_tools.statistics(band);n=stats['pixels'];pixels+=n;high+=stats['highlights']*n;low+=stats['shadows']*n;hist+=np.array(stats['histogram'])
            encoded=np.clip(band,0,1) if settings['encoding']=='linear' else raw_engine.srgb_encode(band)
            result[y-border:stop-border]=np.rint(encoded*65535).astype(np.uint16)
            update(progress=80+round((stop-border)/(h-2*border)*16),message=f'Encoding TIFF · {round((stop-border)/(h-2*border)*100)}%',phase='export')
        result.flush();del result;check_cancel()
        with output.open('xb') as destination:
            created=True
            with partial.open('rb') as source:shutil.copyfileobj(source,destination,16*1024*1024)
        partial.unlink()
        report={**report,'settings':settings,'meta':meta,'cache':str(cache) if cache else None,
            'width':w-2*border,'height':h-2*border,'full_width':w,'full_height':h,
            'clipping':{'highlights':high/pixels,'shadows':low/pixels,'histogram':hist.tolist()},'output_encoding':settings['encoding']}
        path=Path(str(output)+'.json')
        with path.open('x') as f:json.dump(report,f,indent=2)
        return path
    except Exception:
        if created:output.unlink(missing_ok=True)
        raise
    finally:partial.unlink(missing_ok=True)

def merge(config):
    cache=None;success=False
    try:
        settings=photo_settings.validate(config.get('settings'));folder=Path(config['folder']).expanduser().resolve();count=int(config.get('count',16))
        if count not in (4,16):raise ValueError('Choose 4 or 16 frames.')
        if config.get('capture'):
            with SCAN_LOCK:paths=list(CAPTURES.get(config['capture'],[]))
        else:paths=sorted(folder.glob('*.ARW'))
        if len(paths)!=count:raise ValueError('Select a complete verified capture group.')
        paths,group=raw_engine.verify(paths,count)
        output=output_path(config.get('output',''),paths,count,settings);scale=2 if count==16 else 1
        h,w=BASE_H*scale,BASE_W*scale
        needed=count*6400*9600*2+h*w*21+BASE_H*BASE_W*12+1_000_000_000
        if shutil.disk_usage(output.parent).free<needed:raise ValueError(f'Allow {needed/1e9:.1f} GB free space including the editable RAW cache.')
        import uuid
        cache=output.parent/'.pixelshift-data'/uuid.uuid4().hex;cache.mkdir(parents=True)
        shifts=[];metas=[];reference=None;regions=None;warnings=[]
        for i,path in enumerate(paths):
            check_cancel();update(progress=round(i/count*20),phase='alignment',frame=i+1,total=count,message=f'Reading and aligning frame {i+1}/{count}')
            sensor,meta=raw_engine.read_sensor(path)
            if i==0:
                regions=raw_engine.select_regions(sensor,meta,settings);reference=raw_engine.gray_patches(sensor,meta,regions)
                shift={'x':0.,'y':0.,'confidence':1.,'patch_spread':0.,'patch_estimates':[]};write_reference(sensor,meta,cache)
            else:shift=raw_engine.alignment(reference,raw_engine.gray_patches(sensor,meta,regions),settings)
            shift['name']=path.name;shifts.append(shift);metas.append(meta)
            if i and any(meta[k]!=metas[0][k] for k in ('crop','pattern','black','white','wb','transform','flip')):raise ValueError('Calibration changes within the capture group.')
            if shift.get('warning'):warnings.append(f'{path.name}: {shift["warning"]}')
            update(diagnostics=shifts.copy(),warnings=warnings.copy())
            np.save(cache/f'sensor-{i}.npy',sensor);del sensor
        sensors=[np.load(cache/f'sensor-{i}.npy',mmap_mode='r') for i in range(count)]
        linear=np.lib.format.open_memmap(cache/'linear.npy',mode='w+',shape=(h,w,3),dtype='float32')
        maps=np.lib.format.open_memmap(cache/'maps.npy',mode='w+',shape=(h,w,3),dtype='uint8')
        fallback=0;update(phase='reconstruction',phase_started=time.monotonic(),phase_fraction=0)
        for y in range(0,h,96):
            check_cancel();rows=min(96,h-y)
            rgb,missing,diagnostic=raw_engine.reconstruct_band(sensors,metas,shifts,scale,y,rows,w,settings=settings,diagnostics=True)
            fallback+=missing;linear[y:y+rows]=rgb
            maps[y:y+rows]=np.rint(np.stack([diagnostic['coverage'],diagnostic['confidence'],diagnostic['fallback']],axis=2)*255).astype(np.uint8)
            fraction=(y+rows)/h;update(progress=20+round(fraction*59),phase_fraction=fraction,message=f'Reconstructing RAW samples · {round(fraction*100)}%')
        linear.flush();maps.flush();del maps,sensors
        for i in range(count):(cache/f'sensor-{i}.npy').unlink(missing_ok=True)
        report={'engine':'raw-cfa-normalized-convolution-v3','frames':shifts,'registration_regions':regions,'scale':scale,
            'sources':[str(p) for p in paths],'sony_group':group,'alignment_warnings':warnings,
            'fallback_channel_fraction':fallback/(h*w*3),'reconstruction_settings':{k:settings[k] for k in RECONSTRUCTION_KEYS}}
        path=export_file(linear,metas[0],settings,output,report,cache);del linear
        info=remember(path);success=True
        update(status='done',phase='done',progress=100,message=f'Saved {output}',output=str(output),warnings=warnings,inspection=info,ended=time.monotonic())
    except InterruptedError as e:update(status='cancelled',phase='cancelled',message=str(e),ended=time.monotonic())
    except Exception as e:update(status='error',phase='error',message=str(e),ended=time.monotonic());traceback.print_exc()
    finally:
        if cache and not success:shutil.rmtree(cache,ignore_errors=True)

def export_active(config):
    try:
        settings=photo_settings.validate(config.get('settings'))
        with inspector.LOCK:
            job=inspector.require();report=job['report'].copy();linear=job['linear'];meta=job['meta'];cache=job['cache']
            for key in RECONSTRUCTION_KEYS:
                if settings[key]!=job['settings'][key]:raise ValueError('Alignment and reconstruction settings changed. Build a new composite instead of exporting.')
            sources=[Path(p) for p in report.get('sources',[])] or [Path(job['output'].stem)]
            output=output_path(config.get('output',''),sources,len(report.get('frames',[])) or 16,settings)
            legacy=job['legacy']
        if legacy:
            # Legacy TIFFs lack editable camera data. Work in disk-backed linear sRGB.
            import tempfile
            tmp=Path(tempfile.mkdtemp(prefix='.pixelshift-export-',dir=output.parent));temp=tmp/'linear.npy'
            try:
                converted=np.lib.format.open_memmap(temp,mode='w+',shape=linear.shape,dtype='float32')
                for y in range(0,linear.shape[0],128):
                    band=np.asarray(linear[y:y+128],np.float32)/65535
                    converted[y:y+128]=band if report.get('output_encoding','srgb')=='linear' else raw_engine.srgb_decode(band)
                path=export_file(converted,meta,settings,output,report,None);del converted
            finally:shutil.rmtree(tmp,ignore_errors=True)
        else:path=export_file(linear,meta,settings,output,report,cache)
        info=remember(path);update(status='done',phase='done',progress=100,message=f'Saved {output}',output=str(output),inspection=info,ended=time.monotonic())
    except InterruptedError as e:update(status='cancelled',phase='cancelled',message=str(e),ended=time.monotonic())
    except Exception as e:update(status='error',phase='error',message=str(e),ended=time.monotonic());traceback.print_exc()

def launch(target,config):
    with LOCK:
        if STATE['status']=='running':raise ValueError('A processing job is already running.')
        STATE.update(mode=config.get('mode','pixelshift'),status='running',phase='validation',progress=0,message='Checking source and settings…',started=time.monotonic(),ended=None,warnings=[],diagnostics=[],output=None)
    CANCEL.clear();threading.Thread(target=target,args=(config,),daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,data,status=200):
        body=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def binary(self,body,content_type,stats=None):
        self.send_response(200);self.send_header('Content-Type',content_type);self.send_header('Cache-Control','no-store')
        if stats:self.send_header('X-Pixelshift-Stats',json.dumps(stats,separators=(',',':')))
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        parsed=urlparse(self.path);query={k:v[0] for k,v in parse_qs(parsed.query).items()}
        try:
            if parsed.path=='/api/status':return self.reply(status())
            if parsed.path=='/api/library':return self.reply(library())
            if parsed.path=='/api/presets':return self.reply(presets())
            if parsed.path=='/api/tile':
                body,stats=inspector.tile(query);return self.binary(body,'image/png',stats)
            if parsed.path=='/api/source-overview':return self.binary((focus_overview if query.get('mode')=='focus' else source_overview)(query.get('capture')),'image/png')
            if parsed.path=='/':
                body=(ROOT/'static/index.html').read_text().replace('__TOKEN__',TOKEN).replace('__DEFAULT_OUTPUT__',json.dumps(str(DEFAULT_OUTPUT))).replace('__DEFAULT_SETTINGS__',json.dumps(photo_settings.DEFAULTS))
                return self.binary(body.encode(),'text/html; charset=utf-8')
            file=(ROOT/'static'/parsed.path.lstrip('/')).resolve()
            if file.is_relative_to((ROOT/'static').resolve()) and file.is_file():return self.binary(file.read_bytes(),mimetypes.guess_type(file)[0] or 'text/plain')
            self.send_error(404)
        except (ValueError,KeyError,OSError) as e:self.reply({'error':str(e)},400)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_POST(self):
        if self.headers.get('X-Local-Token')!=TOKEN:return self.reply({'error':'Unauthorized'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=65536:raise ValueError('Invalid request size.')
            config=json.loads(self.rfile.read(size))
            if self.path=='/api/cancel':CANCEL.set();return self.reply({'ok':True})
            if self.path=='/api/pick-folder':return self.reply({'folder':pick_folder()})
            if self.path=='/api/pick':
                folder=pick_folder()
                return self.reply((scan_focus(folder) if config.get('mode')=='focus' else scan_captures(folder,int(config.get('count',16)))) if folder else {'cancelled':True})
            if self.path=='/api/scan':return self.reply(scan_focus(config['folder']) if config.get('mode')=='focus' else scan_captures(config['folder'],int(config.get('count',16))))
            if self.path=='/api/presets':return self.reply(save_preset(config))
            if self.path=='/api/open':
                if STATE['status']=='running':raise ValueError('Wait for the current job to finish.')
                library();path=RESULTS.get(config.get('id'))
                if not path:raise ValueError('Result not found.')
                return self.reply(remember(path))
            if self.path=='/api/view':return self.reply(inspector.view(config.get('settings',{})))
            if self.path=='/api/start':photo_settings.validate(config.get('settings'));launch(focus_merge if config.get('mode')=='focus' else merge,config);return self.reply({'ok':True})
            if self.path=='/api/export':photo_settings.validate(config.get('settings'));launch(export_active,config);return self.reply({'ok':True})
            self.send_error(404)
        except (ValueError,KeyError,OSError,TypeError) as e:self.reply({'error':str(e)},400)

if __name__=='__main__':
    items=library()
    if items:
        try:
            info=remember(RESULTS[items[0]['id']]);STATE.update(status='done',phase='saved',progress=100,message='Opened '+Path(info['output']).name,warnings=info.get('warnings',[]),mode=info.get('kind','pixelshift'))
        except Exception:pass
    server=ThreadingHTTPServer(('127.0.0.1',int(os.environ.get('PIXELSHIFT_PORT','0'))),Handler)
    url=f'http://127.0.0.1:{server.server_port}';print(f'Pixelshift: {url}\nKeep this window open. Control-C to stop.',flush=True)
    if not os.environ.get('PIXELSHIFT_NO_BROWSER'):webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:CANCEL.set();server.server_close()
