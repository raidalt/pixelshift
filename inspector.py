"""Disk-backed full-image inspection. Original photos never leave loopback."""
import json, threading
from pathlib import Path
import numpy as np
import cv2
import tifffile
import raw_engine as engine
import image_tools
from photo_settings import validate

LOCK=threading.RLock()
ACTIVE=None

def open_report(path):
    global ACTIVE
    path=Path(path);report=json.loads(path.read_text());output=Path(str(path)[:-5])
    if not output.is_file():raise ValueError('The output TIFF is unavailable.')
    image=tifffile.memmap(output)
    cache=Path(report.get('cache') or '')
    has_raw=bool(report.get('cache')) and (cache/'linear.npy').is_file()
    linear=np.load(cache/'linear.npy',mmap_mode='r') if has_raw else image
    meta=report.get('meta') if has_raw else {'space':'linear_srgb'}
    settings=validate(report.get('settings',{}))
    if not has_raw:settings['white_balance']='camera';settings['border']=0
    with LOCK:
        ACTIVE={'report':report,'output':output,'image':image,'linear':linear,'meta':meta,'cache':cache if has_raw else None,
                'legacy':not has_raw,'settings':settings,'views':{},'source_cache':{},'version':output.stat().st_mtime_ns}
    return describe()

def require():
    if ACTIVE is None:raise ValueError('Build a composite or open an existing result first.')
    return ACTIVE

def describe():
    with LOCK:
        job=require();r=job['report'];h,w=job['linear'].shape[:2]
        crop=job['settings']['border']
        return {'width':w,'height':h,'border':crop,'scale':r.get('scale',2 if w>10000 else 1),
                'frames':r.get('frames',[]),'regions':r.get('registration_regions',[]),'settings':job['settings'],
                'maps':bool(job['cache'] and (job['cache']/'maps.npy').is_file()),'raw_controls':not job['legacy'],
                'output':str(job['output']),'version':job['version'],'sources':len(r.get('sources',[])),
                'source_available':[Path(p).is_file() or (i==0 and bool(job['cache']) and (job['cache']/'reference.npy').is_file()) for i,p in enumerate(r.get('sources',[]))],
                'sensor_crop':job['meta'].get('crop',[0,0]),
                'message':'Legacy TIFF: exposure and finishing use its existing rendered color. RAW white balance and support maps require a new build.' if job['legacy'] else 'Full-resolution RAW reconstruction cache available.'}

def view(settings):
    import secrets
    with LOCK:
        job=require();settings=validate(settings)
        if job['legacy'] and settings['white_balance']!='camera':raise ValueError('Custom temperature requires a fresh RAW reconstruction.')
        key=secrets.token_hex(8);job['views'][key]=settings
        while len(job['views'])>12:job['views'].pop(next(iter(job['views'])))
        return {'view':key,**describe()}

def region(query):
    with LOCK:
        job=require();h,w=job['linear'].shape[:2]
        x=max(0,min(w-1,int(float(query.get('x',0)))));y=max(0,min(h-1,int(float(query.get('y',0)))))
        rw=max(1,min(w-x,int(float(query.get('w',w)))));rh=max(1,min(h-y,int(float(query.get('h',h)))))
        ow=max(1,min(2048,int(query.get('ow',1024))));oh=max(1,min(2048,int(query.get('oh',768))))
        return x,y,rw,rh,ow,oh

def sample(array,bounds,padding=0):
    x,y,w,h,ow,oh=bounds
    step=max(1,int(min(w/max(1,ow),h/max(1,oh))))
    h0,w0=array.shape[:2]
    x0=max(0,x-padding*step);y0=max(0,y-padding*step);x1=min(w0,x+w+padding*step);y1=min(h0,y+h+padding*step)
    pixels=np.asarray(array[y0:y1:step,x0:x1:step]).astype(np.float32)
    return pixels,step,(x-x0)//step,(y-y0)//step

def composite(bounds,settings):
    job=require();pixels,step,dx,dy=sample(job['linear'],bounds,32)
    if job['legacy']:
        pixels/=65535
        if job['report'].get('output_encoding','srgb')!='linear':pixels=engine.srgb_decode(pixels)
    tuned=engine.render_float(pixels,job['meta'],{**settings,'sharpening_radius':max(.3,settings['sharpening_radius']/step)})
    _,_,w,h,ow,oh=bounds
    tuned=tuned[dy:dy+max(1,(h+step-1)//step),dx:dx+max(1,(w+step-1)//step)]
    return cv2.resize(tuned,(ow,oh),interpolation=cv2.INTER_AREA if step>1 else cv2.INTER_LINEAR)

def original(bounds,settings,index=0,aligned=True):
    job=require();report=job['report'];sources=report.get('sources',[]);scale=report.get('scale',2 if job['linear'].shape[1]>10000 else 1)
    index=max(0,min(len(sources)-1,index)) if sources else 0
    cache=job['cache'];first=cache/'reference.npy' if cache else None
    x,y,w,h,ow,oh=bounds;shift=report.get('frames',[{'x':0,'y':0}])[index] if report.get('frames') else {'x':0,'y':0}
    sx=shift.get('x',0) if aligned else 0;sy=shift.get('y',0) if aligned else 0
    left=max(0,int(x/scale+sx)-32)//2*2;top=max(0,int(y/scale+sy)-32)//2*2
    nw=int(w/scale)+72;nh=int(h/scale)+72
    if index==0 and first and first.is_file():
        array=np.load(first,mmap_mode='r');meta=report['meta']
        patch=np.asarray(array[top:min(array.shape[0],top+nh),left:min(array.shape[1],left+nw)]).copy()
    else:
        if not sources or not Path(sources[index]).is_file():raise ValueError('Original RAW is offline. Reconnect its source drive, then rebuild to cache the first reference.')
        path=sources[index]
        if job['source_cache'].get('path')!=path:
            sensor,meta=engine.read_sensor(Path(path));job['source_cache']={'path':path,'sensor':sensor,'meta':meta}
        src=job['source_cache'];meta=src['meta'];cl,ct,cw,ch=meta['crop']
        # Padded crop preserves CFA parity and demosaicing boundary behavior.
        right=min(cw,left+nw);bottom=min(ch,top+nh)
        patch=engine.demosaic(src['sensor'][ct+top:ct+bottom,cl+left:cl+right],meta)
    if patch.size==0:raise ValueError('Original crop is outside the source image.')
    # Match positions in the reference grid. Equal finishing is applied before display.
    mapx=np.linspace(x/scale+sx-left,(x+w-1)/scale+sx-left,ow,dtype=np.float32)
    mapy=np.linspace(y/scale+sy-top,(y+h-1)/scale+sy-top,oh,dtype=np.float32)
    rgb=cv2.remap(patch,np.broadcast_to(mapx,(oh,ow)).copy(),np.broadcast_to(mapy[:,None],(oh,ow)).copy(),cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
    return engine.render_float(rgb,meta,{**settings,'sharpening_radius':max(.3,settings.get('sharpening_radius',.8)/max(1,w/ow))})

def tile(query):
    with LOCK:
        job=require();settings=job['views'].get(query.get('view'))
        if settings is None:raise ValueError('Preview settings expired. Refresh the inspector.')
        bounds=region(query);kind=query.get('kind','composite');index=int(query.get('reference',0))
        if kind=='original':linear=original(bounds,settings,index)
        elif kind=='alignment':
            base=engine.srgb_encode(original(bounds,settings,0))
            other=engine.srgb_encode(original(bounds,settings,index if index else min(1,len(job['report'].get('sources',[]))-1)))
            rgb=np.stack([base.mean(axis=2),other.mean(axis=2),other.mean(axis=2)],axis=2)
            return image_tools.png(rgb),{}
        elif kind in ('coverage','confidence','fallback'):
            cache=job['cache']
            if not cache or not (cache/'maps.npy').is_file():raise ValueError('Support maps require a new RAW build.')
            maps=np.load(cache/'maps.npy',mmap_mode='r');pixels,step,dx,dy=sample(maps,bounds);pixels/=255
            channel={'coverage':0,'confidence':1,'fallback':2}[kind]
            value=cv2.resize(pixels[:,:,channel],(bounds[4],bounds[5]),interpolation=cv2.INTER_NEAREST)
            rgb=np.stack([1-value,value,np.zeros_like(value)],axis=2) if kind!='fallback' else np.stack([value,value*.25,np.zeros_like(value)],axis=2)
            return image_tools.png(rgb),{'legend':'Red: weak support / Green: strong support' if kind!='fallback' else 'Orange: wider interpolation used; black: compact support'}
        else:linear=composite(bounds,settings)
        stats=image_tools.statistics(linear)
        rgb=engine.srgb_encode(linear)
        if kind=='difference':
            before=engine.srgb_encode(original(bounds,settings,index));rgb=np.clip(np.abs(rgb-before)*max(1,min(32,float(query.get('gain',4)))),0,1)
        overlay=query.get('clip','off')
        if overlay in ('high','both'):rgb[np.any(linear>=1,axis=2)]=[1,.1,.1]
        if overlay in ('low','both'):rgb[np.all(linear<=0,axis=2)]=[.1,.3,1]
        return image_tools.png(rgb),stats
