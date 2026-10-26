"""Independent RAW-sample reconstruction. No developed RGB is merged."""
import json, shutil, subprocess
from pathlib import Path
import numpy as np
import cv2
import rawpy

XYZ_FROM_SRGB=np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]])

def metadata(paths):
    exe=shutil.which('exiftool')
    if not exe: raise ValueError('ExifTool is required to verify Sony pixel-shift sequences. Install ExifTool, then restart the app.')
    rows=[]
    for start in range(0,len(paths),128):
        run=subprocess.run([exe,'-j','-n','-Model','-BodySerialNumber','-DateTimeOriginal','-PixelShiftInfo','-ExposureTime','-ISO','-FNumber',*[str(p) for p in paths[start:start+128]]],capture_output=True,text=True)
        if run.returncode not in (0,1): raise ValueError('Could not read capture metadata: '+run.stderr[:200])
        rows.extend(json.loads(run.stdout or '[]'))
    return rows

def group_rows(rows,count):
    groups={}
    for row in rows:
        info=str(row.get('PixelShiftInfo','')).split()
        if len(info)!=4 or row.get('Model')!='ILCE-7RM5': continue
        try: shot,total=int(info[1]),int(info[2])
        except ValueError: continue
        if total!=count or not 1<=shot<=total: continue
        key=(row['Model'],row.get('BodySerialNumber',''),str(row.get('DateTimeOriginal',''))[:10],info[0],info[3],total)
        groups.setdefault(key,[]).append((shot,row))
    valid=[]; incomplete=0
    for key,items in groups.items():
        if sorted(n for n,_ in items)!=list(range(1,count+1)):
            incomplete+=1;continue
        ordered=[r for _,r in sorted(items,key=lambda pair:pair[0])]
        if any(len({r.get(tag) for r in ordered})!=1 for tag in ('ExposureTime','ISO','FNumber')):
            incomplete+=1;continue
        valid.append((key,ordered))
    return valid,incomplete

def verify(paths,count):
    groups,_=group_rows(metadata(paths),count)
    if len(groups)!=1 or len(groups[0][1])!=len(paths):
        raise ValueError('Files are not one complete, matching Sony pixel-shift group. Rescan the drive; ordinary bursts and incomplete sets are excluded.')
    return [Path(row['SourceFile']) for row in groups[0][1]],groups[0][0]

def read_sensor(path):
    with rawpy.imread(str(path)) as r:
        sizes=r.sizes
        if (getattr(sizes,'crop_width',0),getattr(sizes,'crop_height',0))==(9504,6336):
            crop=(sizes.crop_left_margin-sizes.left_margin,sizes.crop_top_margin-sizes.top_margin,9504,6336)
        elif (sizes.width,sizes.height)==(9504,6336):crop=(0,0,9504,6336)
        else:raise ValueError(f'{path.name}: unsupported RAW dimensions; use full-size a7R V RAWs.')
        pattern=r.raw_colors_visible[:2,:2].copy()
        if r.color_desc!=b'RGBG' or sorted(pattern.ravel().tolist())!=[0,1,2,3]:
            raise ValueError('Unsupported color filter array.')
        cam=np.asarray(r.rgb_xyz_matrix[:3],np.float64)@XYZ_FROM_SRGB
        cam/=cam.sum(axis=1,keepdims=True)
        transform=np.linalg.inv(cam).astype(np.float32)
        wb=np.array(r.camera_whitebalance,np.float32);wb/=wb[1]
        if not np.all(np.isfinite(wb)) or np.min(wb)<=0:raise ValueError('Missing camera white balance.')
        meta={'crop':crop,'pattern':pattern.tolist(),'black':list(r.black_level_per_channel),
              'white':r.white_level,'wb':wb.tolist(),'transform':transform.tolist(),'camera_xyz':np.asarray(r.rgb_xyz_matrix[:3]).tolist(),'flip':sizes.flip}
        return r.raw_image_visible.copy(),meta

def select_regions(sensor,meta,settings=None):
    settings=settings or {}
    """Select focused texture on the reference, not preset background corners."""
    h,w=sensor.shape;size=min(640,h//3,w//3)//2*2
    greens=[normalized_plane(sensor,meta,py,px) for py in range(2) for px in range(2) if meta['pattern'][py][px] in (1,3)]
    green=(greens[0]+greens[1])*.5
    lap=cv2.Laplacian(cv2.GaussianBlur(green,(0,0),.8),cv2.CV_32F)
    window=max(3,int(size*.4375))
    energy=cv2.boxFilter(lap*lap,-1,(window,window))
    margin=size//4+40;step=max(16,size//4)
    candidates=[]
    for y in range(margin,energy.shape[0]-margin,step):
        for x in range(margin,energy.shape[1]-margin,step):
            candidates.append((float(energy[y,x]),y*2,x*2))
    masks=settings.get('regions',[]);includes=[r for r in masks if r['kind']=='include'];excludes=[r for r in masks if r['kind']=='exclude']
    left,top,cw,ch=meta.get('crop',(0,0,w,h))
    def inside(r,x,y):return r['x']<=(x-left)/cw<=r['x']+r['w'] and r['y']<=(y-top)/ch<=r['y']+r['h']
    chosen=[]
    for score,y,x in sorted(candidates,reverse=True):
        if any(inside(r,x,y) for r in excludes):continue
        if settings.get('region_mode')=='manual' and not any(inside(r,x,y) for r in includes):continue
        if chosen and score<chosen[0]['texture_score']*.25:break
        if all(np.hypot(y-(r['top']+size/2),x-(r['left']+size/2))>size*.9 for r in chosen):
            chosen.append({'top':int(y-size/2)//2*2,'left':int(x-size/2)//2*2,'size':size,'texture_score':score})
        if len(chosen)==5:break
    if len(chosen)<2:raise ValueError('Insufficient textured area to estimate pixel-shift alignment.')
    return chosen

def gray_patches(sensor,meta,regions):
    patches=[]
    for region in regions:
        top,left,size=region['top'],region['left'],region['size']
        patch=sensor[top:top+size,left:left+size]
        # Demosaicing is used ONLY for estimating displacement.
        rgb=demosaic(patch,meta)
        patches.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
    return patches


def alignment(reference,current,settings=None):
    settings=settings or {}
    estimates=[]
    for a,b in zip(reference,current):
        if float(np.std(a))<.003:continue
        try:
            # Tripod pixel shifts are small. Starting near zero avoids phase
            # correlation jumping to another repeat of a printed pattern.
            warp=np.float32([[1,0,0],[0,1,0]])
            quality,warp=cv2.findTransformECC(a,b,warp,cv2.MOTION_TRANSLATION,(3,100,1e-6),None,5)
            if quality>settings.get('minimum_correlation',.90) and np.max(np.abs(warp[:,2]))<settings.get('maximum_shift',16):
                estimates.append((float(warp[0,2]),float(warp[1,2]),quality))
        except cv2.error:continue
    if len(estimates)<2:raise ValueError('Insufficient stationary detail for reliable RAW alignment. Use a stable tripod and still subject.')
    arr=np.asarray(estimates); shift=np.median(arr[:,:2],axis=0)
    spread=float(np.max(np.linalg.norm(arr[:,:2]-shift,axis=1)))
    warning=(f'Alignment regions disagree by {spread:.2f} px. Continuing with the median translation; the estimate is uncertain and may produce blur, ghosting, or color artifacts. This does not prove the camera moved.' if spread>settings.get('disagreement',1.5) else None)
    if warning and settings.get('alignment_mode')=='strict':raise ValueError('Strict alignment: '+warning.split(' Continuing')[0])
    return {'x':float(shift[0]),'y':float(shift[1]),'confidence':float(np.median(arr[:,2])),'patch_spread':spread,'warning':warning,'patch_estimates':arr.tolist()}

def normalized_plane(sensor,meta,py,px):
    channel=meta['pattern'][py][px]
    black=meta['black'][channel]
    return np.maximum(sensor[py::2,px::2].astype(np.float32)-black,0)/(meta['white']-black)

def demosaic(sensor,meta):
    """Conventional single-frame baseline, same calibration as the composite."""
    norm=np.zeros(sensor.shape,np.float32)
    for py in range(2):
        for px in range(2):norm[py::2,px::2]=normalized_plane(sensor,meta,py,px)
    pattern=meta['pattern']
    colors=''.join('RGBG'[c] for row in pattern for c in row)
    code=getattr(cv2,'COLOR_Bayer'+colors+'2RGB')
    return cv2.cvtColor(np.clip(norm*65535,0,65535).astype(np.uint16),code).astype(np.float32)/65535

def reconstruct_band(sensors,metas,shifts,scale,y,rows,width,radius=None,settings=None,diagnostics=False):
    """Normalized convolution of original CFA samples, with measured support maps."""
    settings=settings or {}
    radius=radius or settings.get('support',1.)/scale
    sums=np.zeros((rows,width,3),np.float32);weights=np.zeros_like(sums);confidence=np.zeros_like(sums)
    offsets=(-1,0,1) if radius>1 else (0,)
    def kernel(distance):
        t=np.abs(distance)/radius
        return np.where(t<1, .5+.5*np.cos(np.pi*t),0) if settings.get('kernel')=='cosine' else np.maximum(0,1-t)
    for sensor,meta,shift in zip(sensors,metas,shifts):
        left,top,_,_=meta['crop']
        sx=np.arange(width,dtype=np.float32)/scale+left+shift['x']
        sy=np.arange(y,y+rows,dtype=np.float32)/scale+top+shift['y']
        for py in range(2):
            base_y=np.floor((sy-py)/2+.5).astype(np.int32)
            for px in range(2):
                plane=sensor[py::2,px::2];sh,sw=plane.shape
                base_x=np.floor((sx-px)/2+.5).astype(np.int32)
                channel=meta['pattern'][py][px];c=1 if channel==3 else channel;black=meta['black'][channel]
                for dy in offsets:
                    iy=base_y+dy;wy=kernel(sy-(iy*2+py))
                    if not np.any(wy):continue
                    for dx in offsets:
                        ix=base_x+dx;wx=kernel(sx-(ix*2+px))
                        if not np.any(wx):continue
                        weight=wy[:,None]*wx[None,:]
                        weight*=((iy[:,None]>=0)&(iy[:,None]<sh)&(ix[None,:]>=0)&(ix[None,:]<sw))
                        values=plane[np.clip(iy,0,sh-1)[:,None],np.clip(ix,0,sw-1)[None,:]].astype(np.float32)
                        values=np.maximum(values-black,0)/(meta['white']-black)
                        sums[:,:,c]+=values*weight;weights[:,:,c]+=weight
                        confidence[:,:,c]+=weight*shift.get('confidence',1.)
    missing=weights<1e-5
    fallback_radius=settings.get('fallback_support',1.5)
    if np.any(missing):
        if radius>=max(fallback_radius,settings.get('support',1.)/scale):raise ValueError('RAW samples do not cover the requested image grid.')
        fallback,_=reconstruct_band(sensors,metas,shifts,scale,y,rows,width,fallback_radius,settings)
    result=sums/np.maximum(weights,1e-10)
    if np.any(missing):result[missing]=fallback[missing]
    count=int(np.count_nonzero(missing))
    if diagnostics:
        coverage=np.clip(np.min(weights/np.array([1,2,1],np.float32),axis=2),0,1)
        reliability=np.min(confidence/np.maximum(weights,1e-10),axis=2)
        return result,count,{'coverage':coverage,'confidence':reliability,'fallback':np.any(missing,axis=2)}
    return result,count

def temperature_wb(meta,temperature,tint):
    t=float(temperature)
    x=(-.2661239e9/t**3-.2343580e6/t**2+.8776956e3/t+.179910) if t<=4000 else (-3.0258469e9/t**3+2.1070379e6/t**2+.2226347e3/t+.240390)
    if t<=2222:y=-1.1063814*x**3-1.34811020*x*x+2.18555832*x-.20219683
    elif t<=4000:y=-.9549476*x**3-1.37418593*x*x+2.09137015*x-.16748867
    else:y=3.0817580*x**3-5.87338670*x*x+3.75112997*x-.37001483
    matrix=np.asarray(meta.get('camera_xyz'),np.float32)
    if matrix.shape!=(3,3):raise ValueError('Custom temperature requires the original camera calibration. Rebuild from RAW.')
    response=matrix@np.array([x/y,1,(1-x-y)/y])
    gains=1/np.maximum(response,1e-5);gains/=gains[1];gains[1]*=2**(-float(tint)/150)
    return gains

def render_float(rgb,meta,settings=None):
    settings=settings or {}
    if meta.get('space')=='linear_srgb':
        if settings.get('white_balance','camera')!='camera':raise ValueError('Custom white balance requires RAW calibration.')
        linear=rgb.copy()*2**settings.get('exposure',0)
    else:
        wb=temperature_wb(meta,settings.get('temperature',6500),settings.get('tint',0)) if settings.get('white_balance')=='custom' else np.array(meta['wb'][:3],np.float32)
        linear=((rgb*wb)@np.array(meta['transform'],np.float32).T)*2**settings.get('exposure',0)
    if settings.get('denoise',0)>0:
        linear=cv2.bilateralFilter(np.ascontiguousarray(linear),5,float(settings['denoise'])/100*.05,1.5)
    if settings.get('sharpening',0)>0:
        blurred=cv2.GaussianBlur(linear,(0,0),settings.get('sharpening_radius',.8))
        linear=linear+(linear-blurred)*settings['sharpening']
    return linear

def srgb_encode(linear):
    linear=np.clip(linear,0,1)
    return np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055)

def srgb_decode(encoded):return np.where(encoded<=.04045,encoded/12.92,((encoded+.055)/1.055)**2.4)

def render(rgb,meta,settings=None):
    settings=settings or {}
    linear=render_float(rgb,meta,settings)
    encoded=np.clip(linear,0,1) if settings.get('encoding')=='linear' else srgb_encode(linear)
    return np.rint(np.clip(encoded,0,1)*65535).astype(np.uint16)
