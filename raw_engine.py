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
              'white':r.white_level,'wb':wb.tolist(),'transform':transform.tolist(),'flip':sizes.flip}
        return r.raw_image_visible.copy(),meta

def select_regions(sensor,meta):
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
    chosen=[]
    for score,y,x in sorted(candidates,reverse=True):
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


def alignment(reference,current):
    estimates=[]
    for a,b in zip(reference,current):
        if float(np.std(a))<.003:continue
        try:
            # Tripod pixel shifts are small. Starting near zero avoids phase
            # correlation jumping to another repeat of a printed pattern.
            warp=np.float32([[1,0,0],[0,1,0]])
            quality,warp=cv2.findTransformECC(a,b,warp,cv2.MOTION_TRANSLATION,(3,100,1e-6),None,5)
            if quality>.90 and np.max(np.abs(warp[:,2]))<16:
                estimates.append((float(warp[0,2]),float(warp[1,2]),quality))
        except cv2.error:continue
    if len(estimates)<2:raise ValueError('Insufficient stationary detail for reliable RAW alignment. Use a stable tripod and still subject.')
    arr=np.asarray(estimates); shift=np.median(arr[:,:2],axis=0)
    spread=float(np.max(np.linalg.norm(arr[:,:2]-shift,axis=1)))
    warning=(f'Alignment regions disagree by {spread:.2f} px. Continuing with the median translation; the estimate is uncertain and may produce blur, ghosting, or color artifacts. This does not prove the camera moved.' if spread>1.5 else None)
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

def reconstruct_band(sensors,metas,shifts,scale,y,rows,width,radius=None):
    """Compact normalized convolution of original per-color sensel values.

    At integer/half-pixel ideal offsets this reduces exactly to CFA interleaving
    with two-green averaging. Fractional displacement uses tent-weighted RAW
    samples. No demosaiced image enters this operation.
    """
    sums=np.zeros((rows,width,3),np.float32); weights=np.zeros_like(sums)
    radius=radius or 1/scale
    for sensor,meta,shift in zip(sensors,metas,shifts):
        left,top,_,_=meta['crop']
        sx=np.arange(width,dtype=np.float32)/scale+left+shift['x']
        sy=np.arange(y,y+rows,dtype=np.float32)/scale+top+shift['y']
        for py in range(2):
            iy=np.floor((sy-py)/2+.5).astype(np.int32)
            wy=np.maximum(0,1-np.abs(sy-(iy*2+py))/radius)
            for px in range(2):
                ix=np.floor((sx-px)/2+.5).astype(np.int32)
                wx=np.maximum(0,1-np.abs(sx-(ix*2+px))/radius)
                weight=wy[:,None]*wx[None,:]
                sh,sw=sensor[py::2,px::2].shape
                valid=(iy[:,None]>=0)&(iy[:,None]<sh)&(ix[None,:]>=0)&(ix[None,:]<sw)
                weight*=valid
                channel=meta['pattern'][py][px]; c=1 if channel==3 else channel
                values=sensor[py::2,px::2][np.clip(iy,0,sh-1)[:,None],np.clip(ix,0,sw-1)[None,:]].astype(np.float32)
                black=meta['black'][channel]
                values=np.maximum(values-black,0)/(meta['white']-black)
                sums[:,:,c]+=values*weight;weights[:,:,c]+=weight
    missing=weights<1e-5
    if np.any(missing):
        if radius>=1.5:raise ValueError('RAW samples do not cover the requested image grid.')
        fallback,_=reconstruct_band(sensors,metas,shifts,scale,y,rows,width,1.5)
    result=sums/np.maximum(weights,1e-10)
    if np.any(missing):result[missing]=fallback[missing]
    return result,int(np.count_nonzero(missing))

def render(rgb,meta):
    wb=np.array(meta['wb'][:3],np.float32)
    linear=(rgb*wb)@np.array(meta['transform'],np.float32).T
    linear=np.clip(linear,0,1)
    encoded=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055)
    return np.rint(np.clip(encoded,0,1)*65535).astype(np.uint16)
