"""Local focus stacking: similarity registration and multiscale sharpness fusion."""
import json,re,shutil,subprocess,tempfile
from functools import lru_cache
import heif_native
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import rawpy
import capture_metadata
import raw_engine

EXTENSIONS={'.arw','.jpg','.jpeg','.hif','.heic','.heif','.tif','.tiff'}

def metadata(paths):
    return capture_metadata.read(paths)

def timestamp(row):
    try:return datetime.strptime(str(row['DateTimeOriginal']),'%Y:%m:%d %H:%M:%S').timestamp()+float('0.'+str(row.get('SubSecTimeOriginal','0')))
    except (KeyError,ValueError):return 0

def group_rows(rows):
    """Candidate groups, never a claim of verified focus-bracket metadata."""
    unique={}
    for row in rows:
        p=Path(row['SourceFile']);info=str(row.get('PixelShiftInfo','')).split()
        if len(info)==4 and info[1] not in ('0','0/0'):continue
        if not timestamp(row):continue
        key=(str(p.parent),p.stem.lower())
        rank={'.arw':0,'.hif':1,'.heic':1,'.heif':1,'.tif':2,'.tiff':2,'.jpg':3,'.jpeg':3}.get(p.suffix.lower(),4)
        if key not in unique or rank<unique[key][0]:unique[key]=(rank,row)
    ordered=sorted((v[1] for v in unique.values()),key=lambda r:(str(Path(r['SourceFile']).parent),timestamp(r),r['SourceFile']))
    groups=[];batch=[]
    def finish():
        if len(batch)>=2:groups.append(batch.copy())
    for row in ordered:
        if batch:
            prev=batch[-1];gap=timestamp(row)-timestamp(prev)
            try:seq=int(row.get('SequenceNumber',0));last=int(prev.get('SequenceNumber',0))
            except (TypeError,ValueError):seq=last=0
            compatible=all(row.get(k)==prev.get(k) for k in ('Model','ImageWidth','ImageHeight','FNumber','ISO','ExposureTime'))
            if Path(row['SourceFile']).parent!=Path(prev['SourceFile']).parent or gap>15 or gap<0 or not compatible or (seq>0 and last>0 and seq!=last+1) or len(batch)>=100:
                finish();batch=[]
        batch.append(row)
    finish()
    return sorted(groups,key=lambda group:timestamp(group[-1]),reverse=True)

@lru_cache(maxsize=10)
def linear_lut(bits,quantized=False):
    # Match heif-convert's PNG precision exactly (high-bit inputs are left-shifted).
    code=np.arange(1<<bits,dtype=np.uint16)
    encoded=code.astype(np.float32)/255 if bits==8 else np.left_shift(code,16-bits).astype(np.float32)/65535
    linear=raw_engine.srgb_decode(encoded)
    return np.rint(linear*65535).astype(np.uint16) if quantized else linear

def decode_to_cache(path,target,check_cancel=lambda:None):
    """Decode directly into a disk-backed linear reference with bounded conversion buffers."""
    path=Path(path)
    if path.suffix.lower() in ('.hif','.heif','.heic'):
        try:
            with heif_native.pixels(path) as (pixels,bits):
                result=np.lib.format.open_memmap(target,mode='w+',dtype='uint16',shape=pixels.shape)
                lut=linear_lut(bits,True)
                for y in range(0,pixels.shape[0],128):
                    check_cancel();result[y:y+128]=lut[pixels[y:y+128]]
                result.flush()
                # Registration must see the same float preview as the original pipeline.
                # Resizing quantized uint16 references perturbs ORB matches on defocused scenes.
                thumbnail=preview(linear_lut(bits)[pixels])
                np.save(Path(target).with_suffix('.preview.npy'),gray(thumbnail))
                del result
                return thumbnail,'libheif-direct'
        except heif_native.Unavailable:pass
    rgb=decode(path);result=np.lib.format.open_memmap(target,mode='w+',dtype='uint16',shape=rgb.shape)
    thumbnail=preview(rgb)
    for y in range(0,rgb.shape[0],128):
        check_cancel();result[y:y+128]=np.rint(np.clip(rgb[y:y+128],0,1)*65535).astype(np.uint16)
    result.flush();np.save(Path(target).with_suffix('.preview.npy'),gray(thumbnail));del result,rgb
    return thumbnail,'libraw' if path.suffix.lower()=='.arw' else 'image-decoder'


def decode(path):
    """Return full-resolution scene-linear sRGB, using recorded SDR encoding."""
    path=Path(path);ext=path.suffix.lower()
    if ext=='.arw':
        with rawpy.imread(str(path)) as raw:
            rgb=raw.postprocess(use_camera_wb=True,no_auto_bright=True,output_bps=16,gamma=(1,1),output_color=rawpy.ColorSpace.sRGB,user_flip=0)
            sizes=raw.sizes
            if getattr(sizes,'crop_width',0)==9504:
                x=max(0,sizes.crop_left_margin-sizes.left_margin);y=max(0,sizes.crop_top_margin-sizes.top_margin);rgb=rgb[y:y+6336,x:x+9504]
        return rgb.astype(np.float32)/65535
    if ext in ('.hif','.heif','.heic'):
        try:
            with heif_native.pixels(path) as (pixels,bits):return linear_lut(bits)[pixels]
        except heif_native.Unavailable:pass
        exe=shutil.which('heif-convert')
        if not exe:raise ValueError('HIF decoding needs libheif. Install with: brew install libheif')
        with tempfile.TemporaryDirectory(prefix='pixelshift-heif-') as directory:
            dest=Path(directory)/'image.png'
            r=subprocess.run([exe,'--quiet','--png-compression-level','0',str(path),str(dest)],capture_output=True,text=True)
            if r.returncode:raise ValueError(f'Cannot decode {path.name}: {r.stderr[-240:]}')
            image=cv2.imread(str(dest),cv2.IMREAD_UNCHANGED)
    else:image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3 or image.shape[2]<3:raise ValueError(f'Cannot decode RGB image {path.name}.')
    rgb=cv2.cvtColor(image[:,:,:3],cv2.COLOR_BGR2RGB).astype(np.float32)/(65535 if image.dtype==np.uint16 else 255)
    return raw_engine.srgb_decode(rgb)

def check_color(rows):
    for row in rows:
        if Path(row['SourceFile']).suffix.lower()=='.arw':continue
        # Avoid treating HDR transfer functions or wide-gamut values as sRGB.
        tr=row.get('TransferCharacteristics');prim=row.get('ColorPrimaries');space=row.get('ColorSpace');profile=str(row.get('ProfileDescription','')).lower()
        if profile and ('srgb' not in profile or 'linear' in profile):raise ValueError('Convert profiled rendered images to standard display-encoded sRGB before focus stacking.')
        if tr is not None and str(tr) not in ('13','sRGB or sYCC'):raise ValueError('Focus stacking currently supports SDR sRGB HIFs; convert HDR/HLG/PQ files to sRGB first.')
        if prim is not None and str(prim) not in ('1','BT.709'):raise ValueError('Convert wide-gamut images to sRGB before focus stacking.')
        if space is not None and str(space) not in ('1','sRGB'):raise ValueError('Focus stacking requires sRGB rendered images or ARW sources.')

def preview(linear,width=1600):
    h,w=linear.shape[:2];return cv2.resize(linear,(min(width,w),round(h*min(width,w)/w)),interpolation=cv2.INTER_AREA)

def gray(rgb):
    if rgb.ndim==2:return rgb
    return cv2.cvtColor(raw_engine.srgb_encode(rgb).astype(np.float32),cv2.COLOR_RGB2GRAY)

def register(reference,current):
    """Similarity transform maps reference coordinates into the source (focus breathing)."""
    a=gray(reference);b=gray(current);orb=cv2.ORB_create(nfeatures=6000,fastThreshold=8)
    ka,da=orb.detectAndCompute(np.uint8(a*255),None);kb,db=orb.detectAndCompute(np.uint8(b*255),None)
    matrix=np.eye(2,3,dtype=np.float32);inliers=0
    if da is not None and db is not None:
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
        if len(good)>=8:
            estimate,mask=cv2.estimateAffinePartial2D(np.float32([ka[m.queryIdx].pt for m in good]),np.float32([kb[m.trainIdx].pt for m in good]),method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=4000)
            if estimate is not None:matrix=estimate.astype(np.float32);inliers=int(mask.sum())
    # ECC refines translation/scale/rotation when enough common scene texture remains.
    if inliers<8:
        try:
            _,matrix=cv2.findTransformECC(a,b,matrix,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),None,5)
        except cv2.error:raise ValueError('Cannot align focus frames. Use a stable camera and overlapping focus steps.')
    scale=float(np.sqrt(abs(np.linalg.det(matrix[:,:2]))))
    if not .8<scale<1.2 or abs(matrix[0,1]+matrix[1,0])>.04 or np.linalg.norm(matrix[:,2])>max(a.shape)*.25:raise ValueError('Frame geometry differs too much to form a reliable focus stack.')
    warped=cv2.warpAffine(b,matrix,(a.shape[1],a.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT_101)
    mask=cv2.warpAffine(np.ones_like(b),matrix,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)>.5
    correlation=float(np.corrcoef(a[mask],warped[mask])[0,1]) if mask.sum()>100 else 0.
    if not np.isfinite(correlation) or correlation<.35:raise ValueError('Focus-frame alignment has too little common detail.')
    return matrix,correlation,scale,inliers

def warp_band(source,matrix,y,rows,width):
    xs=np.arange(width,dtype=np.float32)[None,:];ys=np.arange(y,y+rows,dtype=np.float32)[:,None]
    mx=matrix[0,0]*xs+matrix[0,1]*ys+matrix[0,2];my=matrix[1,0]*xs+matrix[1,1]*ys+matrix[1,2]
    valid=(mx>=1)&(my>=1)&(mx<source.shape[1]-2)&(my<source.shape[0]-2)
    # Read only the intersecting source band, rather than copying a full image.
    low=max(0,int(my.min())-3);high=min(source.shape[0],int(my.max())+4)
    if high<=low:return np.zeros((rows,width,3),np.float32),np.zeros((rows,width),bool)
    patch=np.asarray(source[low:high]);rgb=cv2.remap(patch,mx,my-low,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)/65535
    return rgb,valid

def sharpness(rgb,radius=3):
    luma=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    smooth=cv2.GaussianBlur(luma,(0,0),.7)
    # Modified Laplacian suppresses sensor noise before measuring focused edges.
    dx=np.abs(cv2.filter2D(smooth,-1,np.array([[-1,2,-1]],np.float32)))
    dy=np.abs(cv2.filter2D(smooth,-1,np.array([[-1],[2],[-1]],np.float32)))
    return cv2.GaussianBlur(dx+dy,(0,0),float(radius))

def fuse(images,validity,settings):
    # Streaming top-two reduction avoids a full per-pixel sort/copy of the stack.
    scores=[];best=np.zeros(images[0].shape[:2],np.float32);second=best.copy();scratch=best.copy()
    winner=np.zeros(best.shape,np.uint8)
    for i,(rgb,valid) in enumerate(zip(images,validity)):
        score=sharpness(rgb,settings['focus_radius']);score*=valid;scores.append(score)
        better=score>best;np.copyto(winner,i,where=better)
        np.minimum(best,score,out=scratch);np.maximum(second,scratch,out=second);np.maximum(best,score,out=best)
    confidence=(best-second)/(best+1e-7)
    weights=np.stack(scores);del scores
    weights+=1e-8;np.divide(weights,best[None]+1e-8,out=weights)
    np.power(weights,settings['focus_selectivity'],out=weights)
    for weight,valid in zip(weights,validity):weight*=valid
    norm=weights.sum(axis=0);none=norm<1e-12;weights[0,none]=1;norm[none]=1
    weights/=norm[None]
    # Laplacian pyramid fusion avoids hard cut seams while retaining focused detail.
    levels=3;total=[None]*(levels+1)
    sums=[None]*(levels+1)
    for i,rgb in enumerate(images):
        gp=[rgb];wp=[weights[i]]
        for _ in range(levels):gp.append(cv2.pyrDown(gp[-1]));wp.append(cv2.pyrDown(wp[-1]))
        lp=[gp[k]-cv2.pyrUp(gp[k+1],dstsize=(gp[k].shape[1],gp[k].shape[0])) for k in range(levels)]+[gp[-1]]
        for k in range(levels+1):
            value=lp[k]*wp[k][:,:,None]
            if total[k] is None:total[k]=value;sums[k]=wp[k].copy()
            else:total[k]+=value;sums[k]+=wp[k]
    fused=total[-1]/np.maximum(sums[-1][:,:,None],1e-8)
    for k in reversed(range(levels)):fused=cv2.pyrUp(fused,dstsize=(total[k].shape[1],total[k].shape[0]))+total[k]/np.maximum(sums[k][:,:,None],1e-8)
    return np.clip(fused,0,1),winner.astype(np.uint8),confidence,none

def band_geometry(settings,height):
    """Pyramid-aligned bands with enough halo for the largest supported filters."""
    halo=int(np.ceil((settings['focus_radius']*4+40)/8)*8)
    core=256
    for y in range(0,height,core):yield y,max(0,y-halo),min(height,y+core+halo),min(y+core,height)
