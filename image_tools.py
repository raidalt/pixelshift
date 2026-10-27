"""Color profiles and inspection statistics shared by preview and export."""
import struct
import numpy as np
import raw_engine

def profile(linear=False):
    def fixed(x):return struct.pack('>i',round(x*65536))
    def xyz(v):return b'XYZ '+b'\0'*4+b''.join(fixed(x) for x in v)
    def text(value):
        value=value.encode('utf-16-be')
        return b'mluc'+b'\0'*4+struct.pack('>II',1,12)+b'enUS'+struct.pack('>II',len(value),28)+value
    trc=b'curv'+b'\0'*4+struct.pack('>IH',1,256) if linear else b'para'+b'\0'*4+struct.pack('>HH',4,0)+b''.join(fixed(x) for x in (2.4,1/1.055,.055/1.055,1/12.92,.04045,0,0))
    tags={b'desc':text('Pixelshift Linear sRGB' if linear else 'Pixelshift sRGB'),b'cprt':text('Public colorimetric matrix profile'),
          b'wtpt':xyz((.9642,1,.8249)),b'rXYZ':xyz((.4360747,.2225045,.0139322)),
          b'gXYZ':xyz((.3850649,.7168786,.0971045)),b'bXYZ':xyz((.1430804,.0606169,.7141733)),
          b'rTRC':trc,b'gTRC':trc,b'bTRC':trc}
    offset=128+4+len(tags)*12;records=[];data=b''
    for key,value in tags.items():
        records.append(key+struct.pack('>II',offset+len(data),len(value)))
        data+=value+b'\0'*((-len(value))%4)
    header=bytearray(128);struct.pack_into('>I',header,0,offset+len(data));header[4:8]=b'pxsh'
    struct.pack_into('>I',header,8,0x04300000);header[12:24]=b'mntrRGB XYZ '
    struct.pack_into('>6H',header,24,2026,1,1,0,0,0);header[36:40]=b'acsp';header[40:44]=b'APPL'
    struct.pack_into('>I',header,64,1);header[68:80]=b''.join(fixed(x) for x in (.9642,1,.8249));header[80:84]=b'pxsh'
    body=struct.pack('>I',len(tags))+b''.join(records)+data
    # ICC.1 permits an all-zero optional profile ID. Color transforms use the tags.
    return bytes(header)+body

def statistics(linear,encoded=None):
    clipped_high=np.any(linear>=1,axis=2);clipped_low=np.all(linear<=0,axis=2)
    if encoded is None:encoded=raw_engine.srgb_encode(linear)
    hist=[]
    for c in range(3):
        values=encoded[:,:,c].ravel();values=values[np.isfinite(values)]
        bins=np.minimum((values*64).astype(np.int32),63)
        hist.append(np.bincount(bins,minlength=64).tolist())
    return {'highlights':float(clipped_high.mean()),'shadows':float(clipped_low.mean()),'histogram':hist,'pixels':int(clipped_high.size)}

def png(rgb):
    import cv2
    if rgb.dtype!=np.uint8:rgb=np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
    ok,buffer=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if not ok:raise ValueError('Preview encoding failed.')
    return buffer.tobytes()
