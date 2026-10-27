"""Local processing benchmark. Uses synthetic pixels unless --report is supplied."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
import focus_engine,performance_runtime,photo_settings

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Existing focus-stack JSON report; references are opened read-only')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--runs',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.runs<=20:parser.error('Use 1–20 runs.')
    settings=photo_settings.DEFAULTS
    if args.report:
        report=json.loads(args.report.read_text());cache=Path(report['cache']);settings=photo_settings.validate(report.get('settings'))
        if report.get('kind')!='focus':parser.error('Choose a focus-stack report.')
        arrays=[np.load(cache/f'frame-{i}.npy',mmap_mode='r') for i in range(len(report['frames']))];matrices=[np.array(f['matrix'],np.float32) for f in report['frames']]
    else:
        rng=np.random.default_rng(1);scene=rng.integers(1000,50000,(768,1536,3),dtype=np.uint16)
        arrays=[scene,cv2.GaussianBlur(scene,(0,0),2)];matrices=[np.eye(2,3,dtype=np.float32)]*2
    height,width=arrays[0].shape[:2];core=min(256,height);halo=int(np.ceil((settings['focus_radius']*4+40)/8)*8)
    middle=max(0,(height-core)//2//8*8);geometry=[(middle,max(0,middle-halo),min(height,middle+core+halo),min(height,middle+core))]*args.workers
    def band(bounds):
        y,start,end,stop=bounds;images=[];valid=[]
        for source,matrix in zip(arrays,matrices):
            image,mask=focus_engine.warp_band(source,matrix,start,end-start,width);images.append(image);valid.append(mask)
        return float(focus_engine.fuse(images,valid,settings)[0][y-start:stop-start].mean())
    samples=[]
    for _ in range(args.runs):
        start=time.perf_counter();list(performance_runtime.ordered_map(band,geometry,args.workers));samples.append(time.perf_counter()-start)
    print(json.dumps({'frames':len(arrays),'width':width,'core_rows':core,'workers':args.workers,'runs_seconds':samples,'median_seconds_per_band':float(np.median(samples))/args.workers},indent=2))

if __name__=='__main__':main()
