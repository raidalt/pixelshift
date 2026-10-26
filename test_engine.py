import tempfile, unittest, threading, urllib.request, urllib.error
from pathlib import Path
from unittest.mock import patch
import numpy as np
import app, raw_engine

class Tests(unittest.TestCase):
    def fixture(self,scale):
        rng=np.random.default_rng(5)
        h,w=40,48
        scene=rng.integers(1000,50000,(h*scale,w*scale,3),dtype=np.uint16)
        base=[(0,0),(0,-1),(1,-1),(1,0)]
        shifts=[{'x':x+fx,'y':y+fy} for fx,fy in ([(0,0)] if scale==1 else [(0,0),(.5,0),(0,.5),(.5,.5)]) for x,y in base]
        meta={'crop':(8,8,24,20),'pattern':[[0,1],[3,2]],'black':[0]*4,'white':65535,'wb':[1]*4,'transform':np.eye(3).tolist(),'flip':0}
        sensors=[]
        for shift in shifts:
            yy=np.clip(np.arange(h)*scale-int(shift['y']*scale),0,h*scale-1)
            xx=np.clip(np.arange(w)*scale-int(shift['x']*scale),0,w*scale-1)
            sensor=np.zeros((h,w),np.uint16)
            for py in range(2):
                for px in range(2):
                    c=meta['pattern'][py][px];c=1 if c==3 else c
                    sensor[py::2,px::2]=scene[yy[py::2,None],xx[None,px::2],c]
            sensors.append(sensor)
        return scene,sensors,meta,shifts

    def test_exact_raw_reconstruction_4_and_16(self):
        for scale in (1,2):
            scene,sensors,meta,shifts=self.fixture(scale)
            rgb,missing=raw_engine.reconstruct_band(sensors,[meta]*len(sensors),shifts,scale,0,20*scale,24*scale)
            expected=scene[8*scale:28*scale,8*scale:32*scale].astype(np.float32)/65535
            self.assertEqual(missing,0)
            np.testing.assert_allclose(rgb,expected,atol=1e-7)
            # A high-frequency random scene cannot be recovered by ordinary upscaling.

    def test_black_and_white_correction(self):
        _,sensors,meta,shifts=self.fixture(1)
        sensors=[(s//4+512).astype(np.uint16) for s in sensors]
        meta.update(black=[512]*4,white=16895)
        rgb,missing=raw_engine.reconstruct_band(sensors,[meta]*4,shifts,1,0,20,24)
        self.assertTrue(np.isfinite(rgb).all());self.assertEqual(missing,0)

    def test_single_frame_color_order(self):
        meta={'pattern':[[0,1],[3,2]],'black':[0]*4,'white':65535}
        sensor=np.zeros((20,20),np.uint16)
        sensor[0::2,0::2]=10000;sensor[0::2,1::2]=20000;sensor[1::2,0::2]=20000;sensor[1::2,1::2]=30000
        rgb=raw_engine.demosaic(sensor,meta)
        np.testing.assert_allclose(rgb[5,5],np.array([10000,20000,30000])/65535,atol=1e-6)

    def test_sequence_metadata_order_and_missing(self):
        rows=[{'SourceFile':str(i),'Model':'ILCE-7RM5','PixelShiftInfo':f'12345678 {i} 4 0x1','ISO':100} for i in [4,2,1,3]]
        groups,bad=raw_engine.group_rows(rows,4)
        self.assertEqual([r['SourceFile'] for r in groups[0][1]],['1','2','3','4'])
        self.assertEqual(raw_engine.group_rows(rows[:-1],4)[0],[])
        self.assertEqual(raw_engine.group_rows(rows,16)[0],[])
        self.assertEqual(raw_engine.group_rows(rows+[rows[0]],4)[0],[])
        rows[0]['ISO']=200
        self.assertEqual(raw_engine.group_rows(rows,4)[0],[])

    def test_nested_cross_folder_metadata_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);deep=root
            for i in range(40):deep=deep/str(i)
            deep=deep/'.hidden';deep.mkdir(parents=True)
            paths=[deep/'unrelated-name.ARW',root/'z.ARW',deep/'other.ARW',root/'x.ARW']
            for p in paths:p.touch()
            (deep/'cycle').symlink_to(root,target_is_directory=True)
            rows=[{'SourceFile':str(p),'Model':'ILCE-7RM5','PixelShiftInfo':f'12345678 {i+1} 4 0x1'} for i,p in enumerate(paths)]
            with patch.object(raw_engine,'metadata',return_value=rows):report=app.scan_captures(root,4)
            self.assertEqual(report['total'],4);self.assertEqual(len(report['sets']),1)
            self.assertTrue(report['sets'][0]['verified']);self.assertEqual(report['folders'],42)

    def test_pictures_default_and_unique_name(self):
        with tempfile.TemporaryDirectory() as directory:
            default=Path(directory)/'Pictures'/'Pixel Shift'
            with patch.object(app,'DEFAULT_OUTPUT',default):
                p=app.output_path('',[Path('FRAME0001.ARW')],4)
                self.assertEqual(p.parent,default.resolve());p.touch()
                self.assertNotEqual(p,app.output_path('',[Path('FRAME0001.ARW')],4))

    def test_alignment_disagreement_warns_and_continues(self):
        rng=np.random.default_rng(1)
        patches=[rng.random((32,32),dtype=np.float32) for _ in range(3)]
        warps=[(.99,np.float32([[1,0,x],[0,1,0]])) for x in (0,1,4)]
        with patch.object(raw_engine.cv2,'phaseCorrelate',return_value=((0,0),1)),patch.object(raw_engine.cv2,'findTransformECC',side_effect=warps):
            result=raw_engine.alignment(patches,patches)
        self.assertEqual(result['x'],1)
        self.assertEqual(result['patch_spread'],3)
        self.assertIn('Continuing',result['warning'])

    def test_alignment_still_rejects_unusable_data(self):
        patches=[np.zeros((32,32),np.float32)]*3
        with self.assertRaisesRegex(ValueError,'Insufficient stationary detail'):
            raw_engine.alignment(patches,patches)

    def test_texture_regions_avoid_blank_background(self):
        rng=np.random.default_rng(7)
        sensor=np.full((1200,1600),512,np.uint16)
        sensor[400:1000,400:1200]=rng.integers(1000,15000,(600,800),dtype=np.uint16)
        meta={'pattern':[[0,1],[3,2]],'black':[512]*4,'white':16383}
        regions=raw_engine.select_regions(sensor,meta)
        self.assertGreaterEqual(len(regions),2)
        for region in regions:
            cy=region['top']+region['size']/2;cx=region['left']+region['size']/2
            block=sensor[region['top']:region['top']+region['size'],region['left']:region['left']+region['size']]
            self.assertGreaterEqual(float((block!=512).mean()),.25)
            self.assertGreater(region['texture_score'],0)
        patches=raw_engine.gray_patches(sensor,meta,regions)
        self.assertEqual(len(patches),len(regions))

    def test_small_shift_registration_on_textured_regions(self):
        import cv2
        rng=np.random.default_rng(8)
        a=cv2.GaussianBlur(rng.random((256,256),dtype=np.float32),(0,0),1.2)
        b=cv2.warpAffine(a,np.float32([[1,0,.9],[0,1,-1]]),(256,256),borderMode=cv2.BORDER_REFLECT)
        result=raw_engine.alignment([a,a,a],[b,b,b])
        self.assertLess(abs(result['x']-.9),.08)
        self.assertLess(abs(result['y']+1),.08)
        self.assertIsNone(result['warning'])

    def test_http_interface_and_token(self):
        server=app.ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(url) as response:html=response.read()
            self.assertIn(b'Before and after scrub position',html)
            self.assertNotIn(b'__DEFAULT_OUTPUT__',html)
            request=urllib.request.Request(url+'/api/start',data=b'{}',method='POST')
            with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(request)
            self.assertEqual(e.exception.code,403);e.exception.close()
        finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
