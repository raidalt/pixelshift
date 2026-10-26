import unittest,tempfile,json,struct,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
import app,raw_engine,photo_settings,image_tools,inspector
import test_engine

class ControlsTests(unittest.TestCase):
    def tearDown(self):inspector.ACTIVE=None
    def test_validation(self):
        for values in ({'exposure':float('nan')},{'filename':'../../private'},{'kernel':'bad'},{'region_mode':'manual'},{'regions':[{'kind':'exclude','x':.9,'y':.1,'w':.5,'h':.2}]}):
            with self.assertRaises(ValueError):photo_settings.validate(values)
        self.assertEqual(photo_settings.validate({'exposure':1.5})['exposure'],1.5)
    def test_strict_threshold(self):
        a=np.random.default_rng(1).random((32,32),dtype=np.float32);warps=[(.99,np.float32([[1,0,x],[0,1,0]])) for x in (0,.2,.4)]
        with patch.object(raw_engine.cv2,'findTransformECC',side_effect=warps):
            with self.assertRaisesRegex(ValueError,'Strict alignment'):raw_engine.alignment([a]*3,[a]*3,{'alignment_mode':'strict','disagreement':.1})
    def test_manual_masks(self):
        sensor=np.random.default_rng(7).integers(1000,15000,(1200,1600),dtype=np.uint16)
        meta={'crop':(0,0,1600,1200),'pattern':[[0,1],[3,2]],'black':[512]*4,'white':16383}
        options={'region_mode':'manual','regions':[{'kind':'include','x':0,'y':0,'w':.8,'h':1},{'kind':'exclude','x':0,'y':0,'w':.3,'h':1}]}
        regions=raw_engine.select_regions(sensor,meta,options);self.assertGreaterEqual(len(regions),2)
        for r in regions:self.assertTrue(.3<(r['left']+r['size']/2)/1600<=.8)
    def test_both_kernels_and_maps(self):
        scene,sensors,meta,shifts=test_engine.Tests().fixture(2)
        for kernel in ('triangle','cosine'):
            rgb,missing,maps=raw_engine.reconstruct_band(sensors,[meta]*16,shifts,2,0,40,48,settings={'kernel':kernel},diagnostics=True)
            np.testing.assert_allclose(rgb,scene[16:56,16:64]/65535,atol=1e-7)
            self.assertEqual(missing,0);self.assertTrue(np.all(maps['coverage']>.99));self.assertFalse(maps['fallback'].any())
    def test_color_controls(self):
        rgb=np.full((16,16,3),.1,np.float32);meta={'wb':[1]*4,'transform':np.eye(3).tolist(),'camera_xyz':np.eye(3).tolist()}
        np.testing.assert_allclose(raw_engine.render_float(rgb,meta,{'exposure':2}),rgb*4)
        a=raw_engine.render_float(rgb,meta,{'white_balance':'custom','temperature':3000});b=raw_engine.render_float(rgb,meta,{'white_balance':'custom','temperature':9000})
        self.assertFalse(np.allclose(a,b));self.assertTrue(np.isfinite(raw_engine.render_float(rgb,meta,{'denoise':30,'sharpening':.4})).all())
    def test_icc_profiles(self):
        for linear in (False,True):
            p=image_tools.profile(linear);self.assertEqual(struct.unpack_from('>I',p)[0],len(p));self.assertEqual(p[36:40],b'acsp')
            n=struct.unpack_from('>I',p,128)[0];tags={p[132+i*12:136+i*12]:struct.unpack_from('>II',p,136+i*12) for i in range(n)}
            offset,_=tags[b'rTRC'];self.assertEqual(p[offset:offset+4],b'curv' if linear else b'para')
    def test_preview_export_finishing_and_crop(self):
        linear=np.random.default_rng(4).random((128,160,3),dtype=np.float32)*.3;meta={'wb':[1]*4,'transform':np.eye(3).tolist(),'crop':[0,0,160,128]}
        settings=photo_settings.validate({'exposure':.5,'sharpening':.4,'denoise':15,'border':4})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir();np.save(cache/'linear.npy',linear);out=root/'image.tif'
            report={'frames':[{'name':'FRAME0001.ARW','x':0,'y':0,'confidence':1,'patch_spread':0}],'scale':1,'sources':[]}
            path=app.export_file(linear,meta,settings,out,report,cache);actual=tifffile.imread(out)
            self.assertEqual(actual.shape,(120,152,3));np.testing.assert_allclose(actual,raw_engine.render(linear,meta,settings)[4:-4,4:-4],atol=1)
            with tifffile.TiffFile(out) as f:self.assertIn(34675,f.pages[0].tags)
            inspector.open_report(path);preview=inspector.composite((4,4,152,120,152,120),settings)
            np.testing.assert_allclose(np.rint(raw_engine.srgb_encode(preview)*65535),actual,atol=1)
    def test_small_end_to_end_build_and_offline_reference(self):
        scene,sensors,meta,shifts=test_engine.Tests().fixture(1)
        for shift in shifts:shift.update(confidence=1.,patch_spread=0.)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();paths=[source/f'FRAME{i:04}.ARW' for i in range(4)]
            for path in paths:path.touch()
            out=root/'result.tif'
            with patch.object(app,'BASE_H',20),patch.object(app,'BASE_W',24),patch.object(raw_engine,'verify',return_value=(paths,['synthetic'])),patch.object(raw_engine,'read_sensor',side_effect=[(s.copy(),copy.deepcopy(meta)) for s in sensors]),patch.object(raw_engine,'select_regions',return_value=[]),patch.object(raw_engine,'gray_patches',return_value=[]),patch.object(raw_engine,'alignment',side_effect=shifts[1:]):
                app.CANCEL.clear();app.merge({'folder':str(source),'count':4,'output':str(out),'settings':{}})
            self.assertEqual(app.STATE['status'],'done',app.STATE)
            report=json.loads(Path(str(out)+'.json').read_text());self.assertTrue((Path(report['cache'])/'maps.npy').exists())
            for path in paths:path.unlink()
            inspector.open_report(str(out)+'.json');self.assertTrue(inspector.describe()['source_available'][0]);v=inspector.view({})
            body,stats=inspector.tile({'view':v['view'],'kind':'original','x':0,'y':0,'w':24,'h':20,'ow':24,'oh':20})
            self.assertTrue(body.startswith(b'\x89PNG'));self.assertIn('histogram',stats)
    def test_legacy_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'legacy.tif';tifffile.imwrite(out,np.full((32,48,3),32768,np.uint16),photometric='rgb');r=Path(str(out)+'.json');r.write_text(json.dumps({'width':48,'height':32,'frames':[]}))
            info=inspector.open_report(r);self.assertFalse(info['raw_controls']);self.assertFalse(info['maps'])
            view=inspector.view({'exposure':1});body,stats=inspector.tile({'view':view['view'],'kind':'composite','ow':48,'oh':32});self.assertTrue(body.startswith(b'\x89PNG'))
            with self.assertRaisesRegex(ValueError,'RAW'):inspector.view({'white_balance':'custom'})
    def test_legacy_linear_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'linear.tif';tifffile.imwrite(out,np.full((16,24,3),16384,np.uint16),photometric='rgb')
            report=Path(str(out)+'.json');report.write_text(json.dumps({'output_encoding':'linear','frames':[]}))
            inspector.open_report(report)
            np.testing.assert_allclose(inspector.composite((0,0,24,16,24,16),photo_settings.DEFAULTS),16384/65535,atol=1e-6)

    def test_presets(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app,'DEFAULT_OUTPUT',Path(directory)):
                data=app.save_preset({'name':'Synthetic test','settings':{'exposure':1}});self.assertEqual(data['Synthetic test']['exposure'],1)
                self.assertEqual(app.presets()['Synthetic test']['exposure'],1);self.assertNotIn('Synthetic test',app.save_preset({'name':'Synthetic test','delete':True}))

if __name__=='__main__':unittest.main()
