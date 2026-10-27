"""Numerical and resource-bound regressions for optimized processing paths."""
import copy,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
import cv2,numpy as np
import reference_cache
import app,capture_metadata,focus_engine as focus,heif_native,image_tools,inspector,performance_runtime as runtime,photo_settings

class PerformanceTests(unittest.TestCase):
    def test_native_integer_luts_preserve_previous_conversion(self):
        import raw_engine
        for bits in (8,10,12,16):
            values=np.arange(1<<bits,dtype=np.uint16)
            encoded=values.astype(np.float32)/255 if bits==8 else (values<<(16-bits)).astype(np.float32)/65535
            expected=raw_engine.srgb_decode(encoded)
            np.testing.assert_array_equal(focus.linear_lut(bits),expected)
            np.testing.assert_array_equal(focus.linear_lut(bits,True),np.rint(expected*65535).astype('uint16'))
    def test_direct_decode_writes_banded_cache(self):
        pixels=np.random.default_rng(3).integers(0,1024,(263,320,3),dtype=np.uint16)
        @contextmanager
        def native(path):yield pixels,10
        with tempfile.TemporaryDirectory() as directory,patch.object(heif_native,'pixels',native):
            dest=Path(directory)/'cache.npy';thumbnail,backend=focus.decode_to_cache(Path('test.hif'),dest)
            np.testing.assert_array_equal(np.load(dest),focus.linear_lut(10,True)[pixels]);self.assertEqual(backend,'libheif-direct');self.assertEqual(thumbnail.shape,pixels.shape)
            np.testing.assert_array_equal(thumbnail,focus.preview(focus.linear_lut(10)[pixels]))
            np.testing.assert_array_equal(np.load(dest.with_suffix('.preview.npy')),focus.gray(thumbnail))
    def test_unavailable_native_decoder_keeps_fallback(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(heif_native,'pixels',side_effect=heif_native.Unavailable()),patch.object(focus,'decode',return_value=np.full((12,16,3),.2,np.float32)):
            dest=Path(directory)/'cache.npy';_,kind=focus.decode_to_cache('test.hif',dest);self.assertEqual(kind,'image-decoder');self.assertTrue(dest.exists())
    def test_decode_cancellation_does_not_fall_back(self):
        @contextmanager
        def native(path):yield np.zeros((300,16,3),np.uint16),10
        def cancel():raise InterruptedError('cancel')
        with tempfile.TemporaryDirectory() as directory,patch.object(heif_native,'pixels',native),patch.object(focus,'decode') as fallback:
            with self.assertRaises(InterruptedError):focus.decode_to_cache('test.hif',Path(directory)/'cache.npy',cancel)
            fallback.assert_not_called()
    def test_optimized_geometry_matches_whole_image(self):
        rng=np.random.default_rng(4);a=rng.random((768,96,3),dtype=np.float32)*.5;b=cv2.GaussianBlur(a,(0,0),2)
        a[384:]=b[384:];valid=np.ones(a.shape[:2],bool)
        for radius in (1.,3.,12.):
            settings={**photo_settings.DEFAULTS,'focus_radius':radius};whole=focus.fuse([a,b],[valid,valid],settings)[0];actual=np.empty_like(whole)
            for y,start,end,stop in focus.band_geometry(settings,a.shape[0]):
                band=focus.fuse([a[start:end],b[start:end]],[valid[start:end]]*2,settings)[0];actual[y:stop]=band[y-start:stop-start]
            np.testing.assert_allclose(actual,whole,atol=2e-6)
    def test_histogram_fast_path_matches_numpy(self):
        import raw_engine
        rgb=np.random.default_rng(6).random((101,97,3),dtype=np.float32)*1.3-.1
        encoded=raw_engine.srgb_encode(rgb);encoded[0,0]=[0,.5,1]
        stats=image_tools.statistics(rgb,encoded)
        self.assertEqual(stats['histogram'],[np.histogram(encoded[:,:,c],64,range=(0,1))[0].tolist() for c in range(3)])
    def test_metadata_cache_invalidates_changed_files(self):
        capture_metadata.CACHE.clear()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.jpg';path.write_bytes(b'a')
            result=type('Result',(),{'returncode':0,'stdout':json.dumps([{'SourceFile':str(path),'Model':'Test'}]),'stderr':''})()
            with patch.object(capture_metadata.shutil,'which',return_value='exiftool'),patch.object(capture_metadata.subprocess,'run',return_value=result) as run:
                first=capture_metadata.read([path]);first[0]['Model']='mutated'
                self.assertEqual(capture_metadata.read([path])[0]['Model'],'Test');self.assertEqual(run.call_count,1)
                path.write_bytes(b'changed');capture_metadata.read([path]);self.assertEqual(run.call_count,2)
    def test_metadata_cache_limit_does_not_drop_requested_rows(self):
        capture_metadata.CACHE.clear()
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/f'{i}.jpg' for i in range(4)]
            for path in paths:path.touch()
            result=type('Result',(),{'returncode':0,'stdout':json.dumps([{'SourceFile':str(path)} for path in paths]),'stderr':''})()
            with patch.object(capture_metadata,'LIMIT',2),patch.object(capture_metadata.shutil,'which',return_value='exiftool'),patch.object(capture_metadata.subprocess,'run',return_value=result):self.assertEqual(len(capture_metadata.read(paths)),4)
    def test_bounded_parallel_results_keep_input_order(self):
        def task(i):time.sleep(.002*(4-i));return i*i
        self.assertEqual(list(runtime.ordered_map(task,range(5),2)),[0,1,4,9,16])
    def test_parallel_cancellation_and_worker_bounds(self):
        with patch.dict(os.environ,{'PIXELSHIFT_WORKERS':'200'}):self.assertEqual(runtime.workers(),2)
        with patch.dict(os.environ,{'PIXELSHIFT_WORKERS':'bad'}):self.assertEqual(runtime.workers(),1)
        with self.assertRaises(InterruptedError):list(runtime.ordered_map(lambda i:i,range(5),2,lambda:(_ for _ in ()).throw(InterruptedError())))
    def test_export_exclusive_copy_fallback_and_existing_file(self):
        rgb=np.full((32,48,3),.2,np.float32);meta={'space':'linear_srgb'}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'image.tif'
            with patch.object(app.os,'link',side_effect=OSError('unsupported')):app.export_file(rgb,meta,{},out,{'frames':[]},None)
            original=out.read_bytes()
            with self.assertRaises(ValueError):app.export_file(rgb,meta,{},out,{'frames':[]},None)
            self.assertEqual(out.read_bytes(),original)
    def test_reference_reuse_requires_unchanged_source_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'photo.hif';source.write_bytes(b'photo');cache=root/'.pixelshift-data'/'one';cache.mkdir(parents=True);frame=cache/'frame-0.npy';np.save(frame,np.zeros((12,16,3),np.uint16))
            reference_cache.register([source],cache)
            self.assertEqual(reference_cache.find([source],root),[frame])
            source.write_bytes(b'changed');self.assertEqual(reference_cache.find([source],root),[None])
            reference_cache.register([source],cache);frame.touch();self.assertEqual(reference_cache.find([source],root),[None])
    def test_complete_rebuild_reuses_references_and_exact_previews(self):
        import tifffile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sources=[root/f'F{i}.HIF' for i in range(2)]
            for source in sources:source.write_bytes(b'synthetic')
            rows=[{'SourceFile':str(p),'ImageWidth':64,'ImageHeight':48,'FocusPosition2':i} for i,p in enumerate(sources)]
            def decode(path,target,check_cancel):
                i=int(Path(path).stem[1:]);rgb=np.full((48,64,3),.2+i*.1,np.float32);np.save(target,np.rint(rgb*65535).astype('uint16'));preview=focus.gray(focus.preview(rgb));np.save(Path(target).with_suffix('.preview.npy'),preview);return preview,'test'
            with patch.dict(app.FOCUS_CAPTURES,{'reuse-test':rows}),patch.object(focus,'decode_to_cache',side_effect=decode),patch.object(focus,'register',return_value=(np.eye(2,3,dtype=np.float32),1.,1.,20)):
                app.CANCEL.clear();app.focus_merge({'capture':'reuse-test','output':str(root/'first.tif'),'settings':{}})
            self.assertEqual(app.STATE['status'],'done',app.STATE['message']);one=json.loads((root/'first.tif.json').read_text())
            with patch.dict(app.FOCUS_CAPTURES,{'reuse-test':rows}),patch.object(focus,'decode_to_cache',side_effect=AssertionError('unchanged references must be reused')),patch.object(focus,'decode',side_effect=AssertionError('exact previews must be reused')),patch.object(focus,'register',return_value=(np.eye(2,3,dtype=np.float32),1.,1.,20)):
                app.focus_merge({'capture':'reuse-test','output':str(root/'second.tif'),'settings':{}})
            self.assertEqual(app.STATE['status'],'done',app.STATE['message']);two=json.loads((root/'second.tif.json').read_text());self.assertEqual(two['performance']['reused_references'],2)
            self.assertEqual((Path(one['cache'])/'frame-0.npy').stat().st_ino,(Path(two['cache'])/'frame-0.npy').stat().st_ino)
            np.testing.assert_array_equal(tifffile.imread(root/'first.tif'),tifffile.imread(root/'second.tif'))

    def test_modified_registration_preview_invalidates_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'photo.hif';source.touch();cache=root/'.pixelshift-data'/'one';cache.mkdir(parents=True);np.save(cache/'frame-0.npy',np.zeros((12,16,3),np.uint16));preview=cache/'frame-0.preview.npy';np.save(preview,np.zeros((12,16),np.float32));reference_cache.register([source],cache)
            self.assertIsNotNone(reference_cache.find([source],root)[0]);preview.touch();self.assertEqual(reference_cache.find([source],root),[None])

    def test_changed_source_is_not_registered_under_a_new_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'photo.hif';source.write_bytes(b'old');old=reference_cache.key(source);cache=root/'.pixelshift-data'/'one';cache.mkdir(parents=True);np.save(cache/'frame-0.npy',np.zeros((12,16,3),np.uint16));source.write_bytes(b'new input')
            reference_cache.register([source],cache,[old]);self.assertEqual(reference_cache.find([source],root),[None])
    def test_reference_cache_handles_unsupported_hard_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'photo.hif';source.touch();cache=root/'.pixelshift-data'/'one';cache.mkdir(parents=True);np.save(cache/'frame-0.npy',np.zeros((12,16,3),np.uint16));reference_cache.register([source],cache)
            with patch.object(reference_cache.os,'link',side_effect=OSError('unsupported')):self.assertEqual(reference_cache.find([source],root),[None])

    def test_identical_tiles_reuse_encoded_preview(self):
        import tifffile
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'image.tif';tifffile.imwrite(out,np.full((32,48,3),20000,np.uint16),photometric='rgb');path=Path(str(out)+'.json');path.write_text(json.dumps({'frames':[]}));inspector.open_report(path);view=inspector.view({})['view']
            with patch.object(inspector,'_tile',wraps=inspector._tile) as render:
                query={'view':view,'kind':'composite','ow':48,'oh':32};first=inspector.tile(query);self.assertEqual(first,inspector.tile(query));self.assertEqual(render.call_count,1)
                other=inspector.view({'exposure':1})['view'];inspector.tile({**query,'view':other});self.assertEqual(render.call_count,2)

if __name__=='__main__':unittest.main()
