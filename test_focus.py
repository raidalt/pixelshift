"""Focus-stack math and workflow checks, without private camera files."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import app,focus_engine as focus,photo_settings,inspector

class FocusTests(unittest.TestCase):
    def test_candidate_sequences_deduplicate_and_exclude_pixelshift(self):
        rows=[]
        for i in range(3):
            rows.append({'SourceFile':f'/card/a/F{i}.HIF','DateTimeOriginal':'2026:01:01 01:00:00','SubSecTimeOriginal':str(100+i*200),'SequenceNumber':i+1,'PixelShiftInfo':'00000000 0 0 0x0'})
        rows.append({**rows[0],'SourceFile':'/card/a/F0.JPG'})
        rows.append({**rows[-2],'SourceFile':'/card/a/P.ARW','PixelShiftInfo':'001 1 16 0xa'})
        groups=focus.group_rows(rows);self.assertEqual(len(groups),1);self.assertEqual(len(groups[0]),3)
        rows.append({**rows[0],'SourceFile':'/card/a/N0.HIF','DateTimeOriginal':'2026:01:01 01:00:30'})
        self.assertEqual(len(focus.group_rows(rows)),1)
    def test_large_card_preserves_latest_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);groups=[[{'SourceFile':str(root/f'B{j}-{i}.JPG')} for i in range(2)] for j in range(180)]
            with patch.object(focus,'metadata',return_value=[]),patch.object(focus,'group_rows',return_value=groups):
                found=app.scan_focus(str(root))
            self.assertIn(found['sets'][0]['id'],app.FOCUS_CAPTURES)
            self.assertEqual(len(found['sets']),180)

    def test_hdr_is_not_silently_treated_as_srgb(self):
        focus.check_color([{'SourceFile':'a.hif','TransferCharacteristics':13,'ColorPrimaries':1,'ColorSpace':1}])
        with self.assertRaisesRegex(ValueError,'HDR'):focus.check_color([{'SourceFile':'a.hif','TransferCharacteristics':18}])
    def scene(self):
        a=np.random.default_rng(9).random((256,384,3),dtype=np.float32)*.7+.1
        return cv2.GaussianBlur(a,(0,0),.5)
    def test_fusion_recovers_complementary_focused_regions(self):
        sharp=self.scene();blur=cv2.GaussianBlur(sharp,(0,0),3)
        a=sharp.copy();a[:,192:]=blur[:,192:];b=sharp.copy();b[:,:192]=blur[:,:192]
        result,winner,confidence,missing=focus.fuse([a,b],[np.ones(a.shape[:2],bool)]*2,photo_settings.DEFAULTS)
        core=np.s_[32:-32,32:-32]
        self.assertLess(np.mean((result[core]-sharp[core])**2),np.mean(((a[core]+b[core])/2-sharp[core])**2)*.2)
        self.assertLess(winner[32:-32,32:150].mean(),.02);self.assertGreater(winner[32:-32,240:-32].mean(),.98)
        self.assertFalse(missing.any());self.assertTrue((confidence>=0).all())
    def test_similarity_recovers_focus_breathing(self):
        a=self.scene();m=cv2.getRotationMatrix2D((192,128),.5,1.025);m[:,2]+=[3,-2]
        b=cv2.warpAffine(a,m,(384,256),borderMode=cv2.BORDER_REFLECT_101)
        actual,corr,scale,_=focus.register(a,b);np.testing.assert_allclose(actual,m,atol=.7);self.assertGreater(corr,.85);self.assertAlmostEqual(scale,1.025,delta=.01)
    def test_inverse_warp_band(self):
        rgb=self.scene();encoded=np.rint(rgb*65535).astype(np.uint16)
        actual,valid=focus.warp_band(encoded,np.array([[1,0,4],[0,1,3]],np.float32),12,50,384)
        np.testing.assert_allclose(actual[:,:-4],rgb[15:65,4:],atol=1/65535);self.assertFalse(valid[:,-1].any())
    def test_small_full_stack_and_offline_inspector(self):
        sharp=self.scene()[:96,:128];blur=cv2.GaussianBlur(sharp,(0,0),3);a=sharp.copy();b=sharp.copy();a[:,64:]=blur[:,64:];b[:,:64]=blur[:,:64]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[{'SourceFile':str(root/f'F{i}.JPG'),'ImageWidth':128,'ImageHeight':96,'FocusPosition2':100+i} for i in range(2)]
            with patch.dict(app.FOCUS_CAPTURES,{'test':rows}),patch.object(focus,'decode',side_effect=[a,b]),patch.object(focus,'register',return_value=(np.eye(2,3,dtype=np.float32),1.,1.,20)):
                app.CANCEL.clear();app.focus_merge({'capture':'test','output':str(root/'stack.tif'),'settings':{}})
            self.assertEqual(app.STATE['status'],'done',app.STATE['message'])
            info=inspector.describe();self.assertEqual(info['kind'],'focus');self.assertFalse(info['raw_controls']);self.assertTrue(info['focus_maps']);self.assertEqual(info['source_available'],[True,True])
            v=inspector.view({})['view']
            for kind in ['original','composite','difference','alignment','focusmap','focusconfidence']:
                png,_=inspector.tile({'view':v,'kind':kind,'reference':1,'ow':128,'oh':96});self.assertTrue(png.startswith(b'\x89PNG'))
            with self.assertRaises(ValueError):inspector.view({'white_balance':'custom'})
    def test_multiscale_band_boundaries_match_whole_image(self):
        a=self.scene();b=cv2.GaussianBlur(a,(0,0),2);a=a.copy();a[:,192:]=b[:,192:]
        valid=np.ones(a.shape[:2],bool);expected=focus.fuse([a,b],[valid,valid],photo_settings.DEFAULTS)[0]
        actual=np.empty_like(expected)
        for y in range(0,a.shape[0],96):
            lo=max(0,y-96);hi=min(a.shape[0],y+192)
            band=focus.fuse([a[lo:hi],b[lo:hi]],[valid[lo:hi],valid[lo:hi]],photo_settings.DEFAULTS)[0]
            stop=min(y+96,a.shape[0]);actual[y:stop]=band[y-lo:stop-lo]
        np.testing.assert_allclose(actual,expected,atol=2e-6)

    def test_focus_setting_validation(self):
        for setting in [{'focus_radius':float('nan')},{'focus_selectivity':100},{'focus_alignment':'fake'}]:
            with self.assertRaises(ValueError):photo_settings.validate(setting)

if __name__=='__main__':unittest.main()
