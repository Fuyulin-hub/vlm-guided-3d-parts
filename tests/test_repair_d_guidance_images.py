import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

class RepairGuideTests(unittest.TestCase):
    def test_white_haze_and_jpeg_white_are_background_but_dark_regions_survive(self):
        from repair_d_guidance_images import foreground_mask
        rgba = np.array([[[255,255,255,255],[250,250,250,180],[0,0,0,255],
                          [229,81,22,255],[229,81,22,80]]], dtype=np.uint8)
        self.assertEqual(foreground_mask(rgba).tolist(), [[False,False,True,True,False]])

    def test_gradients_of_two_hues_yield_two_regions(self):
        from repair_d_guidance_images import infer_regions
        rgba = np.zeros((60,120,4),dtype=np.uint8)
        for x in range(60):
            rgba[:,x,:3] = np.array([240,80,20])*(0.55+0.45*x/59)
            rgba[:,x+60,:3] = np.array([20,80,240])*(0.55+0.45*x/59)
        rgba[...,3]=255
        labels, colors, info = infer_regions(rgba,{})
        self.assertEqual(len(colors),2)
        self.assertEqual(len(np.unique(labels[:,:60])),1)
        self.assertEqual(len(np.unique(labels[:,60:])),1)
        self.assertNotEqual(labels[0,0], labels[0,119])

    def test_near_exact_palette_retains_original_ids(self):
        from repair_d_guidance_images import infer_regions
        rgba=np.zeros((30,60,4),dtype=np.uint8)
        rgba[:,:30]=[229,81,22,255]
        rgba[:,30:]=[22,81,229,255]
        rgba[:,29,:3]=[228,81,22]
        labels, colors, info=infer_regions(rgba,{7:(229,81,22),12:(22,81,229)})
        self.assertEqual(set(np.unique(labels)),{7,12})
        self.assertEqual(info['method'],'active_source_palette')

    def test_uniform_alignment_preserves_left_right_order(self):
        from repair_d_guidance_images import align_to_raw
        labels=np.zeros((20,30),dtype=np.uint16)
        labels[5:15,5:15]=1
        labels[5:15,15:25]=2
        target=np.zeros((80,120),dtype=bool)
        target[20:60,20:100]=True
        aligned,meta=align_to_raw(labels,target)
        self.assertEqual(aligned.shape,target.shape)
        self.assertEqual(aligned[40,40],1)
        self.assertEqual(aligned[40,80],2)
        self.assertEqual(meta['rotation_degrees'],0)

    def test_small_solid_boundary_blend_does_not_become_extra_region(self):
        from repair_d_guidance_images import infer_regions
        rgba=np.zeros((40,81,4),dtype=np.uint8)
        rgba[:,:40]=[240,80,20,255]
        rgba[:,40]=[20,240,20,255]
        rgba[:,41:]=[20,80,240,255]
        labels,colors,info=infer_regions(rgba,{},stable_color_seeds=True)
        self.assertEqual(len(colors),2)

if __name__=='__main__':
    unittest.main()
