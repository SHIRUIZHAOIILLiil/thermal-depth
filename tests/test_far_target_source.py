import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_far_target_source as tool
from tools.diagnose_depth_chain import read_jsonl, select_rows, TRAIN_SEQUENCES


def row(source, band, bias, rmse, split='train', q=1., stage='vae', mask='heldout', mode=tool.HELDOUT_ALIGN):
    return dict(split=split, source=source, q=q, stage=stage, mode=mode, mask=mask, band=band,
                frames=9, points=99, bias_m=bias, rmse_m=rmse)


def table(far, near):
    rows = []
    for source in tool.SOURCES:
        rows.append(row(source, '60-80', far[source], 9.))
        rows.append(row(source, '0-40', -.1, near[source]))
    return rows


FAR = dict(anythermal=-10., bts=-9., adabins=-9., newcrf=-9.)
NEAR = dict.fromkeys(tool.SOURCES, 2.)


class FarTargetSource(unittest.TestCase):
    def test_holdout_is_deterministic_whole_cells_and_near_requested_share(self):
        valid = np.zeros((256, 640), bool)
        valid[::3, ::3] = True
        held = tool.holdout_mask(valid, '2021-08-06-10-59-33_000001')
        np.testing.assert_array_equal(held, tool.holdout_mask(valid, '2021-08-06-10-59-33_000001'))
        self.assertFalse(np.array_equal(held, tool.holdout_mask(valid, '2021-08-06-10-59-33_000002')))
        self.assertFalse((held & ~valid).any())
        self.assertAlmostEqual(held.sum()/valid.sum(), .2, delta=.03)
        cells = np.add.reduceat(np.add.reduceat(held.astype(int), np.arange(0, 256, 8), 0), np.arange(0, 640, 8), 1)
        full = np.add.reduceat(np.add.reduceat(valid.astype(int), np.arange(0, 256, 8), 0), np.arange(0, 640, 8), 1)
        self.assertTrue(np.all((cells == 0) | (cells == full)))

    def test_frame_records_fit_only_and_keep_masks_separate(self):
        gt = np.array([[5, 20, 50, 70, 75, 0]], np.float32)
        valid = gt > 0
        held = np.array([[0, 0, 0, 0, 1, 0]], bool)
        rows = tool.frame_records(gt, valid, held, gt.copy(), np.log(np.maximum(gt, 1e-6)), None)
        by = {(r['mode'], r['mask'], r['band']): r for r in rows}
        self.assertEqual(by[tool.HELDOUT_ALIGN, 'heldout', '60-80']['points'], 1)
        self.assertEqual(by[tool.HELDOUT_ALIGN, 'fit', '60-80']['points'], 1)
        self.assertAlmostEqual(by[tool.HELDOUT_ALIGN, 'heldout', '60-80']['bias_m'], 0, places=2)
        self.assertNotIn((tool.HELDOUT_ALIGN, 'heldout', '0-40'), by)

    def test_heldout_gt_cannot_change_target_or_alignment(self):
        gt = np.array([[5, 20, 50, 65, 75, 0]], np.float32)
        valid = gt > 0
        held = np.array([[0, 0, 0, 0, 1, 0]], bool)
        fit = valid & ~held
        raw = np.array([[2, 8, 17, 20, 25, 15]], np.float32)
        changed = gt.copy()
        changed[held] = 61
        config = dict(min_fit_pixels=3)
        before = tool.make_target(raw,gt,fit,config)
        after = tool.make_target(raw,changed,fit,config)
        for a,b in zip(before,after):
            np.testing.assert_array_equal(a,b)
        log_value = np.log(before[0])
        original = tool.chain.aligned_depth
        with patch.object(tool.chain,'aligned_depth',wraps=original) as alignment:
            rows1 = tool.frame_records(gt,valid,held,before[0],log_value,None)
            rows2 = tool.frame_records(changed,valid,held,after[0],log_value,None)
            for call in alignment.call_args_list:
                np.testing.assert_array_equal(call.args[2],fit)
        fit1 = [r for r in rows1 if r['mask']=='fit']
        fit2 = [r for r in rows2 if r['mask']=='fit']
        self.assertEqual(fit1,fit2)
        held1 = next(r for r in rows1 if r['mask']=='heldout' and r['mode']==tool.HELDOUT_ALIGN)
        held2 = next(r for r in rows2 if r['mask']=='heldout' and r['mode']==tool.HELDOUT_ALIGN)
        # Only the scoring label changed: the fitted prediction stays identical.
        self.assertAlmostEqual(held2['bias_m']-held1['bias_m'],14.,places=8)

    def test_clean_source_is_chosen_before_newcrf(self):
        far = dict(FAR, bts=-6.9, adabins=-6.8, newcrf=-5.)
        decision = tool.decide(table(far, NEAR), True)
        self.assertEqual((decision['status'], decision['chosen']), ('clean_source_passed', 'adabins'))

    def test_only_newcrf_needs_the_user(self):
        decision = tool.decide(table(dict(FAR, newcrf=-6.), NEAR), True)
        self.assertEqual(decision['status'], 'only_newcrf_passed_needs_user_decision')
        self.assertIsNone(decision['chosen'])

    def test_boundaries_and_near_guard(self):
        self.assertTrue(tool.decide(table(dict(FAR, bts=-7.), NEAR), True)['sources']['bts']['passed'])
        self.assertTrue(tool.decide(table(dict(FAR,bts=-7.),dict(NEAR,bts=2.02)),True)['sources']['bts']['passed'])
        self.assertFalse(tool.decide(table(dict(FAR, bts=-7.01), NEAR), True)['sources']['bts']['passed'])
        decision = tool.decide(table(dict(FAR, bts=-5.), dict(NEAR, bts=2.03)), True)
        self.assertEqual((decision['status'], decision['chosen']), ('no_source_passed', None))

    def test_failed_control_blocks_every_source(self):
        decision = tool.decide(table(dict(FAR, bts=-5.), NEAR), False)
        self.assertEqual((decision['status'], decision['chosen']), ('control_failed', None))

    def test_control_reads_only_separate_legacy_all_pixels(self):
        common = dict(split='val', mask='control_all', mode='ssi_log')
        rows = [row('anythermal', '60-80', -8.3, 9., q=None, stage='source', **common),
                row('anythermal', '60-80', -12.5, 9., q=.98, **common)]
        self.assertTrue(tool.control(rows)['passed'])
        # A good fit-only row cannot substitute for a failed historical control.
        rows.append(row('anythermal','60-80',-12.86,9.,split='val',q=.98,mask='fit'))
        rows[1]['bias_m'] = -14.
        self.assertFalse(tool.control(rows)['passed'])

    def test_legacy_control_matches_original_full_lidar_target_chain(self):
        class IdentityVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()),requires_grad=False)
                self.config = SimpleNamespace(scaling_factor=.18215)
                self.inputs = []
            def encode(self,x):
                self.inputs.append(x.clone())
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:x))
            def decode(self,z):
                return SimpleNamespace(sample=z)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root/'raw'
            raw_dir.mkdir()
            thermal = np.arange(64*160,dtype=np.uint16).reshape(64,160)+20000
            Image.fromarray(thermal).save(root/'thermal.png')
            depth = np.tile(np.linspace(3,79,160,dtype=np.float32),(64,1))
            gt = np.where(np.indices(depth.shape)[1]%3==0,depth,0)
            Image.fromarray((gt*256).astype(np.uint16)).save(root/'gt.png')
            manifest = root/'m.jsonl'
            manifest.write_text(json.dumps(dict(id='a',thermal_path='thermal.png',thermal_depth_path='gt.png'))+'\n')
            np.save(raw_dir/'a.npy',depth**.8)
            frames = [SimpleNamespace(sample_id='a',thermal_gt_path=str(root/'gt.png'))]
            config = dict(min_fit_pixels=16,dtype='float16')
            vae = IdentityVAE()
            with torch.inference_mode():
                rows = tool.legacy_control_records(frames,manifest,raw_dir,config,
                    SimpleNamespace(out=root,data_root=root),vae)
            measured_gt = tool.load_ms2_gt(root/'gt.png',256.)[1]
            valid = tool.official_valid_mask(measured_gt,.001,80.)
            expected,_ = tool.calibrate_pseudo_depth(depth**.8,measured_gt,valid,min_fit_pixels=16)
            stored = np.load(root/'legacy_control_pseudo/a.npy')
            np.testing.assert_array_equal(stored,expected.astype(np.float16))
            dataset = tool.MS2ThermalDataset(manifest,root,root/'legacy_control_pseudo',
                norm_type='log_truncnorm',truncnorm_min=.02,truncnorm_max=.98)
            torch.testing.assert_close(vae.inputs[0][0],dataset[0]['depth_values'],rtol=0,atol=0)
            self.assertEqual(vae.inputs[0].shape[0],2)  # same completed/pure batch as the legacy chain
            unit = ((vae.inputs[0]*.18215/.18215).mean(1)[0]/2+.5).numpy()
            maps = {stage:unit for stage in tool.val_chain.STAGES}
            maps['pseudo'] = stored.astype(np.float32)
            bounds = {stage:dataset[0]['norm_bounds'].numpy() for stage in maps}
            legacy = tool.val_chain.score_frame(measured_gt,maps,bounds)
            for stage,new_stage in (('pseudo','source'),('vae_mode','vae')):
                want = next(r for r in legacy if r['stage']==stage and r['mode']=='ssi_log')
                actual = next(r for r in rows if r['stage']==new_stage and r['band']=='60-80')
                self.assertEqual(actual['mask'],'control_all')
                for key in ('points','bias_m','rmse_m'):
                    self.assertAlmostEqual(actual[key],want[key],places=8)

    def test_aggregate_is_frame_weighted(self):
        base = dict(split='train', source='bts', q=1., stage='vae', mode='ssi_log', mask='heldout', band='60-80')
        out = tool.aggregate([dict(base, id='a', points=1, bias_m=-4., rmse_m=4.),
                              dict(base, id='b', points=100, bias_m=-2., rmse_m=2.)])
        self.assertEqual((out[0]['frames'], out[0]['points'], out[0]['bias_m']), (2, 101, -3.))

    def test_target_dataset_matches_training_normalisation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'m.jsonl'
            manifest.write_text(json.dumps(dict(id='a', thermal_path='t.png', thermal_depth_path='thr/d.png'))+'\n')
            dataset = tool.TargetDataset(manifest, root, root, norm_type='log_truncnorm',
                                         truncnorm_min=.02, truncnorm_max=1.)
            with self.assertRaises(ValueError):
                dataset._completed_depth(dataset.rows[0])
            dense = np.linspace(1, 70, 64, dtype=np.float32).reshape(8, 8)
            dataset.dense = dense
            depth, sparse, mask = dataset._completed_depth(dataset.rows[0])
            self.assertEqual(depth.shape, (1, 8, 8))
            torch.testing.assert_close(depth[0], torch.from_numpy(dense))
            self.assertEqual(float(sparse.sum()+mask.sum()), 0.)

    def test_select_rows_refuses_test_sequences(self):
        row_id = '2021-08-13-21-18-04_000001'
        rows = [dict(id=row_id, thermal_path=f'{row_id[:19]}/thr/img_left/000001.png',
                     thermal_depth_path=f'{row_id[:19]}/thr/depth_filtered/000001.png', caption='x')]
        with self.assertRaises(ValueError):
            select_rows(rows, 1, 1, TRAIN_SEQUENCES)


if __name__ == '__main__':
    unittest.main()
