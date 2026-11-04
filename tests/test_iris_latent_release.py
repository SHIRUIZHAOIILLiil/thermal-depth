import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools.iris_latent_release import release_cells,latent_loss,select,START,STEPS,BANDS
from tools import train_iris_latent_release as train
from tools import test_iris_latent_release as runner


def curve():
    base = dict(metrics=dict(abs_rel=.079,rmse=2.74,a1=.945),
                bands={f'{lo}-{hi}':dict(abs_rel=.07,frames=100) for lo,hi in BANDS})
    return {s:copy.deepcopy(base) for s in (START,*STEPS)}


class LatentReleaseTests(unittest.TestCase):
    def test_only_valid_far_returns_beyond_frame_upper_bound(self):
        gt = torch.zeros(2,1,16,32)
        valid = torch.ones_like(gt,dtype=torch.bool)
        gt[0,0,0,0] = 60
        gt[0,0,0,8] = 59.99
        gt[0,0,0,16] = 80
        gt[0,0,0,24] = 70
        valid[0,0,0,24] = False
        gt[1,0,0,0] = 70
        gt[1,0,8,8] = 79
        gt[1,0,8,24] = float('nan')
        bounds = torch.tensor([[1.,np.log(55.)],[1.,np.log(75.)]])
        mask = release_cells(gt,valid,bounds)
        expected = torch.zeros(2,1,2,4,dtype=torch.bool)
        expected[0,0,0,0]=True
        expected[1,0,1,1]=True
        torch.testing.assert_close(mask,expected)
        # Flips move masks with supervision, independently of caption and prediction.
        torch.testing.assert_close(release_cells(gt.flip(-1),valid.flip(-1),bounds),expected.flip(-1))

    def test_original_denominator_and_gradients(self):
        pred=torch.tensor([[[[1.,2.,3.,4.]]]],requires_grad=True)
        target=torch.zeros_like(pred)
        keep=torch.tensor([[[[True,True,True,False]]]])
        released=torch.tensor([[[[False,True,False,True]]]])
        control=latent_loss(pred,target,keep,released,'control')
        masked=latent_loss(pred,target,keep,released,'masked')
        torch.testing.assert_close(control,torch.nn.functional.mse_loss(pred[keep],target[keep]))
        self.assertAlmostEqual(masked.item(),10/3,places=6)
        masked.backward()
        torch.testing.assert_close(pred.grad,torch.tensor([[[[2/3,0.,2.,0.]]]]))
        all_removed=latent_loss(pred,target,keep,torch.ones_like(released),'masked')
        self.assertEqual(all_removed.item(),0)
        self.assertTrue(all_removed.requires_grad)
        with self.assertRaises(ValueError):
            latent_loss(pred,target,torch.zeros_like(keep),released,'control')

    def test_selection_references_and_band_guard(self):
        rows=curve()
        original=dict(abs_rel=.07936,a1=.94227)
        self.assertEqual(select(rows,original),START)
        rows[4250]['metrics']['rmse']=2.70
        self.assertEqual(select(rows,original),4250)
        rows[4500]['metrics']['rmse']=2.68
        rows[4500]['bands']['20-40']['abs_rel']=.07051
        self.assertEqual(select(rows,original),4250)
        rows[5000]['metrics'].update(rmse=2.67,abs_rel=.0796)
        self.assertEqual(select(rows,original),4250) # current-start AbsRel is stricter
        rows[5000]['metrics'].update(abs_rel=.079,a1=.9439)
        self.assertEqual(select(rows,original),4250) # current-start d1 is stricter
        rows[5000]['metrics']['a1']=.945
        self.assertEqual(select(rows,original),5000)
        rows[5000]['metrics']['rmse']=float('nan')
        with self.assertRaises(ValueError):
            select(rows,original)

    def test_selected_start_hash_must_match_and_complete_curve_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'control').mkdir()
            conv=root/'parent.pt'
            conv.write_bytes(b'weights')
            rows=curve()
            train.pair.json_write(root/'smoke_pass.json',dict(baseline=rows[START]))
            summary=dict(curve=rows,baseline=rows[START],selected=START,
                         selected_conv=str(conv),selected_sha256=train.pair.sha256_file(conv))
            path=root/'control/summary.json'
            train.pair.json_write(path,summary)
            inputs=dict(original=dict(abs_rel=.07936,a1=.94227),conv=conv)
            self.assertEqual(train.selected_weights(root,'control',inputs),(START,conv))
            conv.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                train.selected_weights(root,'control',inputs)
            del rows[4250]
            with self.assertRaises(ValueError):
                select(rows,inputs['original'])

    def test_test_entry_point_reuses_verified_parent_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parent,ancestor,paired=[root/k for k in ('parent','ancestor','paired')]
            folder=paired/'train/frame'
            folder.mkdir(parents=True)
            parent.mkdir()
            ancestor.mkdir()
            (root/'control').mkdir()
            for path in (root/'contract.json',root/'control/summary.json',parent/'summary.json',
                         ancestor/'summary.json',paired/'paired.json',folder/'contract.json',folder/'summary.json'):
                path.write_text('{}')
            conv=parent/'conv_4000.pt'
            conv.write_bytes(b'weights')
            contract=dict(source_sha256='sourcehash')
            inputs=dict(parent=parent,ancestor=ancestor,paired=paired,source=root/'source.pt',
                        conv=conv,contract=contract)
            sample=SimpleNamespace(sample_id='frame1')
            manifest={'sha256':'abc'}
            pilot=runner.parent_test.pilot
            destination=parent/'test_selected_v1/rain'
            destination.mkdir(parents=True)
            frames=destination/'frame_frames.csv'
            values={k:.1 for k in pilot.OFFICIAL_METRICS}
            with frames.open('w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['id',*pilot.OFFICIAL_METRICS])
                writer.writerow(['frame1',*values.values()])
            provenance=dict(manifest=manifest,condition='rain',alignment='ssi_log',valid_gt_m=[.001,80.],
                code_sha256=train.pair.sha256_file(pilot.__file__),paired_sha256=train.pair.sha256_file(paired/'paired.json'),
                frame=dict(source=str(inputs['source']),source_sha256='sourcehash',step=START,
                    conv_sha256=train.pair.sha256_file(conv),contract_sha256=train.pair.sha256_file(folder/'contract.json'),
                    validation_sha256=train.pair.sha256_file(folder/'summary.json')),
                selection_sha256=train.pair.sha256_file(parent/'summary.json'),
                parent_selection_sha256=train.pair.sha256_file(ancestor/'summary.json'),
                runner_sha256=train.pair.sha256_file(runner.parent_test.__file__))
            cache=destination/'summary.json'
            payload=dict(provenance=provenance,frames=1,common_frames=1,missing_official_ids=[],
                         frame_csv=str(frames),results={'frame':{'full':values,'official_common':values}})
            train.pair.json_write(cache,payload)
            args=SimpleNamespace(runs=root,run=root,arm='control',condition='rain',data_root=root,manifests=root)
            with patch.object(train,'parent_inputs',return_value=inputs), \
                 patch.object(train,'contract_for',return_value={}), \
                 patch.object(train,'verify_pair'), \
                 patch.object(train,'selected_weights',return_value=(START,conv)), \
                 patch.dict(pilot.TESTS,{'rain':('rain','abc',1,[])}), \
                 patch.dict(pilot.COMMON,{'rain':(1,1)}), \
                 patch.object(pilot,'load_manifest',return_value=([sample],manifest)), \
                 patch.object(pilot,'validate_samples',return_value={'frame1'}), \
                 patch.object(pilot,'official_frames',return_value=[{'id':'frame1'}]), \
                 patch.object(pilot,'load_selected',side_effect=AssertionError('Must not infer')):
                runner.run(args)
                result=train.pair.read(root/'control/test_selected_v1/rain/summary.json')
                self.assertTrue(result['reused'])
                self.assertEqual(result['results'],payload['results'])
                # Completed-output retry also verifies the referenced CSV.
                runner.run(args)
                frames.write_text('corrupted')
                with self.assertRaises(ValueError):
                    runner.run(args)


if __name__=='__main__':
    unittest.main()
