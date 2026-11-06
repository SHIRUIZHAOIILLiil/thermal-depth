"""Necessary CPU checks for the new q100 decoder experiment; no model downloads."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from tools import train_iris_tmax1_decoder as runner


class TMaxDecoderTests(unittest.TestCase):
    def test_dataset_keeps_far_returns_and_same_near_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'pseudo').mkdir()
            thermal = np.arange(64*160, dtype=np.uint16).reshape(64,160)+20000
            Image.fromarray(thermal).save(root/'thermal.png')
            dense = np.full((64,160), 20., dtype=np.float32)
            dense[:, :10] = np.linspace(1.,10.,10)
            dense[:, -2:] = [65.,79.]  # 1.25%: lost at q98, distinct at q100
            gt = dense.copy()
            gt[:, 10:120] = 0
            Image.fromarray((gt*256).astype(np.uint16)).save(root/'gt.png')
            np.save(root/'pseudo/f0.npy',dense)
            manifest = root/'train.jsonl'
            manifest.write_text(json.dumps(dict(id='f0',thermal_path='thermal.png',
                thermal_depth_path='gt.png',caption='a road'))+'\n')
            contract = dict(train={'path':str(manifest)},pseudo=str(root/'pseudo'))
            ds = runner.make_dataset(contract,root)
            example = ds[0]
            old_ds = type(ds)(manifest,root,root/'pseudo',norm_type='log_truncnorm',
                              truncnorm_min=.02,truncnorm_max=.98,use_captions=True)
            before = old_ds[0]
            self.assertEqual((ds.truncnorm_min,ds.truncnorm_max),(.02,1.))
            self.assertEqual(float(example['norm_bounds'][0]),float(before['norm_bounds'][0]))
            self.assertAlmostEqual(float(example['norm_bounds'][1]),math.log(79.),places=5)
            self.assertTrue(torch.all(before['depth_values'][0,:,-2:]==1))
            self.assertTrue(torch.all(example['depth_values'][0,:,-2]<example['depth_values'][0,:,-1]))
            torch.testing.assert_close(example['gt_depth_values'],torch.from_numpy(gt)[None])
            torch.testing.assert_close(example['gt_valid_values'],before['gt_valid_values'])
            self.assertFalse(bool(example['gt_valid_values'][0,0,20]))

    def test_source_and_original_start_are_pinned(self):
        source = Path('runs')/runner.tmax.SOURCE_RUN/'converted/step16000_weights.pt'
        self.assertEqual(runner.tmax.check_source(source),source)
        with self.assertRaises(ValueError):
            runner.tmax.check_source(source.with_name('conv_4000.pt'))
        runner.tmax.require_start(runner.tmax.EXPECTED)
        with self.assertRaises(ValueError):
            runner.tmax.require_start(dict(abs_rel=.07753,rmse=2.67544,a1=.94693))

    def test_comparator_checks_budget_data_and_deployed_weights(self):
        current = dict(train={'path':'train'},val={'path':'val','sample_count':14249},
                       recipe={'route':'b_thermal_unet'},pseudo='calibrated',pseudo_config='p')
        reference = {'metrics':dict(abs_rel=.08,rmse=2.47,a1=.95)}
        contract = dict(train=current['train'],val=current['val'],recipe=current['recipe'],
            pseudo=current['pseudo'],pseudo_config_sha256='p',newcrf=reference,
            lr=runner.old.LR,batch=runner.old.BATCH,accumulation=runner.old.ACCUM,budget=1000,
            steps=list(runner.old.STEPS),seed=43,image_cursor=48000,weight_decay=.01,
            augmentation='none: deployment input',pixel_loss_weight=.05,
            pair_weight=runner.old.bias.PAIR_WEIGHT,bias_weight=runner.old.bias.BIAS_WEIGHT)
        runner.check_comparator_contract(contract,current,reference)
        for key,value in [('budget',2000),('val',{'path':'different'}),('pseudo_config_sha256','wrong')]:
            changed = copy.deepcopy(contract)
            changed[key] = value
            with self.assertRaises(ValueError):
                runner.check_comparator_contract(changed,current,reference)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'train/test_selected_v1'
            folder.mkdir(parents=True)
            weights = root/'train/decoder_1000.pt'
            weights.write_bytes(b'original historical weights')
            metrics = dict(abs_rel=.07753,rmse=2.67544,a1=.94693)
            summary = dict(curve={str(s):dict(metrics=metrics,samples=14249) for s in (0,500,1000)})
            runner.pair.json_write(root/'contract.json',contract)
            runner.pair.json_write(root/'train/summary.json',summary)
            reproduction = dict(step=1000,decoder_sha256=runner.pair.sha256_file(weights),measured=metrics)
            runner.pair.json_write(folder/'reproduced.json',reproduction)
            with patch.object(runner.prior_eval,'selection',return_value=(contract,metrics,1000,weights)):
                result = runner.read_comparator(root,current,reference)
                self.assertEqual(result['selected'],1000)
                # No test scene artifacts exist: historical validation alone is sufficient.
                reproduction['decoder_sha256'] = 'wrong'
                runner.pair.json_write(folder/'reproduced.json',reproduction)
                with self.assertRaises(ValueError):
                    runner.read_comparator(root,current,reference)

    def test_performance_comparison_does_not_hide_tradeoffs(self):
        ref = dict(abs_rel=.08,rmse=2.47,a1=.95)
        old = dict(abs_rel=.07753,rmse=2.67544,a1=.94693)
        tradeoff = dict(abs_rel=.079,rmse=2.60,a1=.946)
        self.assertEqual(runner.compare_metrics(tradeoff,old,ref)['status'],'tradeoff_primary3')
        better = dict(abs_rel=.077,rmse=2.60,a1=.948)
        self.assertEqual(runner.compare_metrics(better,old,ref)['status'],'candidate_dominates_primary3')
        self.assertEqual(runner.compare_metrics(old,old,ref)['status'],'equal_primary3')

    def test_submission_only_original_tmax_smoke_then_decoder(self):
        bash = 'D:/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).exists():
            self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory(prefix='tmax_decoder_submit_',dir=ROOT) as tmp:
            root = Path(tmp)
            base = root/'runs/outputs/lotus_line_v2'
            history = base/'iris_vae_decoder_only_v1/run_historical'
            files = {
                history/'jobs.txt':'val reproduction -> 9213840\n',
                history/'contract.json':'{}', history/'train/summary.json':'{}',
                history/'train/test_selected_v1/reproduced.json':'{}',
                base/'iris_far_pairs_l050_frame4k_v1/job_9163317/summary.json':'{}',
                base/'iris_pair_rmse_val_v1/job_9167854/newcrf_moments.npz':'placeholder',
                root/'runs/iris_ms2'/runner.tmax.SOURCE_RUN/'converted/step16000_weights.pt':'weights',
            }
            for file,text in files.items():
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_text(text)
            binary = root/'bin'
            binary.mkdir()
            stub = binary/'sbatch'
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\necho 12345\n',newline='\n')
            stub.chmod(0o755)
            calls = root/'calls'
            env = dict(os.environ,IRIS_REPO=ROOT.as_posix(),IRIS_RUNS=(root/'runs').as_posix(),
                       SCRATCH=(root/'scratch').as_posix(),CALLS=calls.as_posix(),MOCK_BIN=binary.as_posix())
            result = subprocess.run([bash,'-c','export PATH="$(cd "$MOCK_BIN" && pwd):$PATH"; bash slurm/submit_iris_tmax1_decoder.sh'],
                cwd=ROOT,env=env,text=True,encoding='utf8',errors='replace',capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            lines = calls.read_text().splitlines()
            self.assertEqual(len(lines),2)
            self.assertIn(' smoke ',lines[0])
            self.assertIn(' train ',lines[1])
            self.assertIn('--dependency=afterok:12345',lines[1])
            for line in lines:
                self.assertIn(runner.tmax.SOURCE_RUN+'/converted/step16000_weights.pt',line)
                self.assertIn('run_historical',line)
                self.assertNotIn('test',line)


if __name__=='__main__':
    unittest.main()
