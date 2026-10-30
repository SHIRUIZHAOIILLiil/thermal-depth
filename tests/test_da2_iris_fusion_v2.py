"""Matched checkpoint guards, permutation integrity and dependency-chain tests."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.train_da2_iris_fusion_v2 import (validate_iris_payload, matched_sources, permutation_tokens,
    pair_smoke, content_validation, EXPERIMENT, IRIS_RUNS, SMOKE_STEPS, OFFICIAL_METRICS)


def samples_and_tokens(n=9):
    samples=[SimpleNamespace(sample_id=f'frame-{i}',caption=f'caption {i}') for i in range(n)]
    tokens=dict(input_ids=torch.arange(n)[:,None].expand(n,77).clone(),
                attention_mask=torch.ones(n,77,dtype=torch.bool))
    return samples,tokens


class MatchedTests(unittest.TestCase):
    def test_weight_prompt_guards_and_old_experiment_rejection(self):
        for arm,mode in (('frame','correct'),('empty','empty')):
            payload=dict(route='b_thermal_unet',global_step=18000,caption_mode=mode,state_dicts={'unet':{}})
            validate_iris_payload(payload,arm)
            with self.assertRaises(ValueError): validate_iris_payload(payload,'empty' if arm=='frame' else 'frame')
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for arm in ('frame','empty'):
                p=Path(tmp)/IRIS_RUNS[arm]/'converted/step18000_weights.pt'
                p.parent.mkdir(parents=True); p.write_bytes(arm.encode()); paths.append(p)
            sources=matched_sources(*paths)
            self.assertNotEqual(sources['frame']['sha256'],sources['empty']['sha256'])
            with self.assertRaises(ValueError): matched_sources(paths[1],paths[0])
            paths[1].write_bytes(paths[0].read_bytes())
            with self.assertRaises(ValueError): matched_sources(*paths)

    def test_derangement_preserves_marginal_duplicates_and_rng(self):
        samples,tokens=samples_and_tokens()
        # Repeated captions / truncation collisions are reported, not hidden.
        for s in samples: s.caption='same description'
        tokens['input_ids'].zero_()
        torch.manual_seed(42); before=torch.get_rng_state().clone()
        a,info,pairs=permutation_tokens(samples,tokens)
        b,other,again=permutation_tokens(samples,tokens)
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        self.assertEqual(info,other); self.assertEqual(pairs,again)
        self.assertTrue(all(x!=y for x,y in pairs))
        self.assertEqual(sorted(y for _,y in pairs),sorted(s.sample_id for s in samples))
        self.assertEqual(info['identical_token_count'],len(samples))
        self.assertEqual(info['identical_text_count'],len(samples))
        samples,tokens=samples_and_tokens()
        original=tokens['input_ids'].clone()
        perm,info,pairs=permutation_tokens(samples,tokens)
        self.assertEqual(info['identical_token_count'],0)
        torch.testing.assert_close(tokens['input_ids'],original,rtol=0,atol=0)
        for i,(_,donor) in enumerate(pairs):
            j=int(donor.split('-')[-1])
            torch.testing.assert_close(perm['input_ids'][i],tokens['input_ids'][j])
            torch.testing.assert_close(perm['attention_mask'][i],tokens['attention_mask'][j])
        with self.assertRaises(ValueError): permutation_tokens(*samples_and_tokens(1))

    def test_pair_allows_only_intended_iris_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            sources={a:dict(run=IRIS_RUNS[a],sha256=a,trained_prompt='correct' if a=='frame' else 'empty') for a in IRIS_RUNS}
            for arm in IRIS_RUNS:
                d=root/arm; d.mkdir()
                (d/'contract.json').write_text(json.dumps(dict(arm=arm,experiment=EXPERIMENT,iris_sources=sources,
                    initial_iris_sha256=arm,initial_da2_sha256='same',initial_fusion_sha256='same')))
                (d/'completed.json').write_text(json.dumps(dict(mode='smoke',step=SMOKE_STEPS,smoke_checks_passed=True,
                    initial_prediction_sha256='same',initial_baseline_sha256='same')))
                (d/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s),
                    input_sha256='same',noise_sha256='same')) for s in range(1,SMOKE_STEPS+1)))
            pair_smoke(root)
            path=root/'empty/contract.json'; c=json.loads(path.read_text()); c['initial_da2_sha256']='wrong'
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError): pair_smoke(root)
            c['initial_da2_sha256']='same'; c['experiment']='da2_iris_fusion_pair_v1'; path.write_text(json.dumps(c))
            with self.assertRaises(ValueError): pair_smoke(root)

    def test_content_validation_reuses_model_images_and_selected_step(self):
        samples,tokens=samples_and_tokens(); perm,info,pairs=permutation_tokens(samples,tokens)
        validation=SimpleNamespace(samples=samples)
        # Provide len without touching image files in this orchestration test.
        class Data:
            def __init__(self): self.samples=samples
            def __len__(self): return len(samples)
        validation=Data()
        m=torch.nn.Module(); m.fusions=torch.nn.Linear(1,1)
        correct=dict(step=14000,samples=len(samples),metrics={mode:{k:1. for k in OFFICIAL_METRICS} for mode in ('ssi','none')})
        def fake_evaluate(got, data, out, step):
            self.assertIs(got,m); self.assertIs(data.samples,samples)
            self.assertIs(data.tokens,perm); self.assertEqual(step,14000)
            self.assertEqual(out.name,'permuted_selected')
            return dict(step=step,samples=len(samples),metrics={mode:{k:2. for k in OFFICIAL_METRICS} for mode in ('ssi','none')})
        with tempfile.TemporaryDirectory() as tmp, patch('tools.train_da2_iris_fusion_v2.evaluate',fake_evaluate):
            report=content_validation(m,validation,perm,Path(tmp),14000,correct,info)
            self.assertEqual(report['delta_permuted_minus_correct']['ssi']['abs_rel'],1.)
            self.assertTrue((Path(tmp)/'caption_content.json').is_file())
            with self.assertRaises(ValueError): content_validation(m,validation,perm,Path(tmp),20000,correct,info)

    @unittest.skipUnless(shutil.which('bash'),'Requires bash')
    def test_new_submission_requires_both_sources_and_chains_two_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'slurm').mkdir(); (root/'bin').mkdir()
            for name in ('submit_da2_iris_fusion_v2.sh','da2_iris_fusion_v2.sbatch'):
                shutil.copyfile(ROOT/'slurm'/name,root/'slurm'/name)
            runs=root/'runs'; paths=[]
            names=['da2_caption_v1/no-caption_s43/step018000.pt','da2_caption_assets/assets.json']
            names += [f'iris_ms2/{IRIS_RUNS[a]}/converted/step18000_weights.pt' for a in ('frame','empty')]
            for name in names:
                p=runs/name; p.parent.mkdir(parents=True,exist_ok=True); p.touch(); paths.append(p)
            calls=root/'calls'; fake=root/'bin/sbatch'
            fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\nprintf "45678\\n"\n'); fake.chmod(0o755)
            env=dict(os.environ,IRIS_REPO=str(root),IRIS_RUNS=str(runs),CALLS=str(calls),PATH=str(root/'bin')+os.pathsep+os.environ['PATH'])
            for key in ('DA2_CAPTION_ASSETS','IRIS_CAP_FEATURE_CKPT','IRIS_EMPTY_FEATURE_CKPT'): env.pop(key,None)
            command=['bash',str(root/'slurm/submit_da2_iris_fusion_v2.sh')]
            subprocess.run(command,env=env,check=True,capture_output=True)
            lines=calls.read_text().splitlines(); self.assertEqual(len(lines),3)
            self.assertIn('da2_if2_smoke',lines[0])
            for arm,line in zip(('frame','empty'),lines[1:]):
                self.assertIn('--dependency=afterok:45678',line); self.assertIn(f'ARM={arm}',line)
            # An explicit unreadable no-caption source must fail before any submission.
            env['IRIS_EMPTY_FEATURE_CKPT']=str(root/'missing')
            result=subprocess.run(command,env=env,capture_output=True)
            self.assertNotEqual(result.returncode,0); self.assertEqual(calls.read_text().splitlines(),lines)


if __name__=='__main__': unittest.main()
