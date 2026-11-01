import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import review_train80_captions as review


class TrainReviewTests(unittest.TestCase):
    def test_manifest_guard_and_repeatable_spaced_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/review.MANIFEST
            with manifest.open('w', encoding='utf-8') as f:
                for j,seq in enumerate(review.SEQUENCES):
                    for i in range(9461):  # 9461*8 = 75,688
                        frame = f'{i:06d}'
                        path = f'sync_data/_{seq}/thr/img_left/{frame}.png'
                        f.write(json.dumps(dict(id=f'{seq}_{frame}', split='train', thermal_path=path,
                                                caption=f'original {j}'))+'\n')
            digest = review.sha(manifest)
            picks = review.sample_train(manifest, root, digest)
            self.assertEqual(picks, review.sample_train(manifest, root, digest))
            self.assertEqual(len(picks), 80)
            for j,seq in enumerate(review.SEQUENCES):
                subset = [(int(s.split('_')[-1]),side) for s,p,c,side in picks if s.startswith(seq)]
                nums = sorted(i for i,_ in subset)
                self.assertEqual(len(nums), 10)
                self.assertTrue(all(b-a >= 2 for a,b in zip(nums,nums[1:])))
                self.assertEqual(sum(side=='A' for _,side in subset), 5)
                self.assertTrue(all(c == f'original {j}' for s,p,c,side in picks if s.startswith(seq)))
            with self.assertRaisesRegex(ValueError,'exact train manifest'):
                review.sample_train(manifest, root, 'other')
            manifest.write_text(json.dumps(dict(id='2021-08-13-16-08-46_000000', split='val', caption='x')))
            with self.assertRaisesRegex(ValueError, 'Unexpected train record'):
                review.sample_train(manifest, root, review.sha(manifest))

    def test_blind_export_and_resume_without_repeated_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root/'captioning/scripts'
            scripts.mkdir(parents=True)
            script = scripts/'generate_captions.py'
            script.write_text('# synthetic generator fixture')
            image_path = root/'thermal.png'
            Image.fromarray(np.array([[300,600],[350,500]], dtype=np.uint16)).save(image_path)
            b = np.array([[0,255],[42,170]], dtype=np.uint8)
            mock_backend = SimpleNamespace(__file__=str(script),
                  load_image=lambda p,thermal_render: Image.fromarray(b).convert('RGB'))
            class Model:
                default_model_id = 'OpenGVLab/InternVL3-8B'
                calls = loads = 0
                def __init__(self, **kwargs): pass
                def load(self): Model.loads += 1
                def generate(self, image, *, prompt, prompt_type):
                    self.assertions(image, prompt)
                    Model.calls += 1
                    return 'Shorter & grounded.'
                @staticmethod
                def assertions(image, prompt):
                    np.testing.assert_array_equal(np.asarray(image)[...,0], b)
                    assert prompt == review.PROMPT
            mock_backend.InternVLCaptioner = Model
            groups = {s:{i:(f'{s}_{i:06d}',image_path,'Original <description>.') for i in range(100)}
                      for s in review.SEQUENCES}
            picks = review.choose(groups)
            args = SimpleNamespace(out=root/'out', data_root=root, manifest=root/review.MANIFEST,
                                   caption_repo=root/'captioning', snapshot=root, train_contract=root/'train.json')
            args.train_contract.write_text(json.dumps({'train':{'sha256':'verified'}}))
            with patch.dict(sys.modules, {'generate_captions':mock_backend}), patch.object(review,'sample_train',return_value=picks):
                review.build(args)
                page = (args.out/'review.html').read_text(encoding='utf-8')
                review.build(args)
                self.assertEqual(page,(args.out/'review.html').read_text(encoding='utf-8'))
            self.assertEqual((Model.loads,Model.calls), (1,80))
            self.assertEqual(page.count('<tr data-id='), 80)
            self.assertEqual(page.count('data:image/png;base64,'), 80)
            self.assertNotIn(picks[0][0],page)
            self.assertNotIn('answer_key',page)
            with (args.out/'private/answer_key.csv').open(encoding='utf-8-sig',newline='') as f:
                key = list(csv.DictReader(f))
            self.assertEqual(len(key),80)
            self.assertEqual({r['train_frame_id'] for r in key},{s for s,_,_,_ in picks})
            for r in key:
                row = page.split(f'<tr data-id="{r["review_id"]}">')[1].split('</tr>')[0]
                captions = [part.split('</p>')[0] for part in row.split('<p class="caption">')[1:]]
                self.assertEqual(captions['AB'.index(r['new_caption_side'])], 'Shorter &amp; grounded.')


if __name__ == '__main__':
    unittest.main()
