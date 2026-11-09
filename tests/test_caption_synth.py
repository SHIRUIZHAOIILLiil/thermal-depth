import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'lotus'))
from tools import caption_synth as tool
from tools import synth_data


def tree(d):
    hyp = Path(d)/'hypersim'
    for scene in ('ai_001_001', 'ai_002_001'):
        p = hyp/'train'/scene/'images'/'scene_cam_00_final_preview'
        p.mkdir(parents=True)
        (p/'frame.0000.tonemap.jpg').write_bytes(b'x')
        (p/'frame.0000.color.jpg').write_bytes(b'x')
    vk = Path(d)/'vkitti'
    for s in synth_data.VKITTI_SCENES:
        for c in synth_data.VKITTI_CONDITIONS:
            for cam in synth_data.VKITTI_CAMERAS:
                q = vk/f'Scene{s}'/c/'frames'/'rgb'/f'Camera_{cam}'
                q.mkdir(parents=True)
                (q/'rgb_00000.jpg').write_bytes(b'x')
    return str(hyp), str(vk)


def write_shards(out, rows, count, drop=None):
    (out/'shards').mkdir(parents=True, exist_ok=True)
    for i in range(count):
        with open(tool.shard_path(out, i, count), 'w') as f:
            for kind, key in rows[i::count]:
                ok = (kind, key) != drop
                f.write(json.dumps(dict(kind=kind, key=key, description='a road' if ok else '',
                                        status='ok' if ok else 'error: x')) + '\n')


class Captions(unittest.TestCase):
    def test_hypersim_keys_match_the_training_loader(self):
        from utils.hypersim_dataset import get_hypersim_dataset_depth_normal
        import inspect
        src = inspect.getsource(get_hypersim_dataset_depth_normal)
        self.assertIn('os.walk(split_dir)', src)
        self.assertIn('file.endswith("tonemap.jpg")', src)
        self.assertIn('text_descriptions.get(image_path', src)
        with tempfile.TemporaryDirectory() as d:
            hyp, _ = tree(d)
            keys = tool.hypersim_images(hyp)
            self.assertEqual(len(keys), 2)
            self.assertTrue(all(k.startswith(hyp + '/train/') and k.endswith('tonemap.jpg') for k in keys))

    def test_vkitti_keys_match_the_training_loader(self):
        from utils.vkitti_dataset import VKITTIDataset
        with tempfile.TemporaryDirectory() as d:
            _, vk = tree(d)
            for s in synth_data.VKITTI_SCENES:
                for c in synth_data.VKITTI_CONDITIONS:
                    for cam in synth_data.VKITTI_CAMERAS:
                        q = Path(vk)/f'Scene{s}'/c/'frames'/'depth'/f'Camera_{cam}'
                        q.mkdir(parents=True)
            ds = VKITTIDataset(vk)
            self.assertEqual(sorted(ds.image_path), [k for kind, k in tool.all_rows(d + '/none', vk) if kind == 'vkitti'])

    def test_finalize_writes_iris_format_and_refuses_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            hyp, vk = tree(d)
            out = Path(d)/'captions'
            rows = tool.all_rows(hyp, vk)
            a = argparse.Namespace(hypersim=hyp, vkitti=vk, out=out, count=3)
            write_shards(out, rows, 3, drop=rows[5])
            with self.assertRaises(ValueError):
                tool.run_finalize(a)
            write_shards(out, rows, 3)
            tool.run_finalize(a)
            table = json.loads((out/'hypersim_depth_descriptions.json').read_text())
            self.assertEqual(set(table), {k for kind, k in rows if kind == 'hypersim'})
            self.assertEqual(next(iter(table.values())), {'description': 'a road'})
            self.assertEqual(len(json.loads((out/'vkitti_depth_descriptions.json').read_text())), 80)

    def test_prompt_is_iris_prompt(self):
        src = (ROOT/'internVL'/'scripts'/'lotus_inference'/'inference_hypersim.py').read_text()
        self.assertIn(f'prompt = "{tool.PROMPT}"', src)
        self.assertIn('generation_config = dict(max_new_tokens=77, do_sample=True)', src)
        self.assertIn("path = 'OpenGVLab/InternVL3-8B'", src)


if __name__ == '__main__':
    unittest.main()
