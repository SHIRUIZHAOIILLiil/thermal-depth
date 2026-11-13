import argparse
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import caption_eval_sets as tool
from tools import caption_synth as cap

SCRIPTS = ROOT/'internVL'/'scripts'/'lotus_inference'


def iris_paths(stem):
    src = (SCRIPTS/f'inference_{stem}.py').read_text(encoding='utf-8')
    tar = re.search(r"tar_path = '([^']+)'", src).group(1)
    lst = re.search(r"filename_list_path = '([^']+)'", src).group(1)
    prompt = re.search(r'prompt = "(.*?)"', src).group(1)
    return src, tar, lst, prompt


class MatchesIris(unittest.TestCase):
    def test_tars_lists_prompt_and_generation_match_each_iris_script(self):
        for name, (tar_rel, lst_rel, stem) in tool.SETS.items():
            src, tar, lst, prompt = iris_paths(stem)
            self.assertEqual(tar, 'lotus/datasets/eval/depth/' + tar_rel, name)
            self.assertEqual(lst, 'lotus/datasets/eval/depth/data_split/' + lst_rel, name)
            self.assertEqual(prompt, cap.PROMPT, name)
            self.assertIn('generation_config = dict(max_new_tokens=77, do_sample=True)', src)
            self.assertIn('rgb_path = parts[0]', src)
            self.assertIn('extractfile("./" + image_path)', src)

    def test_config_names_and_output_files_match_the_eval_side(self):
        import yaml
        cfg_dir = ROOT/'lotus'/'datasets'/'eval'/'depth'/'configs'
        names = {yaml.safe_load(open(p))['name']: p.name for p in cfg_dir.glob('*.yaml')}
        self.assertEqual(set(names), set(tool.SETS))
        for name, (tar_rel, lst_rel, _) in tool.SETS.items():
            cfg = yaml.safe_load(open(cfg_dir/names[name]))
            self.assertEqual(cfg['dir'], tar_rel)
            self.assertEqual(cfg['filenames'], 'datasets/eval/depth/data_split/' + lst_rel)
        sh = (ROOT/'lotus'/'eval_scripts'/'eval-depth-g.sh').read_text()
        for _, _, stem in tool.SETS.values():
            self.assertIn(f'/{stem}_depth_descriptions.json', sh)
        rng = (ROOT/'lotus'/'eval_rng.py').read_text()
        for name in tool.SETS:
            self.assertIn(f'"{name}":', rng)

    def test_every_split_list_is_present_and_non_empty(self):
        rows = tool.all_rows()
        for name in tool.SETS:
            self.assertGreater(sum(n == name for n, _ in rows), 0, name)


class TarAndFinalize(unittest.TestCase):
    def make(self, d):
        data, splits = Path(d)/'data', Path(d)/'splits'
        for name, (tar_rel, lst_rel, _) in tool.SETS.items():
            (splits/lst_rel).parent.mkdir(parents=True, exist_ok=True)
            (splits/lst_rel).write_text(f'{name}/b.png {name}/b_d.png\n{name}/a.png {name}/a_d.png\n')
            (data/'depth'/tar_rel).parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(data/'depth'/tar_rel, 'w') as t:
                for k in ('a', 'b'):
                    blob = f'{name}{k}'.encode()
                    info = tarfile.TarInfo('./' + f'{name}/{k}.png')
                    info.size = len(blob)
                    t.addfile(info, io.BytesIO(blob))
        return data, splits

    def test_check_reads_members_and_refuses_missing(self):
        with tempfile.TemporaryDirectory() as d:
            data, splits = self.make(d)
            tool.run_check(argparse.Namespace(data=data, splits=splits))
            with tarfile.open(data/'depth'/tool.SETS['kitti'][0]) as t:
                self.assertEqual(tool.image_bytes(t, 'kitti/a.png'), b'kittia')
            lst = splits/tool.SETS['eth3d'][1]
            lst.write_text(lst.read_text() + 'eth3d/c.png x\n')
            with self.assertRaises(ValueError):
                tool.run_check(argparse.Namespace(data=data, splits=splits))

    def test_finalize_keys_are_first_columns_and_gaps_fail(self):
        with tempfile.TemporaryDirectory() as d:
            data, splits = self.make(d)
            out = Path(d)/'out'
            a = argparse.Namespace(data=data, splits=splits, out=out, count=2)
            rows = tool.all_rows(splits)
            self.assertEqual([k for n, k in rows if n == 'nyu_v2'], ['nyu_v2/a.png', 'nyu_v2/b.png'])
            for i in range(2):
                p = cap.shard_path(out, i, 2)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, 'w') as f:
                    for n, k in rows[i::2]:
                        if (n, k) != ('diode', 'diode/b.png'):
                            f.write(json.dumps(dict(kind=n, key=k, description='x', status='ok')) + '\n')
            with self.assertRaises(ValueError):
                tool.run_finalize(a)
            with open(cap.shard_path(out, 1, 2), 'a') as f:
                f.write(json.dumps(dict(kind='diode', key='diode/b.png', description='y', status='ok')) + '\n')
            tool.run_finalize(a)
            nyu = json.load(open(out/'nyu_depth_descriptions.json'))
            self.assertEqual(sorted(nyu), ['nyu_v2/a.png', 'nyu_v2/b.png'])
            self.assertEqual(nyu['nyu_v2/a.png'], {'description': 'x'})


if __name__ == '__main__':
    unittest.main()
