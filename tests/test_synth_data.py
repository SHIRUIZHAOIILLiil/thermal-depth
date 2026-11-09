import argparse
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import synth_data as tool


def h5_bytes(array):
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        f.create_dataset('dataset', data=array)
    return buf.getvalue()


def scene_zip(scene, frames, nan_frames=()):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fid in frames:
            depth = np.full((4, 4), np.nan if fid in nan_frames else 3., np.float16)
            zf.writestr(f'{scene}/images/scene_cam_00_final_preview/frame.{fid:04d}.tonemap.jpg', b'jpg%d' % fid)
            zf.writestr(f'{scene}/images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5', h5_bytes(depth))
            zf.writestr(f'{scene}/images/scene_cam_00_geometry_hdf5/frame.{fid:04d}.normal_cam.hdf5',
                        h5_bytes(np.zeros((4, 4, 3), np.float16)))
            zf.writestr(f'{scene}/images/scene_cam_00_final_hdf5/frame.{fid:04d}.color.hdf5', b'unused')
    return buf.getvalue()


def rows(scene, frames, split='train', public='True'):
    return [dict(scene_name=scene, camera_name='cam_00', frame_id=str(f),
                 included_in_public_release=public, split_partition_name=split) for f in frames]


def write_csv(path, all_rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)


class RangeReads(unittest.TestCase):
    def test_zip_read_through_ranges_matches_local(self):
        data = scene_zip('ai_001_001', [0, 1])
        calls = []

        def fetch(s, e):
            calls.append((s, e))
            return data[s:e+1]
        zf = zipfile.ZipFile(io.BufferedReader(tool.RangeFile(len(data), fetch), buffer_size=64))
        name = 'ai_001_001/images/scene_cam_00_final_preview/frame.0001.tonemap.jpg'
        self.assertEqual(zf.read(name), b'jpg1')
        self.assertTrue(calls and all(0 <= s <= e < len(data) for s, e in calls))


class Hypersim(unittest.TestCase):
    def test_scene_keeps_clean_frames_and_drops_nan_depth(self):
        data = scene_zip('ai_001_001', [0, 1, 2], nan_frames=(1,))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            open_zip = lambda url: zipfile.ZipFile(io.BytesIO(data))
            rec = tool.fetch_scene('ai_001_001', rows('ai_001_001', [0, 1, 2]), out/'train', open_zip)
            self.assertEqual(len(rec['kept']), 2)
            self.assertEqual(rec['dropped_nan'], ['ai_001_001/images/scene_cam_00_final_preview/frame.0001.tonemap.jpg'])
            self.assertFalse(list((out/'train').rglob('frame.0001.*')))
            self.assertFalse(list((out/'train').rglob('*.part')))
            self.assertFalse(list((out/'train').rglob('*color*')))
            again = tool.fetch_scene('ai_001_001', [], out/'train', None)
            self.assertEqual(again, rec)

    def test_finalize_requires_every_scene_and_writes_metadata_in_csv_order(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            all_rows = rows('ai_001_001', [2, 0]) + rows('ai_001_002', [0]) + rows('ai_001_003', [0], split='val')
            write_csv(out/tool.CSV_NAME, all_rows)
            a = argparse.Namespace(out=out)
            def fetch(scene, frames):
                data = scene_zip(scene, frames)
                tool.fetch_scene(scene, rows(scene, frames), out/'train', lambda url: zipfile.ZipFile(io.BytesIO(data)))
            fetch('ai_001_001', [2, 0])
            with self.assertRaises(ValueError):
                tool.run_finalize(a)
            fetch('ai_001_002', [0])
            tool.run_finalize(a)
            meta = [json.loads(l) for l in open(out/'train'/'metadata.jsonl')]
            self.assertEqual([m['file_name'].split('/')[-1] for m in meta],
                             ['frame.0002.tonemap.jpg', 'frame.0000.tonemap.jpg', 'frame.0000.tonemap.jpg'])
            self.assertEqual(meta[0]['depth'], 'ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0002.depth_meters.hdf5')

    def test_frame_missing_from_zip_fails_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            write_csv(out/tool.CSV_NAME, rows('ai_001_001', [0, 5]))
            data = scene_zip('ai_001_001', [0])
            tool.fetch_scene('ai_001_001', rows('ai_001_001', [0, 5]), out/'train', lambda url: zipfile.ZipFile(io.BytesIO(data)))
            with self.assertRaises(ValueError):
                tool.run_finalize(argparse.Namespace(out=out))


class VKitti(unittest.TestCase):
    def test_pairs_follow_the_loader_rewrite_and_root_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'vkitti'
            for s in tool.VKITTI_SCENES:
                for c in tool.VKITTI_CONDITIONS:
                    (root/f'Scene{s}'/c).mkdir(parents=True)
                    (root/f'Scene{s}'/c/'intrinsic.txt').write_text('frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n')
                    for cam in tool.VKITTI_CAMERAS:
                        for kind, ext in (('rgb', 'jpg'), ('depth', 'png')):
                            p = root/f'Scene{s}'/c/'frames'/kind/f'Camera_{cam}'
                            p.mkdir(parents=True)
                            (p/f'{kind}_00000.{ext}').write_bytes(b'x')
            tool.run_vkitti_check(argparse.Namespace(out=root, normals=False))
            pairs = tool.vkitti_frames(str(root))
            self.assertEqual(len(pairs), 4*10*2)
            self.assertTrue(pairs[0][1].endswith('frames/depth/Camera_0/depth_00000.png'))
            with self.assertRaises(ValueError):
                tool.run_vkitti_check(argparse.Namespace(out=root, normals=True))
            for rgb, _ in pairs:
                n = Path(rgb.replace('rgb', 'normal').replace('.jpg', '.png'))
                n.parent.mkdir(parents=True, exist_ok=True)
                n.write_bytes(b'x')
            tool.run_vkitti_check(argparse.Namespace(out=root, normals=True))
            (root/'Scene02'/'fog'/'frames'/'depth'/'Camera_1'/'depth_00000.png').unlink()
            with self.assertRaises(ValueError):
                tool.run_vkitti_check(argparse.Namespace(out=root, normals=False))
            with self.assertRaises(ValueError):
                tool.run_vkitti_check(argparse.Namespace(out=Path(d)/'vkitti_rgb', normals=False))


if __name__ == '__main__':
    unittest.main()
