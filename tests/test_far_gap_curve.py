from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_far_gap_curve as tool


def rows(step, far, mid=(-.5, -4.)):
    out = []
    for split, (f, m) in dict(train=(far[0], mid[0]), val=(far[1], mid[1])).items():
        for band, bias in (('60-80', f), ('40-60', m)):
            out.append(dict(step=step, split=split, mode=tool.gap_tool.HELD, mask='heldout', band=band,
                            frames=9, points=99, bias_m=bias, rmse_m=abs(bias)+1.))
    return out


HEAD = dict(train=dict(abs_rel=.05, rmse=2., a1=.97), val=dict(abs_rel=.08, rmse=2.7, a1=.94))


class Curve(unittest.TestCase):
    def test_rows_report_the_gap_by_step(self):
        table = rows(2000, (-5., -7.))+rows(16000, (-2., -12.))
        out = tool.curve_rows(table, {2000: HEAD, 16000: HEAD})
        self.assertEqual([r['step'] for r in out], [2000, 16000])
        self.assertAlmostEqual(out[0]['far_gap_m'], -2.)
        self.assertAlmostEqual(out[1]['far_gap_m'], -10.)
        self.assertEqual(out[1]['val']['abs_rel'], .08)

    def test_missing_band_is_an_error(self):
        table = [r for r in rows(2000, (-5., -7.)) if r['band'] != '40-60']
        with self.assertRaises(ValueError):
            tool.curve_rows(table, {2000: HEAD})

    def test_anchor_must_match_the_earlier_job(self):
        reference = dict(decomposition=[dict(band='60-80', model_train_m=-2.13, model_val_m=-12.21)])
        out = tool.curve_rows(rows(16000, (-2.13, -12.21)), {16000: HEAD})
        self.assertTrue(tool.anchor_check(out, reference)['passed'])
        out = tool.curve_rows(rows(16000, (-2.13, -12.5)), {16000: HEAD})
        self.assertFalse(tool.anchor_check(out, reference)['passed'])

    def test_payload_step_is_checked(self):
        payload = dict(route='b_thermal_unet', caption_mode='correct', global_step=4000, state_dicts=dict(unet={}))
        tool.check_payload(payload, 4000)
        with self.assertRaises(ValueError):
            tool.check_payload(payload, 8000)

    def test_steps_include_the_anchor(self):
        self.assertIn(tool.ANCHOR, tool.STEPS)


if __name__ == '__main__':
    unittest.main()
