import unittest

import torch

from models.anythermal_lotus_bridge import AnyThermalLotusBridge
from models.anythermal_lotus_conditioner import AnyThermalDirectConditioner


class AnyThermalDirectConditionerTest(unittest.TestCase):
    def test_matches_direct_bridge_on_final_feature(self):
        torch.manual_seed(7)
        features = [torch.randn(2, 768, 3, 5) for _ in range(4)]
        target_size = (4, 8)
        conditioner = AnyThermalDirectConditioner()
        expected = AnyThermalLotusBridge()(features[-1], target_size, output_channels=4)
        actual = conditioner(features, target_size)
        torch.testing.assert_close(actual, expected)

    def test_has_no_parameters(self):
        conditioner = AnyThermalDirectConditioner()
        self.assertEqual(sum(parameter.numel() for parameter in conditioner.parameters()), 0)
        self.assertEqual(conditioner.state_dict(), {})

    def test_rejects_empty_feature_sequence(self):
        with self.assertRaisesRegex(ValueError, "at least one feature map"):
            AnyThermalDirectConditioner()([], (4, 8))


if __name__ == "__main__":
    unittest.main()
