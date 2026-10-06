"""Contracts for the explicitly adapted SpinQuant-inspired rotations."""

import copy
import unittest

import torch
from torch import nn

from s1q.baselines import quantize_baseline_model
from s1q.quantization import CalibrationCollector
from s1q.spinquant_proxy import _hadamard, _rotate_blocks


class SpinQuantProxyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)

    def test_rotation_pair_is_exact_before_quantization(self):
        for width, block in ((8, 4), (7, 2)):
            inputs = torch.randn(5, width)
            weight = torch.randn(3, width)
            parameter = torch.randn(block, block)
            skew = parameter - parameter.T
            eye = torch.eye(block)
            learned = torch.linalg.solve(eye + skew, eye - skew)
            for rotation in (learned, learned @ _hadamard(block, device=inputs.device)):
                self.assertTrue(torch.allclose(rotation.T @ rotation, eye, atol=1e-5))
                rotated = nn.functional.linear(_rotate_blocks(inputs, rotation),
                                               _rotate_blocks(weight, rotation))
                original = nn.functional.linear(inputs, weight)
                self.assertTrue(torch.allclose(rotated, original, atol=1e-5))

    def test_both_spin_variants_are_reversible_and_honest_about_artifacts(self):
        model = nn.Sequential(nn.Linear(8, 9), nn.Tanh(), nn.Linear(9, 4))
        original = copy.deepcopy(model.state_dict())
        inputs = torch.randn(15, 8)
        with CalibrationCollector(model, exclude_patterns=(), reservoir_size=15) as collector:
            model(inputs)
        statistics = collector.statistics()
        for method, activation_bits in (("spinquant-nohad-adapted", None),
                                        ("spinquant-had-adapted", 4)):
            with quantize_baseline_model(model, method=method, bits=3, group_size=4,
                                         statistics=statistics, activation_bits=activation_bits,
                                         exclude_patterns=(), spin_steps=2) as session:
                result = model(inputs)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(set(session.layers), set(statistics))
                self.assertFalse(session.report()["official_spinquant"])
                self.assertGreater(session.report()["extra_rotation_bytes_fp32"], 0)
                self.assertEqual(session.report()["rotation_execution"],
                                 "online_float_hook_before_activation_qdq")
                for layer in session.layers.values():
                    self.assertEqual(layer.method, method)
                    self.assertTrue(torch.isfinite(layer.effective_weight()).all())
                    qmax = (1 << (layer.bits - 1)) - 1
                    self.assertGreaterEqual(layer.integer_weight.min().item(), -qmax)
                    self.assertLessEqual(layer.integer_weight.max().item(), qmax)
                    identity = torch.eye(layer.rotation.shape[0])
                    self.assertTrue(torch.allclose(layer.rotation.T @ layer.rotation,
                                                    identity, atol=1e-4))
                with self.assertRaisesRegex(NotImplementedError, "would not reproduce"):
                    session.artifact()
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]), name)

    def test_missing_statistics_is_rejected(self):
        model = nn.Linear(4, 3)
        with self.assertRaisesRegex(ValueError, "statistics"):
            quantize_baseline_model(model, method="spinquant-nohad-adapted",
                                    bits=4, group_size=4, statistics=None,
                                    activation_bits=4, exclude_patterns=())


if __name__ == "__main__":
    unittest.main()
