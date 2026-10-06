"""Behavioral checks for controlled, architecture-neutral PTQ adaptations."""

import copy
import unittest

import torch
from torch import nn

from s1q.baselines import quantize_baseline_model, quantize_baseline_weight
from s1q.quantization import CalibrationCollector, load_quantized_artifact, quantize_weight


class _TinyDecision(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(6, 9), nn.Tanh(), nn.Linear(9, 5))
        self.decision_head = nn.Linear(5, 3)

    def forward(self, x):
        return self.decision_head(self.backbone(x))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(221)

    def _statistics(self, model, inputs):
        with CalibrationCollector(model, include_prefixes=("backbone",), reservoir_size=64, seed=9) as collector:
            model(inputs)
        return collector.statistics()

    def test_rtn_reuses_identical_quantizer(self):
        weight = torch.randn(4, 7)
        direct = quantize_weight(weight, bits=4, group_size=3)
        adapted = quantize_baseline_weight(weight, method="rtn", bits=4, group_size=3)
        self.assertTrue(torch.equal(direct.integer_weight, adapted.integer_weight))
        self.assertTrue(torch.equal(direct.scales, adapted.scales))

    def test_awq_adapted_search_includes_rtn_on_calibration_inputs(self):
        layer = nn.Linear(6, 4, bias=False)
        inputs = torch.randn(40, 6) * torch.tensor([1., 0.05, 4., 0.2, 8., 0.02])
        with CalibrationCollector(layer, exclude_patterns=(), reservoir_size=40) as collector:
            layer(inputs)
        stats = collector.statistics()[""]
        rtn = quantize_baseline_weight(layer.weight, method="rtn", bits=4,
                                       group_size=3, statistics=stats)
        awq = quantize_baseline_weight(layer.weight, method="awq-adapted", bits=4,
                                       group_size=3, statistics=stats)
        self.assertLessEqual(awq.reservoir_relative_error, rtn.reservoir_relative_error + 1e-6)
        self.assertTrue(torch.isfinite(awq.effective_weight()).all())

    def test_gptq_variants_produce_valid_packed_codes_and_restore(self):
        model = _TinyDecision()
        inputs = torch.randn(18, 6)
        statistics = self._statistics(model, inputs)
        original = copy.deepcopy(model.state_dict())
        for method, bits in (("gptq-blockdiag-adapted", 3), ("gptq-full-adapted", 4)):
            duplicate = copy.deepcopy(model)
            with quantize_baseline_model(model, method=method, bits=bits, group_size=3,
                                         statistics=statistics,
                                         include_prefixes=("backbone",)) as session:
                self.assertEqual(set(session.layers), set(statistics))
                self.assertTrue(torch.equal(model.decision_head.weight, original["decision_head.weight"]))
                prediction = model(inputs).detach()
                self.assertTrue(torch.isfinite(prediction).all())
                for layer in session.layers.values():
                    qmax = (1 << (bits - 1)) - 1
                    self.assertGreaterEqual(layer.integer_weight.min().item(), -qmax)
                    self.assertLessEqual(layer.integer_weight.max().item(), qmax)
                    self.assertTrue((layer.scales > 0).all())
                with load_quantized_artifact(duplicate, session.artifact()):
                    self.assertTrue(torch.equal(duplicate(inputs), prediction))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]), name)

    def test_smoothquant_activation_hook_and_artifact_replay(self):
        model = _TinyDecision()
        duplicate = copy.deepcopy(model)
        inputs = torch.randn(16, 6)
        statistics = self._statistics(model, inputs)
        with quantize_baseline_model(model, method="smoothquant-adapted", bits=4,
                                     group_size=3, statistics=statistics,
                                     include_prefixes=("backbone",), activation_bits=4) as session:
            prediction = model(inputs).detach()
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertEqual(session.activation_bits, 4)
            self.assertTrue(any(not torch.allclose(layer.input_scale, torch.ones_like(layer.input_scale))
                                for layer in session.layers.values()))
            with load_quantized_artifact(duplicate, session.artifact()):
                self.assertTrue(torch.equal(duplicate(inputs), prediction))
        with quantize_baseline_model(model, method="smoothquant-adapted", bits=3,
                                     group_size=3, statistics=statistics,
                                     include_prefixes=("backbone",), activation_bits=None) as weight_only:
            self.assertIsNone(weight_only.activation_bits)
            self.assertTrue(torch.isfinite(model(inputs)).all())

    def test_required_calibration_and_active_session_guards(self):
        model = _TinyDecision()
        inputs = torch.randn(8, 6)
        statistics = self._statistics(model, inputs)
        with self.assertRaises(ValueError):
            quantize_baseline_model(model, method="gptq-full-adapted", bits=4, group_size=3)
        with quantize_baseline_model(model, method="rtn", bits=4, group_size=3,
                                     include_prefixes=("backbone",)):
            with self.assertRaises(RuntimeError):
                quantize_baseline_model(model, method="awq-adapted", bits=4,
                                        group_size=3, statistics=statistics,
                                        include_prefixes=("backbone",))


if __name__ == "__main__":
    unittest.main()
