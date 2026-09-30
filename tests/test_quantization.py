"""CPU checks of quantization mathematics, scope, restoration and artifact IO."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from s1q.quantization import (
    CalibrationCollector,
    InputStatistics,
    dequantize_groupwise,
    fake_quantize_activation,
    load_quantized_artifact,
    pack_int4,
    quantize_model,
    quantize_weight,
    selected_linear_modules,
    unpack_int4,
)


class TinyDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 4))
        self.pointer_head = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.pointer_head(self.backbone(inputs))


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)

    def calibrate(self, model, inputs):
        with CalibrationCollector(model, reservoir_size=16, seed=41) as collector:
            model(inputs)
        return collector.statistics()

    def test_groupwise_remainder_and_zero_groups(self):
        weight = torch.tensor([[0., 0., 0., 0., -0.8, 0.4, 0.2],
                               [1., -1., 0.25, 0.5, 0., 0., 0.]])
        quantized = quantize_weight(weight, bits=4, group_size=4)
        self.assertEqual(quantized.integer_weight.shape, weight.shape)
        self.assertEqual(quantized.scales.shape, (2, 2))
        self.assertTrue(torch.isfinite(quantized.scales).all())
        self.assertTrue((quantized.scales > 0).all())
        self.assertTrue(torch.equal(quantized.integer_weight[0, :4], torch.zeros(4, dtype=torch.int8)))
        restored = dequantize_groupwise(quantized.integer_weight, quantized.scales, 4)
        self.assertTrue(torch.allclose(restored, quantized.effective_weight()))
        self.assertLessEqual((restored[0, 4:] - weight[0, 4:]).abs().max().item(), 0.8 / 14 + 1e-6)

    def test_single_remainder_group_smaller_than_group_size(self):
        weight = torch.randn(3, 5)
        quantized = quantize_weight(weight, bits=8, group_size=128)
        self.assertEqual(quantized.scales.shape, (3, 1))
        bound = weight.abs().amax(dim=1, keepdim=True) / (2 * 127)
        self.assertTrue(((quantized.effective_weight() - weight).abs() <= bound + 1e-6).all())

    def test_unobserved_channels_and_zero_inputs_do_not_produce_invalid_scales(self):
        weight = torch.randn(3, 5)
        stats = InputStatistics(8, torch.zeros(5), torch.zeros(5))
        quantized = quantize_weight(weight, statistics=stats, method="s1q", group_size=3)
        rtn = quantize_weight(weight, method="rtn", group_size=3)
        self.assertTrue(torch.equal(quantized.integer_weight, rtn.integer_weight))
        self.assertTrue(torch.equal(quantized.input_scale, torch.ones(5)))
        self.assertEqual(quantized.clipping_ratio, 1.)
        self.assertTrue(torch.isfinite(quantized.transformed_weight()).all())

    def test_s1q_recovers_rtn_candidate_and_improves_anisotropic_error(self):
        weight = torch.tensor([[0.12, 3.0, -0.2, 0.1], [0.15, -2.5, 0.2, -0.1]])
        moments = torch.tensor([1000., 0.001, 0.01, 0.01])
        stats = InputStatistics(100, moments, moments.sqrt() * 3)
        rtn = quantize_weight(weight, group_size=4, statistics=stats, method="rtn")
        s1q = quantize_weight(weight, group_size=4, statistics=stats, method="s1q")
        self.assertLess(s1q.weighted_error, rtn.weighted_error * 0.5)
        recovered = quantize_weight(weight, group_size=4, statistics=stats, method="s1q",
                                    alphas=(0.,), clipping_ratios=(1.,))
        self.assertTrue(torch.equal(recovered.integer_weight, rtn.integer_weight))
        self.assertEqual(recovered.weighted_error, rtn.weighted_error)

    def test_diagonal_compensation_matches_effective_linear(self):
        layer = nn.Linear(5, 3)
        inputs = torch.randn(12, 5) * torch.tensor([0.1, 1., 4., 8., 0.01])
        with CalibrationCollector(layer, reservoir_size=6) as collector:
            layer(input=inputs)
        statistics = collector.statistics()
        original_bias = layer.bias.detach().clone()
        session = quantize_model(layer, group_size=3, method="s1q", statistics=statistics,
                                 alphas=(0.5,), clipping_ratios=(1.,))
        quantized = session.layers[""]
        expected = F.linear(inputs, quantized.effective_weight(), original_bias)
        self.assertTrue(torch.allclose(layer(input=inputs), expected, atol=1e-6, rtol=1e-5))
        # The compensated input/weight transformation itself is an exact identity.
        original = session._original_weights[""]
        scale = quantized.input_scale
        self.assertTrue(torch.allclose(F.linear(inputs / scale, original * scale, original_bias),
                                       F.linear(inputs, original, original_bias), atol=1e-6, rtol=1e-5))
        session.restore()

    def test_activation_quantization_is_per_token_and_zero_safe(self):
        inputs = torch.tensor([[[0., 0., 0.], [0.1, 1., -0.4]],
                               [[0.2, 10., -4.], [2., -2., 0.]]], dtype=torch.float16)
        quantized = fake_quantize_activation(inputs, 4)
        self.assertEqual(quantized.dtype, inputs.dtype)
        self.assertTrue(torch.equal(quantized[0, 0], inputs[0, 0]))
        self.assertEqual(quantized[0, 1, 1].item(), 1.)
        self.assertEqual(quantized[1, 0, 1].item(), 10.)
        q8 = fake_quantize_activation(inputs, 8)
        self.assertLessEqual((q8.float() - inputs.float()).square().sum().item(),
                             (quantized.float() - inputs.float()).square().sum().item())

    def test_collector_moments_reservoir_and_hook_cleanup(self):
        layer = nn.Linear(5, 2)
        first = torch.randn(2, 3, 5)
        second = torch.randn(4, 5)
        baseline = layer(first)
        with CalibrationCollector(layer, reservoir_size=3, seed=7) as collector:
            self.assertTrue(torch.equal(layer(first), baseline))
            layer(second)
        stats = collector.statistics()[""]
        rows = torch.cat((first.reshape(-1, 5), second))
        self.assertEqual(stats.count, 10)
        self.assertTrue(torch.allclose(stats.mean_square, rows.square().mean(dim=0)))
        self.assertTrue(torch.equal(stats.absmax, rows.abs().amax(dim=0)))
        self.assertEqual(stats.reservoir.shape, (3, 5))
        self.assertFalse(layer._forward_pre_hooks)
        with CalibrationCollector(layer, reservoir_size=3, seed=7) as repeated:
            layer(first)
            layer(second)
        self.assertTrue(torch.equal(stats.reservoir, repeated.statistics()[""].reservoir))

    def test_scope_and_exact_restore_leave_head_unchanged(self):
        model = TinyDecisionModel()
        inputs = torch.randn(20, 5)
        expected = model(inputs).detach()
        original = copy.deepcopy(model.state_dict())
        stats = self.calibrate(model, inputs)
        self.assertEqual(set(stats), {"backbone.0", "backbone.2"})
        session = quantize_model(model, method="s1q", statistics=stats, group_size=3)
        self.assertEqual(set(session.layers), {"backbone.0", "backbone.2"})
        self.assertTrue(torch.equal(model.pointer_head.weight, original["pointer_head.weight"]))
        self.assertTrue(torch.isfinite(model(inputs)).all())
        session.restore()
        session.restore()  # idempotent
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, original[name]), name)
        self.assertTrue(torch.equal(model(inputs), expected))
        self.assertFalse(model.backbone[0]._forward_pre_hooks)

    def test_prefix_preservation_and_explicit_head_opt_in(self):
        model = TinyDecisionModel()
        selected = selected_linear_modules(model, include_prefixes=("backbone",),
                                            preserve_prefixes=("backbone.0",))
        self.assertEqual(set(selected), {"backbone.2"})
        self.assertEqual(len(selected_linear_modules(model, exclude_patterns=())), 3)

    def test_tied_parameter_cannot_silently_mutate_an_excluded_head(self):
        model = nn.Module()
        model.backbone = nn.Linear(3, 3)
        model.pointer_head = nn.Linear(3, 3)
        model.pointer_head.weight = model.backbone.weight
        original = model.pointer_head.weight.detach().clone()
        with self.assertRaises(ValueError):
            quantize_model(model)
        self.assertTrue(torch.equal(model.pointer_head.weight, original))

    def test_sensitive_fraction_preserves_reported_layers(self):
        model = TinyDecisionModel()
        stats = self.calibrate(model, torch.randn(30, 5))
        original = copy.deepcopy(model.state_dict())
        session = quantize_model(model, method="s1q", statistics=stats,
                                 sensitive_fraction=0.5, group_size=3)
        self.assertEqual(len(session.layers), 1)
        self.assertEqual(len(session.preserved_layers), 1)
        preserved = next(iter(session.preserved_layers))
        self.assertTrue(torch.equal(model.get_submodule(preserved).weight, original[preserved + ".weight"]))
        session.restore()

    def test_missing_stats_and_nested_sessions_do_not_mutate_model(self):
        model = TinyDecisionModel()
        original = copy.deepcopy(model.state_dict())
        with self.assertRaises(ValueError):
            quantize_model(model, method="s1q", statistics={})
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, original[name]))
        session = quantize_model(model, group_size=3)
        quantized = copy.deepcopy(model.state_dict())
        with self.assertRaises(RuntimeError):
            quantize_model(model, group_size=3)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, quantized[name]))
        session.restore()

    def test_int4_packing_signed_range_odd_length_and_corruption(self):
        integer = torch.tensor([-8, -7, -1, 0, 1, 6, 7], dtype=torch.int8)
        packed = pack_int4(integer)
        self.assertEqual(packed.numel(), 4)
        self.assertEqual(packed[0].item(), 0x98)
        self.assertTrue(torch.equal(unpack_int4(packed, integer.numel()), integer))
        self.assertEqual(pack_int4(torch.empty(0, dtype=torch.int8)).numel(), 0)
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([65536], dtype=torch.int64))
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([0.5]))
        with self.assertRaises(ValueError):
            unpack_int4(packed, 3)

    def test_packed_artifact_roundtrip_including_compensation_and_a4(self):
        for bits in (4, 8):
            model = TinyDecisionModel()
            base = copy.deepcopy(model)
            inputs = torch.randn(20, 5)
            stats = self.calibrate(model, inputs)
            session = quantize_model(model, bits=bits, method="s1q", statistics=stats,
                                     group_size=3, activation_bits=4)
            expected = model(inputs).detach()
            self.assertEqual(session.report()["execution"], "dequantized_weights_with_optional_fake_activation")
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "weights.pt"
                session.export(path)
                loaded = load_quantized_artifact(base, path)
                self.assertTrue(torch.equal(base(inputs), expected))
                loaded.restore()
            session.restore()
            self.assertTrue(torch.equal(base(inputs), model(inputs)))

    def test_artifact_prevalidation_is_transactional(self):
        model = TinyDecisionModel()
        base = copy.deepcopy(model)
        original = copy.deepcopy(base.state_dict())
        session = quantize_model(model, group_size=3)
        artifact = session.artifact()
        artifact["layers"]["backbone.2"]["scales"][0, 0] = 0
        with self.assertRaises(ValueError):
            load_quantized_artifact(base, artifact)
        for name, tensor in base.state_dict().items():
            self.assertTrue(torch.equal(tensor, original[name]))
        self.assertFalse(base.backbone[0]._forward_pre_hooks)
        session.restore()

    def test_invalid_configuration_and_nonfinite_calibration(self):
        weight = torch.randn(2, 3)
        stats = InputStatistics(3, torch.ones(3), torch.ones(3))
        for kwargs in ({"bits": 3}, {"group_size": 0}, {"method": "unknown"},
                       {"method": "s1q"},
                       {"method": "s1q", "statistics": stats, "alphas": ()},
                       {"method": "s1q", "statistics": stats, "clipping_ratios": (0.,)}):
            with self.assertRaises(ValueError):
                quantize_weight(weight, **kwargs)
        layer = nn.Linear(3, 2)
        with self.assertRaises(ValueError):
            with CalibrationCollector(layer) as collector:
                layer(torch.tensor([[float("nan"), 1., 2.]]))
        self.assertFalse(layer._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
