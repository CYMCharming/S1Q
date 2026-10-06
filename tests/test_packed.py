"""Packed-storage parity, persistent-memory, and artifact compatibility checks."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from s1q.packed import PACKED_EXECUTION, PackedLinear, apply_packed, packed_report
from s1q.quantization import CalibrationCollector, load_quantized_artifact, quantize_model


class SmallDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 4))
        self.pointer_head = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.pointer_head(self.backbone(inputs))


class PackedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(921)

    def artifact(self, model, inputs, *, bits=4, activation_bits=None, group_size=3):
        with CalibrationCollector(model, reservoir_size=8) as collector:
            model(inputs)
        session = quantize_model(model, method="s1q", bits=bits,
                                 statistics=collector.statistics(), group_size=group_size,
                                 alphas=(0.5,), clipping_ratios=(1.,),
                                 activation_bits=activation_bits)
        expected = model(inputs).detach().clone()
        artifact = session.artifact()
        session.restore()
        return artifact, expected

    def test_packed_matches_compensated_dequantized_forward_w4_w8_a4_a8(self):
        inputs = torch.randn(11, 5) * torch.tensor([0.02, 1., 4., 0.5, 8.])
        for bits in (4, 8):
            for activation_bits in (None, 4, 8):
                with self.subTest(bits=bits, activation_bits=activation_bits):
                    model = SmallDecisionModel()
                    artifact, expected = self.artifact(model, inputs, bits=bits,
                                                       activation_bits=activation_bits)
                    model = apply_packed(model, artifact)
                    self.assertTrue(torch.equal(model(inputs), expected))
                    self.assertIsInstance(model.backbone[0], PackedLinear)
                    self.assertIsInstance(model.backbone[2], PackedLinear)

    def test_w2_w3_artifact_is_actually_bit_packed_and_matches_dense_forward(self):
        inputs = torch.randn(11, 5) * torch.tensor([0.02, 1., 4., 0.5, 8.])
        for bits in (2, 3):
            for activation_bits in (None, 4, 8):
                with self.subTest(bits=bits, activation_bits=activation_bits):
                    model = SmallDecisionModel()
                    artifact, expected = self.artifact(model, inputs, bits=bits,
                                                       activation_bits=activation_bits)
                    self.assertEqual(artifact["format"], "s1q.packed_linear.v2")
                    for record in artifact["layers"].values():
                        count = math.prod(record["shape"])
                        self.assertEqual(record["payload"].numel(), math.ceil(count * bits / 8))
                        self.assertEqual(record["bit_order"], "lsb_first_twos_complement")
                    with load_quantized_artifact(model, artifact):
                        self.assertTrue(torch.equal(model(inputs), expected))
                    model = apply_packed(model, artifact)
                    self.assertTrue(torch.equal(model(inputs), expected))
                    self.assertEqual(model.backbone[0].bits, bits)

    def test_untouched_head_preserves_object_and_tensors(self):
        model = SmallDecisionModel()
        head = model.pointer_head
        original_head = copy.deepcopy(head.state_dict())
        inputs = torch.randn(6, 5)
        artifact, _ = self.artifact(model, inputs)
        packed = apply_packed(model, artifact)
        self.assertIs(packed, model)
        self.assertIs(model.pointer_head, head)
        for name, tensor in head.state_dict().items():
            self.assertTrue(torch.equal(tensor, original_head[name]))

    def test_root_linear_returns_replacement_and_handles_no_bias(self):
        original = nn.Linear(5, 3, bias=False)
        inputs = torch.randn(8, 5)
        artifact, expected = self.artifact(original, inputs)
        replacement = apply_packed(original, artifact)
        self.assertIsInstance(replacement, PackedLinear)
        self.assertIsNot(replacement, original)
        self.assertIsNone(replacement.bias)
        self.assertTrue(torch.equal(replacement(input=inputs), expected))

    def test_weight_property_folds_in_compensation_without_retaining_dense_tensor(self):
        original = nn.Linear(5, 4)
        inputs = torch.randn(20, 5) * torch.tensor([0.01, 1., 5., 2., 0.2])
        artifact, expected = self.artifact(original, inputs)
        packed = apply_packed(original, artifact)
        before = packed.retained_bytes()
        first, second = packed.weight, packed.weight
        self.assertTrue(torch.equal(first, second))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.allclose(F.linear(inputs, first, packed.bias), expected, atol=1e-6, rtol=1e-5))
        self.assertEqual(packed.retained_bytes(), before)
        self.assertEqual(list(packed.parameters()), [])
        self.assertFalse(any(tuple(buffer.shape) == (4, 5) and buffer.is_floating_point()
                             for buffer in packed.buffers()))

    def test_persistent_storage_and_serialized_artifact_are_smaller_than_dense_weights(self):
        original = nn.Linear(257, 64)
        inputs = torch.randn(4, 257)
        artifact, expected = self.artifact(original, inputs, group_size=128)
        original_bytes = sum(tensor.numel() * tensor.element_size() for tensor in original.state_dict().values())
        packed = apply_packed(original, artifact)
        report = packed_report(packed)
        self.assertEqual(report["execution"], PACKED_EXECUTION)
        self.assertFalse(report["native_integer_gemm"])
        self.assertEqual(report["packed_linear_count"], 1)
        self.assertEqual(report["packed_payload_bytes"], math.ceil(257 * 64 / 2))
        self.assertLess(report["model_resident_tensor_bytes"], original_bytes * 0.25)
        self.assertEqual(report["model_resident_tensor_bytes"], packed.retained_bytes())
        self.assertTrue(torch.equal(packed(inputs), expected))
        with tempfile.TemporaryDirectory() as temporary:
            packed_path = Path(temporary) / "packed.pt"
            dense_path = Path(temporary) / "dense.pt"
            torch.save(artifact, packed_path)
            torch.save(original.state_dict(), dense_path)
            self.assertLess(packed_path.stat().st_size, dense_path.stat().st_size * 0.4)

    def test_serialized_artifact_loading_preserves_outputs(self):
        model = SmallDecisionModel()
        inputs = torch.randn(9, 5)
        artifact, expected = self.artifact(model, inputs, activation_bits=8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "packed.pt"
            torch.save(artifact, path)
            loaded = apply_packed(model, path)
        self.assertTrue(torch.equal(loaded(inputs), expected))

    def test_dtype_moves_keep_scale_precision_and_integer_codes(self):
        original = nn.Linear(5, 3)
        artifact, _ = self.artifact(original, torch.randn(7, 5))
        packed = apply_packed(original, artifact)
        scales, input_scale = packed.scales.clone(), packed.input_scale.clone()
        codes = packed.qweight.clone()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            packed.to(dtype=dtype)
            self.assertEqual(packed.weight.dtype, dtype)
            self.assertEqual(packed.bias.dtype, dtype)
            self.assertEqual(packed.scales.dtype, torch.float32)
            self.assertEqual(packed.input_scale.dtype, torch.float32)
            self.assertTrue(torch.equal(packed.scales, scales))
            self.assertTrue(torch.equal(packed.input_scale, input_scale))
            self.assertTrue(torch.equal(packed.qweight, codes))

    def test_active_quantization_session_must_be_restored_before_replacement(self):
        model = SmallDecisionModel()
        session = quantize_model(model, group_size=3)
        artifact = session.artifact()
        with self.assertRaisesRegex(RuntimeError, "Restore"):
            apply_packed(model, artifact)
        self.assertIsInstance(model.backbone[0], nn.Linear)
        session.restore()
        apply_packed(model, artifact)
        self.assertIsInstance(model.backbone[0], PackedLinear)

    def test_malformed_later_layer_never_partly_replaces_model(self):
        model = SmallDecisionModel()
        artifact, _ = self.artifact(model, torch.randn(5, 5))
        first, second = model.backbone[0], model.backbone[2]
        artifact["layers"]["backbone.2"]["scales"][0, 0] = 0
        with self.assertRaises(ValueError):
            apply_packed(model, artifact)
        self.assertIs(model.backbone[0], first)
        self.assertIs(model.backbone[2], second)

    def test_payload_range_padding_and_shape_validation(self):
        original = nn.Linear(3, 3)
        artifact, _ = self.artifact(original, torch.randn(3, 3))
        record = artifact["layers"][""]
        mutations = []
        invalid_range = copy.deepcopy(record)
        invalid_range["payload"][0] = 8  # -8 is outside this +/-7 quantizer.
        mutations.append(invalid_range)
        invalid_padding = copy.deepcopy(record)
        invalid_padding["payload"][-1] |= 16
        mutations.append(invalid_padding)
        invalid_scales = copy.deepcopy(record)
        invalid_scales["scales"][0, 0] = float("nan")
        mutations.append(invalid_scales)
        invalid_shape = copy.deepcopy(record)
        invalid_shape["shape"] = [3, 4]
        mutations.append(invalid_shape)
        invalid_input_scale = copy.deepcopy(record)
        invalid_input_scale["input_scale"][0] = 0
        mutations.append(invalid_input_scale)
        for index, bad in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises(ValueError):
                PackedLinear(bad)

    def test_replacement_rejects_existing_hooks_and_incompatible_model(self):
        model = SmallDecisionModel()
        artifact, _ = self.artifact(model, torch.randn(5, 5))
        handle = model.backbone[0].register_forward_hook(lambda module, inputs, output: output)
        with self.assertRaisesRegex(ValueError, "hooks"):
            apply_packed(model, artifact)
        handle.remove()
        wrong_model = SmallDecisionModel()
        wrong_model.backbone[2] = nn.Linear(7, 5)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            apply_packed(wrong_model, artifact)

    def test_shared_parameter_cannot_be_replaced_and_change_scope_semantics(self):
        first = nn.Linear(3, 3)
        artifact, _ = self.artifact(first, torch.randn(5, 3))
        model = nn.Module()
        model.backbone = first
        model.pointer_head = nn.Linear(3, 3)
        model.pointer_head.weight = first.weight
        artifact["layers"]["backbone"] = artifact["layers"].pop("")
        with self.assertRaisesRegex(ValueError, "shares"):
            apply_packed(model, artifact)
        self.assertIs(model.backbone, first)

    def test_empty_artifact_leaves_model_untouched_and_reports_zero_packed_layers(self):
        model = SmallDecisionModel()
        artifact, _ = self.artifact(model, torch.randn(5, 5))
        artifact["layers"] = {}
        self.assertIs(apply_packed(model, artifact), model)
        report = packed_report(model)
        self.assertEqual(report["packed_linear_count"], 0)
        self.assertEqual(report["packed_payload_bytes"], 0)
        self.assertGreater(report["model_resident_tensor_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
