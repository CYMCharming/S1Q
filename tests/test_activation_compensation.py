"""Checks for activation-error compensation, independent of decision labels."""

import unittest

import torch
import torch.nn.functional as F

from s1q.activation_compensation import compensate_weight
from s1q.quantization import fake_quantize_activation


class ActivationCompensationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(271)
        self.weight = torch.randn(7, 19)
        self.inputs = torch.randn(11, 19) * torch.linspace(0.1, 5.0, 19)
        self.scale = torch.linspace(0.2, 1.8, 19)

    def test_dual_solution_matches_independent_primal_ridge_solution(self):
        corrected, metadata = compensate_weight(self.weight, self.inputs, self.scale, 4,
                                                ridge=0.2, max_relative_correction=10.0)
        transformed = self.weight * self.scale
        z = fake_quantize_activation(self.inputs / self.scale, 4)
        residual = F.linear(self.inputs, self.weight) - F.linear(z, transformed)
        regularization = metadata["regularization_lambda"]
        expected_delta = torch.linalg.solve(z.T @ z + regularization * torch.eye(z.shape[1]),
                                            z.T @ residual).T
        torch.testing.assert_close(corrected, transformed + expected_delta, atol=2e-5, rtol=2e-5)
        self.assertLess(metadata["after_relative_output_mse"], metadata["before_relative_output_mse"])
        self.assertFalse(metadata["uses_gold_labels"])
        self.assertEqual(metadata["extra_inference_parameters"], 0)

    def test_token_weighted_solution_matches_weighted_normal_equations(self):
        weights = torch.linspace(0.0, 3.0, self.inputs.shape[0])
        corrected, metadata = compensate_weight(self.weight, self.inputs, self.scale, 4,
                                                ridge=0.1, max_relative_correction=10.0,
                                                token_weights=weights)
        z = fake_quantize_activation(self.inputs / self.scale, 4)
        residual = F.linear(self.inputs, self.weight) - F.linear(z, self.weight * self.scale)
        normalized = weights / weights.mean()
        expected_delta = torch.linalg.solve(z.T @ (normalized[:, None] * z)
                                            + metadata["regularization_lambda"] * torch.eye(z.shape[1]),
                                            z.T @ (normalized[:, None] * residual)).T
        torch.testing.assert_close(corrected, self.weight * self.scale + expected_delta,
                                   atol=3e-5, rtol=3e-5)
        self.assertTrue(metadata["token_weighting"])

    def test_native_activation_and_disabled_controls_are_exact_noops(self):
        for bits, kwargs, reason in ((None, {}, "native_activation"),
                                    (4, {"strength": 0}, "zero_strength"),
                                    (8, {"max_relative_correction": 0}, "zero_trust_cap")):
            corrected, metadata = compensate_weight(self.weight, self.inputs, self.scale, bits, **kwargs)
            self.assertTrue(torch.equal(corrected, self.weight * self.scale))
            self.assertFalse(metadata["correction_applied"])
            self.assertEqual(metadata["no_op_reason"], reason)

    def test_trust_cap_and_strength_shrink_the_correction(self):
        transformed = self.weight * self.scale
        uncapped, _ = compensate_weight(self.weight, self.inputs, self.scale, 4,
                                        max_relative_correction=10.0)
        shrunk, _ = compensate_weight(self.weight, self.inputs, self.scale, 4,
                                      strength=0.3, max_relative_correction=10.0)
        torch.testing.assert_close(shrunk - transformed, 0.3 * (uncapped - transformed))
        capped, metadata = compensate_weight(self.weight, self.inputs, self.scale, 4,
                                            max_relative_correction=0.001)
        ratio = torch.linalg.vector_norm(capped - transformed) / torch.linalg.vector_norm(transformed)
        self.assertLessEqual(ratio.item(), 0.001001)
        self.assertLess(metadata["trust_cap_multiplier"], 1.0)
        self.assertLessEqual(metadata["after_relative_output_mse"], metadata["before_relative_output_mse"])

    def test_input_and_weight_are_unmodified_and_fp32_is_returned(self):
        weight = self.weight.to(torch.bfloat16)
        inputs = self.inputs.to(torch.bfloat16)
        before_weight, before_inputs = weight.clone(), inputs.clone()
        corrected, _ = compensate_weight(weight, inputs, self.scale, 8)
        self.assertEqual(corrected.dtype, torch.float32)
        self.assertFalse(corrected.requires_grad)
        self.assertTrue(torch.equal(weight, before_weight))
        self.assertTrue(torch.equal(inputs, before_inputs))

    def test_zero_weights_and_zero_inputs_are_finite(self):
        for weight, inputs in ((torch.zeros_like(self.weight), self.inputs),
                               (self.weight, torch.zeros_like(self.inputs))):
            corrected, metadata = compensate_weight(weight, inputs, self.scale, 4)
            self.assertTrue(torch.isfinite(corrected).all())
            self.assertFalse(metadata["correction_applied"])

    def test_invalid_dimensions_values_and_configuration_fail_loudly(self):
        cases = [
            {"ridge": 0}, {"ridge": float("nan")}, {"strength": -1},
            {"max_relative_correction": -1}, {"activation_bits": 3},
            {"token_weights": torch.zeros(self.inputs.shape[0])},
            {"token_weights": torch.ones(self.inputs.shape[0] + 1)},
            {"token_weights": torch.full((self.inputs.shape[0],), -1.0)},
            {"inputs": self.inputs[:, :-1]}, {"inputs": self.inputs[:0]},
            {"input_scale": torch.zeros_like(self.scale)},
            {"input_scale": torch.full_like(self.scale, float("inf"))},
            {"weight": self.weight.to(torch.int32)},
        ]
        for changes in cases:
            arguments = {"weight": self.weight, "inputs": self.inputs,
                         "input_scale": self.scale, "activation_bits": 4, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                compensate_weight(**arguments)


if __name__ == "__main__":
    unittest.main()
