"""Reversible-state, exact-QDQ, and scale-folding checks for optional repair."""

import copy
import unittest

import torch
from torch import nn

from s1q.baselines import quantize_baseline_model
from s1q.decision_repair import _STEInputHook, repair_session
from s1q.quantization import (CalibrationCollector, _InputTransform,
                             load_quantized_artifact, quantize_model)
from s1q.spinquant_proxy import _RotateInput, _hadamard


class TinyAdapter:
    def __init__(self, disconnected=False):
        self.model = nn.Module()
        self.model.backbone = nn.Sequential(nn.Linear(9, 11), nn.Tanh(), nn.Linear(11, 5))
        self.model.head = nn.Linear(5, 3)
        self.backbone = self.model.backbone
        self.disconnected = disconnected

    def infer(self, record):
        logits = self.model.head(self.backbone(record["input"])).reshape(-1)
        return [logits.detach() if self.disconnected else logits]


class DecisionRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(713)
        self.adapter = TinyAdapter()
        self.records = [{"input": torch.randn(1, 9) * 3, "label": "must not be used"} for _ in range(8)]
        with torch.no_grad():
            self.teachers = [[z.clone() for z in self.adapter.infer(record)] for record in self.records]
        with CalibrationCollector(self.adapter.backbone, reservoir_size=16) as collector:
            for record in self.records:
                self.adapter.infer(record)
        self.statistics = collector.statistics()

    def test_ste_has_exact_original_forward_and_nonzero_input_gradient(self):
        for original in (_InputTransform(torch.linspace(.2, 2., 8), 4),
                         _RotateInput(_hadamard(4, device=torch.device("cpu")), 4)):
            value = torch.randn(2, 8, requires_grad=True)
            forward_args, _ = original(None, (value,), {})
            ste_args, _ = _STEInputHook(original)(None, (value,), {})
            self.assertTrue(torch.equal(forward_args[0], ste_args[0]))
            ste_args[0].sum().backward()
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0.)

    def test_repair_folds_without_changing_codes_biases_or_parameter_flags(self):
        native = copy.deepcopy(self.adapter.model.state_dict())
        duplicate = copy.deepcopy(self.adapter)
        self.adapter.model.train()
        self.adapter.model.backbone[0].eval()
        self.adapter.model.backbone[0].bias.requires_grad_(False)
        self.adapter.model.head.weight.grad = torch.full_like(self.adapter.model.head.weight, 7.)
        saved_grad = self.adapter.model.head.weight.grad.clone()
        flags = [p.requires_grad for p in self.adapter.model.parameters()]
        modes = [m.training for m in self.adapter.model.modules()]
        with quantize_model(self.adapter.backbone, method="s1q2", bits=3, group_size=4,
                            statistics=self.statistics, activation_bits=4) as session:
            codes = {name: layer.integer_weight.clone() for name, layer in session.layers.items()}
            hook_ids = [handle.id for handle in session._handles]
            original_hooks = [self.adapter.backbone.get_submodule(name)._forward_pre_hooks[handle.id]
                              for (name, _), handle in zip(session.layers.items(), session._handles)]
            metadata = repair_session(self.adapter, session, self.records, self.teachers,
                                      seed=3, epochs=2, max_records=8,
                                      accept_only_if_improved=False)
            self.assertTrue(metadata["accepted"])
            self.assertEqual(metadata["steps"], 16)
            self.assertEqual(metadata["trainable_parameters"], 16)
            self.assertEqual(metadata["extra_inference_parameters"], 0)
            self.assertFalse(metadata["uses_gold_labels"])
            for (name, layer), identifier, original_hook in zip(session.layers.items(), hook_ids, original_hooks):
                self.assertTrue(torch.equal(layer.integer_weight, codes[name]))
                module = self.adapter.backbone.get_submodule(name)
                self.assertIs(module._forward_pre_hooks[identifier], original_hook)
                self.assertFalse(module._forward_hooks)
                self.assertTrue(torch.equal(module.bias, native[f"backbone.{name}.bias"]))
            self.assertEqual(flags, [p.requires_grad for p in self.adapter.model.parameters()])
            self.assertEqual(modes, [m.training for m in self.adapter.model.modules()])
            self.assertTrue(torch.equal(saved_grad, self.adapter.model.head.weight.grad))
            with torch.no_grad():
                folded_predictions = [self.adapter.infer(record)[0].clone() for record in self.records]
            with load_quantized_artifact(duplicate.backbone, session.artifact()):
                with torch.no_grad():
                    for record, expected in zip(self.records, folded_predictions):
                        self.assertTrue(torch.equal(duplicate.infer(record)[0], expected))
        for name, value in self.adapter.model.state_dict().items():
            self.assertTrue(torch.equal(value, native[name]), name)

    def test_folded_weight_gain_matches_output_hook_in_float32(self):
        module = nn.Linear(9, 5)
        inputs = torch.randn(3, 9)
        gain = torch.linspace(.93, 1.07, 5)
        output = module(inputs)
        hook_output = output + (output - module.bias) * (gain - 1.)
        folded_output = nn.functional.linear(inputs, module.weight * gain[:, None], module.bias)
        torch.testing.assert_close(hook_output, folded_output, atol=1e-6, rtol=1e-6)

    def test_outer_inference_mode_and_inference_caches_do_not_break_training(self):
        with quantize_model(self.adapter.backbone, bits=4, activation_bits=4) as session:
            for module in session._modules.values():
                hook = next(iter(module._forward_pre_hooks.values()))
                with torch.inference_mode():
                    hook._cache[(torch.device("cpu"), torch.float32)] = hook.input_scale.clone()
                self.assertTrue(torch.is_inference(next(iter(hook._cache.values()))))
            with torch.inference_mode():
                result = repair_session(self.adapter, session, self.records, self.teachers,
                                        epochs=1, max_records=4, accept_only_if_improved=False)
            self.assertTrue(result["accepted"])
            for module in session._modules.values():
                hook = next(iter(module._forward_pre_hooks.values()))
                self.assertTrue(torch.is_inference(next(iter(hook._cache.values()))))

    def test_failure_restores_flags_hooks_and_keeps_original_quantized_weights(self):
        self.adapter.disconnected = True
        flags = [p.requires_grad for p in self.adapter.model.parameters()]
        with quantize_model(self.adapter.backbone, bits=4, activation_bits=4) as session:
            before = {name: module.weight.clone() for name, module in session._modules.items()}
            original_hooks = {name: dict(module._forward_pre_hooks) for name, module in session._modules.items()}
            with self.assertRaisesRegex(RuntimeError, "no finite gradient"):
                repair_session(self.adapter, session, self.records, self.teachers)
            self.assertEqual(flags, [p.requires_grad for p in self.adapter.model.parameters()])
            for name, module in session._modules.items():
                self.assertTrue(torch.equal(module.weight, before[name]))
                self.assertEqual(module._forward_pre_hooks, original_hooks[name])
                self.assertFalse(module._forward_hooks)

    def test_spinquant_proxy_session_retains_rotations_and_restores(self):
        native = copy.deepcopy(self.adapter.model.state_dict())
        with quantize_baseline_model(self.adapter.backbone, method="spinquant-had-adapted",
                                     bits=4, group_size=4, statistics=self.statistics,
                                     activation_bits=4, spin_steps=2) as session:
            rotations = {name: layer.rotation.clone() for name, layer in session.layers.items()}
            report = repair_session(self.adapter, session, self.records, self.teachers,
                                    epochs=1, max_records=4, accept_only_if_improved=False)
            self.assertTrue(report["accepted"])
            for name, layer in session.layers.items():
                self.assertTrue(torch.equal(layer.rotation, rotations[name]))
                self.assertIsInstance(next(iter(session._modules[name]._forward_pre_hooks.values())), _RotateInput)
        for name, value in self.adapter.model.state_dict().items():
            self.assertTrue(torch.equal(value, native[name]), name)

    def test_zero_epochs_is_exact_unchanged_control(self):
        with quantize_model(self.adapter.backbone, bits=3, activation_bits=4) as session:
            weights = {name: module.weight.clone() for name, module in session._modules.items()}
            report = repair_session(self.adapter, session, self.records, self.teachers, epochs=0)
            self.assertFalse(report["accepted"])
            self.assertEqual(report["before"], report["after"])
            for name, module in session._modules.items():
                self.assertTrue(torch.equal(module.weight, weights[name]))

    def test_mismatched_teacher_or_restored_session_fails_before_changes(self):
        session = quantize_model(self.adapter.backbone, bits=4)
        with self.assertRaises(ValueError):
            repair_session(self.adapter, session, self.records, self.teachers[:-1])
        session.restore()
        with self.assertRaises(RuntimeError):
            repair_session(self.adapter, session, self.records, self.teachers)


if __name__ == "__main__":
    unittest.main()
