"""Checks for the opt-in empirical and labeled-calibration research modes."""

import copy
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from s1q.decision_stats import (DecisionFisherCollector,
                                correct_boundary_question_weights,
                                decision_fisher_objective)
from s1q.quantization import InputStatistics, load_quantized_artifact, quantize_model, quantize_weight


class _TinyAdapter:
    def __init__(self):
        self.model = nn.Module()
        self.model.backbone = nn.Linear(2, 3)
        self.model.head = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.model.backbone.weight.zero_()
            self.model.backbone.bias.zero_()
            self.model.head.weight.copy_(torch.tensor([[1., 2., 0.], [0., 0., 0.]]))

    def linear_modules(self):
        return {"backbone": self.model.backbone}

    def infer(self, _record):
        hidden = self.model.backbone(torch.tensor([[1., 2.]]))
        return [self.model.head(hidden).reshape(-1)]


class S1Q2MethodTests(unittest.TestCase):
    def test_correlated_inputs_change_search_without_changing_bit_budget(self):
        weight = torch.tensor([
            [-2.4124324, -0.4943543, 0.3754711, 0.00041137, 0.0455685, 0.0542803],
            [0.5686941, 0.2115024, -0.1552247, 0.6216088, -0.1127751, 0.1453021],
        ])
        base = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.],
                             [0., 0., 1.], [0., 0., -1.], [1., 1., 1.], [-1., -1., -1.]])
        x = torch.stack((base[:, 0], base[:, 0], base[:, 1], -base[:, 1],
                         base[:, 2], -base[:, 2]), dim=1)
        stats = InputStatistics(len(x), x.square().mean(0), x.abs().amax(0), x)
        old = quantize_weight(weight, method="s1q", group_size=6, statistics=stats)
        zero_blend = quantize_weight(weight, method="s1q2", group_size=6, statistics=stats,
                                     reservoir_blend=0)
        empirical = quantize_weight(weight, method="s1q2", group_size=6, statistics=stats,
                                    reservoir_blend=.75)
        self.assertTrue(torch.equal(old.integer_weight, zero_blend.integer_weight))
        self.assertTrue(torch.equal(old.scales, zero_blend.scales))
        self.assertTrue(torch.equal(old.input_scale, zero_blend.input_scale))
        old_loss = F.linear(x, old.effective_weight() - weight).square().mean()
        new_loss = F.linear(x, empirical.effective_weight() - weight).square().mean()
        self.assertLess(new_loss.item(), old_loss.item() * .5)
        self.assertEqual(empirical.integer_weight.shape, old.integer_weight.shape)
        self.assertEqual(empirical.scales.shape, old.scales.shape)
        self.assertEqual(empirical.report()["search_reservoir_rows"], len(x))
        self.assertEqual(empirical.report()["reservoir_blend"], .75)

    def test_empirical_search_requires_calibration_rows_and_roundtrips(self):
        model = nn.Module()
        model.backbone = nn.Linear(6, 2)
        model.head = nn.Linear(2, 2)
        baseline = copy.deepcopy(model)
        x = torch.arange(1., 25.).reshape(4, 6) / 10
        stats = InputStatistics(len(x), x.square().mean(0), x.abs().amax(0), x)
        with self.assertRaisesRegex(ValueError, "reservoir"):
            quantize_weight(model.backbone.weight, method="s1q2", statistics=InputStatistics(
                len(x), x.square().mean(0), x.abs().amax(0)))
        with self.assertRaises(ValueError):
            quantize_weight(model.backbone.weight, method="s1q2", statistics=stats,
                            reservoir_blend=1.1)
        session = quantize_model(model, method="s1q2", group_size=6,
                                 statistics={"backbone": stats}, reservoir_blend=.5)
        expected = model.head(model.backbone(x)).detach()
        artifact = session.artifact()
        self.assertEqual(artifact["layers"]["backbone"]["reservoir_blend"], .5)
        restored = load_quantized_artifact(baseline, artifact)
        torch.testing.assert_close(baseline.head(baseline.backbone(x)), expected)
        restored.restore()
        session.restore()

    def test_correct_boundary_weights_do_not_preserve_teacher_errors_equally(self):
        logits = [torch.tensor([0., 0.]), torch.tensor([0., 1.])]
        record = {"questions": {
            "correct": {"type": "choice", "criteria": ["a", "b"], "label": 0},
            "wrong": {"type": "choice", "criteria": ["a", "b"], "label": 0},
        }}
        self.assertEqual(correct_boundary_question_weights(logits, record), [2., .25])
        with self.assertRaisesRegex(ValueError, "labels"):
            correct_boundary_question_weights(logits, {"questions": {
                "correct": {"type": "choice", "criteria": ["a", "b"], "label": 0},
                "wrong": {"type": "choice", "criteria": ["a", "b"]},
            }})

    def test_weighted_fisher_covariance_and_default_mode_are_separate(self):
        correct_record = {"questions": {"q": {"type": "choice", "criteria": ["a", "b"], "label": 0}}}
        wrong_record = {"questions": {"q": {"type": "choice", "criteria": ["a", "b"], "label": 1}}}
        observations = []
        for record in (correct_record, wrong_record):
            with DecisionFisherCollector(_TinyAdapter(), seed=22, probes_per_record=16,
                                         decision_weighting="correct_boundary") as collector:
                collector.collect(record)
            self.assertTrue(collector.metadata()["uses_gold_labels"])
            observations.append(collector.statistics()["backbone"].mean_square)
        torch.testing.assert_close(observations[0], observations[1] * 8, atol=1e-6, rtol=1e-5)
        with DecisionFisherCollector(_TinyAdapter(), seed=22, probes_per_record=16) as collector:
            collector.collect(wrong_record)
        self.assertFalse(collector.metadata()["uses_gold_labels"])
        torch.testing.assert_close(collector.statistics()["backbone"].mean_square,
                                   observations[1] * 4, atol=1e-6, rtol=1e-5)

        z = torch.tensor([0., 0.], requires_grad=True)
        unweighted = decision_fisher_objective([z], generator=torch.Generator().manual_seed(5))
        original_gradient = torch.autograd.grad(unweighted, z)[0]
        weighted = decision_fisher_objective([z], generator=torch.Generator().manual_seed(5),
                                              question_weights=[4.])
        torch.testing.assert_close(torch.autograd.grad(weighted, z)[0], 2 * original_gradient)
        with self.assertRaises(ValueError):
            decision_fisher_objective([z], generator=torch.Generator(), question_weights=[0.])


if __name__ == "__main__":
    unittest.main()
