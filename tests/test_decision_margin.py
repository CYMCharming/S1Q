"""Aligned native-decision calibration mathematics and state restoration."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from s1q.decision_margin import (DecisionMarginCollector, DecisionMarginStatistics,
                                 margin_token_weights, teacher_margin_objective)


class TokenAdapter:
    def __init__(self, *, disconnected=False, opposite_questions=False,
                 common_shift=False, zero_gradient=False):
        self.model = nn.Module()
        self.model.backbone = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.backbone.weight.zero_()
        self.backbone = self.model.backbone
        self.disconnected = disconnected
        self.opposite_questions = opposite_questions
        self.common_shift = common_shift
        self.zero_gradient = zero_gradient

    def linear_modules(self):
        return {"projection": self.backbone}

    def infer(self, record):
        hidden = self.backbone(torch.tensor([[1., 0.], [0., 2.]]))
        score = hidden[1, 0]
        shared = hidden.sum() if self.common_shift or self.zero_gradient else 0.
        if self.zero_gradient:
            logits = torch.stack((shared, shared))
        else:
            logits = torch.stack((score + shared, -score + shared))
        if self.disconnected:
            logits = logits.detach()
        return [logits, -logits] if self.opposite_questions else [logits]


def test_margin_objective_has_detached_scale_and_ignores_common_logit_shift():
    logits = torch.tensor([4., 3., -7.], requires_grad=True)
    objective = teacher_margin_objective(logits, margin_floor=1.)
    objective.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([.5, -.5, 0.]))
    assert float(logits.grad.sum()) == 0
    assert objective.item() == pytest.approx(.5)


def test_bounded_weights_are_mean_one_and_declared_cap_survives_normalization():
    energy = torch.zeros(1000)
    energy[0] = 1.
    weights = margin_token_weights(energy, shrinkage=.5, cap=4.)
    assert weights.mean().item() == pytest.approx(1., abs=1e-6)
    assert weights.max().item() <= 4
    assert weights.min().item() >= .5
    assert weights[0].item() == pytest.approx(4.)
    assert margin_token_weights(torch.zeros(4)).tolist() == [1.] * 4
    assert margin_token_weights(energy, shrinkage=1).tolist() == [1.] * 1000


def test_moments_and_saved_importance_align_with_decisive_token_inputs():
    with DecisionMarginCollector(TokenAdapter(), reservoir_size=2, seed=1) as collector:
        collector.collect({"label": "ignored"})
    statistics = collector.statistics()["projection"]
    torch.testing.assert_close(statistics.mean_square, torch.tensor([.25, 3.]))
    for row, weight in zip(statistics.reservoir, statistics.reservoir_weights):
        assert weight.item() == pytest.approx(.5 if row[0] else 1.5)
    assert statistics.count == 2 and statistics.records == 1
    assert statistics.effective_weight_sum == pytest.approx(2.)
    assert statistics.output_mean_square is None
    assert collector.metadata()["backward_probes"] == 1
    assert collector.metadata()["uses_gold_labels"] is False
    assert collector.metadata()["is_categorical_fisher"] is False


def test_opposing_question_gradients_do_not_cancel_decisive_token_importance():
    results = []
    for opposite in (False, True):
        with DecisionMarginCollector(TokenAdapter(opposite_questions=opposite), reservoir_size=2) as collector:
            collector.collect({})
        results.append(collector.statistics()["projection"].mean_square)
        assert collector.metadata()["backward_probes"] == (2 if opposite else 1)
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[1], torch.tensor([.25, 3.]))


def test_common_shift_does_not_change_margin_statistics():
    results = []
    for shift in (False, True):
        with DecisionMarginCollector(TokenAdapter(common_shift=shift), reservoir_size=2) as collector:
            collector.collect({})
        results.append(collector.statistics()["projection"].mean_square)
    torch.testing.assert_close(results[0], results[1])


def test_zero_decision_gradient_conservatively_uses_uniform_inputs():
    with DecisionMarginCollector(TokenAdapter(zero_gradient=True), reservoir_size=2) as collector:
        collector.collect({})
    statistics = collector.statistics()["projection"]
    torch.testing.assert_close(statistics.mean_square, torch.tensor([.5, 2.]))
    assert statistics.reservoir_weights.tolist() == [1., 1.]


def test_collector_restores_modes_flags_and_existing_gradients_under_outer_inference():
    adapter = TokenAdapter()
    adapter.model.train()
    adapter.backbone.eval()
    adapter.backbone.weight.grad = torch.full_like(adapter.backbone.weight, 7.)
    modes = [m.training for m in adapter.model.modules()]
    flags = [p.requires_grad for p in adapter.model.parameters()]
    with torch.inference_mode():
        with DecisionMarginCollector(adapter) as collector:
            logits = collector.collect({})
            assert not logits[0].requires_grad
            assert all(not p.requires_grad for p in adapter.model.parameters())
    assert [p.requires_grad for p in adapter.model.parameters()] == flags
    assert [m.training for m in adapter.model.modules()] == modes
    assert not adapter.backbone._forward_hooks
    torch.testing.assert_close(adapter.backbone.weight.grad, torch.full_like(adapter.backbone.weight, 7.))


def test_disconnected_forward_fails_and_restores_hooks_and_flags():
    adapter = TokenAdapter(disconnected=True)
    with pytest.raises(RuntimeError, match="differentiable path"):
        with DecisionMarginCollector(adapter) as collector:
            collector.collect({})
    assert adapter.backbone.weight.requires_grad
    assert not adapter.backbone._forward_hooks
    assert collector.records == 0 and collector.failed_records == 1
    assert collector.statistics(require_all=False) == {}


def test_failure_before_commit_does_not_change_statistics_or_random_state(monkeypatch):
    adapter = TokenAdapter()
    with DecisionMarginCollector(adapter, reservoir_size=1, seed=5) as collector:
        state = collector._generator.get_state().clone()
        update = collector._update

        def fail_after_update(*args):
            update(*args)
            raise RuntimeError("fixture request failure")

        monkeypatch.setattr(collector, "_update", fail_after_update)
        with pytest.raises(RuntimeError, match="fixture request failure"):
            collector.collect({})
        torch.testing.assert_close(collector._generator.get_state(), state, rtol=0, atol=0)
        assert collector.statistics(require_all=False) == {}
        assert collector.backward_probes == 1
        assert collector.records == 0 and collector.failed_records == 1


def test_seed_reproducibility_labels_ignored_and_uniform_sampling_identity():
    results = []
    for label in ("A", "B"):
        with DecisionMarginCollector(TokenAdapter(), reservoir_size=1, seed=19,
                                     reservoir_sampling="uniform_priority") as collector:
            collector.collect({"label": label})
        results.append(collector.statistics()["projection"])
    torch.testing.assert_close(results[0].reservoir, results[1].reservoir, rtol=0, atol=0)
    assert results[0].reservoir_sampling == "uniform_priority"
    assert results[0].reservoir_weights.shape == (1,)


def test_multiple_calls_of_same_linear_preserve_input_gradient_alignment():
    adapter = TokenAdapter()

    def repeated(_):
        first = adapter.backbone(torch.tensor([[1., 0.]]))
        second = adapter.backbone(torch.tensor([[0., 2.]]))
        return [torch.stack((second[0, 0] + first[0, 0] * 0, -second[0, 0]))]

    adapter.infer = repeated
    with DecisionMarginCollector(adapter, reservoir_size=2) as collector:
        collector.collect({})
    torch.testing.assert_close(collector.statistics()["projection"].mean_square, torch.tensor([.25, 3.]))


@pytest.mark.parametrize("options", [{"shrinkage": 0}, {"cap": .5}, {"margin_floor": 0},
                                    {"reservoir_size": -1}, {"reservoir_sampling": "bad"}])
def test_invalid_configuration_fails(options):
    with pytest.raises(ValueError):
        DecisionMarginCollector(TokenAdapter(), **options)


def test_statistics_reject_misaligned_importance():
    statistics = DecisionMarginStatistics(count=1, mean_square=torch.ones(2), absmax=torch.ones(2),
        reservoir=torch.ones(1, 2), reservoir_weights=torch.ones(2), effective_weight_sum=1., records=1)
    with pytest.raises(ValueError, match="align"):
        statistics.validate(2)
