"""Mathematical and reversible-state checks for decision Fisher statistics."""
import pytest
import torch
from dataclasses import dataclass
from torch import nn

from s1q.decision_stats import (DecisionFisherCollector, DecisionGradientStatistics,
                                attach_input_statistics, categorical_fisher_projection, decision_fisher_objective)


class TinyAdapter:
    def __init__(self, *, shared_shift=False, disconnected=False):
        self.model = nn.Module()
        self.model.backbone = nn.Linear(2, 3)
        self.model.head = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.model.backbone.weight.zero_()
            self.model.backbone.bias.zero_()
            self.model.head.weight.copy_(torch.tensor([[1., 2., 3. if shared_shift else 0.],
                                                       [0., 0., 3. if shared_shift else 0.]]))
        self.backbone, self.disconnected = self.model.backbone, disconnected

    def linear_modules(self):
        return {"projection": self.backbone}

    def infer(self, record):
        hidden = self.backbone(torch.tensor([[1., 2.]]))
        out = self.model.head(hidden).reshape(-1)
        if self.disconnected:
            out = out.detach()
        return [out]


def test_projection_has_exact_categorical_fisher_covariance_in_expectation():
    p = torch.tensor([.2, .3, .5])
    noise = torch.randn(30000, 3, generator=torch.Generator().manual_seed(21))
    v = categorical_fisher_projection(p.expand_as(noise), noise)
    torch.testing.assert_close(v.sum(-1), torch.zeros(v.shape[0]), atol=1e-6, rtol=0)
    observed = v.T @ v / len(v)
    expected = torch.diag(p) - p[:, None] * p[None, :]
    torch.testing.assert_close(observed, expected, atol=.005, rtol=.03)


def test_gradient_estimate_matches_analytic_row_fisher_and_ignores_common_shift():
    adapter = TinyAdapter(shared_shift=True)
    with DecisionFisherCollector(adapter, probes_per_record=1024, seed=4) as collector:
        collector.collect({"label": "this is deliberately ignored"})
    stat = collector.statistics()["projection"]
    # z=[0,0], so p=[.5,.5]. Row Fisher is .25*(head[0]-head[1])².
    torch.testing.assert_close(stat.mean_square, torch.tensor([.25, 1., 0.]), atol=.04, rtol=.15)
    assert stat.count == 1024
    assert collector.metadata()["uses_gold_labels"] is False
    assert collector.metadata()["backward_probes"] == 1024


def test_collection_restores_flags_modes_and_existing_parameter_gradients():
    adapter = TinyAdapter()
    adapter.model.train()
    adapter.model.head.eval()
    adapter.model.backbone.bias.requires_grad_(False)
    adapter.model.backbone.weight.grad = torch.full_like(adapter.model.backbone.weight, 7.)
    flags = [p.requires_grad for p in adapter.model.parameters()]
    modes = [m.training for m in adapter.model.modules()]
    saved_grad = adapter.model.backbone.weight.grad.clone()
    with torch.inference_mode():
        with DecisionFisherCollector(adapter) as collector:
            values = collector.collect({})
            assert all(not p.requires_grad for p in adapter.model.parameters())
            assert not values[0].requires_grad
    assert [p.requires_grad for p in adapter.model.parameters()] == flags
    assert [m.training for m in adapter.model.modules()] == modes
    assert not adapter.model.backbone._forward_hooks
    torch.testing.assert_close(adapter.model.backbone.weight.grad, saved_grad)
    assert adapter.model.head.weight.grad is None


def test_disconnected_forward_fails_and_restores_model():
    adapter = TinyAdapter(disconnected=True)
    original = [p.requires_grad for p in adapter.model.parameters()]
    with pytest.raises(RuntimeError, match="differentiable path"):
        with DecisionFisherCollector(adapter) as collector:
            collector.collect({})
    assert [p.requires_grad for p in adapter.model.parameters()] == original
    assert not adapter.model.backbone._forward_hooks
    assert collector.metadata()["autograd_fallback"] == "none_fail_loudly"
    assert collector.metadata()["failed_records"] == 1


def test_failed_request_does_not_commit_partial_gradient_statistics(monkeypatch):
    import s1q.decision_stats as decision_stats
    original_objective = decision_stats.decision_fisher_objective
    calls = 0

    def fail_second_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture backward failure")
        return original_objective(*args, **kwargs)

    monkeypatch.setattr(decision_stats, "decision_fisher_objective", fail_second_probe)
    with DecisionFisherCollector(TinyAdapter(), probes_per_record=2) as collector:
        with pytest.raises(RuntimeError, match="fixture backward"):
            collector.collect({})
    assert collector.records == 0
    assert collector.failed_records == 1
    assert collector.backward_probes == 1
    assert collector.statistics(require_all=False) == {}


def test_row_weights_are_mean_one_and_zero_estimate_falls_back_conservatively():
    stat = DecisionGradientStatistics(10, torch.tensor([0., 1., 3.]))
    weights = stat.row_weights(shrinkage=.1)
    assert weights.mean().item() == pytest.approx(1.)
    assert weights[0].item() == pytest.approx(.1)
    assert weights[2] > weights[1]
    assert DecisionGradientStatistics(10, torch.zeros(3)).row_weights().tolist() == [1., 1., 1.]


def test_seed_reproducibility_and_label_independence():
    values = []
    for label in ("A", "B"):
        adapter = TinyAdapter()
        with DecisionFisherCollector(adapter, probes_per_record=4, seed=8) as collector:
            collector.collect({"label": label})
        values.append(collector.statistics()["projection"].mean_square)
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)


def test_attach_requires_matching_module_names_and_preserves_input_stats():
    @dataclass
    class InputStat:
        mean_square: torch.Tensor
        output_mean_square: torch.Tensor | None = None

    inputs = {"projection": InputStat(torch.tensor([1., 2.]))}
    gradients = {"projection": DecisionGradientStatistics(4, torch.tensor([.1, .3, .6]))}
    combined = attach_input_statistics(inputs, gradients)
    torch.testing.assert_close(combined["projection"].output_mean_square, gradients["projection"].mean_square)
    assert inputs["projection"].output_mean_square is None
    with pytest.raises(ValueError, match="missing"):
        attach_input_statistics(inputs, {})


def test_multiple_linears_keep_the_original_gradient_chain():
    adapter = TinyAdapter()
    adapter.model.extra = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        adapter.model.extra.weight.copy_(2 * torch.eye(3))
    adapter.linear_modules = lambda: {"first": adapter.backbone, "second": adapter.model.extra}
    adapter.infer = lambda _: [adapter.model.head(adapter.model.extra(
        adapter.backbone(torch.tensor([[1., 2.]])))).reshape(-1)]
    with DecisionFisherCollector(adapter, probes_per_record=8) as collector:
        collector.collect({})
    stats = collector.statistics()
    # Detaching every hooked output would incorrectly erase this factor of four.
    torch.testing.assert_close(stats["first"].mean_square, 4 * stats["second"].mean_square)


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_invalid_temperatures_rejected(temperature):
    with pytest.raises(ValueError):
        decision_fisher_objective([torch.zeros(2)], generator=torch.Generator(), temperature=temperature)
