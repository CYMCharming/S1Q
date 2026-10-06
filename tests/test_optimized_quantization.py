"""Optimized quantization replay, scope and fit/select separation checks."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

import s1q.optimized_quantization as optimized
from s1q.decision_margin import DecisionMarginStatistics
from s1q.quantization import (CalibrationCollector, InputStatistics,
                              fake_quantize_activation, load_quantized_artifact,
                              selected_linear_modules)


@pytest.fixture(scope="module", autouse=True)
def compact_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 5)
        self.norm = nn.LayerNorm(5)
        self.proj = nn.Linear(5, 7)
        self.gate = nn.Linear(7, 3)
        self.head = nn.Linear(3, 2, bias=False)

    def forward(self, inputs):
        return self.head(self.gate(torch.tanh(self.proj(self.norm(inputs)))))


def fixture_model():
    torch.manual_seed(62)
    model = TinyDecisionModel().eval()
    inputs = torch.randn(20, 5)
    inputs[:, 0] *= 5
    with CalibrationCollector(model, reservoir_size=16, seed=12) as collector:
        model(inputs)
    return model, inputs, collector.statistics()


@pytest.mark.parametrize("method", ["s1q-mac", "awq-mac", "gptq-mac",
                                    "spinquant-nohad-mac", "spinquant-had-mac"])
def test_exact_scope_no_extra_preservation_and_exception_safe_restore(method):
    model, inputs, statistics = fixture_model()
    originals = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before = model(inputs).detach()
    session, metadata = optimized.quantize_optimized_model(model, statistics=statistics,
        method=method, bits=4, group_size=4, activation_bits=4)
    assert set(session.layers) == set(selected_linear_modules(model)) == {"proj", "gate"}
    assert session.preserved_layers == {}
    assert set(metadata) == {"proj", "gate"}
    with pytest.raises(RuntimeError, match="fixture failure"):
        with session:
            assert model(inputs).shape == (20, 2)
            assert torch.isfinite(model(inputs)).all()
            for name in ("embedding.weight", "norm.weight", "norm.bias", "head.weight", "proj.bias", "gate.bias"):
                torch.testing.assert_close(model.state_dict()[name], originals[name], rtol=0, atol=0)
            assert model.proj._forward_pre_hooks and model.gate._forward_pre_hooks
            raise RuntimeError("fixture failure")
    for name, original in originals.items():
        torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)
    assert not model.proj._forward_pre_hooks and not model.gate._forward_pre_hooks
    assert not hasattr(model.proj, "_s1q_active_quantization")
    torch.testing.assert_close(model(inputs), before, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["s1q-mac", "awq-mac", "gptq-mac"])
@pytest.mark.parametrize("bits", [3, 4])
def test_packed_artifact_replays_compensated_a4_path_exactly(method, bits):
    model, inputs, statistics = fixture_model()
    replay = copy.deepcopy(model)
    native = replay(inputs).detach()
    session, _ = optimized.quantize_optimized_model(model, statistics=statistics,
        method=method, bits=bits, group_size=4, activation_bits=4)
    with session:
        expected = model(inputs).detach()
        artifact = session.artifact()
        assert artifact["activation_bits"] == 4
        assert artifact["preserved_layers"] == {}
        assert set(artifact["layers"]) == {"proj", "gate"}
        assert all(record["payload"].dtype == torch.uint8 for record in artifact["layers"].values())
    with load_quantized_artifact(replay, artifact):
        torch.testing.assert_close(replay(inputs), expected, rtol=0, atol=0)
    torch.testing.assert_close(replay(inputs), native, rtol=0, atol=0)


def test_compensation_receives_fit_half_only_and_selection_error_uses_holdout(monkeypatch):
    model, _, statistics = fixture_model()
    stat = statistics["proj"]
    fit, select = optimized._split_inputs(stat, "cpu")
    seen = []
    original = optimized.compensate_weight

    def record_fit(weight, inputs, scale, activation_bits, **kwargs):
        seen.append(inputs.detach().clone())
        return original(weight, inputs, scale, activation_bits, **kwargs)

    monkeypatch.setattr(optimized, "compensate_weight", record_fit)
    layer, metadata = optimized.quantize_mac_weight(model.proj.weight, statistics=stat,
        bits=4, group_size=4, activation_bits=4)
    assert seen
    for observed in seen:
        torch.testing.assert_close(observed, fit, rtol=0, atol=0)
    reference = torch.nn.functional.linear(select, model.proj.weight.detach().float())
    prediction = torch.nn.functional.linear(fake_quantize_activation(select / layer.input_scale, 4),
                                           layer.transformed_weight())
    score = (prediction - reference).square().mean() / reference.square().mean().clamp_min(1e-12)
    assert layer.relative_error == pytest.approx(score.item(), rel=1e-6)
    assert metadata["fit_token_rows"] == len(fit)
    assert metadata["select_token_rows"] == len(select)


def test_scale_candidate_fitting_uses_fit_rows_not_holdout_or_full_moments(monkeypatch):
    model, _, statistics = fixture_model()
    stat = statistics["proj"]
    fit, _ = optimized._split_inputs(stat, "cpu")
    # Deliberately poison unused full-reservoir moments; scale construction
    # should be derived from the fit inputs rather than these moments.
    stat = replace(stat, mean_square=torch.full_like(stat.mean_square, 1e6))
    observed_bases = []
    original = optimized._normalized_scale

    def record_candidate(magnitude, observed):
        observed_bases.append(magnitude.detach().clone())
        return original(magnitude, observed)

    monkeypatch.setattr(optimized, "_normalized_scale", record_candidate)
    optimized.quantize_mac_weight(model.proj.weight, statistics=stat, bits=4,
        group_size=4, activation_bits=4, compensation=False)
    expected_rms = fit.square().mean(0).sqrt().clamp_min(1e-8).pow(.25)
    torch.testing.assert_close(observed_bases[1], expected_rms, rtol=0, atol=0)
    expected_mean_abs = fit.abs().mean(0).clamp_min(1e-8).pow(.25)
    torch.testing.assert_close(observed_bases[6], expected_mean_abs, rtol=0, atol=0)


def test_spin_rotation_training_receives_fit_tokens_only(monkeypatch):
    model, _, statistics = fixture_model()
    received = []
    original = optimized.quantize_spinquant_proxy_weight

    def record_rotation(weight, **kwargs):
        received.append(kwargs["statistics"].reservoir.detach().clone())
        return original(weight, **kwargs)

    monkeypatch.setattr(optimized, "quantize_spinquant_proxy_weight", record_rotation)
    session, _ = optimized.quantize_optimized_model(model, statistics=statistics,
        method="spinquant-had-mac", bits=4, group_size=4, activation_bits=4)
    session.restore()
    for observed, (name, stat) in zip(received, statistics.items()):
        fit, _ = optimized._split_inputs(stat, "cpu")
        torch.testing.assert_close(observed, fit, rtol=0, atol=0)


def test_no_correction_candidate_remains_available_on_selection_holdout():
    model, _, statistics = fixture_model()
    uncorrected, _ = optimized.quantize_mac_weight(model.proj.weight,
        statistics=statistics["proj"], group_size=4, compensation=False)
    corrected, metadata = optimized.quantize_mac_weight(model.proj.weight,
        statistics=statistics["proj"], group_size=4, compensation=True)
    assert corrected.relative_error <= uncorrected.relative_error + 1e-8
    assert metadata["candidate_count"] == 44


def test_gptq_transformed_fit_statistics_accept_aligned_margin_subclass():
    model, _, statistics = fixture_model()
    plain = statistics["proj"]
    margin = DecisionMarginStatistics(count=plain.count, mean_square=plain.mean_square,
        absmax=plain.absmax, reservoir=plain.reservoir, reservoir_weights=torch.ones(len(plain.reservoir)),
        effective_weight_sum=float(plain.count), records=1)
    layer, metadata = optimized.quantize_gptq_mac_weight(model.proj.weight,
        statistics=margin, bits=4, group_size=4, activation_bits=4)
    assert layer.integer_weight.shape == model.proj.weight.shape
    assert layer.scales.shape == (7, 2)
    assert torch.isfinite(layer.transformed_weight()).all()
    assert metadata["fit_token_rows"] == len(plain.reservoir) // 2


@pytest.mark.parametrize("method", ["gptq-mac", "spinquant-nohad-mac", "spinquant-had-mac"])
def test_ragged_groups_and_rotation_reconstruct_correct_input_shape(method):
    model, inputs, statistics = fixture_model()
    session, _ = optimized.quantize_optimized_model(model, statistics=statistics,
        method=method, bits=3, group_size=4, activation_bits=4)
    with session:
        prediction = model(inputs)
        assert prediction.shape == (20, 2)
        assert torch.isfinite(prediction).all()
        for name, layer in session.layers.items():
            width = getattr(model, name).in_features
            assert layer.integer_weight.shape[1] == width
            assert layer.input_scale.shape == (width,)
            assert layer.effective_weight().shape == getattr(model, name).weight.shape
        if method.startswith("spinquant"):
            with pytest.raises(NotImplementedError, match="not represented"):
                session.artifact()


def test_scope_mismatch_and_active_session_fail_before_model_mutation():
    model, _, statistics = fixture_model()
    with pytest.raises(ValueError, match="scope exactly"):
        optimized.quantize_optimized_model(model, statistics={"proj": statistics["proj"]})
    session, _ = optimized.quantize_optimized_model(model, statistics=statistics, group_size=4)
    with session:
        current = model.proj.weight.detach().clone()
        with pytest.raises(RuntimeError, match="active quantization"):
            optimized.quantize_optimized_model(model, statistics=statistics)
        torch.testing.assert_close(model.proj.weight, current, rtol=0, atol=0)


def test_split_reservoir_rejects_insufficient_rows():
    with pytest.raises(ValueError, match="at least four"):
        optimized._split_inputs(InputStatistics(count=3, mean_square=torch.ones(2),
            absmax=torch.ones(2), reservoir=torch.ones(3, 2)), "cpu")
