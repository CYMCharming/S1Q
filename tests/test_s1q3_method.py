"""Joint W/A search tests; model-accuracy claims require separate development data."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from s1q.quantization import (InputStatistics, fake_quantize_activation,
                              load_quantized_artifact, quantize_model,
                              quantize_weight)


def _stats(inputs):
    return InputStatistics(len(inputs), inputs.square().mean(0), inputs.abs().amax(0), inputs)


def _joint_error(inputs, weight, layer, bits):
    transformed = fake_quantize_activation(inputs / layer.input_scale, bits)
    return F.linear(transformed, layer.transformed_weight()).sub(F.linear(inputs, weight)).square().mean()


def test_s1q3_optimizes_the_actual_w4a4_linear_output_on_calibration_rows():
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(17, 12, generator=generator)
    inputs[:, 0] *= 12
    inputs[:, 1] *= 0.02
    weight = torch.randn(5, 12, generator=generator)
    statistics = _stats(inputs)
    old = quantize_weight(weight, bits=4, group_size=6, method="s1q",
                          statistics=statistics)
    joint = quantize_weight(weight, bits=4, group_size=6, method="s1q3",
                            statistics=statistics, activation_bits=4,
                            reservoir_blend=1, max_reservoir_rows=len(inputs))
    assert _joint_error(inputs, weight, joint, 4) <= _joint_error(inputs, weight, old, 4) + 1e-6
    assert joint.report()["selection_objective"] == "diagonal_and_joint_weight_activation_output_reconstruction"
    assert joint.report()["search_activation_bits"] == 4
    assert joint.report()["scale_family"] in ("rms", "mean_abs")
    assert joint.report()["search_reservoir_rows"] == len(inputs)


def test_s1q3_requires_a_precision_and_replays_packed_artifact():
    model = nn.Module()
    model.backbone = nn.Linear(6, 4)
    model.head = nn.Linear(4, 2)
    baseline = copy.deepcopy(model)
    inputs = torch.arange(1., 37.).reshape(6, 6) / 10
    statistics = _stats(inputs)
    with pytest.raises(ValueError, match="A4 or A8"):
        quantize_weight(model.backbone.weight, method="s1q3", statistics=statistics)
    with pytest.raises(ValueError, match="reservoir"):
        quantize_weight(model.backbone.weight, method="s1q3", activation_bits=4,
                        statistics=InputStatistics(len(inputs), inputs.square().mean(0), inputs.abs().amax(0)))
    session = quantize_model(model, method="s1q3", bits=4, group_size=3,
                             activation_bits=4, statistics={"backbone": statistics})
    expected = model.head(model.backbone(inputs)).detach()
    artifact = session.artifact()
    assert artifact["activation_bits"] == 4
    assert artifact["layers"]["backbone"]["scale_family"] in ("rms", "mean_abs")
    restored = load_quantized_artifact(baseline, artifact)
    torch.testing.assert_close(baseline.head(baseline.backbone(inputs)), expected)
    restored.restore()
    session.restore()
