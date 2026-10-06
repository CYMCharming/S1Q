"""Public calibration-and-quantization entry point for the current S1Q."""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from .decision_margin import DecisionMarginCollector
from .optimized_quantization import quantize_optimized_model
from .quantization import selected_linear_modules


def quantize_s1q(adapter, calibration_records: Iterable[dict], *, bits=4,
                 activation_bits=4, group_size=128, reservoir_size=128,
                 seed=20261004):
    """Apply S1Q to an evaluation-mode native adapter and return its session.

    Only native-admitted calibration requests belong here. Gold labels and
    targets are stripped from independent copies before sensitivity collection.
    The returned session is already active; use it as a context manager or call
    ``restore()``. Calibration performs backward passes for token importance;
    inference remains floating-point with weight/activation QDQ.
    """
    if adapter.model.training:
        raise ValueError("Put the native model in eval mode before S1Q calibration")
    count = 0
    with DecisionMarginCollector(adapter, modules=selected_linear_modules(adapter.backbone),
                                 reservoir_size=reservoir_size, seed=seed) as collector:
        for record in calibration_records:
            request = deepcopy(record)
            for question in request["questions"].values():
                question.pop("label", None)
                question.pop("target", None)
            collector.collect(request)
            count += 1
    if count == 0:
        raise ValueError("S1Q calibration requires at least one admitted request")
    session, layer_metadata = quantize_optimized_model(
        adapter.backbone, method="s1q", statistics=collector.statistics(),
        bits=bits, group_size=group_size, activation_bits=activation_bits,
        compensation=True)
    return session, {"method": "S1Q", "calibration": collector.metadata(),
                     "layers": layer_metadata,
                     "activation_execution": "floating QDQ; no integer kernel"}
