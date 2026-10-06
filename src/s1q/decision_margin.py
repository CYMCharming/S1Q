"""Aligned decision-token calibration for post-training quantization.

Each native question contributes the gradient of its teacher top-one versus
runner-up logit margin. Questions are differentiated separately, so opposing
margin directions cannot cancel before squaring. Squared gradient norms give a
token importance aligned with the Linear input that produced that output.
This is a margin-Jacobian diagonal proxy, not the categorical Fisher matrix.

The teacher uses no gold labels. Within each request/layer, importance is
normalized, shrunk toward uniform and bounded. Request normalization means
margin normalization affects the relative contributions of multiple questions;
it does not introduce between-request confidence weighting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .quantization import InputStatistics, _get_input


@dataclass(frozen=True)
class DecisionMarginStatistics(InputStatistics):
    """Input moments with aligned importance and reservoir sampling identity.

    With ``weighted_priority`` sampling, plain reservoir reconstruction already
    emphasizes important tokens. Applying ``reservoir_weights`` a second time
    would intentionally add a second weighting, not an unbiased weighted-MSE
    estimate. With ``uniform_priority``, apply saved weights once to estimate
    the importance-weighted reconstruction objective.
    """
    reservoir_weights: Tensor | None = None
    reservoir_sampling: str = "weighted_priority"
    effective_weight_sum: float = 0.0
    records: int = 0

    def validate(self, in_features: int) -> None:
        super().validate(in_features)
        if self.reservoir_sampling not in ("weighted_priority", "uniform_priority"):
            raise ValueError("Unknown decision reservoir sampling policy.")
        if self.effective_weight_sum <= 0 or not math.isfinite(self.effective_weight_sum):
            raise ValueError("Decision statistics require a positive finite weight sum.")
        if self.records <= 0:
            raise ValueError("Decision statistics require at least one request.")
        if self.reservoir is None:
            if self.reservoir_weights is not None:
                raise ValueError("Reservoir weights require reservoir inputs.")
        elif self.reservoir_weights is None or self.reservoir_weights.shape != (self.reservoir.shape[0],):
            raise ValueError("Reservoir importance must align with input rows.")
        elif not torch.isfinite(self.reservoir_weights).all() or (self.reservoir_weights <= 0).any():
            raise ValueError("Reservoir importance must be finite and positive.")


def margin_token_weights(energy: Tensor, *, shrinkage: float = 0.5,
                         cap: float = 4.0) -> Tensor:
    """Return mean-one token weights in [shrinkage, cap].

    A bounded normalization is used rather than clipping then renormalizing,
    which can accidentally exceed the declared cap. Entirely zero decision
    sensitivity falls back to uniform importance.
    """
    if energy.ndim != 1 or not energy.numel() or not torch.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("Token gradient energy must be a nonempty finite nonnegative vector.")
    if not math.isfinite(shrinkage) or not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must lie in (0,1].")
    if not math.isfinite(cap) or cap < 1:
        raise ValueError("cap must be finite and at least one.")
    value = energy.detach().double()
    mean = value.mean()
    if mean <= 0 or shrinkage == 1 or cap == 1:
        return torch.ones_like(energy, dtype=torch.float32)
    raw = shrinkage + (1 - shrinkage) * value / mean
    # Find the multiplier that restores mean one under fixed lower/upper bounds.
    lower, upper = 0.0, 1.0
    while float((raw * upper).clamp(shrinkage, cap).mean()) < 1:
        upper *= 2
    for _ in range(50):
        midpoint = (lower + upper) / 2
        if float((raw * midpoint).clamp(shrinkage, cap).mean()) < 1:
            lower = midpoint
        else:
            upper = midpoint
    return (raw * ((lower + upper) / 2)).clamp(shrinkage, cap).float()


def teacher_margin_objective(logit: Tensor, *, margin_floor: float = 1.0) -> Tensor:
    """Detached teacher pair and scale; common-logit shifts have zero gradient."""
    if logit.ndim != 1 or logit.numel() < 2 or not torch.isfinite(logit).all():
        raise ValueError("Each decision requires at least two finite logits.")
    if not math.isfinite(margin_floor) or margin_floor <= 0:
        raise ValueError("margin_floor must be finite and positive.")
    indices = logit.detach().float().topk(2).indices
    margin = logit.float()[indices[0]] - logit.float()[indices[1]]
    return margin / (margin.detach().abs() + margin_floor)


@dataclass
class _Call:
    inputs: Tensor
    energy: Tensor


@dataclass
class _Accumulation:
    count: int
    weight_sum: float
    sum_square: Tensor
    absmax: Tensor
    reservoir: Tensor | None
    reservoir_weights: Tensor | None
    priorities: Tensor | None
    records: int


class DecisionMarginCollector:
    """Reversible, request-atomic aligned margin-gradient calibration hooks."""

    def __init__(self, adapter: Any, *, modules: Mapping[str, nn.Linear] | None = None,
                 reservoir_size: int = 128, seed: int = 0, margin_floor: float = 1.0,
                 shrinkage: float = 0.5, cap: float = 4.0,
                 reservoir_sampling: str = "weighted_priority"):
        if type(reservoir_size) is not int or reservoir_size < 0:
            raise ValueError("reservoir_size must be a nonnegative integer.")
        if not math.isfinite(margin_floor) or margin_floor <= 0:
            raise ValueError("margin_floor must be finite and positive.")
        if not math.isfinite(shrinkage) or not 0 < shrinkage <= 1 or not math.isfinite(cap) or cap < 1:
            raise ValueError("Require shrinkage in (0,1] and finite cap >= 1.")
        if reservoir_sampling not in ("weighted_priority", "uniform_priority"):
            raise ValueError("Unknown decision reservoir sampling policy.")
        self.adapter = adapter
        self.modules = dict(modules if modules is not None else adapter.linear_modules())
        if not self.modules or any(not isinstance(m, nn.Linear) for m in self.modules.values()):
            raise ValueError("At least one native Linear module is required.")
        self.reservoir_size, self.seed = reservoir_size, seed
        self.margin_floor, self.shrinkage, self.cap = margin_floor, shrinkage, cap
        self.reservoir_sampling = reservoir_sampling
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._handles: list[Any] = []
        self._parameters: list[tuple[nn.Parameter, bool]] = []
        self._training: list[tuple[nn.Module, bool]] = []
        self._calls: dict[str, list[_Call]] = {}
        self._accumulators: dict[str, _Accumulation] = {}
        self._active = self._collecting = False
        self.records = self.questions = self.backward_probes = self.failed_records = 0

    def start(self) -> DecisionMarginCollector:
        if self._active:
            raise RuntimeError("DecisionMarginCollector is already active.")
        self._active = True
        try:
            self._parameters = [(p, p.requires_grad) for p in self.adapter.model.parameters()]
            self._training = [(m, m.training) for m in self.adapter.model.modules()]
            for parameter, _ in self._parameters:
                parameter.requires_grad_(False)
            self.adapter.model.eval()
            for name, module in self.modules.items():
                def observe(current, args, kwargs, output, layer_name=name):
                    if not self._collecting:
                        return
                    if not torch.is_grad_enabled() or not isinstance(output, Tensor):
                        raise RuntimeError(f"Native forward disables autograd at {layer_name}.")
                    value = _get_input(args, kwargs)
                    if value.shape[-1] != current.in_features or output.shape[-1] != current.out_features:
                        raise ValueError(f"Linear width mismatch at {layer_name}.")
                    rows = value.detach().reshape(-1, current.in_features)
                    if rows.shape[0] != output.numel() // current.out_features or not torch.isfinite(rows).all():
                        raise ValueError(f"Nonfinite or unaligned Linear inputs at {layer_name}.")
                    call = _Call(rows, torch.zeros(rows.shape[0], device=rows.device, dtype=torch.float64))
                    self._calls.setdefault(layer_name, []).append(call)
                    if not output.requires_grad:
                        output.requires_grad_(True)

                    def gradient_hook(gradient, captured=call, width=current.out_features):
                        if not self._collecting:
                            return
                        gradients = gradient.detach().reshape(-1, width).float()
                        if gradients.shape[0] != captured.inputs.shape[0] or not torch.isfinite(gradients).all():
                            raise ValueError(f"Nonfinite or unaligned margin gradients at {layer_name}.")
                        captured.energy.add_(gradients.double().square().sum(dim=1))

                    output.register_hook(gradient_hook)
                self._handles.append(module.register_forward_hook(observe, with_kwargs=True))
        except BaseException:
            self.close()
            raise
        return self

    def _update(self, calls: list[_Call], previous: _Accumulation | None) -> _Accumulation:
        inputs = torch.cat([c.inputs.float().cpu() for c in calls], dim=0)
        energy = torch.cat([c.energy.cpu() for c in calls])
        if not inputs.shape[0]:
            raise ValueError("Selected Linear received no token rows.")
        weights = margin_token_weights(energy, shrinkage=self.shrinkage, cap=self.cap)
        squares = (inputs.double().square() * weights.double()[:, None]).sum(dim=0)
        maximum = inputs.abs().amax(dim=0)
        count, mass, records = inputs.shape[0], float(weights.double().sum()), 1
        selected = selected_weights = selected_priorities = None
        if self.reservoir_size:
            # Exponential random races implement weighted sampling without
            # replacement. Lower -log(U)/weight priorities are retained.
            uniform = torch.rand(count, generator=self._generator, dtype=torch.float64).clamp_min(1e-30)
            priority = -uniform.log()
            if self.reservoir_sampling == "weighted_priority":
                priority /= weights.double()
            keep = min(count, self.reservoir_size)
            selected_priorities, indices = priority.topk(keep, largest=False)
            selected, selected_weights = inputs[indices], weights[indices]
            if previous is not None and previous.reservoir is not None:
                selected = torch.cat((previous.reservoir, selected))
                selected_weights = torch.cat((previous.reservoir_weights, selected_weights))
                selected_priorities = torch.cat((previous.priorities, selected_priorities))
                keep = min(self.reservoir_size, selected.shape[0])
                selected_priorities, indices = selected_priorities.topk(keep, largest=False)
                selected, selected_weights = selected[indices], selected_weights[indices]
        if previous is not None:
            count += previous.count
            mass += previous.weight_sum
            records += previous.records
            squares += previous.sum_square
            maximum = torch.maximum(maximum, previous.absmax)
        return _Accumulation(count, mass, squares, maximum, selected,
                             selected_weights, selected_priorities, records)

    def collect(self, record: Mapping[str, Any]) -> list[Tensor]:
        """Use native teacher logits only; supplied dataset labels are ignored."""
        if not self._active:
            raise RuntimeError("Use the collector as a context manager or call start().")
        if self._collecting:
            raise RuntimeError("Nested decision-margin collection is unsupported.")
        self._collecting, self._calls = True, {}
        random_state = self._generator.get_state()
        try:
            with torch.inference_mode(False), torch.enable_grad():
                logits = self.adapter.infer(record)
                if not logits or any(not z.requires_grad for z in logits):
                    raise RuntimeError("Native logits have no differentiable path to selected linears.")
                objectives = [teacher_margin_objective(z, margin_floor=self.margin_floor) for z in logits]
                for index, objective in enumerate(objectives):
                    objective.backward(retain_graph=index + 1 < len(objectives))
                    self.backward_probes += 1
            updates = {name: self._update(calls, self._accumulators.get(name))
                       for name, calls in self._calls.items()}
            self._accumulators.update(updates)
            self.records += 1
            self.questions += len(logits)
            return [z.detach() for z in logits]
        except BaseException:
            self._generator.set_state(random_state)
            self.failed_records += 1
            raise
        finally:
            self._collecting, self._calls = False, {}

    def statistics(self, *, require_all: bool = True) -> dict[str, DecisionMarginStatistics]:
        missing = set(self.modules) - set(self._accumulators)
        if require_all and missing:
            raise ValueError(f"Selected layers were never observed: {sorted(missing)}")
        result = {}
        for name, item in self._accumulators.items():
            statistics = DecisionMarginStatistics(
                count=item.count, mean_square=(item.sum_square / item.weight_sum).float().clone(),
                absmax=item.absmax.clone(), reservoir=item.reservoir.clone() if item.reservoir is not None else None,
                reservoir_weights=item.reservoir_weights.clone() if item.reservoir_weights is not None else None,
                reservoir_sampling=self.reservoir_sampling, effective_weight_sum=item.weight_sum,
                records=item.records)
            statistics.validate(self.modules[name].in_features)
            result[name] = statistics
        return result

    def metadata(self) -> dict[str, Any]:
        return {"objective": "exact_teacher_top1_runnerup_margin_jacobian",
                "uses_gold_labels": False, "margin_floor": self.margin_floor,
                "question_gradient_combination": "sum_squared_norms_after_separate_backward",
                "importance_alignment": "same_forward_call_input_and_output_gradient_token",
                "token_weight_normalization": "mean_one_per_request_per_layer_bounded",
                "shrinkage": self.shrinkage, "token_weight_cap": self.cap,
                "reservoir_size": self.reservoir_size, "reservoir_sampling": self.reservoir_sampling,
                "seed": self.seed, "records": self.records, "questions": self.questions,
                "backward_probes": self.backward_probes, "failed_records": self.failed_records,
                "failed_request_statistics": "discarded_atomically_with_rng_restoration",
                "all_zero_gradient_fallback": "uniform_token_weights",
                "autograd_fallback": "none_fail_loudly",
                "is_categorical_fisher": False}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for parameter, requires_grad in self._parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in self._training:
            module.training = training
        self._parameters.clear()
        self._training.clear()
        self._calls.clear()
        self._active = self._collecting = False

    def __enter__(self) -> DecisionMarginCollector:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()
