"""Teacher-distribution Fisher sensitivity for native decision outputs.

This collector uses no gold labels. A randomized centered projection estimates
the categorical Fisher metric at the full-precision teacher distribution. The
squared gradient at each Linear output channel weights that channel's local
reconstruction error. It is an approximation to output KL sensitivity, not a
guarantee of better accuracy, calibration or transfer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


def categorical_fisher_projection(probabilities: Tensor, noise: Tensor) -> Tensor:
    """Return v with E[v v^T] = diag(p) - p p^T for standard Gaussian noise.

    The last dimension holds options. Leading dimensions may batch independent
    questions/probes. This construction also removes the common-logit shift,
    whose gradient must have zero decision sensitivity.
    """
    if probabilities.shape != noise.shape or probabilities.ndim == 0:
        raise ValueError("probabilities and noise need matching option dimensions")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.isfinite(noise).all():
        raise ValueError("projection noise must be finite")
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError("probabilities must sum to one")
    a = probabilities.sqrt() * noise
    return a - probabilities * a.sum(-1, keepdim=True)


def decision_fisher_objective(logits: Sequence[Tensor], *, generator: torch.Generator,
                              temperature: float = 1.0) -> Tensor:
    """A differentiable scalar with unbiased categorical Fisher gradient covariance.

    Independent question projections are summed with 1/sqrt(Q), so their expected
    squared gradient represents mean per-question sensitivity rather than growing
    with the number of questions in a request. Teacher probabilities and random
    projection coefficients are detached from the gradient graph.
    """
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("nonempty logits and a finite positive temperature are required")
    terms = []
    for z in logits:
        if z.ndim != 1 or z.numel() == 0 or not torch.isfinite(z).all():
            raise ValueError("each question needs finite nonempty one-dimensional logits")
        scaled = z.float() / temperature
        p = torch.softmax(scaled.detach(), -1)
        # CPU RNG gives a stable seed sequence across GPU models and devices.
        noise = torch.randn(p.shape, generator=generator, device="cpu", dtype=torch.float32).to(p.device)
        v = categorical_fisher_projection(p, noise).detach()
        terms.append((scaled * v).sum())
    return torch.stack(terms).sum() / math.sqrt(len(terms))


@dataclass(frozen=True)
class DecisionGradientStatistics:
    """CPU moments of d(randomized decision objective)/d(linear output).

    ``count`` is flattened output token rows across backward probes. It includes
    zero-gradient padded rows. ``mean_square`` has one entry per weight output
    row, corresponding to the row dimension of a Linear weight matrix.
    """
    count: int
    mean_square: Tensor

    def validate(self, out_features: int) -> None:
        if self.count <= 0 or self.mean_square.shape != (out_features,):
            raise ValueError("decision statistics have invalid count or output width")
        if not torch.isfinite(self.mean_square).all() or (self.mean_square < 0).any():
            raise ValueError("decision gradient moments must be finite and nonnegative")

    def row_weights(self, shrinkage: float = 0.05) -> Tensor:
        """Mean-one row importance with optional shrinkage toward uniform weights.

        An entirely zero sensitivity estimate falls back to uniform weighting;
        it cannot justify discarding all local reconstruction errors. Shrinkage
        is a method hyperparameter and must be recorded and selected on dev data.
        """
        if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must lie in [0,1]")
        self.validate(self.mean_square.numel())
        mean = self.mean_square.mean()
        if mean <= 0:
            return torch.ones_like(self.mean_square)
        return (1 - shrinkage) * self.mean_square / mean + shrinkage


@dataclass
class _GradientAccumulator:
    count: int
    sum_square: Tensor


class DecisionFisherCollector:
    """Reversible gradient hooks for native, differentiable decision adapters.

    Parameters are temporarily frozen to avoid allocating model-weight gradients.
    Each observed Linear output starts/reuses a gradient graph without detaching
    an existing graph. The collector keeps no activations after a request.
    Original requires_grad flags and training modes are restored on every exit.
    Native forward implementations using inference_mode/no_grad internally cannot
    be used; their adapter must expose the differentiable raw forward path.
    """

    def __init__(self, adapter: Any, *, modules: Mapping[str, nn.Linear] | None = None,
                 seed: int = 0, temperature: float = 1.0, probes_per_record: int = 1):
        if isinstance(probes_per_record, bool) or not isinstance(probes_per_record, int) or probes_per_record <= 0:
            raise ValueError("probes_per_record must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.adapter = adapter
        self.modules = dict(modules if modules is not None else adapter.linear_modules())
        if not self.modules or any(not isinstance(m, nn.Linear) for m in self.modules.values()):
            raise ValueError("at least one native Linear module is required")
        self.temperature, self.probes_per_record, self.seed = temperature, probes_per_record, seed
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._handles: list[Any] = []
        self._parameters: list[tuple[nn.Parameter, bool]] = []
        self._training: list[tuple[nn.Module, bool]] = []
        self._accumulators: dict[str, _GradientAccumulator] = {}
        self._pending: dict[str, _GradientAccumulator] = {}
        self._active = False
        self._collecting = False
        self.records, self.backward_probes, self.failed_records = 0, 0, 0

    def _observe_gradient(self, name: str, width: int, gradient: Tensor) -> None:
        if not self._collecting:
            return
        rows = gradient.detach().reshape(-1, width).float()
        if rows.numel() == 0:
            return
        if not torch.isfinite(rows).all():
            raise ValueError(f"Nonfinite decision gradients for {name}")
        summed = rows.square().sum(0, dtype=torch.float64).cpu()
        accumulator = self._pending.get(name)
        if accumulator is None:
            accumulator = _GradientAccumulator(0, torch.zeros_like(summed))
            self._pending[name] = accumulator
        accumulator.count += rows.shape[0]
        accumulator.sum_square += summed

    def start(self) -> DecisionFisherCollector:
        if self._active:
            raise RuntimeError("DecisionFisherCollector is already active")
        self._active = True
        try:
            self._parameters = [(p, p.requires_grad) for p in self.adapter.model.parameters()]
            self._training = [(m, m.training) for m in self.adapter.model.modules()]
            for parameter, _ in self._parameters:
                parameter.requires_grad_(False)
            self.adapter.model.eval()
            for name, module in self.modules.items():
                def attach(current, args, output, layer_name=name):
                    if not self._collecting:
                        return None
                    if not torch.is_grad_enabled() or not isinstance(output, Tensor):
                        raise RuntimeError(f"Native forward disables autograd at {layer_name}")
                    if output.shape[-1] != current.out_features:
                        raise ValueError(f"Linear output width mismatch at {layer_name}")
                    if not output.requires_grad:
                        output.requires_grad_(True)
                    output.register_hook(lambda gradient, n=layer_name, w=current.out_features:
                                         self._observe_gradient(n, w, gradient))
                    return None
                self._handles.append(module.register_forward_hook(attach))
        except BaseException:
            self.close()
            raise
        return self

    def collect(self, record: Mapping[str, Any]) -> list[Tensor]:
        """Observe one calibration request, returning detached teacher logits."""
        if not self._active:
            raise RuntimeError("Use the collector as a context manager or call start()")
        if self._collecting:
            raise RuntimeError("Nested decision-statistic collection is unsupported")
        self._collecting = True
        self._pending = {}
        try:
            # Override an outer evaluation context only for this calibration pass.
            with torch.inference_mode(False), torch.enable_grad():
                logits = self.adapter.infer(record)
                if not logits or any(not z.requires_grad for z in logits):
                    raise RuntimeError("Native logits have no differentiable path to selected linears")
                for probe in range(self.probes_per_record):
                    objective = decision_fisher_objective(logits, generator=self._generator,
                                                          temperature=self.temperature)
                    objective.backward(retain_graph=probe + 1 < self.probes_per_record)
                    self.backward_probes += 1
                for name, update in self._pending.items():
                    accumulator = self._accumulators.get(name)
                    if accumulator is None:
                        self._accumulators[name] = update
                    else:
                        accumulator.count += update.count
                        accumulator.sum_square += update.sum_square
                self.records += 1
                return [z.detach() for z in logits]
        except BaseException:
            self.failed_records += 1
            raise
        finally:
            self._collecting = False
            self._pending = {}

    def statistics(self, *, require_all: bool = True) -> dict[str, DecisionGradientStatistics]:
        missing = set(self.modules) - set(self._accumulators)
        if require_all and missing:
            raise ValueError(f"No decision-gradient observations for selected layers: {sorted(missing)}")
        return {name: DecisionGradientStatistics(a.count, (a.sum_square / a.count).float().clone())
                for name, a in self._accumulators.items()}

    def attach_input_statistics(self, input_statistics: Mapping[str, Any], *, require_all: bool = True):
        """Attach raw output moments to quantization.InputStatistics by module name."""
        return attach_input_statistics(input_statistics, self.statistics(require_all=require_all),
                                       require_all=require_all)

    def metadata(self) -> dict[str, Any]:
        return {"objective": "categorical_teacher_fisher_random_projection",
                "seed": self.seed, "temperature": self.temperature,
                "probes_per_record": self.probes_per_record,
                "records": self.records, "backward_probes": self.backward_probes,
                "failed_records": self.failed_records,
                "question_normalization": "sum_div_sqrt_question_count",
                "gradient_reduction": "mean_square_over_output_token_rows",
                "uses_gold_labels": False,
                "autograd_fallback": "none_fail_loudly",
                "failed_request_statistics": "discarded_atomically"}

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
        self._collecting, self._active = False, False

    def __enter__(self) -> DecisionFisherCollector:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()


def collect_decision_statistics(adapter: Any, records: Iterable[Mapping[str, Any]], **kwargs):
    """Convenience function returning per-layer moments and collector metadata."""
    with DecisionFisherCollector(adapter, **kwargs) as collector:
        for record in records:
            collector.collect(record)
    return collector.statistics(), collector.metadata()


def attach_input_statistics(input_statistics: Mapping[str, Any],
                            decision_statistics: Mapping[str, DecisionGradientStatistics], *,
                            require_all: bool = True) -> dict[str, Any]:
    """Return replaced input-stat dataclasses with output_mean_square attached.

    Names must be relative to the same backbone in both collectors. Raw Fisher
    moments are retained; the quantizer performs its declared normalization and
    floor. Missing gradients raise by default rather than silently becoming a
    different method on unobserved layers.
    """
    missing = set(input_statistics) - set(decision_statistics)
    if require_all and missing:
        raise ValueError(f"Decision gradients missing for calibrated layers: {sorted(missing)}")
    result = {}
    for name, item in input_statistics.items():
        if name not in decision_statistics:
            result[name] = item
            continue
        gradient = decision_statistics[name]
        gradient.validate(gradient.mean_square.numel())
        result[name] = replace(item, output_mean_square=gradient.mean_square.clone())
    return result
