"""Reversible research quantization for typed decision models.

RTN is groupwise symmetric round-to-nearest. The proposed S1Q weight search
combines AWQ-style diagonal input/weight compensation with activation-second-
moment-weighted clipping. This module does *not* implement integer GEMM: weights
are dequantized into the model's floating point dtype and optional A4/A8 is
simulated by an input hook. Packed artifacts measure storage, not speed.

Typical use::

    with CalibrationCollector(model, include_prefixes=("backbone",)) as collector:
        for batch in calibration_batches:
            model(**batch)
    stats = collector.statistics()
    session = quantize_model(model, method="s1q", bits=4, statistics=stats,
                             include_prefixes=("backbone",))
    # Evaluate, then restore before another candidate.
    session.restore()

Calibration data and global decision-level model selection belong in the
evaluation pipeline; they must be disjoint from the final test split.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bitpack import pack_signed_codes, unpack_signed_codes


DEFAULT_EXCLUDE_PATTERNS = (r"(^|\.)([^.]*head|classifier|pooler)(\.|$)",)
_ACTIVE_ATTRIBUTE = "_s1q_quantization_active"
_EXECUTION = "dequantized_weights_with_optional_fake_activation"


def selected_linear_modules(
    model: nn.Module,
    *,
    include_prefixes: Sequence[str] | None = None,
    preserve_prefixes: Sequence[str] = (),
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> dict[str, nn.Linear]:
    """Select linears by module path; heads are preserved by default.

    Prefixes match a complete module path or its descendants. Pass
    ``exclude_patterns=()`` explicitly to include classifier/decision heads.
    The root module has the empty-string name and can itself be a Linear.
    """
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in exclude_patterns]

    def matches(name: str, prefix: str) -> bool:
        return not prefix or name == prefix or name.startswith(prefix + ".")

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (include_prefixes is None or any(matches(name, p) for p in include_prefixes))
        and not any(matches(name, p) for p in preserve_prefixes)
        and not any(pattern.search(name) for pattern in patterns)
    }


def _validate_unshared_weights(model: nn.Module, selected: Mapping[str, nn.Linear]) -> None:
    # Mutating a shared Parameter would also mutate an excluded head/embedding;
    # distinct input scales on tied layers would invalidate compensation.
    owners: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, nn.Parameter):
            owners.setdefault(id(weight), []).append(name)
    for name, module in selected.items():
        aliases = owners[id(module.weight)]
        if len(aliases) > 1:
            raise ValueError(
                f"Selected layer {name!r} shares its weight with {aliases}; "
                "preserve these modules or untie weights explicitly before quantizing."
            )


@dataclass(frozen=True)
class InputStatistics:
    """CPU calibration moments; count is flattened input rows, not examples."""

    count: int
    mean_square: Tensor
    absmax: Tensor
    reservoir: Tensor | None = None
    output_mean_square: Tensor | None = None

    def validate(self, in_features: int) -> None:
        if self.count <= 0:
            raise ValueError("Calibration statistics require at least one input row.")
        for name, value in (("mean_square", self.mean_square), ("absmax", self.absmax)):
            if value.shape != (in_features,):
                raise ValueError(f"{name} must have shape ({in_features},), got {tuple(value.shape)}.")
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must contain finite, nonnegative values.")
        if self.reservoir is not None:
            if self.reservoir.ndim != 2 or self.reservoir.shape[1] != in_features:
                raise ValueError("Calibration reservoir has an incompatible input width.")
            if not torch.isfinite(self.reservoir).all():
                raise ValueError("Calibration reservoir must be finite.")


@dataclass
class _Accumulator:
    count: int
    sum_square: Tensor
    absmax: Tensor
    reservoir: Tensor | None = None
    priorities: Tensor | None = None


def _get_input(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Tensor:
    value = args[0] if args else kwargs.get("input")
    if not isinstance(value, Tensor):
        raise TypeError("Selected Linear must receive a Tensor as its input.")
    return value


class CalibrationCollector:
    """Capture Linear input moments without changing the model computation.

    A bounded reservoir is sampled using random priorities, so every flattened
    input row has equal inclusion probability. Statistics include padded tokens
    unless the caller excludes them before forwarding; document that choice.
    ``reservoir_size=0`` disables storage of input rows entirely.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        include_prefixes: Sequence[str] | None = None,
        preserve_prefixes: Sequence[str] = (),
        exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
        reservoir_size: int = 128,
        seed: int = 0,
    ) -> None:
        if reservoir_size < 0:
            raise ValueError("reservoir_size must be nonnegative.")
        self.modules = selected_linear_modules(
            model, include_prefixes=include_prefixes,
            preserve_prefixes=preserve_prefixes, exclude_patterns=exclude_patterns,
        )
        self.reservoir_size = reservoir_size
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._accumulators: dict[str, _Accumulator] = {}
        self._handles: list[Any] = []
        self._started = False

    def start(self) -> CalibrationCollector:
        if self._started:
            raise RuntimeError("CalibrationCollector is already active.")
        self._started = True
        for name, module in self.modules.items():
            def observe(
                current: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any],
                layer_name: str = name,
            ) -> None:
                value = _get_input(args, kwargs)
                if value.shape[-1] != current.in_features:
                    raise ValueError(f"Calibration input width mismatch for {layer_name!r}.")
                rows = value.detach().reshape(-1, current.in_features).float()
                if rows.shape[0] == 0:
                    return
                if not torch.isfinite(rows).all():
                    raise ValueError(f"Nonfinite calibration inputs for {layer_name!r}.")
                sum_square = rows.square().sum(dim=0, dtype=torch.float64).cpu()
                absmax = rows.abs().amax(dim=0).cpu()
                accumulator = self._accumulators.get(layer_name)
                if accumulator is None:
                    accumulator = _Accumulator(0, torch.zeros_like(sum_square), torch.zeros_like(absmax))
                    self._accumulators[layer_name] = accumulator
                accumulator.count += rows.shape[0]
                accumulator.sum_square += sum_square
                accumulator.absmax = torch.maximum(accumulator.absmax, absmax)
                if self.reservoir_size:
                    priorities = torch.rand(rows.shape[0], generator=self._generator)
                    take = min(self.reservoir_size, rows.shape[0])
                    local_priorities, indices = priorities.topk(take)
                    selected = rows.index_select(0, indices.to(rows.device)).cpu()
                    if accumulator.reservoir is not None:
                        selected = torch.cat((accumulator.reservoir, selected), dim=0)
                        local_priorities = torch.cat((accumulator.priorities, local_priorities))
                    keep = min(self.reservoir_size, selected.shape[0])
                    accumulator.priorities, indices = local_priorities.topk(keep)
                    accumulator.reservoir = selected.index_select(0, indices)

            self._handles.append(module.register_forward_pre_hook(observe, with_kwargs=True))
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._started = False

    def statistics(self, *, require_all: bool = True) -> dict[str, InputStatistics]:
        missing = set(self.modules) - set(self._accumulators)
        if require_all and missing:
            raise ValueError(f"Selected layers were never calibrated: {sorted(missing)}")
        return {
            name: InputStatistics(
                count=a.count,
                mean_square=(a.sum_square / a.count).float().clone(),
                absmax=a.absmax.clone(),
                reservoir=a.reservoir.clone() if a.reservoir is not None else None,
            )
            for name, a in self._accumulators.items()
        }

    def __enter__(self) -> CalibrationCollector:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()


def _validate_bits(bits: int) -> None:
    if bits not in (2, 3, 4, 8):
        raise ValueError("Only signed symmetric 2-, 3-, 4- and 8-bit weight quantization is supported.")


def fake_quantize_activation(value: Tensor, bits: int) -> Tensor:
    """Per-token dynamic symmetric fake quantization across the last dimension.

    The positive/negative range is +/-7 for A4 and +/-127 for A8; no integer
    activation kernel or latency improvement is implied by this operation.
    """
    if bits not in (4, 8):
        raise ValueError("Only 4-bit and 8-bit activation QDQ is supported.")
    if not value.is_floating_point() or value.ndim == 0:
        raise ValueError("Activations must be a floating Tensor with a feature dimension.")
    if value.numel() == 0:
        return value.clone()
    floating = value.float()
    qmax = (1 << (bits - 1)) - 1
    maximum = floating.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    quantized = (floating / scale).round().clamp(-qmax, qmax)
    return (quantized * scale).to(value.dtype)


def _groupwise_quantize(weight: Tensor, bits: int, group_size: int, clipping_ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return unpadded integer codes, scales, and transformed dequantized weights."""
    width = weight.shape[1]
    groups = math.ceil(width / group_size)
    padding = groups * group_size - width
    grouped = F.pad(weight, (0, padding)).reshape(weight.shape[0], groups, group_size)
    maximum = grouped.abs().amax(dim=-1, keepdim=True) * clipping_ratio
    qmax = (1 << (bits - 1)) - 1
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    integer = (grouped / scale).round().clamp(-qmax, qmax).to(torch.int8)
    dequantized = (integer.float() * scale).reshape(weight.shape[0], -1)[:, :width]
    integer = integer.reshape(weight.shape[0], -1)[:, :width].contiguous()
    return integer, scale.squeeze(-1), dequantized


def dequantize_groupwise(integer_weight: Tensor, scales: Tensor, group_size: int) -> Tensor:
    """Reconstruct a floating [out_features, in_features] weight including remainder groups."""
    if integer_weight.ndim != 2 or group_size <= 0:
        raise ValueError("Expected a 2D integer weight and a positive group_size.")
    out_features, width = integer_weight.shape
    groups = math.ceil(width / group_size)
    if scales.shape != (out_features, groups):
        raise ValueError("Scale shape is incompatible with weight shape and group_size.")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Quantization scales must be finite and positive.")
    expanded = scales.repeat_interleave(group_size, dim=1)[:, :width]
    return integer_weight.float() * expanded.to(integer_weight.device)


@dataclass(frozen=True)
class QuantizedLinear:
    """CPU codes and scales plus diagnostics for one compensated Linear."""

    integer_weight: Tensor
    scales: Tensor
    input_scale: Tensor
    bits: int
    group_size: int
    method: str
    alpha: float
    clipping_ratio: float
    weighted_error: float
    relative_error: float
    calibration_rows: int
    reservoir_relative_error: float | None = None
    reservoir_blend: float | None = None
    search_reservoir_rows: int | None = None
    scale_family: str | None = None
    search_activation_bits: int | None = None

    def transformed_weight(self, *, device: Any = None, dtype: torch.dtype = torch.float32) -> Tensor:
        # W' is used with x' = x / input_scale.
        integer = self.integer_weight.to(device=device)
        scales = self.scales.to(device=device)
        return dequantize_groupwise(integer, scales, self.group_size).to(dtype)

    def effective_weight(self, *, device: Any = None, dtype: torch.dtype = torch.float32) -> Tensor:
        return (self.transformed_weight(device=device) / self.input_scale.to(device=device)).to(dtype)

    def report(self) -> dict[str, Any]:
        report = {
            "method": self.method, "bits": self.bits, "group_size": self.group_size,
            "shape": list(self.integer_weight.shape), "alpha": self.alpha,
            "clipping_ratio": self.clipping_ratio,
            "weighted_error": self.weighted_error, "relative_error": self.relative_error,
            "calibration_rows": self.calibration_rows,
            "reservoir_relative_error": self.reservoir_relative_error,
        }
        if self.method == "s1q2":
            report["selection_objective"] = "diagonal_and_empirical_output_reconstruction"
            report["reservoir_blend"] = self.reservoir_blend
            report["search_reservoir_rows"] = self.search_reservoir_rows
        if self.method == "s1q3":
            report["selection_objective"] = "diagonal_and_joint_weight_activation_output_reconstruction"
            report["reservoir_blend"] = self.reservoir_blend
            report["search_reservoir_rows"] = self.search_reservoir_rows
            report["scale_family"] = self.scale_family
            report["search_activation_bits"] = self.search_activation_bits
        return report


@torch.no_grad()
def quantize_weight(
    weight: Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    method: str = "rtn",
    statistics: InputStatistics | None = None,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    clipping_ratios: Sequence[float] = (0.9, 0.95, 1.0),
    reservoir_blend: float = 0.5,
    max_reservoir_rows: int = 32,
    activation_bits: int | None = None,
) -> QuantizedLinear:
    """Select compensated groupwise weights by a local reconstruction objective.

    S1Q searches s_j = RMS(x_j)**alpha (normalized to unit geometric midrange)
    and clipping ratios. Candidate error is sum_j E[x_j**2] * (Wq_j-W_j)**2,
    where Wq is the *effective* weight after input compensation. Alpha=0 and
    clipping=1 recover RTN. This is a local surrogate, not a decision-quality
    guarantee; select the global configuration on held-out development decisions.
    Opt-in ``s1q2`` uses the same candidates and weight format, but blends this
    diagonal loss with empirical Linear output reconstruction on up to
    ``max_reservoir_rows`` sampled calibration inputs. It captures input-channel
    correlations without allocating a full Hessian. ``reservoir_blend=0``
    recovers the original local search; the empirical term can overfit a small
    reservoir and must be tested on disjoint decisions. Opt-in ``s1q3`` adds
    mean-absolute-activation scaling candidates and searches the actual joint
    weight/activation QDQ Linear output error for A4/A8. Its extra search cost
    and possible calibration overfit must be measured separately.
    """
    _validate_bits(bits)
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer.")
    if weight.ndim != 2 or not weight.is_floating_point() or min(weight.shape) <= 0:
        raise ValueError("Expected a nonempty 2D floating weight.")
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize nonfinite weights.")
    method = method.lower()
    if method not in ("rtn", "s1q", "s1q2", "s1q3"):
        raise ValueError("method must be 'rtn', 's1q', 's1q2' or 's1q3'.")
    if statistics is not None:
        statistics.validate(weight.shape[1])
    if method in ("s1q", "s1q2", "s1q3") and statistics is None:
        raise ValueError("S1Q requires calibration input statistics for each quantized layer.")
    if method in ("s1q", "s1q2", "s1q3"):
        if not alphas or any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas):
            raise ValueError("alphas must be a nonempty sequence in [0, 1].")
        if not clipping_ratios or any(not math.isfinite(c) or not 0 < c <= 1 for c in clipping_ratios):
            raise ValueError("clipping_ratios must be a nonempty sequence in (0, 1].")
    else:
        alphas, clipping_ratios = (0.0,), (1.0,)
    if method in ("s1q2", "s1q3"):
        if not math.isfinite(reservoir_blend) or not 0 <= reservoir_blend <= 1:
            raise ValueError("reservoir_blend must be finite and in [0, 1].")
        if isinstance(max_reservoir_rows, bool) or not isinstance(max_reservoir_rows, int) or max_reservoir_rows <= 0:
            raise ValueError("max_reservoir_rows must be a positive integer.")
        if (reservoir_blend > 0 or method == "s1q3") and (statistics is None or statistics.reservoir is None or statistics.reservoir.shape[0] == 0):
            raise ValueError("Empirical reconstruction requires a nonempty calibration reservoir.")
    if method == "s1q3" and activation_bits not in (4, 8):
        raise ValueError("S1Q3 joint W/A search requires A4 or A8 activation QDQ.")

    floating = weight.detach().float()
    importance = (
        statistics.mean_square.to(device=weight.device, dtype=torch.float32)
        if statistics is not None else torch.ones(weight.shape[1], device=weight.device)
    )
    # Normalizing moments changes neither candidate ranking nor relative error.
    importance = importance / importance.mean().clamp_min(1e-30)
    row_importance = torch.ones(weight.shape[0],device=weight.device)
    if statistics is not None and statistics.output_mean_square is not None:
        fisher = statistics.output_mean_square
        if fisher.shape != (weight.shape[0],) or not torch.isfinite(fisher).all() or (fisher < 0).any():
            raise ValueError("Output-gradient statistics must be finite nonnegative per-output-channel values")
        if fisher.max() > 0:
            row_importance = (fisher.to(device=weight.device,dtype=torch.float32) / fisher.mean().clamp_min(1e-30)).clamp_min(1e-4)
    denominator = (floating.square() * importance * row_importance[:,None]).sum(dtype=torch.float64)
    search_inputs = None
    reservoir_reference = None
    if method in ("s1q2", "s1q3") and reservoir_blend > 0:
        # A bounded, uniformly sampled subset keeps candidate GEMMs feasible for
        # wide layers. The diagonal term regularizes the noisy empirical estimate.
        search_inputs = statistics.reservoir[:max_reservoir_rows].to(device=weight.device, dtype=torch.float32)
        reference_output = F.linear(search_inputs, floating)
        reservoir_reference = (reference_output.square() * row_importance).mean(dtype=torch.float64).clamp_min(1e-30)
    rms = importance.sqrt()
    scale_candidates = [("rms", alpha, rms) for alpha in alphas]
    if method == "s1q3":
        # AWQ-style mean absolute activation scaling is a second family of
        # candidates, searched under the same signed group format and A QDQ.
        mean_abs = statistics.reservoir[:max_reservoir_rows].to(
            device=weight.device, dtype=torch.float32).abs().mean(dim=0)
        scale_candidates.extend(("mean_abs", step / 20, mean_abs) for step in range(1, 20))
    best: tuple[float, Tensor, Tensor, Tensor, float, float, str] | None = None
    for scale_family, alpha, scale_base in scale_candidates:
        if alpha == 0:
            input_scale = torch.ones_like(rms)
        else:
            input_scale = scale_base.clamp_min(1e-8).pow(alpha)
            observed_scale = input_scale[importance > 0]
            if observed_scale.numel():
                normalization = (observed_scale.amin() * observed_scale.amax()).sqrt().clamp_min(1e-12)
                input_scale = input_scale / normalization
            input_scale = input_scale.clamp(1e-4, 1e4)
            # Unobserved channels have no trustworthy scale estimate.
            input_scale = torch.where(importance > 0, input_scale, torch.ones_like(input_scale))
        transformed = floating * input_scale
        for ratio in clipping_ratios:
            integer, scales, dequantized = _groupwise_quantize(transformed, bits, group_size, ratio)
            effective = dequantized / input_scale
            error = ((effective - floating).square() * importance * row_importance[:,None]).sum(dtype=torch.float64).item()
            if search_inputs is not None:
                if method == "s1q3":
                    # Runtime applies x/s, dynamic activation QDQ, then Wq*s.
                    # The reference output remains the native unquantized Linear.
                    quantized_input = fake_quantize_activation(
                        search_inputs / input_scale, activation_bits)
                    residual_output = F.linear(quantized_input, dequantized) - reference_output
                else:
                    residual_output = F.linear(search_inputs, effective - floating)
                empirical_error = (residual_output.square() * row_importance).mean(dtype=torch.float64)
                error = (1 - reservoir_blend) * error + reservoir_blend * (
                    empirical_error / reservoir_reference * denominator
                ).item()
            # Prefer less intervention on tied surrogate error, including wholly
            # unobserved channels. This recovers RTN when every candidate ties.
            if best is None or error < best[0] or (
                error == best[0] and (float(alpha), -float(ratio)) < (best[4], -best[5])
            ):
                best = (error, integer, scales, input_scale, float(alpha), float(ratio), scale_family)
    assert best is not None
    error, integer, scales, input_scale, alpha, ratio, scale_family = best
    reservoir_error = None
    if statistics is not None and statistics.reservoir is not None and statistics.reservoir.shape[0]:
        inputs = statistics.reservoir.to(weight.device)
        effective = dequantize_groupwise(integer, scales, group_size) / input_scale
        difference = F.linear(inputs, effective - floating).square().mean().item()
        reference = F.linear(inputs, floating).square().mean().item()
        reservoir_error = difference / reference if reference > 0 else 0.0
    return QuantizedLinear(
        integer_weight=integer.cpu(), scales=scales.cpu(), input_scale=input_scale.cpu(),
        bits=bits, group_size=group_size, method=method, alpha=alpha,
        clipping_ratio=ratio, weighted_error=error,
        relative_error=error / denominator.item() if denominator.item() > 0 else 0.0,
        calibration_rows=statistics.count if statistics is not None else 0,
        reservoir_relative_error=reservoir_error,
        reservoir_blend=reservoir_blend if method in ("s1q2", "s1q3") else None,
        search_reservoir_rows=search_inputs.shape[0] if search_inputs is not None else 0 if method in ("s1q2", "s1q3") else None,
        scale_family=scale_family if method == "s1q3" else None,
        search_activation_bits=activation_bits if method == "s1q3" else None,
    )


class _InputTransform:
    def __init__(self, input_scale: Tensor, activation_bits: int | None) -> None:
        self.input_scale = input_scale
        self.activation_bits = activation_bits
        self._cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    def __call__(self, _: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        value = _get_input(args, kwargs)
        key = (value.device, value.dtype)
        if key not in self._cache:
            self._cache[key] = self.input_scale.to(device=value.device, dtype=value.dtype)
        transformed = value / self._cache[key]
        if self.activation_bits is not None:
            transformed = fake_quantize_activation(transformed, self.activation_bits)
        if args:
            return (transformed, *args[1:]), kwargs
        return args, {**kwargs, "input": transformed}


def pack_int4(integer_weight: Tensor) -> Tensor:
    """Pack signed [-8, 7] codes into uint8; earlier element uses the low nibble."""
    if integer_weight.is_floating_point() or integer_weight.is_complex():
        raise ValueError("int4 packing requires integer codes.")
    flat = integer_weight.detach().to(device="cpu", dtype=torch.int64).flatten()
    if ((flat < -8) | (flat > 7)).any():
        raise ValueError("int4 codes must be in [-8, 7].")
    if flat.numel() % 2:
        flat = F.pad(flat, (0, 1))
    unsigned = flat & 15
    return (unsigned[0::2] | (unsigned[1::2] << 4)).to(torch.uint8)


def unpack_int4(packed: Tensor, numel: int) -> Tensor:
    """Unpack exactly numel signed codes, ignoring an odd-element padding nibble."""
    if packed.dtype != torch.uint8 or packed.ndim != 1 or numel < 0:
        raise ValueError("Expected a flat uint8 payload and nonnegative numel.")
    if packed.numel() != math.ceil(numel / 2):
        raise ValueError("Packed payload length does not match numel.")
    packed = packed.cpu().to(torch.int16)
    unpacked = torch.stack((packed & 15, (packed >> 4) & 15), dim=1).flatten()[:numel]
    return torch.where(unpacked >= 8, unpacked - 16, unpacked).to(torch.int8)


@dataclass
class QuantizationSession:
    """Applied quantization with original CPU weights and removable input hooks."""

    layers: dict[str, QuantizedLinear]
    activation_bits: int | None
    preserved_layers: dict[str, str] = field(default_factory=dict)
    _modules: dict[str, nn.Linear] = field(default_factory=dict, repr=False)
    _original_weights: dict[str, Tensor] = field(default_factory=dict, repr=False)
    _handles: list[Any] = field(default_factory=list, repr=False)
    _restored: bool = field(default=False, repr=False)

    @torch.no_grad()
    def restore(self) -> None:
        if self._restored:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for name, module in self._modules.items():
            module.weight.copy_(self._original_weights[name].to(module.weight.device))
            if hasattr(module, _ACTIVE_ATTRIBUTE):
                delattr(module, _ACTIVE_ATTRIBUTE)
        self._original_weights.clear()
        self._restored = True

    def report(self) -> dict[str, Any]:
        return {
            "execution": _EXECUTION,
            "activation_bits": self.activation_bits,
            "activation_scheme": "dynamic_per_token_symmetric" if self.activation_bits else None,
            "quantized_linear_count": len(self.layers),
            "preserved_layers": self.preserved_layers,
            "layers": {name: layer.report() for name, layer in self.layers.items()},
        }

    def artifact(self) -> dict[str, Any]:
        """Weights-only research artifact; load into the identical base model.

        Unmodified weights, biases, tokenizer, architecture and decision head are
        not included. Loading restores the compensated floating research path.
        """
        low_bit_stream = any(layer.bits in (2, 3) for layer in self.layers.values())
        return {
            "format": "s1q.packed_linear.v2" if low_bit_stream else "s1q.packed_linear.v1",
            "execution": _EXECUTION,
            "activation_bits": self.activation_bits,
            "preserved_layers": dict(self.preserved_layers),
            "layers": {
                name: {
                    **layer.report(),
                    **({"bit_order": "lsb_first_twos_complement"} if layer.bits in (2, 3) else {}),
                    "payload": (pack_signed_codes(layer.integer_weight, layer.bits)
                                if layer.bits in (2, 3) else
                                pack_int4(layer.integer_weight) if layer.bits == 4 else
                                layer.integer_weight.clone()),
                    "scales": layer.scales.clone(), "input_scale": layer.input_scale.clone(),
                }
                for name, layer in self.layers.items()
            },
        }

    def export(self, path: str | Path) -> None:
        torch.save(self.artifact(), path)

    def __enter__(self) -> QuantizationSession:
        if self._restored:
            raise RuntimeError("Cannot re-enter a restored quantization session.")
        return self

    def __exit__(self, *_: Any) -> None:
        self.restore()


@torch.no_grad()
def _apply_layers(
    modules: Mapping[str, nn.Linear], layers: dict[str, QuantizedLinear],
    activation_bits: int | None, preserved: dict[str, str],
) -> QuantizationSession:
    session = QuantizationSession(layers, activation_bits, preserved)
    try:
        for name, layer in layers.items():
            module = modules[name]
            if getattr(module, _ACTIVE_ATTRIBUTE, False):
                raise RuntimeError(f"Layer {name!r} already has an active quantization session.")
            session._original_weights[name] = module.weight.detach().cpu().clone()
            session._modules[name] = module
            module.weight.copy_(layer.transformed_weight(device=module.weight.device, dtype=module.weight.dtype))
            session._handles.append(module.register_forward_pre_hook(
                _InputTransform(layer.input_scale, activation_bits), with_kwargs=True,
            ))
            setattr(module, _ACTIVE_ATTRIBUTE, True)
    except Exception:
        session.restore()
        raise
    return session


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    method: str = "rtn",
    statistics: Mapping[str, InputStatistics] | None = None,
    include_prefixes: Sequence[str] | None = None,
    preserve_prefixes: Sequence[str] = (),
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    clipping_ratios: Sequence[float] = (0.9, 0.95, 1.0),
    reservoir_blend: float = 0.5,
    max_reservoir_rows: int = 32,
    activation_bits: int | None = None,
    sensitive_fraction: float = 0.0,
) -> QuantizationSession:
    """Quantize selected linears reversibly; optionally preserve sensitive layers.

    The most sensitive ceil(fraction * selected_count) layers remain in their
    original dtype, ranked by calibrated RTN relative diagonal error. This is a
    mixed-precision heuristic, not novel by itself. Heads are preserved by
    default. RTN and S1Q comparisons should use the same explicit scope, head
    policy, and activation setting; report any extra S1Q preserved layers.
    """
    _validate_bits(bits)
    if activation_bits is not None:
        if activation_bits not in (4, 8):
            raise ValueError("Only A4 and A8 activation QDQ is supported.")
    if not math.isfinite(sensitive_fraction) or not 0 <= sensitive_fraction <= 1:
        raise ValueError("sensitive_fraction must be in [0, 1].")
    modules = selected_linear_modules(
        model, include_prefixes=include_prefixes, preserve_prefixes=preserve_prefixes,
        exclude_patterns=exclude_patterns,
    )
    if not modules:
        raise ValueError("No Linear layers matched the quantization scope.")
    _validate_unshared_weights(model, modules)
    for name in modules:
        if getattr(modules[name], _ACTIVE_ATTRIBUTE, False):
            raise RuntimeError(f"Layer {name!r} already has an active quantization session.")
        if method.lower() in ("s1q", "s1q2", "s1q3") and (statistics is None or name not in statistics):
            raise ValueError(f"Missing calibration statistics for {name!r}.")
    preserved: dict[str, str] = {}
    if sensitive_fraction:
        if statistics is None or any(name not in statistics for name in modules):
            raise ValueError("Sensitive-layer preservation requires statistics for every selected layer.")
        sensitivity = {
            name: quantize_weight(module.weight, bits=bits, group_size=group_size, method="rtn", statistics=statistics[name]).relative_error
            for name, module in modules.items()
        }
        keep_count = math.ceil(sensitive_fraction * len(modules))
        for name in sorted(sensitivity, key=lambda n: (-sensitivity[n], n))[:keep_count]:
            preserved[name] = "calibrated_rtn_relative_diagonal_error"
    layers = {
        name: quantize_weight(
            module.weight, bits=bits, group_size=group_size, method=method,
            statistics=statistics.get(name) if statistics is not None else None,
            alphas=alphas, clipping_ratios=clipping_ratios,
            reservoir_blend=reservoir_blend, max_reservoir_rows=max_reservoir_rows,
            activation_bits=activation_bits,
        )
        for name, module in modules.items() if name not in preserved
    }
    return _apply_layers(modules, layers, activation_bits, preserved)


@torch.no_grad()
def load_quantized_artifact(model: nn.Module, artifact: Mapping[str, Any] | str | Path) -> QuantizationSession:
    """Apply a packed artifact to an identical base model, with reversible restore.

    A normal model state_dict containing transformed weights is insufficient:
    the diagonal input compensation hooks must be re-established by this loader.
    """
    if isinstance(artifact, (str, Path)):
        artifact = torch.load(artifact, map_location="cpu", weights_only=True)
    if artifact.get("format") not in ("s1q.packed_linear.v1", "s1q.packed_linear.v2") or artifact.get("execution") != _EXECUTION:
        raise ValueError("Unsupported S1Q research artifact format.")
    activation_bits = artifact.get("activation_bits")
    if activation_bits is not None:
        if activation_bits not in (4, 8):
            raise ValueError("Only A4 and A8 activation QDQ is supported.")
    modules = dict(model.named_modules())
    layers: dict[str, QuantizedLinear] = {}
    for name, record in artifact["layers"].items():
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Artifact layer {name!r} is not a Linear in the base model.")
        if getattr(module, _ACTIVE_ATTRIBUTE, False):
            raise RuntimeError(f"Layer {name!r} already has an active quantization session.")
        shape = tuple(record["shape"])
        if shape != tuple(module.weight.shape):
            raise ValueError(f"Artifact shape mismatch for layer {name!r}.")
        bits, group_size = record["bits"], record["group_size"]
        _validate_bits(bits)
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("Artifact group_size must be a positive integer.")
        payload = record["payload"]
        if bits in (2, 3):
            if artifact["format"] != "s1q.packed_linear.v2" or record.get("bit_order") != "lsb_first_twos_complement":
                raise ValueError("Low-bit codes require the v2 LSB-first packed format.")
            integer = unpack_signed_codes(payload, math.prod(shape), bits).reshape(shape)
        elif bits == 4:
            integer = unpack_int4(payload, math.prod(shape)).reshape(shape)
        else:
            if payload.dtype != torch.int8 or tuple(payload.shape) != shape:
                raise ValueError("Invalid 8-bit weight payload.")
            integer = payload.cpu().clone()
        qmax = (1 << (bits - 1)) - 1
        if (integer.to(torch.int16).abs() > qmax).any():
            raise ValueError("Artifact uses codes outside the symmetric quantizer range.")
        scales = record["scales"].cpu().float()
        input_scale = record["input_scale"].cpu().float()
        dequantize_groupwise(integer, scales, group_size)
        if input_scale.shape != (shape[1],) or not torch.isfinite(input_scale).all() or (input_scale <= 0).any():
            raise ValueError("Invalid diagonal input scales.")
        layers[name] = QuantizedLinear(
            integer_weight=integer, scales=scales, input_scale=input_scale,
            bits=bits, group_size=group_size, method=record["method"],
            alpha=record["alpha"], clipping_ratio=record["clipping_ratio"],
            weighted_error=record["weighted_error"], relative_error=record["relative_error"],
            calibration_rows=record["calibration_rows"],
            reservoir_relative_error=record.get("reservoir_relative_error"),
            reservoir_blend=record.get("reservoir_blend"),
            search_reservoir_rows=record.get("search_reservoir_rows"),
            scale_family=record.get("scale_family"),
            search_activation_bits=record.get("search_activation_bits"),
        )
    _validate_unshared_weights(model, {name: modules[name] for name in layers})
    return _apply_layers(modules, layers, activation_bits, dict(artifact.get("preserved_layers", {})))
