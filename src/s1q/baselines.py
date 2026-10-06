"""Architecture-neutral research adaptations of established PTQ baselines.

These are *not* the authors' official AWQ, GPTQ or SmoothQuant implementations.
Their published code targets particular transformer blocks and quantizers. We
apply the central operation of each method to the same ``nn.Linear`` scope,
signed symmetric group quantizer, and optional dynamic activation QDQ used by
S1Q. This makes a controlled comparison on native decision models possible,
but results must be named ``*-adapted`` and cannot be presented as official
reproductions or native integer-kernel performance.

Official references:
  https://github.com/IST-DASLab/gptq
  https://github.com/mit-han-lab/llm-awq
  https://github.com/mit-han-lab/smoothquant

GPTQ adapted: sequential second-order compensation using the Cholesky factor
of an inverse empirical input Hessian. ``gptq-blockdiag-adapted`` keeps 128-column
groups independent for tractable pilots; ``gptq-full-adapted`` includes cross-group
compensation and is considerably more expensive. Both use fixed column order.
AWQ adapted: grid search over activation-mean-absolute-value channel scales,
selecting the lowest single-Linear calibration output reconstruction error.
The official AWQ searches whole transformer blocks and also searches clipping;
this adapter does neither. SmoothQuant adapted: channel balancing from maximum
activation and weight magnitudes, followed by W/A QDQ. Its input division is a
forward hook rather than a fold into architecture-specific predecessor ops.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quantization import (
    DEFAULT_EXCLUDE_PATTERNS,
    InputStatistics,
    QuantizationSession,
    QuantizedLinear,
    _ACTIVE_ATTRIBUTE,
    _apply_layers,
    _groupwise_quantize,
    _validate_unshared_weights,
    dequantize_groupwise,
    quantize_weight,
    selected_linear_modules,
)
from .spinquant_proxy import SPIN_METHODS, quantize_spinquant_proxy_model


BASELINE_METHODS = ("rtn", "gptq-blockdiag-adapted", "gptq-full-adapted", "awq-adapted", "smoothquant-adapted", *SPIN_METHODS)


def _validate(weight: Tensor, bits: int, group_size: int, statistics: InputStatistics | None) -> None:
    if bits not in (2, 3, 4, 8) or type(group_size) is not int or group_size <= 0:
        raise ValueError("Baselines require W2/W3/W4/W8 and a positive group size.")
    if weight.ndim != 2 or not weight.is_floating_point() or min(weight.shape) == 0:
        raise ValueError("Weight must be a nonempty floating matrix.")
    if not torch.isfinite(weight).all():
        raise ValueError("Weight contains nonfinite values.")
    if statistics is not None:
        statistics.validate(weight.shape[1])


def _reservoir(statistics: InputStatistics, device: torch.device) -> Tensor:
    if statistics.reservoir is None or statistics.reservoir.shape[0] == 0:
        raise ValueError("AWQ/GPTQ adaptations require a nonempty calibration input reservoir.")
    return statistics.reservoir.to(device=device, dtype=torch.float32)


def _normalized_scale(scale: Tensor, observed: Tensor) -> Tensor:
    scale = scale.clamp_min(1e-8)
    selected = scale[observed]
    if selected.numel():
        scale = scale / (selected.amin() * selected.amax()).sqrt().clamp_min(1e-12)
    scale = scale.clamp(1e-4, 1e4)
    return torch.where(observed, scale, torch.ones_like(scale))


def _diagnostics(
    weight: Tensor, integer: Tensor, scales: Tensor, input_scale: Tensor,
    statistics: InputStatistics | None, group_size: int,
) -> tuple[float, float, float | None]:
    effective = dequantize_groupwise(integer, scales, group_size) / input_scale
    importance = statistics.mean_square.to(weight.device) if statistics is not None else torch.ones(weight.shape[1], device=weight.device)
    numerator = ((effective - weight).square() * importance).sum(dtype=torch.float64).item()
    denominator = (weight.square() * importance).sum(dtype=torch.float64).item()
    reservoir_error = None
    if statistics is not None and statistics.reservoir is not None and statistics.reservoir.numel():
        inputs = statistics.reservoir.to(device=weight.device, dtype=torch.float32)
        error = F.linear(inputs, effective - weight).square().mean().item()
        reference = F.linear(inputs, weight).square().mean().item()
        reservoir_error = error / reference if reference > 0 else 0.0
    return numerator, numerator / denominator if denominator > 0 else 0.0, reservoir_error


def _finish(
    weight: Tensor, integer: Tensor, scales: Tensor, input_scale: Tensor,
    *, bits: int, group_size: int, method: str, alpha: float,
    statistics: InputStatistics | None,
) -> QuantizedLinear:
    weighted, relative, reservoir = _diagnostics(weight, integer, scales, input_scale, statistics, group_size)
    return QuantizedLinear(
        integer_weight=integer.cpu(), scales=scales.cpu(), input_scale=input_scale.cpu(),
        bits=bits, group_size=group_size, method=method, alpha=alpha,
        clipping_ratio=1.0, weighted_error=weighted, relative_error=relative,
        calibration_rows=statistics.count if statistics is not None else 0,
        reservoir_relative_error=reservoir,
    )


def _inverse_hessian_cholesky(inputs: Tensor, damping: float) -> tuple[Tensor, Tensor]:
    """Return the upper Cholesky factor of the inverse empirical Hessian.

    Dead input channels are handled as in GPTQ: their weights are zeroed for
    reconstruction. Ridge damping is increased only if Cholesky fails.
    """
    hessian = inputs.T @ inputs / inputs.shape[0]
    diagonal = hessian.diagonal()
    dead = diagonal <= 1e-12
    if bool(dead.all()):
        # A calibration split may never exercise a gated projection group.
        # GPTQ has no information about that group; quantizing its zeroed
        # columns is explicit and avoids an unexplained per-layer failure.
        return torch.eye(hessian.shape[0], device=hessian.device), dead
    if bool(dead.any()):
        hessian[dead, :] = 0
        hessian[:, dead] = 0
        hessian[dead, dead] = 1
    base = hessian.diagonal().mean().clamp_min(1e-8)
    eye = torch.eye(hessian.shape[0], device=hessian.device)
    for multiplier in (1.0, 10.0, 100.0):
        chol, info = torch.linalg.cholesky_ex(hessian + damping * multiplier * base * eye)
        if int(info.item()) == 0:
            return torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True), dead
    raise ValueError("Empirical GPTQ Hessian remained non-positive after ridge damping.")


@torch.no_grad()
def quantize_baseline_weight(
    weight: Tensor,
    *,
    method: str,
    bits: int = 4,
    group_size: int = 128,
    statistics: InputStatistics | None = None,
    smooth_alpha: float = 0.5,
    gptq_damping: float = 0.01,
) -> QuantizedLinear:
    """Quantize one Linear weight under a named, controlled baseline setting.

    ``rtn`` delegates to S1Q's existing RTN implementation. For other methods,
    the caller must supply statistics from the *same* calibration records and
    scope used for S1Q. W2/W3/W4/W8 codes and group scales use the same representation.
    AWQ and GPTQ require sampled raw inputs; SmoothQuant uses per-channel maxima.
    """
    method = method.lower()
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline {method!r}; choose from {BASELINE_METHODS}.")
    if method in SPIN_METHODS:
        raise ValueError("SpinQuant adapted rotations require a model-level input hook; use quantize_baseline_model.")
    _validate(weight, bits, group_size, statistics)
    if method == "rtn":
        return quantize_weight(weight, bits=bits, group_size=group_size, method="rtn", statistics=statistics)
    if statistics is None:
        raise ValueError(f"{method} requires calibration statistics.")
    if not math.isfinite(smooth_alpha) or not 0 <= smooth_alpha <= 1:
        raise ValueError("smooth_alpha must lie in [0, 1].")
    if not math.isfinite(gptq_damping) or gptq_damping <= 0:
        raise ValueError("gptq_damping must be positive and finite.")

    floating = weight.detach().float()
    ones = torch.ones(floating.shape[1], device=floating.device)
    if method == "awq-adapted":
        inputs = _reservoir(statistics, floating.device)
        activation = inputs.abs().mean(dim=0)
        observed = activation > 0
        best: tuple[float, float, Tensor, Tensor, Tensor] | None = None
        for step in range(20):
            alpha = step / 20
            scale = _normalized_scale(activation.clamp_min(1e-8).pow(alpha), observed)
            integer, scales, dequantized = _groupwise_quantize(floating * scale, bits, group_size, 1.0)
            effective = dequantized / scale
            error = F.linear(inputs, effective - floating).square().mean().item()
            if best is None or error < best[0]:
                best = (error, alpha, integer, scales, scale)
        assert best is not None
        _, alpha, integer, scales, scale = best
        return _finish(floating, integer, scales, scale, bits=bits, group_size=group_size,
                       method=method, alpha=alpha, statistics=statistics)

    if method == "smoothquant-adapted":
        activation = statistics.absmax.to(device=floating.device, dtype=torch.float32)
        weight_max = floating.abs().amax(dim=0)
        observed = activation > 0
        scale = _normalized_scale(activation.clamp_min(1e-8).pow(smooth_alpha)
                                  / weight_max.clamp_min(1e-8).pow(1 - smooth_alpha), observed)
        integer, scales, _ = _groupwise_quantize(floating * scale, bits, group_size, 1.0)
        return _finish(floating, integer, scales, scale, bits=bits, group_size=group_size,
                       method=method, alpha=smooth_alpha, statistics=statistics)

    # The two GPTQ variants differ only in the Hessian's cross-group scope.
    inputs = _reservoir(statistics, floating.device)
    qmax = (1 << (bits - 1)) - 1
    rows, width = floating.shape
    count = math.ceil(width / group_size)
    working = floating.clone()
    codes = torch.empty_like(working, dtype=torch.int8)
    scales = torch.empty((rows, count), device=working.device)
    if method == "gptq-full-adapted":
        inverse_chol, dead = _inverse_hessian_cholesky(inputs, gptq_damping)
        working[:, dead] = 0
    for group in range(count):
        start, end = group * group_size, min((group + 1) * group_size, width)
        if method == "gptq-blockdiag-adapted":
            inverse_chol, dead = _inverse_hessian_cholesky(inputs[:, start:end], gptq_damping)
            working[:, start:end][:, dead] = 0
        qscale = working[:, start:end].abs().amax(dim=1).div(qmax)
        qscale = torch.where(qscale > 0, qscale, torch.ones_like(qscale))
        scales[:, group] = qscale
        for column in range(start, end):
            local = column if method == "gptq-full-adapted" else column - start
            value = working[:, column]
            code = (value / qscale).round().clamp(-qmax, qmax).to(torch.int8)
            codes[:, column] = code
            remaining = width if method == "gptq-full-adapted" else end
            if column + 1 < remaining:
                error = (value - code.float() * qscale) / inverse_chol[local, local]
                working[:, column + 1:remaining] -= error[:, None] * inverse_chol[local, local + 1:local + remaining - column]
    return _finish(floating, codes, scales, ones, bits=bits, group_size=group_size,
                   method=method, alpha=0.0, statistics=statistics)


@torch.no_grad()
def quantize_baseline_model(
    model: nn.Module,
    *,
    method: str,
    bits: int = 4,
    group_size: int = 128,
    statistics: Mapping[str, InputStatistics] | None = None,
    include_prefixes: Sequence[str] | None = None,
    preserve_prefixes: Sequence[str] = (),
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    activation_bits: int | None = None,
    smooth_alpha: float = 0.5,
    gptq_damping: float = 0.01,
    spin_block_size: int = 16,
    spin_steps: int = 12,
    spin_learning_rate: float = 0.03,
    spin_max_train_rows: int = 16,
    spin_max_train_outputs: int = 32,
    spin_max_train_groups: int = 4,
) -> QuantizationSession:
    """Apply a baseline reversibly to the same Linear scope as ``quantize_model``.

    No additional high-precision layers are silently preserved. Use identical
    ``bits``, ``group_size``, ``activation_bits``, selected linears, calibration
    records and decision evaluation records across candidate methods. The
    SmoothQuant with native activations is a weight-only channel-balancing
    ablation outside its canonical W8A8 setting. SpinQuant variants are
    architecture-neutral proxies, not official reproductions.
    """
    method = method.lower()
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline {method!r}; choose from {BASELINE_METHODS}.")
    if activation_bits not in (None, 4, 8):
        raise ValueError("activation_bits must be None, 4, or 8.")
    if method in SPIN_METHODS:
        return quantize_spinquant_proxy_model(
            model, method=method, bits=bits, group_size=group_size,
            statistics=statistics, activation_bits=activation_bits,
            include_prefixes=include_prefixes, preserve_prefixes=preserve_prefixes,
            exclude_patterns=exclude_patterns, block_size=spin_block_size,
            steps=spin_steps, learning_rate=spin_learning_rate,
            max_train_rows=spin_max_train_rows,
            max_train_outputs=spin_max_train_outputs,
            max_train_groups=spin_max_train_groups,
        )
    modules = selected_linear_modules(model, include_prefixes=include_prefixes,
                                      preserve_prefixes=preserve_prefixes,
                                      exclude_patterns=exclude_patterns)
    if not modules:
        raise ValueError("No Linear modules match the baseline quantization scope.")
    _validate_unshared_weights(model, modules)
    if any(getattr(module, _ACTIVE_ATTRIBUTE, False) for module in modules.values()):
        raise RuntimeError("A selected module already has an active quantization session.")
    if method != "rtn" and (statistics is None or any(name not in statistics for name in modules)):
        raise ValueError(f"{method} requires calibration statistics for every selected Linear.")
    layers = {
        name: quantize_baseline_weight(module.weight, method=method, bits=bits,
                                       group_size=group_size,
                                       statistics=statistics.get(name) if statistics is not None else None,
                                       smooth_alpha=smooth_alpha, gptq_damping=gptq_damping)
        for name, module in modules.items()
    }
    return _apply_layers(modules, layers, activation_bits, preserved={})
