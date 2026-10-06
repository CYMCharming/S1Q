"""Exploratory margin-aware activation compensation (MAC), not official PTQ.

Keep released S1Q/S1Q2 and all adapted baselines unchanged. MAC reconstructs
native Linear outputs through the actual activation QDQ path, optionally
corrects the floating weight target with a bounded dual-ridge solve, and uses
a disjoint half of the saved token reservoir to choose the quantized candidate.
Decision-margin sampling is supplied by decision_margin.py; it is a surrogate
for decision sensitivity, not the full Fisher matrix or a new Fisher theorem.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activation_compensation import compensate_weight
from .baselines import _normalized_scale, quantize_baseline_weight
from .quantization import (InputStatistics, QuantizedLinear, _ACTIVE_ATTRIBUTE,
                           _apply_layers, _groupwise_quantize,
                           _validate_unshared_weights, dequantize_groupwise, fake_quantize_activation,
                           selected_linear_modules)
from .spinquant_proxy import (SpinQuantProxySession, _RotateInput, _rotate_blocks,
                             quantize_spinquant_proxy_weight)


def _split_inputs(statistics: InputStatistics, device) -> tuple[Tensor, Tensor]:
    if statistics.reservoir is None or len(statistics.reservoir) < 4:
        raise ValueError("MAC needs at least four saved calibration token rows")
    rows = statistics.reservoir.to(device=device, dtype=torch.float32)
    # Priority reservoirs are sorted by priority; a fixed permutation prevents
    # the fit/select halves from systematically differing in token importance.
    indices = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261004))
    rows = rows.index_select(0, indices.to(rows.device))
    midpoint = len(rows) // 2
    return rows[:midpoint], rows[midpoint:]


def _qdq(inputs: Tensor, scale: Tensor, activation_bits: int | None) -> Tensor:
    transformed = inputs / scale
    return fake_quantize_activation(transformed, activation_bits) if activation_bits else transformed


@torch.no_grad()
def quantize_mac_weight(weight: Tensor, *, statistics: InputStatistics,
                        bits=4, group_size=128, activation_bits=4,
                        base="s1q", compensation=True, ridge=.1,
                        correction_cap=.1) -> tuple[QuantizedLinear, dict]:
    """Joint search with a bounded correction; no evaluation labels or logits."""
    floating = weight.detach().float()
    fit, select = _split_inputs(statistics, floating.device)
    reference = F.linear(select, floating)
    energy = reference.square().mean().clamp_min(1e-12)
    observed = fit.square().mean(0) > 0
    mean_abs = fit.abs().mean(0)
    rms = fit.square().mean(0).sqrt()
    families = (("mean_abs", mean_abs),) if base == "awq" else (("rms", rms), ("mean_abs", mean_abs))
    alphas = tuple(step / 20 for step in range(20)) if base == "awq" else (0., .25, .5, .75, .9, 1.)
    clips = (1.,) if base == "awq" else (1., .95)
    strengths = (0., 1.) if compensation and activation_bits is not None else (0.,)
    best, candidates = None, 0
    for family, magnitude in families:
        for alpha in alphas:
            if family == "mean_abs" and base == "s1q" and alpha == 0:
                continue
            scale = _normalized_scale(magnitude.clamp_min(1e-8).pow(alpha), observed)
            xq = _qdq(select, scale, activation_bits)
            for strength in strengths:
                target, correction = compensate_weight(
                    floating, fit, scale, activation_bits, ridge=ridge,
                    strength=strength, max_relative_correction=correction_cap)
                for clip in clips:
                    integer, qscale, reconstructed = _groupwise_quantize(target, bits, group_size, clip)
                    score = float(((F.linear(xq, reconstructed) - reference).square().mean() / energy).item())
                    candidates += 1
                    if best is None or score < best[0]:
                        best = (score, integer.cpu(), qscale.cpu(), scale.cpu(), alpha, clip, family, correction)
    assert best is not None
    score, integer, qscale, scale, alpha, clip, family, correction = best
    layer = QuantizedLinear(integer_weight=integer, scales=qscale, input_scale=scale,
                            bits=bits, group_size=group_size, method=f"{base}-mac",
                            alpha=alpha, clipping_ratio=clip, weighted_error=score,
                            relative_error=score, calibration_rows=statistics.count,
                            reservoir_relative_error=score, scale_family=family)
    return layer, {"base": base, "objective": "joint_WA_native_output_MSE_on_reservoir_selection_half",
                   "scale_family": family, "candidate_count": candidates,
                   "fit_token_rows": len(fit), "select_token_rows": len(select),
                   "compensation_enabled": compensation, "correction": correction,
                   "importance_application": "weighted_reservoir_sampling_once_no_second_weighting"}


@torch.no_grad()
def quantize_gptq_mac_weight(weight: Tensor, *, statistics: InputStatistics,
                             bits=4, group_size=128, activation_bits=4,
                             compensation=True) -> tuple[QuantizedLinear, dict]:
    floating = weight.detach().float()
    fit, select = _split_inputs(statistics, floating.device)
    ones = torch.ones(floating.shape[1], device=floating.device)
    xq = _qdq(fit, ones, activation_bits)
    transformed_stats = InputStatistics(count=len(xq), reservoir=xq.cpu(),
                                       mean_square=xq.square().mean(0).cpu(),
                                       absmax=xq.abs().amax(0).cpu())
    reference = F.linear(select, floating)
    energy = reference.square().mean().clamp_min(1e-12)
    best = None
    for strength in ((0., 1.) if compensation and activation_bits else (0.,)):
        target, correction = compensate_weight(floating, fit, ones, activation_bits, strength=strength)
        layer = quantize_baseline_weight(target, method="gptq-blockdiag-adapted",
                                        bits=bits, group_size=group_size,
                                        statistics=transformed_stats)
        predicted = F.linear(_qdq(select, ones, activation_bits), layer.transformed_weight(device=floating.device))
        score = float(((predicted - reference).square().mean() / energy).item())
        if best is None or score < best[0]:
            best = (score, layer, correction)
    assert best is not None
    score, layer, correction = best
    return replace(layer, method="gptq-blockdiag-adapted-mac", reservoir_relative_error=score), {
        "base": "gptq-blockdiag-adapted", "hessian_input": "actual_activation_QDQ_fit_rows",
        "fit_token_rows": len(fit), "select_token_rows": len(select),
        "joint_select_relative_error": score, "correction": correction}


@torch.no_grad()
def quantize_optimized_model(model: nn.Module, *, statistics: Mapping[str, InputStatistics],
                              method="s1q", bits=4, group_size=128, activation_bits=4,
                              compensation=True):
    modules = selected_linear_modules(model)
    _validate_unshared_weights(model, modules)
    if not modules or set(modules) != set(statistics):
        raise ValueError("Optimized statistics must match selected Linear scope exactly")
    if any(getattr(module, _ACTIVE_ATTRIBUTE, False) for module in modules.values()):
        raise RuntimeError("An active quantization session is already present")
    layers, metadata = {}, {}
    spin = method.startswith("spinquant-")
    for name, module in modules.items():
        if spin:
            base = "spinquant-had-adapted" if method.startswith("spinquant-had-") else "spinquant-nohad-adapted"
            raw_fit, raw_select = _split_inputs(statistics[name], module.weight.device)
            rotation_stats = InputStatistics(count=len(raw_fit), reservoir=raw_fit.cpu(),
                mean_square=raw_fit.square().mean(0).cpu(), absmax=raw_fit.abs().amax(0).cpu())
            original = quantize_spinquant_proxy_weight(
                module.weight, method=base, bits=bits, group_size=group_size,
                statistics=rotation_stats, activation_bits=activation_bits)
            floating = _rotate_blocks(module.weight.detach().float(), original.rotation.to(module.weight.device))
            fit, select = raw_fit, raw_select
            rotation = original.rotation.to(floating.device)
            fit, select = _rotate_blocks(fit, rotation), _rotate_blocks(select, rotation)
            reference = F.linear(select, floating)
            ones = torch.ones(floating.shape[1], device=floating.device)
            best = None
            for strength in ((0., 1.) if compensation and activation_bits else (0.,)):
                target, correction = compensate_weight(floating, fit, ones, activation_bits, strength=strength)
                codes, scales, reconstructed = _groupwise_quantize(target, bits, group_size, 1.)
                error = float((F.linear(_qdq(select, ones, activation_bits), reconstructed) - reference).square().mean().item())
                if best is None or error < best[0]:
                    best = (error, codes.cpu(), scales.cpu(), correction)
            error, codes, scales, correction = best
            native_weight = module.weight.detach().float()
            transformed = dequantize_groupwise(codes.to(floating.device), scales.to(floating.device), group_size)
            effective = _rotate_blocks(transformed, rotation.T)
            importance = statistics[name].mean_square.to(floating.device)
            numerator = float(((effective-native_weight).square()*importance).sum(dtype=torch.float64).item())
            denominator = float((native_weight.square()*importance).sum(dtype=torch.float64).item())
            raw_inputs = statistics[name].reservoir.to(floating.device)
            weight_error = float(F.linear(raw_inputs, effective-native_weight).square().mean().item())
            weight_energy = float(F.linear(raw_inputs, native_weight).square().mean().item())
            layers[name] = replace(original, integer_weight=codes, scales=scales, method=method,
                weighted_error=numerator, relative_error=numerator/denominator if denominator else 0.,
                reservoir_relative_error=weight_error/weight_energy if weight_energy else 0.)
            metadata[name] = {"correction": correction, "selection_mse": error,
                              "fit_token_rows": len(fit), "select_token_rows": len(select)}
        elif method.startswith("gptq-"):
            layers[name], metadata[name] = quantize_gptq_mac_weight(
                module.weight, statistics=statistics[name], bits=bits,
                group_size=group_size, activation_bits=activation_bits, compensation=compensation)
        else:
            layers[name], metadata[name] = quantize_mac_weight(
                module.weight, statistics=statistics[name], bits=bits, group_size=group_size,
                activation_bits=activation_bits, base="awq" if method.startswith("awq-") else "s1q",
                compensation=compensation)
    if method == "s1q":
        layers = {name: replace(layer, method="s1q") for name, layer in layers.items()}
    if not spin:
        session = _apply_layers(modules, layers, activation_bits, preserved={})
    else:
        session = SpinQuantProxySession(layers, activation_bits, {}, rotations={n: l.rotation for n,l in layers.items()})
        try:
            for name, layer in layers.items():
                module = modules[name]
                session._original_weights[name] = module.weight.detach().cpu().clone()
                session._modules[name] = module
                module.weight.copy_(layer.transformed_weight(device=module.weight.device, dtype=module.weight.dtype))
                session._handles.append(module.register_forward_pre_hook(_RotateInput(layer.rotation, activation_bits), with_kwargs=True))
                setattr(module, _ACTIVE_ATTRIBUTE, True)
        except BaseException:
            session.restore()
            raise
    return session, metadata
