"""Calibration-only compensation for low-bit activation error.

This module deliberately leaves the frozen S1Q and adapted PTQ baselines intact.
It returns a *floating transformed-domain target* for a subsequent weight
quantizer, rather than a quantized model or an additional inference-time branch.
Given native inputs X, channel scale s, B = W*s and Z = Q_A(X/s), it solves

    min_D ||sqrt(t) (Z (B+D)^T - X W^T)||_F^2 + lambda ||D||_F^2.

The dual formula uses a matrix of calibration-row width rather than feature
width: D = E^T (ZZ^T + lambda I)^-1 Z, with token weights absorbed into Z and E.
The correction is shrunk and bounded before returning it. No labels, evaluation
records, weight gradients, or integer activation kernel are used here.

AWQ-style wrappers can quantize the returned target with their group quantizer.
GPTQ wrappers must also construct their input Hessian from Q_A(X/s), since this
is the input that will execute at inference. Rotation wrappers use rotated X/W
and s=1, then keep their existing inverse rotation/input hook. All wrappers must
report their extra calibration work and retain an unmodified baseline control.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .quantization import fake_quantize_activation


def _validate_scalar(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    invalid = value <= 0 if positive else value < 0
    if invalid:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}.")


@torch.no_grad()
def compensate_weight(
    weight: Tensor,
    inputs: Tensor,
    input_scale: Tensor,
    activation_bits: int | None,
    *,
    ridge: float = 0.1,
    strength: float = 1.0,
    max_relative_correction: float = 0.1,
    token_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Return an activation-compensated FP32 weight target in the W*s domain.

    ``inputs`` is an n-by-in_features raw calibration reservoir. Positive
    ``input_scale`` implements the existing exact scaling equivalence. Optional
    ``token_weights`` is a nonnegative n-vector, normalized to mean one. Weights
    of zero ignore a calibration row; an entirely zero vector is invalid.

    lambda = ridge * trace(Z_weighted Z_weighted^T) / n. The global Frobenius
    trust cap bounds ||strength*D|| / ||W*s||, and therefore does not silently
    allocate extra high-precision layers or an inference-time residual. The
    returned target still needs weight quantization, which may undo a local fit
    improvement; end-to-end superiority must be established separately.
    """
    _validate_scalar("ridge", ridge, positive=True)
    _validate_scalar("strength", strength)
    _validate_scalar("max_relative_correction", max_relative_correction)
    if activation_bits not in (None, 4, 8) or isinstance(activation_bits, bool):
        raise ValueError("activation_bits must be None, 4, or 8.")
    if (weight.ndim != 2 or not weight.is_floating_point() or min(weight.shape) == 0
            or not torch.isfinite(weight).all()):
        raise ValueError("weight must be a nonempty finite floating matrix.")
    if (inputs.ndim != 2 or not inputs.is_floating_point() or inputs.shape[0] == 0
            or inputs.shape[1] != weight.shape[1] or not torch.isfinite(inputs).all()):
        raise ValueError("inputs must be a nonempty finite floating calibration matrix of matching width.")
    if (input_scale.shape != (weight.shape[1],) or not input_scale.is_floating_point()
            or not torch.isfinite(input_scale).all() or (input_scale <= 0).any()):
        raise ValueError("input_scale must be a finite positive feature vector.")

    floating = weight.detach().to(dtype=torch.float32)
    raw_inputs = inputs.detach().to(device=floating.device, dtype=torch.float32)
    scale = input_scale.detach().to(device=floating.device, dtype=torch.float32)
    transformed_weight = floating * scale
    transformed_inputs = raw_inputs / scale
    if not torch.isfinite(transformed_weight).all() or not torch.isfinite(transformed_inputs).all():
        raise ValueError("Channel transformation overflowed FP32.")
    quantized_inputs = (fake_quantize_activation(transformed_inputs, activation_bits)
                        if activation_bits is not None else transformed_inputs)
    count = raw_inputs.shape[0]
    if token_weights is None:
        row_weights = torch.ones(count, device=floating.device, dtype=torch.float32)
    else:
        if (token_weights.shape != (count,) or not token_weights.is_floating_point()
                or not torch.isfinite(token_weights).all() or (token_weights < 0).any()
                or not bool(token_weights.sum() > 0)):
            raise ValueError("token_weights must be a finite nonnegative row vector with positive mass.")
        row_weights = token_weights.detach().to(device=floating.device, dtype=torch.float32)
        row_weights = row_weights / row_weights.mean()
        if not torch.isfinite(row_weights).all():
            raise ValueError("Normalized token weights overflowed FP32.")

    reference = F.linear(raw_inputs, floating)
    before = F.linear(quantized_inputs, transformed_weight)
    row_factor = row_weights.sqrt()[:, None]
    residual = (reference - before) * row_factor
    weighted_inputs = quantized_inputs * row_factor
    reference_energy = (reference.square() * row_weights[:, None]).mean()
    before_error = residual.square().mean()
    norm = torch.linalg.vector_norm(transformed_weight)
    input_energy = transformed_inputs.square().mean()
    input_error = (quantized_inputs - transformed_inputs).square().mean()
    if any(not bool(torch.isfinite(value).all()) for value in
           (reference, before, residual, weighted_inputs, reference_energy,
            before_error, norm, input_energy, input_error)):
        raise ValueError("Activation compensation reconstruction or diagnostics overflowed FP32.")

    metadata: dict[str, Any] = {
        "mechanism": "ridge_activation_error_compensation_dual",
        "target_domain": "transformed_weight_W_times_input_scale",
        "objective": "token_weighted_joint_linear_output_mse_plus_ridge",
        "activation_bits": activation_bits,
        "ridge": float(ridge), "strength": float(strength),
        "max_relative_correction": float(max_relative_correction),
        "calibration_rows": count,
        "token_weighting": token_weights is not None,
        "token_weight_min": float(row_weights.min().item()),
        "token_weight_max": float(row_weights.max().item()),
        "math_dtype": "float32", "solve_domain": "calibration_rows_dual",
        "uses_gold_labels": False, "extra_inference_parameters": 0,
        "activation_execution": "dynamic_per_token_QDQ" if activation_bits else "native_float",
        "activation_relative_input_mse": float((input_error / input_energy.clamp_min(1e-30)).item()),
        "before_relative_output_mse": float((before_error / reference_energy.clamp_min(1e-30)).item()),
        "after_relative_output_mse": float((before_error / reference_energy.clamp_min(1e-30)).item()),
        "regularization_lambda": 0.0, "ridge_escalation_multiplier": 1.0,
        "unscaled_relative_correction": 0.0, "applied_relative_correction": 0.0,
        "trust_cap_multiplier": 1.0, "correction_applied": False,
    }
    # Native-A and zero-strength controls are exact transformed-weight no-ops.
    # Skipping their numerical residual avoids fitting FP32 roundoff error.
    if activation_bits is None or strength == 0 or max_relative_correction == 0 or norm == 0:
        metadata["no_op_reason"] = ("native_activation" if activation_bits is None else
                                     "zero_strength" if strength == 0 else
                                     "zero_trust_cap" if max_relative_correction == 0 else "zero_weight")
        return transformed_weight.clone(), metadata

    gram = weighted_inputs @ weighted_inputs.T
    base = gram.diagonal().mean()
    if not torch.isfinite(gram).all() or not torch.isfinite(residual).all():
        raise ValueError("Activation compensation covariance or residual overflowed FP32.")
    if not bool(base > 0):
        metadata["no_op_reason"] = "zero_quantized_input"
        return transformed_weight.clone(), metadata
    regularization = float(ridge) * base
    identity = torch.eye(count, device=floating.device, dtype=torch.float32)
    factor = None
    multiplier = 1.0
    for multiplier in (1.0, 10.0, 100.0):
        candidate, info = torch.linalg.cholesky_ex(gram + regularization * multiplier * identity)
        if int(info.item()) == 0:
            factor = candidate
            break
    if factor is None:
        raise ValueError("Activation compensation dual covariance was not positive definite after ridge escalation.")
    # Solve for E rather than Z: n*out temporary can be smaller than n*features.
    solved_residual = torch.cholesky_solve(residual, factor)
    correction = solved_residual.T @ weighted_inputs
    if not torch.isfinite(correction).all():
        raise ValueError("Activation compensation solve produced a nonfinite correction.")
    correction_norm = torch.linalg.vector_norm(correction)
    relative = correction_norm / norm.clamp_min(1e-30)
    correction = correction * float(strength)
    applied_norm = correction_norm * float(strength)
    cap_multiplier = torch.minimum(torch.ones_like(norm),
                                   float(max_relative_correction) * norm / applied_norm.clamp_min(1e-30))
    correction = correction * cap_multiplier
    corrected = transformed_weight + correction
    if not torch.isfinite(corrected).all():
        raise ValueError("Activation compensation target overflowed FP32.")
    actual_relative = torch.linalg.vector_norm(corrected - transformed_weight) / norm.clamp_min(1e-30)
    after_error = ((F.linear(quantized_inputs, corrected) - reference).square()
                   * row_weights[:, None]).mean()
    metadata.update({
        "regularization_lambda": float((regularization * multiplier).item()),
        "ridge_escalation_multiplier": multiplier,
        "unscaled_relative_correction": float(relative.item()),
        "applied_relative_correction": float(actual_relative.item()),
        "trust_cap_multiplier": float(cap_multiplier.item()),
        "correction_applied": bool(actual_relative > 0),
        "after_relative_output_mse": float((after_error / reference_energy.clamp_min(1e-30)).item()),
    })
    return corrected, metadata
