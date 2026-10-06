"""Architecture-neutral, explicitly nonofficial SpinQuant-inspired controls.

The official SpinQuant learns architecture-specific, mergeable R1/R2 rotations
and optionally adds online R3/R4 Hadamard transforms. A generic collection of
independent ``nn.Linear`` modules has no equivalent fusion points. Here a
single small orthogonal matrix per Linear is optimized on calibration inputs,
then applied to matching input/weight blocks. The ``-had`` control composes a
fixed normalized Hadamard with that learned matrix. Both require an online
input rotation, so neither is an official SpinQuant reproduction or a native
integer-kernel implementation.

Official method: https://github.com/facebookresearch/SpinQuant
Paper: https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quantization import (
    DEFAULT_EXCLUDE_PATTERNS,
    InputStatistics,
    QuantizationSession,
    QuantizedLinear,
    _ACTIVE_ATTRIBUTE,
    _groupwise_quantize,
    _validate_unshared_weights,
    dequantize_groupwise,
    fake_quantize_activation,
    selected_linear_modules,
)


SPIN_METHODS = ("spinquant-nohad-adapted", "spinquant-had-adapted")


def _hadamard(size: int, *, device: torch.device) -> Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1),
                            torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix / math.sqrt(size)


def _block_size(width: int, group_size: int, requested: int) -> int:
    # A power-of-two divisor of the quantization group keeps every rotation
    # strictly inside one group; ragged final features remain unrotated.
    limit = min(width, group_size, requested)
    size = 1
    while size * 2 <= limit and group_size % (size * 2) == 0:
        size *= 2
    return size


def _rotate_blocks(value: Tensor, rotation: Tensor) -> Tensor:
    size = rotation.shape[0]
    if size == 1:
        return value * rotation[0, 0]
    width = value.shape[-1]
    full = width // size * size
    transformed = torch.matmul(value[..., :full].reshape(*value.shape[:-1], -1, size), rotation)
    if full == width:
        return transformed.reshape(value.shape)
    return torch.cat((transformed.reshape(*value.shape[:-1], full), value[..., full:]), dim=-1)


def _rotation(parameter: Tensor, hadamard: Tensor) -> Tensor:
    skew = parameter - parameter.T
    identity = torch.eye(parameter.shape[0], device=parameter.device, dtype=parameter.dtype)
    cayley = torch.linalg.solve(identity + skew, identity - skew)
    return cayley @ hadamard


def _sample_columns(width: int, group_size: int, max_groups: int, device: torch.device) -> Tensor:
    complete = width // group_size
    if complete == 0:
        return torch.arange(width, device=device)
    chosen = torch.linspace(0, complete - 1, steps=min(complete, max_groups),
                            device=device).round().long().unique()
    offsets = torch.arange(group_size, device=device)
    return (chosen[:, None] * group_size + offsets[None, :]).flatten()


def _sample_rows(count: int, maximum: int, device: torch.device) -> Tensor:
    return torch.linspace(0, count - 1, steps=min(count, maximum),
                          device=device).round().long().unique()


def _proxy_loss(weight: Tensor, inputs: Tensor, rotation: Tensor,
                bits: int, group_size: int, activation_bits: int | None,
                *, training: bool) -> Tensor:
    rotated_weight = _rotate_blocks(weight, rotation)
    _, _, quantized_weight = _groupwise_quantize(rotated_weight.detach(), bits, group_size, 1.0)
    rotated_inputs = _rotate_blocks(inputs, rotation)
    if activation_bits is not None:
        quantized_inputs = fake_quantize_activation(rotated_inputs, activation_bits)
        # Straight-through on activations; weights use an alternating fixed-Q
        # surrogate, avoiding a zero gradient from exact FP rotation symmetry.
        if training:
            quantized_inputs = rotated_inputs + (quantized_inputs - rotated_inputs).detach()
    else:
        quantized_inputs = rotated_inputs
    reference = F.linear(inputs, weight).detach()
    predicted = F.linear(quantized_inputs, quantized_weight.detach())
    normalizer = reference.square().mean().detach().clamp_min(1e-8)
    output_loss = (predicted - reference).square().mean() / normalizer
    if not training:
        return output_loss
    weight_normalizer = weight.square().mean().detach().clamp_min(1e-8)
    weight_loss = (rotated_weight - quantized_weight.detach()).square().mean() / weight_normalizer
    return output_loss + 0.1 * weight_loss


@dataclass(frozen=True)
class RotatedQuantizedLinear(QuantizedLinear):
    rotation: Tensor = field(default_factory=lambda: torch.eye(1))
    rotation_initial_error: float = 0.0
    rotation_selected_error: float = 0.0
    rotation_selected: str = "identity"
    rotation_steps: int = 0
    rotation_train_rows: int = 0
    rotation_train_outputs: int = 0
    rotation_train_groups: int = 0

    def effective_weight(self, *, device: Any = None, dtype: torch.dtype = torch.float32) -> Tensor:
        transformed = self.transformed_weight(device=device, dtype=torch.float32)
        inverse = self.rotation.to(device=transformed.device, dtype=torch.float32).T
        return _rotate_blocks(transformed, inverse).to(dtype)

    def report(self) -> dict[str, Any]:
        result = super().report()
        result.update({
            "official_spinquant": False,
            "rotation_scope": "one_shared_block_orthogonal_matrix_per_linear_online_input_hook",
            "rotation_size": self.rotation.shape[0],
            "rotation_bytes_fp32": self.rotation.numel() * 4,
            "rotation_sha256": hashlib.sha256(self.rotation.contiguous().numpy().tobytes()).hexdigest(),
            "rotation_initial_calibration_error": self.rotation_initial_error,
            "rotation_selected_calibration_error": self.rotation_selected_error,
            "rotation_selected": self.rotation_selected,
            "rotation_steps": self.rotation_steps,
            "rotation_train_rows": self.rotation_train_rows,
            "rotation_train_outputs": self.rotation_train_outputs,
            "rotation_train_groups": self.rotation_train_groups,
        })
        return result


@dataclass
class SpinQuantProxySession(QuantizationSession):
    rotations: dict[str, Tensor] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        result = super().report()
        result.update({
            "official_spinquant": False,
            "adaptation": "calibration_optimized_shared_block_rotation_per_linear",
            "rotation_execution": "online_float_hook_before_activation_qdq",
            "extra_rotation_bytes_fp32": sum(r.numel() * 4 for r in self.rotations.values()),
            "official_r1_r2_r3_r4_mapping": False,
        })
        return result

    def artifact(self) -> dict[str, Any]:
        raise NotImplementedError(
            "SpinQuant adapted rotations are not represented in the generic packed artifact; "
            "export would not reproduce model outputs."
        )


class _RotateInput:
    def __init__(self, rotation: Tensor, activation_bits: int | None) -> None:
        self.rotation = rotation
        self.activation_bits = activation_bits
        self._cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    def __call__(self, _: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        value = args[0] if args else kwargs["input"]
        key = (value.device, value.dtype)
        if key not in self._cache:
            self._cache[key] = self.rotation.to(device=value.device, dtype=torch.float32)
        transformed = _rotate_blocks(value.float(), self._cache[key]).to(value.dtype)
        if self.activation_bits is not None:
            transformed = fake_quantize_activation(transformed, self.activation_bits)
        if args:
            return (transformed, *args[1:]), kwargs
        return args, {**kwargs, "input": transformed}


@torch.no_grad()
def quantize_spinquant_proxy_weight(
    weight: Tensor, *, method: str, bits: int, group_size: int,
    statistics: InputStatistics, activation_bits: int | None,
    block_size: int = 16, steps: int = 12, learning_rate: float = 0.03,
    max_train_rows: int = 16, max_train_outputs: int = 32,
    max_train_groups: int = 4,
) -> RotatedQuantizedLinear:
    if method not in SPIN_METHODS:
        raise ValueError(f"Unknown SpinQuant proxy {method!r}.")
    if bits not in (2, 3, 4, 8) or activation_bits not in (None, 4, 8):
        raise ValueError("SpinQuant proxy supports W2/W3/W4/W8 and native/A4/A8 activations.")
    if group_size <= 0 or block_size <= 0 or steps <= 0 or learning_rate <= 0:
        raise ValueError("SpinQuant proxy group, block, steps and learning rate must be positive.")
    if min(max_train_rows, max_train_outputs, max_train_groups) <= 0:
        raise ValueError("SpinQuant proxy training limits must be positive.")
    if weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("SpinQuant proxy requires a finite 2D floating weight.")
    statistics.validate(weight.shape[1])
    if statistics.reservoir is None or statistics.reservoir.shape[0] == 0:
        raise ValueError("SpinQuant proxy requires a nonempty calibration input reservoir.")
    floating = weight.detach().float()
    device = floating.device
    size = _block_size(floating.shape[1], group_size, block_size)
    had = (_hadamard(size, device=device) if method == "spinquant-had-adapted"
           else torch.eye(size, device=device))
    columns = _sample_columns(floating.shape[1], group_size, max_train_groups, device)
    output_rows = _sample_rows(floating.shape[0], max_train_outputs, device)
    inputs = statistics.reservoir.to(device=device, dtype=torch.float32)
    inputs = inputs[_sample_rows(inputs.shape[0], max_train_rows, device)][:, columns]
    sampled_weight = floating[output_rows][:, columns]
    initial = float(_proxy_loss(sampled_weight, inputs, had, bits, group_size,
                                activation_bits, training=False).item())
    chosen = had.detach()
    selected = "initial_hadamard" if method == "spinquant-had-adapted" else "identity"
    best = initial
    if size > 1:
        with torch.enable_grad():
            parameter = torch.zeros((size, size), device=device, dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.Adam((parameter,), lr=learning_rate)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                candidate = _rotation(parameter, had)
                loss = _proxy_loss(sampled_weight, inputs, candidate, bits, group_size,
                                   activation_bits, training=True)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("SpinQuant proxy rotation optimization became nonfinite.")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    candidate = _rotation(parameter, had).detach()
                    score = float(_proxy_loss(sampled_weight, inputs, candidate, bits, group_size,
                                              activation_bits, training=False).item())
                    if score < best:
                        chosen, best, selected = candidate.clone(), score, "calibration_optimized"
    rotated_weight = _rotate_blocks(floating, chosen)
    integer, scales, dequantized = _groupwise_quantize(rotated_weight, bits, group_size, 1.0)
    effective = _rotate_blocks(dequantized, chosen.T)
    importance = statistics.mean_square.to(device=device)
    numerator = ((effective - floating).square() * importance).sum(dtype=torch.float64).item()
    denominator = (floating.square() * importance).sum(dtype=torch.float64).item()
    reservoir_inputs = statistics.reservoir.to(device=device, dtype=torch.float32)
    output_error = F.linear(reservoir_inputs, effective - floating).square().mean().item()
    output_ref = F.linear(reservoir_inputs, floating).square().mean().item()
    return RotatedQuantizedLinear(
        integer_weight=integer.cpu(), scales=scales.cpu(), input_scale=torch.ones(floating.shape[1]),
        bits=bits, group_size=group_size, method=method, alpha=0.0, clipping_ratio=1.0,
        weighted_error=numerator, relative_error=numerator / denominator if denominator else 0.0,
        calibration_rows=statistics.count,
        reservoir_relative_error=output_error / output_ref if output_ref else 0.0,
        rotation=chosen.cpu().contiguous(), rotation_initial_error=initial,
        rotation_selected_error=best, rotation_selected=selected,
        rotation_steps=steps if size > 1 else 0, rotation_train_rows=inputs.shape[0],
        rotation_train_outputs=sampled_weight.shape[0],
        rotation_train_groups=math.ceil(sampled_weight.shape[1] / group_size),
    )


@torch.no_grad()
def quantize_spinquant_proxy_model(
    model: nn.Module, *, method: str, bits: int, group_size: int,
    statistics: Mapping[str, InputStatistics] | None,
    activation_bits: int | None,
    include_prefixes: Sequence[str] | None = None,
    preserve_prefixes: Sequence[str] = (),
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    block_size: int = 16, steps: int = 12, learning_rate: float = 0.03,
    max_train_rows: int = 16, max_train_outputs: int = 32,
    max_train_groups: int = 4,
) -> SpinQuantProxySession:
    if method not in SPIN_METHODS:
        raise ValueError(f"Unknown SpinQuant proxy {method!r}.")
    modules = selected_linear_modules(model, include_prefixes=include_prefixes,
                                      preserve_prefixes=preserve_prefixes,
                                      exclude_patterns=exclude_patterns)
    if not modules:
        raise ValueError("No Linear modules match SpinQuant proxy scope.")
    _validate_unshared_weights(model, modules)
    if statistics is None or any(name not in statistics for name in modules):
        raise ValueError("SpinQuant proxy requires calibration statistics for every selected Linear.")
    if any(getattr(module, _ACTIVE_ATTRIBUTE, False) for module in modules.values()):
        raise RuntimeError("A selected module already has an active quantization session.")
    layers = {
        name: quantize_spinquant_proxy_weight(
            module.weight, method=method, bits=bits, group_size=group_size,
            statistics=statistics[name], activation_bits=activation_bits,
            block_size=block_size, steps=steps, learning_rate=learning_rate,
            max_train_rows=max_train_rows, max_train_outputs=max_train_outputs,
            max_train_groups=max_train_groups,
        )
        for name, module in modules.items()
    }
    session = SpinQuantProxySession(layers, activation_bits, {}, rotations={
        name: layer.rotation for name, layer in layers.items()
    })
    try:
        for name, layer in layers.items():
            module = modules[name]
            session._original_weights[name] = module.weight.detach().cpu().clone()
            session._modules[name] = module
            module.weight.copy_(layer.transformed_weight(device=module.weight.device, dtype=module.weight.dtype))
            session._handles.append(module.register_forward_pre_hook(
                _RotateInput(layer.rotation, activation_bits), with_kwargs=True))
            setattr(module, _ACTIVE_ATTRIBUTE, True)
    except Exception:
        session.restore()
        raise
    return session
