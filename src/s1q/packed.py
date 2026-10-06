"""Packed-weight storage with transient floating-point Linear inference.

This is a memory/storage research backend, not an INT4/INT8 GEMM backend.
``PackedLinear`` retains integer payloads and small scale/bias buffers, then
unpacks and dequantizes one weight matrix for each call. The transient work can
increase latency; benchmark actual resident and peak memory independently.

Export the weights-only artifact from a quantization session, restore the
session, and discard it before measuring memory::

    artifact = session.artifact()
    session.restore()
    del session  # Its module references would otherwise retain old linears.
    model = apply_packed(model, artifact)

The same base checkpoint, tokenizer, and unmodified decision head are required.
The artifact validates structural compatibility, not base-checkpoint identity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bitpack import packed_nbytes, unpack_signed_codes
from .quantization import (
    _ACTIVE_ATTRIBUTE,
    _EXECUTION,
    _validate_unshared_weights,
    fake_quantize_activation,
    unpack_int4,
)


PACKED_EXECUTION = "packed_storage_transient_float_dequantization"


def _integer_option(value: Any, allowed: tuple[int, ...], label: str) -> int:
    if type(value) is not int or value not in allowed:
        raise ValueError(f"{label} must be one of {allowed}.")
    return value


def _validate_record(record: Mapping[str, Any]) -> tuple[tuple[int, int], int, int, Tensor, Tensor, Tensor]:
    """Validate serialized tensors without retaining an unpacked dense weight."""
    if not isinstance(record, Mapping):
        raise ValueError("Packed layer record must be a mapping.")
    required = {"shape", "bits", "group_size", "payload", "scales", "input_scale"}
    if not required.issubset(record):
        raise ValueError(f"Packed layer is missing fields: {sorted(required - set(record))}.")
    raw_shape = record["shape"]
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        raise ValueError("Packed weight shape must contain two positive integer dimensions.")
    if any(type(dimension) is not int or dimension <= 0 for dimension in raw_shape):
        raise ValueError("Packed weight shape must contain two positive integer dimensions.")
    shape = tuple(raw_shape)
    bits = _integer_option(record["bits"], (2, 3, 4, 8), "weight bits")
    group_size = record["group_size"]
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("Packed group_size must be a positive integer.")
    payload, scales, input_scale = record["payload"], record["scales"], record["input_scale"]
    if not all(isinstance(value, Tensor) for value in (payload, scales, input_scale)):
        raise ValueError("Packed payload, scales and input_scale must be tensors.")
    numel = math.prod(shape)
    if bits in (2, 3):
        if record.get("bit_order") != "lsb_first_twos_complement":
            raise ValueError("Low-bit payload must declare LSB-first two's-complement encoding.")
        if payload.dtype != torch.uint8 or payload.ndim != 1 or payload.numel() != packed_nbytes(numel, bits):
            raise ValueError(f"Invalid packed {bits}-bit payload dtype, shape or length.")
        integer = unpack_signed_codes(payload, numel, bits)
    elif bits == 4:
        if payload.dtype != torch.uint8 or payload.ndim != 1 or payload.numel() != math.ceil(numel / 2):
            raise ValueError("Invalid packed 4-bit payload dtype, shape or length.")
        integer = unpack_int4(payload, numel)
        if numel % 2 and int(payload[-1].item()) >> 4:
            raise ValueError("Odd-length 4-bit payload must have a zero padding nibble.")
    else:
        if payload.dtype != torch.int8 or tuple(payload.shape) != shape:
            raise ValueError("Invalid packed 8-bit payload dtype or shape.")
        integer = payload.detach().cpu().reshape(-1)
    qmax = (1 << (bits - 1)) - 1
    if (integer.to(torch.int16).abs() > qmax).any():
        raise ValueError("Payload codes exceed the signed symmetric quantizer range.")
    if not scales.is_floating_point() or not input_scale.is_floating_point():
        raise ValueError("Quantization scales must have floating point dtypes.")
    scales = scales.detach().cpu().float()
    input_scale = input_scale.detach().cpu().float()
    expected_scales = (shape[0], math.ceil(shape[1] / group_size))
    if tuple(scales.shape) != expected_scales or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid groupwise scale shape or values.")
    if tuple(input_scale.shape) != (shape[1],) or not torch.isfinite(input_scale).all() or (input_scale <= 0).any():
        raise ValueError("Invalid diagonal input scale shape or values.")
    return shape, bits, group_size, payload.detach(), scales, input_scale


class PackedLinear(nn.Module):
    """Inference Linear whose persistent storage contains no dense weight.

    ``weight`` is a compatibility property: accessing it reconstructs the
    *effective* weight with compensation folded in, and never caches it. This
    lets weight-only functional callers obtain the expected compensated weight.
    Such callers bypass ``forward`` and therefore also bypass A4/A8 simulation.
    Verify architecture coverage before making activation-quantization claims.

    Scale buffers stay FP32 through ``half()``, ``bfloat16()`` and ``to()``;
    an empty dtype buffer records the requested floating computation dtype.
    Bias is an inference buffer rather than a trainable Parameter.
    """

    def __init__(
        self,
        record: Mapping[str, Any],
        *,
        bias: Tensor | None = None,
        activation_bits: int | None = None,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        shape, bits, group_size, payload, scales, input_scale = _validate_record(record)
        if activation_bits is not None:
            _integer_option(activation_bits, (4, 8), "activation bits")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("PackedLinear requires a floating computation dtype.")
        if bias is not None:
            if not isinstance(bias, Tensor) or bias.shape != (shape[0],) or not bias.is_floating_point():
                raise ValueError("PackedLinear bias must be a floating vector matching out_features.")
            if not torch.isfinite(bias).all():
                raise ValueError("PackedLinear bias must be finite.")
        self.out_features, self.in_features = shape
        self.bits, self.group_size = bits, group_size
        self.activation_bits = activation_bits
        self.register_buffer("qweight", payload.to(device=device).clone())
        self.register_buffer("scales", scales.to(device=device).clone())
        self.register_buffer("input_scale", input_scale.to(device=device).clone())
        self.register_buffer("bias", bias.detach().to(device=device, dtype=dtype).clone() if bias is not None else None)
        self.register_buffer("_dtype_anchor", torch.empty(0, device=device, dtype=dtype), persistent=False)

    def _apply(self, fn: Any, recurse: bool = True) -> PackedLinear:
        # nn.Module.to casts floating buffers as well as parameters. Preserve
        # original FP32 scale precision while respecting requested device moves.
        original_scales, original_input_scale = self.scales, self.input_scale
        super()._apply(fn, recurse=recurse)
        self.scales = original_scales.to(device=self.qweight.device, dtype=torch.float32)
        self.input_scale = original_input_scale.to(device=self.qweight.device, dtype=torch.float32)
        return self

    def _integer_weight(self) -> Tensor:
        if self.bits == 8:
            return self.qweight
        if self.bits in (2, 3):
            count = self.out_features * self.in_features
            codes = unpack_signed_codes(self.qweight, count, self.bits, validate_padding=False)
            return codes.reshape(self.out_features, self.in_features)
        # GPU-friendly unpacking: quantization.unpack_int4 intentionally returns
        # CPU codes for artifacts, whereas inference must remain on the device.
        payload = self.qweight.to(torch.int16)
        nibbles = torch.stack((payload & 15, (payload >> 4) & 15), dim=1).flatten()
        nibbles = nibbles[:self.out_features * self.in_features]
        signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
        return signed.to(torch.int8).reshape(self.out_features, self.in_features)

    def _transformed_weight(self) -> Tensor:
        integer = self._integer_weight()
        # Validate scales only while loading. Repeating GPU finite/range checks
        # in every Linear call would introduce host synchronization. Broadcasting
        # group scales also avoids materializing an expanded full scale matrix.
        groups = math.ceil(self.in_features / self.group_size)
        padding = groups * self.group_size - self.in_features
        if padding:
            integer = F.pad(integer, (0, padding))
        grouped = integer.reshape(self.out_features, groups, self.group_size)
        dequantized = grouped.float() * self.scales.unsqueeze(-1)
        weight = dequantized.reshape(self.out_features, -1)[:, :self.in_features].contiguous()
        return weight.to(self._dtype_anchor.dtype)

    @property
    def weight(self) -> Tensor:
        transformed = self._transformed_weight()
        return transformed / self.input_scale.to(dtype=transformed.dtype)

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"PackedLinear expects {self.in_features} input features.")
        if input.device != self.qweight.device:
            raise ValueError("PackedLinear input and packed buffers must be on the same device.")
        transformed = input / self.input_scale.to(dtype=input.dtype)
        if self.activation_bits is not None:
            transformed = fake_quantize_activation(transformed, self.activation_bits)
        return F.linear(transformed, self._transformed_weight(), self.bias)

    def retained_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size() for buffer in self.buffers(recurse=False))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"activation_bits={self.activation_bits}, bias={self.bias is not None}, "
            "backend=transient_float_dequantization"
        )


@torch.no_grad()
def apply_packed(model: nn.Module, artifact: Mapping[str, Any] | str | Path) -> nn.Module:
    """Replace artifact linears with PackedLinear after restoring the session.

    Validation and construction finish before any model replacement, so malformed
    artifacts do not partly modify the model. Untouched linears, including heads,
    retain their original objects and tensors. Always use the returned model;
    when the model itself is a Linear, its replacement cannot occur in place.

    This helper does not retain original dense linears for restoration. Rebuild
    the identical base model to return to full precision or another candidate.
    """
    if isinstance(artifact, (str, Path)):
        artifact = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping):
        raise ValueError("Packed artifact must be a mapping.")
    if artifact.get("format") not in ("s1q.packed_linear.v1", "s1q.packed_linear.v2") or artifact.get("execution") != _EXECUTION:
        raise ValueError("Unsupported packed S1Q research artifact format.")
    records = artifact.get("layers")
    if not isinstance(records, Mapping):
        raise ValueError("Packed artifact layers must be a mapping.")
    if "" in records and len(records) != 1:
        raise ValueError("A root Linear artifact cannot also replace descendant modules.")
    activation_bits = artifact.get("activation_bits")
    if activation_bits is not None:
        _integer_option(activation_bits, (4, 8), "activation bits")
    modules = dict(model.named_modules())
    if any(getattr(module, _ACTIVE_ATTRIBUTE, False) for module in modules.values()):
        raise RuntimeError("Restore the active quantization session before applying packed storage.")
    selected: dict[str, nn.Linear] = {}
    for name, record in records.items():
        if not isinstance(name, str):
            raise ValueError("Packed artifact module names must be strings.")
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Packed artifact layer {name!r} is not a base Linear.")
        if type(module) is not nn.Linear:
            raise ValueError(f"Packed replacement requires standard nn.Linear semantics at {name!r}.")
        if module._forward_pre_hooks or module._forward_hooks or module._backward_hooks:
            raise ValueError(f"Remove existing hooks before packed replacement at {name!r}.")
        shape, *_ = _validate_record(record)
        if shape != tuple(module.weight.shape):
            raise ValueError(f"Packed artifact shape mismatch at {name!r}.")
        selected[name] = module
    _validate_unshared_weights(model, selected)
    replacements = {
        name: PackedLinear(
            records[name], bias=module.bias, activation_bits=activation_bits,
            device=module.weight.device, dtype=module.weight.dtype,
        ).train(module.training)
        for name, module in selected.items()
    }
    for name, replacement in replacements.items():
        if name == "":
            return replacement
        parent_name, _, child_name = name.rpartition(".")
        modules[parent_name]._modules[child_name] = replacement
    return model


def packed_report(model: nn.Module) -> dict[str, Any]:
    """Persistent tensor accounting; measure CUDA peak memory separately."""
    layers = {name: module for name, module in model.named_modules() if isinstance(module, PackedLinear)}
    layer_reports: dict[str, dict[str, Any]] = {}
    for name, module in layers.items():
        dense_weight_bytes = module.in_features * module.out_features * module._dtype_anchor.element_size()
        bias_bytes = module.bias.numel() * module.bias.element_size() if module.bias is not None else 0
        payload_bytes = module.qweight.numel() * module.qweight.element_size()
        layer_reports[name] = {
            "shape": [module.out_features, module.in_features], "bits": module.bits,
            "group_size": module.group_size, "activation_bits": module.activation_bits,
            "compute_dtype": str(module._dtype_anchor.dtype),
            "payload_bytes": payload_bytes,
            "auxiliary_bytes": module.retained_bytes() - payload_bytes,
            "retained_bytes": module.retained_bytes(),
            "original_linear_tensor_bytes": dense_weight_bytes + bias_bytes,
        }
    tensors = {id(value): value for value in (*model.parameters(), *model.buffers())}
    return {
        "execution": PACKED_EXECUTION, "native_integer_gemm": False,
        "packed_linear_count": len(layers),
        "packed_payload_bytes": sum(item["payload_bytes"] for item in layer_reports.values()),
        "packed_linear_retained_bytes": sum(item["retained_bytes"] for item in layer_reports.values()),
        "original_linear_tensor_bytes": sum(item["original_linear_tensor_bytes"] for item in layer_reports.values()),
        "model_resident_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
        "layers": layer_reports,
    }
