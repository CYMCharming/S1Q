"""Optional, calibration-only gain repair of an applied quantization session.

The full-precision teacher's detached categorical logits supervise tiny bounded
row gains through the native decision adapter. Backbone/head weights and biases
are frozen. Input QDQ has its exact forward values and a straight-through input
derivative during calibration only; the original inference hooks are restored.
Gains fold into group scales, retaining the integer codes, selected scope and
activation precision. No labels, evaluation split, extra inference branch or
full Fisher matrix are used. This is an experimental distillation adaptation,
not a modification to the frozen S1Q or official AWQ/GPTQ/SpinQuant methods.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quantization import (_ACTIVE_ATTRIBUTE, _InputTransform, _get_input,
                           QuantizationSession)
from .spinquant_proxy import _RotateInput, _rotate_blocks


class _ExactForwardSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre_quantized: Tensor, quantized: Tensor) -> Tensor:
        # Returning the exact QDQ value avoids subtraction/addition roundoff.
        return quantized

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return gradient, None


class _STEInputHook:
    """Preserve a known session hook's exact forward and its handle position."""

    def __init__(self, original: _InputTransform | _RotateInput):
        self.original = original
        # Evaluation may have populated GPU caches inside inference_mode.
        # Such tensors cannot be saved by autograd, even though they are fixed
        # constants. A private equivalent hook clones them into normal tensors;
        # the original inference hook and cache remain untouched by training.
        if isinstance(original, _InputTransform):
            self._forward = _InputTransform(original.input_scale.detach().clone(), original.activation_bits)
        else:
            self._forward = _RotateInput(original.rotation.detach().clone(), original.activation_bits)
        self._forward._cache = {key: value.detach().clone() for key, value in original._cache.items()}

    def __call__(self, module, args, kwargs):
        transformed_args, transformed_kwargs = self._forward(module, args, kwargs)
        if self.original.activation_bits is None:
            return transformed_args, transformed_kwargs
        value = _get_input(args, kwargs)
        key = (value.device, value.dtype)
        if isinstance(self.original, _InputTransform):
            pre_quantized = value / self._forward._cache[key]
        else:
            pre_quantized = _rotate_blocks(value.float(), self._forward._cache[key]).to(value.dtype)
        quantized = _get_input(transformed_args, transformed_kwargs)
        transformed = _ExactForwardSTE.apply(pre_quantized, quantized.detach())
        if args:
            return (transformed, *args[1:]), transformed_kwargs
        return transformed_args, {**transformed_kwargs, "input": transformed}


def _loss(logits: Sequence[Tensor], teacher: Sequence[Tensor], *, margin_weight: float) -> tuple[Tensor, Tensor, Tensor]:
    if len(logits) != len(teacher) or not logits:
        raise ValueError("Student and teacher must have matching nonempty decision questions.")
    kl_terms, margin_terms = [], []
    for student, reference in zip(logits, teacher):
        if (student.ndim != 1 or reference.ndim != 1 or student.shape != reference.shape
                or student.numel() < 2 or not torch.isfinite(student).all()
                or not torch.isfinite(reference).all()):
            raise ValueError("Student/teacher decisions must be finite matching option vectors with at least two options.")
        student = student.float()
        reference = reference.detach().to(device=student.device, dtype=torch.float32)
        log_teacher = F.log_softmax(reference, dim=-1)
        probabilities = log_teacher.exp()
        kl_terms.append((probabilities * (log_teacher - F.log_softmax(student, dim=-1))).sum())
        top = reference.topk(2).indices
        reference_margin = reference[top[0]] - reference[top[1]]
        student_margin = student[top[0]] - student[top[1]]
        normalizer = reference_margin.abs().clamp_min(1.0)
        margin_terms.append(((student_margin - reference_margin) / normalizer).square())
    kl, margin = torch.stack(kl_terms).mean(), torch.stack(margin_terms).mean()
    return kl + margin_weight * margin, kl, margin


def _calibration_metrics(adapter, records, teachers, *, margin_weight):
    losses, kls, margins, flips, questions = 0.0, 0.0, 0.0, 0, 0
    with torch.inference_mode():
        for record, teacher in zip(records, teachers):
            logits = adapter.infer(record)
            loss, kl, margin = _loss(logits, teacher, margin_weight=margin_weight)
            count = len(logits)
            losses += float(loss.item()) * count
            kls += float(kl.item()) * count
            margins += float(margin.item()) * count
            flips += sum(int(student.argmax().item() != reference.argmax().item())
                         for student, reference in zip(logits, teacher))
            questions += count
    if not questions:
        raise ValueError("Calibration repair requires at least one decision question.")
    return {"decision_loss": losses / questions, "teacher_kl": kls / questions,
            "normalized_margin_mse": margins / questions, "teacher_top1_flips": flips,
            "teacher_top1_flip_rate": flips / questions, "decision_questions": questions}


@torch.inference_mode(False)
def repair_session(
    adapter: Any,
    session: QuantizationSession,
    calibration_records: Sequence[Mapping[str, Any]],
    teacher_logits: Sequence[Sequence[Tensor]],
    *,
    seed: int = 0,
    epochs: int = 2,
    max_records: int = 32,
    learning_rate: float = 1e-3,
    margin_weight: float = 0.1,
    max_log_gain: float = 0.1,
    accept_only_if_improved: bool = True,
) -> dict[str, Any]:
    """Fit label-free bounded row gains and fold them into a live session.

    ``teacher_logits[i]`` must have been collected on the native teacher for
    ``calibration_records[i]`` before applying quantization. Labels are never
    inspected. A deterministic shuffled subset of at most ``max_records`` is
    selected before training and reused for fit diagnostics/acceptance; this is
    calibration loss, not held-out validation or an accuracy guarantee.

    A gain scales each Linear's weight contribution while leaving its existing
    bias untouched. This makes gain folding compatible with the session's
    weight-only restoration. The folded, actual inference path determines the
    reported after metrics; BF16 rounding can differ from a training output
    hook. If this calibration objective fails to improve, the complete gain
    change is rejected by default and the original quantized weights retained.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer.")
    if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
        raise ValueError("max_records must be a positive integer.")
    for name, value, positive in (("learning_rate", learning_rate, True),
                                  ("max_log_gain", max_log_gain, True),
                                  ("margin_weight", margin_weight, False)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
            raise ValueError(f"Invalid {name}.")
    if len(calibration_records) != len(teacher_logits) or not calibration_records:
        raise ValueError("Each nonempty calibration record requires matched detached native teacher logits.")
    if session._restored or not session.layers or set(session.layers) != set(session._modules):
        raise RuntimeError("Decision repair requires an applied, live quantization session with matching layers.")
    if any(not getattr(module, _ACTIVE_ATTRIBUTE, False) for module in session._modules.values()):
        raise RuntimeError("Every repair layer must still belong to its active quantization session.")

    indices = list(range(len(calibration_records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:max_records]
    records = [calibration_records[i] for i in indices]
    teachers = []
    for i in indices:
        if not teacher_logits[i]:
            raise ValueError("Native teacher logits must contain decision questions.")
        values = []
        for z in teacher_logits[i]:
            if not isinstance(z, Tensor) or z.ndim != 1 or z.numel() < 2 or not torch.isfinite(z).all():
                raise ValueError("Native teacher logits must be finite option vectors with at least two options.")
            values.append(z.detach().float().cpu().clone())
        teachers.append(values)

    # Inspect hooks before any mutation; leave unrelated user hooks untouched.
    hook_entries = []
    handle_ids = {handle.id for handle in session._handles}
    for name, module in session._modules.items():
        owned = [(identifier, hook) for identifier, hook in module._forward_pre_hooks.items()
                 if identifier in handle_ids]
        if len(owned) != 1 or not isinstance(owned[0][1], (_InputTransform, _RotateInput)):
            raise RuntimeError(f"Unsupported or ambiguous quantization input hook for {name!r}.")
        hook_entries.append((module, owned[0][0], owned[0][1]))
    if len(hook_entries) != len(handle_ids):
        raise RuntimeError("Quantization session has additional unsupported hooks.")

    model = adapter.model
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    original_layers = dict(session.layers)
    gains = {name: nn.Parameter(torch.zeros(module.out_features, device=module.weight.device,
                                          dtype=torch.float32))
             for name, module in session._modules.items()}
    affine_handles = []
    folded = False
    accepted = False
    steps = 0
    epoch_losses = []
    before = after = proposed = None

    def restore_layer_targets():
        session.layers.clear()
        session.layers.update(original_layers)
        with torch.no_grad():
            for name, module in session._modules.items():
                module.weight.copy_(original_layers[name].transformed_weight(
                    device=module.weight.device, dtype=module.weight.dtype))

    def clear_training_hooks():
        for handle in affine_handles:
            handle.remove()
        affine_handles.clear()
        for module, identifier, original in hook_entries:
            if identifier not in module._forward_pre_hooks:
                raise RuntimeError("An active quantization input hook disappeared during decision repair.")
            module._forward_pre_hooks[identifier] = original

    try:
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        model.eval()
        before = _calibration_metrics(adapter, records, teachers, margin_weight=margin_weight)
        for module, identifier, original in hook_entries:
            module._forward_pre_hooks[identifier] = _STEInputHook(original)
        for name, module in session._modules.items():
            parameter = gains[name]

            def apply_gain(current, args, output, log_gain=parameter):
                if not isinstance(output, Tensor) or output.shape[-1] != log_gain.numel():
                    raise RuntimeError("Decision repair requires Tensor outputs of matching Linear width.")
                gain_delta = log_gain.clamp(-max_log_gain, max_log_gain).exp() - 1.0
                floating = output.float()
                bias = current.bias.detach().float() if current.bias is not None else 0.0
                # Identity at log_gain=0, with the bias excluded from scaling.
                return (floating + (floating - bias) * gain_delta).to(output.dtype)

            affine_handles.append(module.register_forward_hook(apply_gain))
        optimizer = torch.optim.Adam(tuple(gains.values()), lr=learning_rate)
        for _ in range(epochs):
            order = list(range(len(records)))
            rng.shuffle(order)
            total, question_count = 0.0, 0
            for index in order:
                optimizer.zero_grad(set_to_none=True)
                # Adapters must expose their native differentiable raw forward.
                with torch.inference_mode(False), torch.enable_grad():
                    logits = adapter.infer(records[index])
                    loss, _, _ = _loss(logits, teachers[index], margin_weight=margin_weight)
                    if not loss.requires_grad or not bool(torch.isfinite(loss)):
                        raise RuntimeError("Native decision path has no finite gradient to row gains.")
                    loss.backward()
                if not any(gain.grad is not None for gain in gains.values()):
                    raise RuntimeError("Native decisions are disconnected from all repair gains.")
                if any(gain.grad is not None and not torch.isfinite(gain.grad).all() for gain in gains.values()):
                    raise RuntimeError("Decision repair produced nonfinite gain gradients.")
                optimizer.step()
                with torch.no_grad():
                    for gain in gains.values():
                        gain.clamp_(-max_log_gain, max_log_gain)
                total += float(loss.detach().item()) * len(logits)
                question_count += len(logits)
                steps += 1
            epoch_losses.append(total / question_count)

        # Remove affine/STE before measuring the folded inference path.
        clear_training_hooks()
        repaired_layers = {}
        with torch.no_grad():
            for name, layer in original_layers.items():
                gain = gains[name].detach().clamp(-max_log_gain, max_log_gain).exp().cpu()
                repaired_layers[name] = replace(layer, scales=layer.scales * gain[:, None])
            session.layers.clear()
            session.layers.update(repaired_layers)
            folded = True
            for name, module in session._modules.items():
                module.weight.copy_(session.layers[name].transformed_weight(
                    device=module.weight.device, dtype=module.weight.dtype))
        proposed = _calibration_metrics(adapter, records, teachers, margin_weight=margin_weight)
        accepted = bool(epochs > 0 and (not accept_only_if_improved
                                      or proposed["decision_loss"] < before["decision_loss"]))
        if not accepted:
            restore_layer_targets()
            folded = False
            after = dict(before)
        else:
            after = proposed
    except BaseException:
        if folded:
            restore_layer_targets()
            folded = False
        raise
    finally:
        clear_training_hooks()
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training

    summaries = {}
    for name, log_gain in gains.items():
        value = log_gain.detach().float().cpu()
        summaries[name] = {
            "channels": value.numel(), "log_gain_min": float(value.min()),
            "log_gain_max": float(value.max()), "log_gain_mean": float(value.mean()),
            "log_gain_sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),
            "at_bound_channels": int((value.abs() >= max_log_gain - 1e-7).sum()),
        }
    return {
        "mechanism": "bounded_gain_only_native_decision_distillation",
        "teacher": "detached_native_categorical_logits_supplied_before_quantization",
        "objective": "teacher_categorical_KL_plus_normalized_top1_runnerup_margin_mse",
        "uses_gold_labels": False, "uses_development_or_test": False,
        "seed": seed, "epochs": epochs, "steps": steps,
        "learning_rate": float(learning_rate), "margin_weight": float(margin_weight),
        "margin_normalization_floor": 1.0, "max_log_gain": float(max_log_gain),
        "calibration_records": len(records), "selected_record_indices": indices,
        "trainable_parameters": sum(gain.numel() for gain in gains.values()),
        "extra_inference_parameters": 0, "integer_codes_unchanged": True,
        "existing_biases_unchanged": True, "gain_fold": "multiply_existing_group_scales_by_row_gain",
        "original_inference_hooks_restored": True,
        "training_activation_gradient": "straight_through_exact_QDQ_forward",
        "accept_only_if_improved": accept_only_if_improved,
        "acceptance_scope": "same_fitting_calibration_subset_not_heldout_validation",
        "accepted": accepted, "before": before, "proposed_folded": proposed,
        "after": after, "epoch_training_losses": epoch_losses, "layers": summaries,
    }
