"""Fixed development-only pilot across matched S1Q and adapted PTQ baselines.

This script never reads the final test split. It is for method screening, not a
publishable test comparison. Every candidate shares accepted request IDs,
calibration records, selected backbone linears, W bits/group size and A QDQ.
Official AWQ/GPTQ/SmoothQuant/SpinQuant code is not used; those method labels
carry 'adapted' in every report. SpinQuant proxies optimize shared per-Linear
block rotations and do not implement the official mergeable R1/R2 or online
R3/R4 architecture transforms. Run with a fresh output directory each time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from s1q import baselines as baseline_module
from s1q.baselines import quantize_baseline_model
from s1q.data import file_sha256, load_records
from s1q.decision_stats import DecisionFisherCollector
from s1q.experiment import evaluate, predict, storage_report, write_json, write_rows
from s1q.metrics import compare
from s1q.models import load_model
from s1q.quantization import CalibrationCollector, quantize_model, selected_linear_modules


METHOD_LABELS = {
    "rtn": "RTN (matched symmetric group quantizer)",
    "s1q-local": "S1Q local v0.1.0",
    "s1q2-beta0": "S1Q2 correlated reconstruction ablation (beta=0)",
    "s1q2-beta05": "S1Q2 correlated reconstruction (beta=0.5)",
    "s1q2-fine-noclip": "S1Q2 alpha grid 0..0.95 step 0.05, no clipping (ablation)",
    "s1q-fisher-boundary": "S1Q local + calibration-label decision-boundary Fisher",
    "s1q2-fisher-teacher": "S1Q2 beta=0.5 + teacher decision Fisher",
    "s1q2-fisher-boundary": "S1Q2 beta=0.5 + calibration-label decision-boundary Fisher",
    "s1q3-joint": "S1Q3 joint W/A reconstruction, RMS and mean-absolute scaling candidates",
    "s1q3-noclip": "S1Q3 joint W/A reconstruction without weight clipping",
    "s1q3-beta1-noclip": "S1Q3 pure joint W/A reconstruction without weight clipping",
    "gptq-blockdiag-adapted": "GPTQ adapted (block-diagonal Hessian)",
    "gptq-full-adapted": "GPTQ adapted (full Hessian)",
    "awq-adapted": "AWQ adapted (single-Linear output search)",
    "smoothquant-adapted": "SmoothQuant adapted (input hook, no fusion)",
    "spinquant-nohad-adapted": "SpinQuant-inspired learned block rotation proxy (no Hadamard; online hook)",
    "spinquant-had-adapted": "SpinQuant-inspired learned block rotation proxy + online Hadamard",
}


def _identity(record: dict, index: int) -> str:
    return str(record.get("id") or (record.get("_meta") or {}).get("id") or index)


def _subset(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("Pilot subset sizes must be positive.")
    ranked = sorted(enumerate(records), key=lambda item: (
        hashlib.sha256(f"{seed}|{_identity(item[1], item[0])}".encode()).hexdigest(), item[0]))
    return [record for _, record in ranked[:limit]]


def _memory(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    return {"allocated_mib": torch.cuda.memory_allocated(device) / 1048576,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1048576}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_pilot(
    *, model_name: str, data_dir: str | Path, output_dir: str | Path,
    methods: tuple[str, ...], bits: int = 4, group_size: int = 128,
    activation_bits: int | None = 4, calibration_count: int = 16,
    development_count: int = 32, reservoir_size: int = 32,
    max_reservoir_rows: int = 32,
    seed: int = 20261001, device: str = "cuda", dtype: str = "bf16",
    source_dir: str | None = None, checkpoint_dir: str | None = None,
    smooth_alpha: float = 0.5,
    gptq_damping: float = 0.01, fisher_probes_per_record: int = 2,
    checkpoint_manifest: str | None = None,
    spin_block_size: int = 16, spin_steps: int = 12,
    spin_learning_rate: float = 0.03,
    spin_max_train_rows: int = 16, spin_max_train_outputs: int = 32,
    spin_max_train_groups: int = 4,
) -> dict:
    if bits not in (2, 3, 4, 8):
        raise ValueError("The shared packed quantizer supports W2/W3/W4/W8.")
    if activation_bits not in (None, 4, 8):
        raise ValueError("activation_bits must be None, 4, or 8.")
    if not methods or any(method not in METHOD_LABELS for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("Provide nonempty, distinct known methods.")
    if group_size <= 0 or reservoir_size <= 0 or max_reservoir_rows <= 0 or fisher_probes_per_record <= 0:
        raise ValueError("group_size, reservoir_size, max_reservoir_rows and fisher_probes_per_record must be positive.")

    data = Path(data_dir)
    calibration_path, development_path = data / "calibration.jsonl", data / "development.jsonl"
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    calibration = _subset(load_records(calibration_path), calibration_count, seed)
    development = _subset(load_records(development_path), development_count, seed)
    output.mkdir(parents=True)

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    adapter = load_model(model_name, device=device, dtype=dtype, source_dir=source_dir,
                         checkpoint_dir=checkpoint_dir,
                         checkpoint_manifest=checkpoint_manifest)
    adapter.model.eval()
    target_device = torch.device(adapter.device)
    _, accepted_calibration, calibration_exclusions, _ = predict(adapter, calibration, allow_rejection=True)
    native_rows, accepted_development, development_exclusions, _ = predict(adapter, development, allow_rejection=True)
    if not accepted_calibration or not accepted_development:
        raise ValueError("No native-admitted calibration or development records in pilot subset.")

    with torch.inference_mode(), CalibrationCollector(
        adapter.backbone, reservoir_size=reservoir_size, seed=seed,
    ) as collector:
        for record in accepted_calibration:
            adapter.infer(record)
    statistics = collector.statistics()
    scope = selected_linear_modules(adapter.backbone)
    if set(statistics) != set(scope):
        raise RuntimeError("Calibration statistics do not cover the selected backbone Linear scope.")
    scope_parameters = sum(module.weight.numel() for module in scope.values())
    identity = {
        "cohort": "fixed_development_pilot_only", "final_test_read": False,
        "model": adapter.metadata, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None,
        "seed": seed, "bits": bits, "activation_bits": activation_bits,
        "group_size": group_size, "reservoir_size": reservoir_size,
        "s1q2_max_reservoir_rows": max_reservoir_rows,
        "fisher_probes_per_record": fisher_probes_per_record,
        "requested_calibration": calibration_count, "accepted_calibration": len(accepted_calibration),
        "requested_development": development_count, "accepted_development": len(accepted_development),
        "calibration_exclusions": calibration_exclusions,
        "development_exclusions": development_exclusions,
        "accepted_calibration_ids": [_identity(record, i) for i, record in enumerate(accepted_calibration)],
        "accepted_development_ids": [_identity(record, i) for i, record in enumerate(accepted_development)],
        "source_sha256": {"calibration": file_sha256(calibration_path),
                          "development": file_sha256(development_path)},
        "selected_linear_count": len(scope), "selected_weight_parameters": scope_parameters,
        "selected_linear_names": sorted(scope),
        "baseline_module_sha256": file_sha256(baseline_module.__file__),
        "implementation_sha256": {
            name: file_sha256(Path(__file__).resolve().parents[1] / relative)
            for name, relative in {
                "quantization.py": "src/s1q/quantization.py",
                "spinquant_proxy.py": "src/s1q/spinquant_proxy.py",
                "models.py": "src/s1q/models.py",
                "experiment.py": "src/s1q/experiment.py",
                "compare_baselines.py": "scripts/compare_baselines.py",
            }.items()
        },
    }
    report = {"identity": identity, "native_development": evaluate(native_rows, {}),
              "methodology": {"official_awq_gptq_smoothquant_spinquant": False,
                              "same_scope_bits_groups_activation_and_records": True,
                              "execution": "floating dequantization and optional activation QDQ; no integer kernel",
                              "spinquant_adaptation": "calibration-optimized shared block rotations per Linear; no official architecture fusion; extra online hook and rotation storage"},
              "methods": {}, "decision_fisher": {}}
    write_rows(output / "native-development.jsonl", native_rows)
    write_json(output / "identity.json", identity)
    write_json(output / "report.json", report)

    fisher_statistics = {}

    def decision_statistics(weighting: str):
        if weighting not in fisher_statistics:
            with DecisionFisherCollector(
                adapter, modules=scope, seed=seed, probes_per_record=fisher_probes_per_record,
                decision_weighting=weighting,
            ) as collector:
                for record in accepted_calibration:
                    collector.collect(record)
            fisher_statistics[weighting] = collector.attach_input_statistics(statistics)
            report["decision_fisher"][weighting] = collector.metadata()
            write_json(output / "report.json", report)
        return fisher_statistics[weighting]

    rows_by_method: dict[str, list[dict]] = {}
    for method in methods:
        print(json.dumps({"stage": "pilot_method", "method": method}), flush=True)
        profile = {"label": METHOD_LABELS[method], "bits": bits, "activation_bits": activation_bits,
                   "group_size": group_size, "calibration_reservoir_size": reservoir_size}
        _sync(target_device)
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
        before = _memory(target_device)
        start = time.perf_counter()
        try:
            if method in ("s1q-local", "s1q2-beta0", "s1q2-beta05", "s1q2-fine-noclip", "s1q3-joint",
                          "s1q3-noclip", "s1q3-beta1-noclip",
                          "s1q-fisher-boundary", "s1q2-fisher-teacher", "s1q2-fisher-boundary"):
                blend = 0.0 if method == "s1q2-beta0" else 1.0 if method == "s1q3-beta1-noclip" else 0.5
                old_s1q = method in ("s1q-local", "s1q-fisher-boundary")
                profile["reservoir_blend"] = None if old_s1q else blend
                weighting = ("correct_boundary" if method.endswith("fisher-boundary") else
                             "teacher" if method.endswith("fisher-teacher") else None)
                profile["decision_weighting"] = weighting
                joint = method.startswith("s1q3-")
                no_clip = "noclip" in method
                fine = method == "s1q2-fine-noclip"
                if fine:
                    profile["alphas"] = [step / 20 for step in range(20)]
                    profile["search_grid_clip_ablation"] = True
                if joint:
                    profile["scale_families"] = ["rms", "mean_abs"]
                    profile["joint_weight_activation_search"] = True
                    profile["clipping_ratios"] = [1.0] if no_clip else [0.9, 0.95, 1.0]
                method_statistics = statistics if weighting is None else decision_statistics(weighting)
                session = quantize_model(
                    adapter.backbone, method="s1q" if old_s1q else "s1q3" if joint else "s1q2",
                    bits=bits, group_size=group_size, statistics=method_statistics,
                    activation_bits=activation_bits,
                    reservoir_blend=blend, max_reservoir_rows=max_reservoir_rows,
                    clipping_ratios=(1.0,) if no_clip else (0.9, 0.95, 1.0),
                    alphas=tuple(step / 20 for step in range(20)) if fine else (0.0, 0.25, 0.5, 0.75),
                )
            else:
                if method.startswith("spinquant-"):
                    profile.update({"official_spinquant": False,
                                    "rotation_block_size": spin_block_size,
                                    "rotation_steps": spin_steps,
                                    "rotation_learning_rate": spin_learning_rate,
                                    "rotation_train_limits": {"rows": spin_max_train_rows,
                                                              "outputs": spin_max_train_outputs,
                                                              "groups": spin_max_train_groups}})
                if method == "smoothquant-adapted" and activation_bits is None:
                    profile["weight_only_channel_balancing_ablation"] = True
                session = quantize_baseline_model(
                    adapter.backbone, method=method, bits=bits,
                    group_size=group_size, statistics=statistics,
                    activation_bits=activation_bits, smooth_alpha=smooth_alpha,
                    gptq_damping=gptq_damping,
                    spin_block_size=spin_block_size, spin_steps=spin_steps,
                    spin_learning_rate=spin_learning_rate,
                    spin_max_train_rows=spin_max_train_rows,
                    spin_max_train_outputs=spin_max_train_outputs,
                    spin_max_train_groups=spin_max_train_groups,
                )
            with session:
                _sync(target_device)
                quantization_seconds = time.perf_counter() - start
                quantization_memory = _memory(target_device)
                if set(session.layers) != set(scope):
                    raise RuntimeError("Baseline used a different selected Linear scope.")
                rows, _, _, _ = predict(adapter, accepted_development)
                _sync(target_device)
                storage = storage_report(adapter, session)
                if method.startswith("spinquant-"):
                    rotation_bytes = session.report()["extra_rotation_bytes_fp32"]
                    storage["rotation_bytes_fp32"] = rotation_bytes
                    storage["estimated_complete_packed_parameters_bytes"] += rotation_bytes
                    storage["estimated_parameter_storage_ratio"] = (
                        storage["estimated_complete_packed_parameters_bytes"] / storage["native_parameters_bytes"])
                    storage["note"] += " SpinQuant proxy rotation matrices and online rotation are included; generic packed artifact export is unsupported."
                result = {"status": "complete", "profile": profile,
                          "quantization_seconds": quantization_seconds,
                          "total_seconds": time.perf_counter() - start,
                          "memory_before": before,
                          "memory_after_quantization": quantization_memory,
                          "memory_peak": _memory(target_device),
                          "quantized_linear_count": len(session.layers),
                          "quantized_weight_parameters": sum(layer.integer_weight.numel() for layer in session.layers.values()),
                          "preserved_layers": dict(session.preserved_layers),
                          "storage": storage,
                          "development": evaluate(rows, {}),
                          "paired_vs_native": compare(native_rows, rows),
                          "quantization": session.report()}
                write_rows(output / f"{method}-development.jsonl", rows)
                rows_by_method[method] = rows
        except Exception as exc:
            _sync(target_device)
            result = {"status": "failed", "profile": profile,
                      "seconds_before_failure": time.perf_counter() - start,
                      "memory_before": before, "memory_at_failure": _memory(target_device),
                      "error_type": type(exc).__name__, "error": str(exc)}
        report["methods"][method] = result
        write_json(output / "report.json", report)
    if "rtn" in rows_by_method:
        rtn_rows = rows_by_method["rtn"]
        for method, rows in rows_by_method.items():
            report["methods"][method]["paired_vs_rtn"] = compare(rtn_rows, rows)
        write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["rtn", "s1q-local", "s1q2-beta0", "s1q2-beta05", "gptq-blockdiag-adapted", "awq-adapted", "smoothquant-adapted", "spinquant-nohad-adapted", "spinquant-had-adapted"],
                        choices=tuple(METHOD_LABELS))
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--no-activation-quantization", action="store_true",
                        help="Weight-only setting (W3): leave activations in native precision")
    parser.add_argument("--calibration-count", type=int, default=16)
    parser.add_argument("--development-count", type=int, default=32)
    parser.add_argument("--reservoir-size", type=int, default=32)
    parser.add_argument("--max-reservoir-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--source-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-manifest",
                        help="Optional JSON of SHA-256 hashes for files relative to --checkpoint-dir")
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--gptq-damping", type=float, default=0.01)
    parser.add_argument("--spin-block-size", type=int, default=16)
    parser.add_argument("--spin-steps", type=int, default=12)
    parser.add_argument("--spin-learning-rate", type=float, default=0.03)
    parser.add_argument("--spin-max-train-rows", type=int, default=16)
    parser.add_argument("--spin-max-train-outputs", type=int, default=32)
    parser.add_argument("--spin-max-train-groups", type=int, default=4)
    parser.add_argument("--fisher-probes-per-record", type=int, default=2)
    args = parser.parse_args()
    result = run_pilot(model_name=args.model, data_dir=args.data_dir,
                       output_dir=args.output_dir, methods=tuple(args.methods),
                       bits=args.bits, group_size=args.group_size,
                       activation_bits=None if args.no_activation_quantization else args.activation_bits,
                       calibration_count=args.calibration_count,
                       development_count=args.development_count,
                       reservoir_size=args.reservoir_size,
                       max_reservoir_rows=args.max_reservoir_rows, seed=args.seed,
                       device=args.device, dtype=args.dtype,
                       source_dir=args.source_dir, checkpoint_dir=args.checkpoint_dir,
                       checkpoint_manifest=args.checkpoint_manifest,
                       smooth_alpha=args.smooth_alpha,
                       gptq_damping=args.gptq_damping,
                       fisher_probes_per_record=args.fisher_probes_per_record,
                       spin_block_size=args.spin_block_size,
                       spin_steps=args.spin_steps,
                       spin_learning_rate=args.spin_learning_rate,
                       spin_max_train_rows=args.spin_max_train_rows,
                       spin_max_train_outputs=args.spin_max_train_outputs,
                       spin_max_train_groups=args.spin_max_train_groups)
    print(json.dumps({"stage": "complete", "output": args.output_dir,
                      "statuses": {key: value["status"] for key, value in result["methods"].items()}}), flush=True)


if __name__ == "__main__":
    main()
