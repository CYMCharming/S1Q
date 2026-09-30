"""Evaluate a locked completed S1Q run on a prepared external-only cohort.

No calibration, temperature fitting, selection, or resampling of the cohort is
performed here. The dense artifact loader reproduces the selected run's forward
path. RTN uses exactly its exported module mask and activation scope. Outputs
include a pre-inference freeze, native eligibility, paired logits, and metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s1q.data import (file_sha256, group_id, load_records, model_request,
                      question_keys, validate_record)
from s1q.experiment import write_json, write_rows
from s1q.metrics import compare, label_index, metrics, paired_accuracy_ci, probabilities
from s1q.models import MODEL_REVISIONS, SOURCE_REVISIONS, UnsupportedRecord, load_model
from s1q.quantization import load_quantized_artifact, quantize_model


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def frozen_temperatures(report: dict, *, split_container: bool = False) -> dict:
    """Recover the already-fitted values; never fit on external predictions."""
    reports = report.get("splits", {}) if split_container else report
    values = [item["temperatures"] for item in reports.values()
              if isinstance(item, dict) and "temperatures" in item]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError("Completed run needs one consistent frozen temperature mapping")
    for kind, temperature in values[0].items():
        if kind not in ("noul", "choice", "score") or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid frozen decision temperature")
    return dict(values[0])


def layer_scope_matches(report: dict, artifact: dict) -> bool:
    scope = report.get("quantization", {})
    layers = scope.get("layers", {})
    if set(layers) != set(artifact["layers"]) or scope.get("activation_bits") != artifact.get("activation_bits"):
        return False
    return all(all(layers[name].get(field) == record.get(field)
                   for field in ("bits", "group_size", "shape"))
               for name, record in artifact["layers"].items())


def lock_inputs(model: str, run_dir: Path, artifact_path: Path, data_dir: Path,
                *, dtype: str, seed: int, bootstrap_samples: int) -> tuple[dict, dict, dict, list[dict]]:
    """Validate immutable inputs before loading a model or seeing predictions."""
    paths = {"run_summary": run_dir / "summary.json", "run_selection": run_dir / "selection.json",
             "run_environment": run_dir / "environment.json", "artifact": artifact_path,
             "external_manifest": data_dir / "manifest.json", "external_data": data_dir / "external_test.jsonl"}
    hashes = {name: {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
              for name, path in paths.items()}
    summary, selection, environment = (read_json(paths[key]) for key in
                                        ("run_summary", "run_selection", "run_environment"))
    manifest = read_json(paths["external_manifest"])
    if summary.get("status") != "complete" or summary.get("model") != model:
        raise ValueError("External evaluation requires a completed run of the requested model")
    if summary.get("selection") != selection or summary.get("environment") != environment:
        raise ValueError("Standalone selection/environment disagree with completed summary")
    if not selection.get("selected_before_quantized_test") or selection.get("selected", {}).get("method") != "s1q":
        raise ValueError("Need a frozen development-selected S1Q profile")
    if model not in MODEL_REVISIONS:
        raise ValueError("Unknown pinned model")
    expected_model = environment.get("model", {})
    family = "kev" if model.startswith("kev-") else "NanoJev" if model == "nanojev" else "laya"
    expected_dtype = {"bf16": "torch.bfloat16", "bfloat16": "torch.bfloat16",
                      "fp16": "torch.float16", "float16": "torch.float16",
                      "fp32": "torch.float32", "float32": "torch.float32"}.get(dtype)
    if expected_dtype is None or expected_model.get("dtype") != expected_dtype:
        raise ValueError("External dtype must match the completed run")
    if (expected_model.get("name") != model or expected_model.get("revision") != MODEL_REVISIONS[model][1]
            or expected_model.get("source_revision") != SOURCE_REVISIONS[family]):
        raise ValueError("Completed run does not identify the currently pinned native model/source")
    selected_name = selection["selected"]["name"]
    selected = summary["quantized"][selected_name]
    if selected.get("profile") != selection["selected"]:
        raise ValueError("Selected result profile differs from the locked selection")
    expected_artifact = selected.get("artifact", {})
    if (expected_artifact.get("sha256") != hashes["artifact"]["sha256"]
            or expected_artifact.get("bytes") != hashes["artifact"]["bytes"]):
        raise ValueError("Selected artifact integrity mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if (not isinstance(artifact, dict) or artifact.get("format") != "s1q.packed_linear.v1"
            or artifact.get("execution") != "dequantized_weights_with_optional_fake_activation"
            or not artifact.get("layers")):
        raise ValueError("Need the selected dense-reference-compatible S1Q artifact")
    if not layer_scope_matches(selected, artifact):
        raise ValueError("Artifact scope differs from selected run result")
    if (artifact.get("activation_bits") != selection["selected"].get("activation_bits")
            or any(r.get("bits") != selection["bits"] or r.get("group_size") != selection["group_size"]
                   for r in artifact["layers"].values())):
        raise ValueError("Artifact precision differs from locked selection")
    if (manifest.get("schema") != "s1q.jevbench-external.v1"
            or manifest.get("role") != "external_test_only" or manifest.get("selection_uses_predictions") is not False):
        raise ValueError("Need a prepared external-only JevBench manifest without prediction selection")
    pinned_data = read_json(ROOT / "configs/jevbench-data-checksums.json")
    if (manifest.get("source_repo") != pinned_data["repo"] or manifest.get("source_revision") != pinned_data["revision"]
            or manifest.get("source_files") != {name: pinned_data["files"][name]
                                               for name in ("datasets/public/original.jsonl", "datasets/public/hard.jsonl")}):
        raise ValueError("External source provenance differs from pinned JevBench release")
    if manifest.get("output", {}).get("external_test.jsonl", {}).get("sha256") != hashes["external_data"]["sha256"]:
        raise ValueError("Prepared external data hash differs from its manifest")
    records = load_records(paths["external_data"])
    for record in records:
        validate_record(record)
        if not (record.get("_meta") or {}).get("id"):
            raise ValueError("External records require stable identities")
    if (not records or len(records) != manifest.get("accepted_records")
            or sum(len(r["questions"]) for r in records) != manifest.get("accepted_questions")):
        raise ValueError("External record/question count differs from prepared manifest")
    baseline_temperature = frozen_temperatures(summary["baseline"])
    selected_temperature = frozen_temperatures(selected, split_container=True)
    rtn_report = next((item for item in summary["quantized"].values()
                       if item.get("profile", {}).get("method") == "rtn" and layer_scope_matches(item, artifact)), None)
    rtn_temperature = frozen_temperatures(rtn_report, split_container=True) if rtn_report else None
    current_packages = {}
    for package, version in environment.get("packages", {}).items():
        try:
            current_packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(f"Locked runtime package missing: {package}") from None
        if current_packages[package] != version:
            raise ValueError(f"Locked runtime package mismatch: {package} expected {version}, found {current_packages[package]}")
    current_sources = {name: file_sha256(ROOT / "src/s1q" / name)
                       for name in ("models.py", "quantization.py", "metrics.py")}
    for name, actual in current_sources.items():
        previous = environment.get("source_files_sha256", {}).get(name)
        if previous is not None and previous != actual:
            raise ValueError(f"Locked inference/evaluation source changed: {name}")
    freeze = {"schema": "s1q.external-freeze.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "model": model, "model_metadata": expected_model, "selection": selection,
              "inputs": hashes, "script_sha256": file_sha256(Path(__file__)),
              "inference_source_files_sha256": current_sources, "packages": current_packages,
              "platform": platform.platform(), "seed": seed, "bootstrap_samples": bootstrap_samples,
              "cohort": {"records": len(records), "questions": manifest["accepted_questions"],
                         "source_revision": manifest["source_revision"], "sampling": "entire prepared cohort; no subsampling"},
              "temperatures": {"baseline": baseline_temperature, "selected": selected_temperature,
                               "rtn_matched": rtn_temperature},
              "rtn_temperature_status": "frozen_from_scope_matched_run" if rtn_report else "unavailable_no_scope_matched_fitted_run",
              "rtn_scope": {"layers": sorted(artifact["layers"]), "bits": selection["bits"],
                            "group_size": selection["group_size"], "activation_bits": artifact.get("activation_bits"),
                            "protected_mask": artifact.get("preserved_layers", {})},
              "policy": {"external_fitting": False, "external_selection": False,
                         "eligibility": "native floating-point admission, frozen before quantized predictions",
                         "tie_break": "first native candidate in insertion order; differs from upstream lexicographic tie break",
                         "probability_gold": "reported separately from authored hard/modal gold; normalized by supplied sum"}}
    return freeze, summary, artifact, records


def verify_frozen_files(freeze: dict) -> None:
    for item in freeze["inputs"].values():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed during evaluation: {item['path']}")


@torch.inference_mode()
def external_predict(adapter, records: list[dict], *, allow_rejection: bool = False):
    rows, accepted, rejected = [], [], []
    for record in records:
        rid = str(record["_meta"]["id"])
        try:
            # Gold, probability targets, provenance, and private rationales never reach infer.
            outputs = adapter.infer(model_request(record))
        except (UnsupportedRecord, ValueError) as error:
            if not allow_rejection:
                raise
            rejected.append({"record_id": rid, "questions": len(record["questions"]),
                             "reason": str(error), "exception": type(error).__name__})
            continue
        if len(outputs) != len(record["questions"]):
            raise ValueError("Native question count changed during external inference")
        accepted.append(record)
        for (qid, question), output in zip(record["questions"].items(), outputs):
            keys = question_keys(question)
            if output.ndim != 1 or len(output) != len(keys) or not torch.isfinite(output).all():
                raise ValueError("Native candidate mapping/nonfinite output in external inference")
            row = {"key": f"{rid}::{qid}", "record_id": rid, "qid": qid, "type": question["type"],
                   "source": question.get("src", record["_meta"].get("source", "unknown")),
                   "tier": record["_meta"].get("tier", "unknown"), "family": record["_meta"].get("family", "unknown"),
                   "cluster_id": group_id(record), "label": label_index(question), "candidate_keys": keys,
                   "label_kind": question.get("label_kind", "authored_scenario_gold"),
                   "logits": output.detach().float().cpu().tolist()}
            if "target_distribution" in question:
                target = question["target_distribution"]
                if (not isinstance(target, dict) or set(target) != set(keys)
                        or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 or x > 1 for x in target.values())
                        or abs(sum(target.values()) - 1) > 1e-3):
                    raise ValueError("Invalid independent soft probability target")
                row["target_distribution"] = target
                row["target_distribution_kind"] = question.get("target_distribution_kind", "unknown")
            rows.append(row)
    return rows, accepted, rejected


def soft_target_metrics(rows: list[dict], temperatures: dict | None = None) -> dict:
    items = [row for row in rows if "target_distribution" in row]
    if not items:
        return {"n": 0}
    temperatures = temperatures or {}
    cross_entropy, kl, squared_error, absolute_error = [], [], [], []
    for row in items:
        target = np.array([row["target_distribution"][k] for k in row["candidate_keys"]], dtype=float)
        target = target / target.sum()
        p = probabilities(row["logits"], temperatures.get(row["type"], 1.0))
        ce = float(-np.sum(target * np.log(np.maximum(p, 1e-15))))
        entropy = float(-np.sum(target * np.log(np.maximum(target, 1e-15))))
        cross_entropy.append(ce); kl.append(max(ce - entropy, 0.0))
        squared_error.append(float(np.sum((p - target) ** 2)))
        absolute_error.append(float(np.sum(np.abs(p - target))))
    return {"n": len(items), "cross_entropy": float(np.mean(cross_entropy)), "kl_target_to_prediction": float(np.mean(kl)),
            "squared_probability_error": float(np.mean(squared_error)), "l1_probability_error": float(np.mean(absolute_error)),
            "interpretation": "distance to state-derivable authored probability, not sampled-event calibration",
            "target_normalization": "divide by original named target sum"}


def external_metrics(rows: list[dict], temperatures: dict | None) -> dict:
    result = {"raw": metrics(rows), "raw_soft_targets": soft_target_metrics(rows),
              "questions": len(rows), "records": len({r["record_id"] for r in rows}),
              "ties_at_max": int(sum(np.count_nonzero(np.asarray(r["logits"]) == max(r["logits"])) > 1 for r in rows))}
    if temperatures is not None:
        missing = sorted({r["type"] for r in rows} - set(temperatures))
        result.update(temperature_calibrated=metrics(rows, temperatures),
                      calibrated_soft_targets=soft_target_metrics(rows, temperatures),
                      temperatures=temperatures, types_without_fitted_temperature=missing)
    else:
        result["temperature_calibrated"] = None
    for field in ("type", "tier", "family", "label_kind"):
        result[f"by_{field}"] = {value: {"raw": metrics([r for r in rows if r[field] == value]),
                                         "raw_soft_targets": soft_target_metrics([r for r in rows if r[field] == value])}
                                 for value in sorted({r[field] for r in rows})}
    return result


def run_external(model: str, run_dir: str | Path, artifact_path: str | Path, data_dir: str | Path,
                 output_dir: str | Path, *, device: str = "cuda", dtype: str = "bf16",
                 source_dir: str | Path | None = None, seed: int = 20261001, bootstrap_samples: int = 2000) -> dict:
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("External output already contains a frozen run; use a fresh directory")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    freeze, _, artifact, records = lock_inputs(model, Path(run_dir), Path(artifact_path), Path(data_dir),
                                               dtype=dtype, seed=seed, bootstrap_samples=bootstrap_samples)
    out.mkdir(parents=True, exist_ok=True)
    # An exclusive file is the immutable pre-inference registration for this run.
    with (out / "freeze.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(freeze, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    freeze_hash = file_sha256(out / "freeze.json")
    torch.manual_seed(seed); torch.set_num_threads(4)
    try:
        print(json.dumps({"stage": "external_load", "model": model, "cohort_records": len(records)}), flush=True)
        adapter = load_model(model, device=device, dtype=dtype, source_dir=source_dir)
        adapter.model.eval()
        actual = {k: v for k, v in adapter.metadata.items() if k != "device"}
        expected = {k: v for k, v in freeze["model_metadata"].items() if k != "device"}
        if actual != expected:
            raise ValueError("Native loaded metadata differs from locked completed run")
        verify_frozen_files(freeze)
        baseline, accepted, exclusions = external_predict(adapter, records, allow_rejection=True)
        write_rows(out / "baseline.jsonl", baseline)
        write_json(out / "eligibility.json", {"accepted": [r["_meta"]["id"] for r in accepted],
                                               "accepted_records": len(accepted), "accepted_questions": len(baseline),
                                               "excluded": exclusions,
                                               "excluded_questions": sum(r["questions"] for r in exclusions)})
        if not baseline:
            raise ValueError("No native-eligible external inputs; inspect eligibility.json")
        with load_quantized_artifact(adapter.backbone, artifact) as selected_session:
            selected, _, _ = external_predict(adapter, accepted)
            write_rows(out / "selected.jsonl", selected)
            selected_quantization = selected_session.report()
        # Exact artifact paths form the matched RTN mask. No sensitivity is recollected.
        selected_modules = dict(adapter.backbone.named_modules())
        if any(not isinstance(selected_modules.get(name), nn.Linear) for name in artifact["layers"]):
            raise ValueError("Selected artifact mask is not native Linear modules after restoration")
        with quantize_model(adapter.backbone, method="rtn", bits=freeze["selection"]["bits"],
                            group_size=freeze["selection"]["group_size"], include_prefixes=tuple(artifact["layers"]),
                            exclude_patterns=(), activation_bits=artifact.get("activation_bits")) as rtn_session:
            if set(rtn_session.layers) != set(artifact["layers"]):
                raise ValueError("Matched RTN module mask differs from frozen artifact")
            rtn, _, _ = external_predict(adapter, accepted)
            write_rows(out / "rtn_matched.jsonl", rtn)
            rtn_quantization = rtn_session.report()
            rtn_quantization["matched_protected_layers"] = artifact.get("preserved_layers", {})
        by_key = [{r["key"]: r for r in items} for items in (baseline, selected, rtn)]
        # compare validates identical unique keys and gold before metrics/paired output.
        comparisons = {"selected_vs_baseline": compare(baseline, selected),
                       "rtn_vs_baseline": compare(baseline, rtn), "selected_vs_rtn": compare(rtn, selected)}
        paired = [{"key": key, "record_id": row["record_id"], "qid": row["qid"], "cluster_id": row["cluster_id"],
                   "candidate_keys": row["candidate_keys"], "label": row["label"], "label_kind": row["label_kind"],
                   "baseline_logits": row["logits"], "selected_logits": by_key[1][key]["logits"],
                   "rtn_matched_logits": by_key[2][key]["logits"]} for key, row in by_key[0].items()]
        write_rows(out / "paired_predictions.jsonl", paired)
        confidence_intervals = {name: paired_accuracy_ci(first, second, seed=seed, samples=bootstrap_samples)
                                for name, first, second in (("selected_vs_baseline", baseline, selected),
                                                           ("rtn_vs_baseline", baseline, rtn), ("selected_vs_rtn", rtn, selected))}
        verify_frozen_files(freeze)
        if file_sha256(out / "freeze.json") != freeze_hash:
            raise ValueError("Pre-inference freeze registration changed")
        result = {"status": "complete", "model": model, "freeze_sha256": freeze_hash,
                  "prepared_records": len(records), "prepared_questions": freeze["cohort"]["questions"],
                  "accepted_records": len(accepted), "accepted_questions": len(baseline),
                  "excluded_records": len(exclusions), "excluded_questions": sum(r["questions"] for r in exclusions),
                  "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
                  "baseline": external_metrics(baseline, freeze["temperatures"]["baseline"]),
                  "selected": external_metrics(selected, freeze["temperatures"]["selected"]),
                  "rtn_matched": external_metrics(rtn, freeze["temperatures"]["rtn_matched"]),
                  "rtn_temperature_status": freeze["rtn_temperature_status"],
                  "paired": comparisons, "paired_accuracy_ci": confidence_intervals,
                  "quantization": {"selected": selected_quantization, "rtn_matched": rtn_quantization},
                  "inference_metadata": adapter.metadata,
                  "limitations": ["Public authored original/hard cohort, not official full JevBench ranking.",
                                  "Authored/modal target agreement is distinct from observed-event calibration.",
                                  "No external calibration, temperature fitting, method or precision selection.",
                                  "Native eligibility is frozen before quantized inference; post-admission failures abort.",
                                  "Dense dequantized artifact execution with optional fake activations; no integer-kernel speed claim.",
                                  "Common cohort across model families requires intersecting their predeclared native eligibility."]}
        write_json(out / "summary.json", result)
        print(json.dumps({"stage": "external_complete", "model": model, "questions": len(baseline)}), flush=True)
        return result
    except BaseException as error:
        write_json(out / "failure.json", {"status": "failed", "freeze_sha256": freeze_hash,
                                          "exception": type(error).__name__, "reason": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_REVISIONS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    run_external(args.model, args.run_dir, args.artifact, args.data_dir, args.out,
                 device=args.device, dtype=args.dtype, source_dir=args.source_dir,
                 seed=args.seed, bootstrap_samples=args.bootstrap_samples)


if __name__ == "__main__":
    main()
