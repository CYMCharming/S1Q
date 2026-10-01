"""Aggregate completed S1Q runs without silently selecting favorable test results.

Default command:
    python scripts/summarize_results.py

Produces docs/results.md, results/aggregate.json and results/metrics.csv. Inputs
are status=complete summary.json files and optional external-summary.json files.
Pairwise intervals require saved paired logits; missing matched RTN comparisons
remain explicitly unavailable. No models or GPUs are loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s1q.metrics import paired_accuracy_ci  # noqa: E402

METRIC_KEYS = ("n", "accuracy", "macro_source_accuracy", "nll", "brier", "ece_15", "score_mae")
SPLIT_ORDER = {"test": 0, "transfer_test": 1, "ood": 2, "external_test": 3}
MODEL_ORDER = {"kev-0.8b": 0, "kev-4b": 1, "kev-9b": 2, "nanojev": 3, "laya": 4}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def metrics_subset(evaluation: dict) -> dict:
    result = {}
    for stage in ("raw", "temperature_calibrated"):
        source = evaluation.get(stage) or {}
        result[stage] = {key: source[key] for key in METRIC_KEYS if key in source}
    result["temperatures"] = evaluation.get("temperatures", {})
    for field in ("label_kind", "type", "tier", "family"):
        key = f"by_{field}"
        if key in evaluation:
            result[key] = {name: metrics_subset(value) for name, value in evaluation[key].items()}
    for key in ("raw_soft_targets", "calibrated_soft_targets", "types_without_fitted_temperature", "ties_at_max"):
        if key in evaluation:
            result[key] = evaluation[key]
    if "paired" in evaluation:
        result["paired_fidelity_to_native"] = evaluation["paired"]
    if "interpretation" in evaluation:
        result["interpretation"] = evaluation["interpretation"]
    return result


def format_signature(report: dict) -> dict:
    """Match actual quantized/protected tensors, not merely profile names."""
    quant = report.get("quantization", {})
    layers = quant.get("layers", {})
    storage = report.get("storage", {})
    return {
        "activation_bits": quant.get("activation_bits", report.get("profile", {}).get("activation_bits")),
        "activation_scheme": quant.get("activation_scheme"),
        "preserved_layers": sorted(quant.get("preserved_layers", {})),
        "layers": {name: {key: layer.get(key) for key in ("bits", "group_size", "shape")}
                   for name, layer in sorted(layers.items())},
        "quantized_parameter_count": storage.get("quantized_parameter_count"),
        "retained_native_parameters_bytes": storage.get("retained_native_parameters_bytes"),
    }


def matched_rtn(quantized: dict, selected_name: str) -> tuple[str | None, dict]:
    selected = quantized.get(selected_name)
    if selected is None:
        return None, {"status": "selected_profile_missing"}
    signature = format_signature(selected)
    if not signature["layers"]:
        return None, {"status": "format_metadata_missing"}
    candidates, rejected = [], {}
    for name, report in quantized.items():
        if report.get("profile", {}).get("method") != "rtn":
            continue
        other = format_signature(report)
        mismatch = [key for key in signature if other[key] != signature[key]]
        if mismatch:
            rejected[name] = mismatch
        else:
            candidates.append(name)
    if not candidates:
        return None, {"status": "no_matched_rtn", "rejected_candidates": rejected}
    candidates.sort(key=lambda name: (name != "rtn-matched", name != "rtn", name))
    return candidates[0], {"status": "matched", "match_basis": "Identical activation scheme/bits, protected layer identities, quantized tensor shapes/bits/groups and retained parameter count.",
                            "rejected_candidates": rejected}


def read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def comparison(reference_path: Path, selected_path: Path, *, seed: int, samples: int) -> dict:
    if not reference_path.exists() or not selected_path.exists():
        return {"status": "paired_predictions_unavailable"}
    reference, selected = read_predictions(reference_path), read_predictions(selected_path)
    left, right = {row["key"]: row for row in reference}, {row["key"]: row for row in selected}
    if len(left) != len(reference) or len(right) != len(selected) or left.keys() != right.keys():
        return {"status": "decision_keys_not_paired"}
    for key in left:
        a, b = left[key], right[key]
        if (a["label"] != b["label"] or len(a["logits"]) != len(b["logits"])
                or a.get("cluster_id", a["record_id"]) != b.get("cluster_id", b["record_id"])):
            return {"status": "target_options_or_clusters_not_paired", "key": key}
        if not np.isfinite(a["logits"]).all() or not np.isfinite(b["logits"]).all():
            return {"status": "nonfinite_prediction", "key": key}
    interval = paired_accuracy_ci(reference, selected, seed=seed, samples=samples)
    interval.update({"status": "computed", "n_decisions": len(selected),
                     "cluster_basis": "saved cluster_id" if all("cluster_id" in row for row in selected) else "record_id fallback in older prediction files",
                     "reference_predictions_sha256": sha256(reference_path),
                     "selected_predictions_sha256": sha256(selected_path)})
    return interval


def study_stage(name: str, *, external: bool = False) -> str:
    if external:
        return "external fixed-profile authored cohort"
    if "pilot" in name:
        return "exploratory pilot"
    if re.search(r"-v([2-9]|\d{2,})(?:$|-)", name):
        return "expanded search after inspected pilot; exploratory"
    return "expanded validation; exploratory"


def load_run(path: Path, *, results_dir: Path, seed: int, samples: int, external: bool = False) -> dict:
    summary = read_json(path)
    if summary.get("status") != "complete":
        raise ValueError("incomplete_status")
    if "freeze_sha256" in summary and "selected" in summary and "rtn_matched" in summary:
        return load_external_run(path, summary=summary, results_dir=results_dir, seed=seed, samples=samples)
    selected = summary["selection"]["selected"]
    selected_name = selected["name"]
    quantized = summary.get("quantized", {})
    rtn_name, match = matched_rtn(quantized, selected_name)
    evaluations = {"native": {split: metrics_subset(value) for split, value in summary.get("baseline", {}).items()
                                if isinstance(value, dict) and "raw" in value and split != "development"}}
    method_meta = {"native": {"profile": {"name": "native", "method": "native"}}}
    for name, report in quantized.items():
        evaluations[name] = {split: metrics_subset(value) for split, value in report.get("splits", {}).items()}
        method_meta[name] = {key: report[key] for key in ("profile", "storage", "artifact") if key in report}
        method_meta[name]["format_signature_sha256"] = hashlib.sha256(json.dumps(format_signature(report), sort_keys=True).encode()).hexdigest()
    paired = {}
    for split in evaluations.get(selected_name, {}):
        native_path = path.parent / "baseline" / f"{split}.jsonl"
        selected_path = path.parent / selected_name / f"{split}.jsonl"
        paired[split] = {"selected_minus_native": comparison(native_path, selected_path, seed=seed, samples=samples),
                         "selected_minus_matched_rtn": comparison(path.parent / rtn_name / f"{split}.jsonl", selected_path, seed=seed, samples=samples)
                         if rtn_name else {"status": "no_matched_rtn"}}
    environment = summary.get("environment", {})
    return {
        "run": relative(path.parent, results_dir), "model": summary["model"],
        "study_stage": study_stage(path.parent.name, external=external), "external": external,
        "summary_path": relative(path, ROOT), "summary_sha256": sha256(path),
        "bits": summary.get("bits"), "group_size": summary.get("group_size"),
        "model_metadata": environment.get("model", {}), "data_manifest_sha256": environment.get("data_manifest_sha256"),
        "gpu": environment.get("gpu"), "source_files_sha256": environment.get("source_files_sha256", {}),
        "selection": summary["selection"], "baseline_development": metrics_subset(summary.get("baseline", {}).get("development", {})),
        "matched_rtn": rtn_name, "rtn_matching": match, "evaluations": evaluations,
        "methods": method_meta, "paired_comparisons": paired, "limitations": summary.get("limitations", []),
    }


def load_external_run(path: Path, *, summary: dict, results_dir: Path, seed: int, samples: int) -> dict:
    """Read the distinct fixed-profile external runner's audited output schema."""
    freeze_path = path.parent / "freeze.json"
    if not freeze_path.exists() or sha256(freeze_path) != summary["freeze_sha256"]:
        raise ValueError("external_freeze_missing_or_hash_mismatch")
    freeze = read_json(freeze_path)
    if freeze.get("policy", {}).get("external_fitting") is not False or freeze.get("policy", {}).get("external_selection") is not False:
        raise ValueError("external_freeze_policy_not_confirmatory")
    selected = freeze["selection"]["selected"]
    selected_name, rtn_name = selected["name"], "rtn-matched"
    quantization = summary["quantization"]
    left = format_signature({"quantization": quantization["selected"], "profile": selected})
    right = format_signature({"quantization": quantization["rtn_matched"], "profile": {**selected, "method": "rtn"}})
    # The external runner enforces a pinned common base and the exact artifact
    # module include mask. Explicit 'preserved' reasons may differ because RTN
    # excludes protected modules through that mask rather than its selector.
    mismatch = [key for key in ("activation_bits", "activation_scheme", "layers") if left[key] != right[key]]
    if sorted(left["layers"]) != sorted(freeze["rtn_scope"]["layers"]):
        mismatch.append("frozen_artifact_layer_mask")
    match = {"status": "matched" if not mismatch else "external_scope_mismatch",
             "match_basis": "Pinned common base and exact frozen artifact tensor mask, layer weight format/group size and activation scheme.",
             "mismatch": mismatch}
    if mismatch:
        rtn_name = None
    evaluations = {"native": {"external_test": metrics_subset(summary["baseline"])},
                   selected_name: {"external_test": metrics_subset(summary["selected"])},
                   "rtn-matched": {"external_test": metrics_subset(summary["rtn_matched"])}}
    paired = {"external_test": {
        "selected_minus_native": comparison(path.parent / "baseline.jsonl", path.parent / "selected.jsonl", seed=seed, samples=samples),
        "selected_minus_matched_rtn": comparison(path.parent / "rtn_matched.jsonl", path.parent / "selected.jsonl", seed=seed, samples=samples)
        if rtn_name else {"status": "external_scope_mismatch"},
    }}
    return {"run": relative(path.parent, results_dir), "model": summary["model"],
            "study_stage": study_stage(path.parent.name, external=True), "external": True,
            "summary_path": relative(path, ROOT), "summary_sha256": sha256(path),
            "freeze_sha256": summary["freeze_sha256"], "freeze": freeze,
            "bits": freeze["selection"]["bits"], "group_size": freeze["selection"]["group_size"],
            "selection": freeze["selection"], "model_metadata": summary["inference_metadata"],
            "matched_rtn": rtn_name, "rtn_matching": match, "evaluations": evaluations,
            "methods": {"native": {"profile": {"name": "native", "method": "native"}},
                        selected_name: {"profile": selected},
                        "rtn-matched": {"profile": {**selected, "name": "rtn-matched", "method": "rtn"}}},
            "paired_comparisons": paired, "baseline_development": {}, "limitations": summary.get("limitations", []),
            "cohort": {key: summary.get(key) for key in ("prepared_records", "accepted_records", "accepted_questions", "excluded_records", "exclusion_counts")},
            "rtn_temperature_status": summary.get("rtn_temperature_status")}


def compact_number(value, *, percent=False, signed=False) -> str:
    if value is None:
        return "—"
    value = float(value) * (100 if percent else 1)
    return f"{value:+.2f}" if signed else f"{value:.2f}" if percent else f"{value:.3f}"


def interval_text(value: dict) -> str:
    if value.get("status") != "computed":
        return "unavailable"
    return f"{compact_number(value['delta'], percent=True, signed=True)} [{compact_number(value['lower_95'], percent=True, signed=True)}, {compact_number(value['upper_95'], percent=True, signed=True)}]"


def latest_rank(run: dict) -> tuple:
    version = re.search(r"-v(\d+)", run["run"])
    n = run["evaluations"].get("native", {}).get("test", {}).get("raw", {}).get("n", 0)
    return (int(version.group(1)) if version else 0, "validation" in run["run"], n, run["run"])


def model_sort(run: dict):
    return (MODEL_ORDER.get(run["model"], 99), run["model"], run["run"])


def selected_rows(runs: list[dict], *, split: str) -> list[list[str]]:
    rows = []
    for run in sorted(runs, key=model_sort):
        evaluations = run["evaluations"]
        selected_name = run["selection"]["selected"]["name"]
        native = evaluations.get("native", {}).get(split, {}).get("raw", {})
        selected = evaluations.get(selected_name, {}).get(split, {}).get("raw", {})
        if not selected:
            continue
        rtn = evaluations.get(run["matched_rtn"], {}).get(split, {}).get("raw", {}) if run["matched_rtn"] else {}
        comparisons = run["paired_comparisons"].get(split, {})
        rows.append([run["model"], run["run"], str(selected.get("n", "—")),
                     compact_number(native.get("accuracy"), percent=True),
                     compact_number(rtn.get("accuracy"), percent=True),
                     compact_number(selected.get("accuracy"), percent=True),
                     interval_text(comparisons.get("selected_minus_matched_rtn", {})),
                     interval_text(comparisons.get("selected_minus_native", {}))])
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["No completed result is available for this cohort.", ""]
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows] + [""]


def observed_limitations(runs: list[dict]) -> list[str]:
    """State measured negatives without selecting or relabeling the results."""
    intervals, native_losses, nll_losses, transfer_losses, fisher_available, fisher_chosen, a4_losses = [], [], [], [], [], [], []
    for run in runs:
        chosen = run["selection"]["selected"]["name"]
        evaluations = run["evaluations"]
        main = evaluations.get(chosen, {}).get("test", {}).get("raw", {})
        native = evaluations.get("native", {}).get("test", {}).get("raw", {})
        rtn = evaluations.get(run["matched_rtn"], {}).get("test", {}).get("raw", {}) if run["matched_rtn"] else {}
        ci = run["paired_comparisons"].get("test", {}).get("selected_minus_matched_rtn", {})
        if ci.get("status") == "computed":
            intervals.append(ci)
        if main and native and native["accuracy"] - main["accuracy"] > 0.01:
            native_losses.append(f"{run['model']} ({(main['accuracy']-native['accuracy'])*100:+.2f} pp)")
        if main and rtn and main["nll"] > rtn["nll"]:
            nll_losses.append(run["model"])
        transfer = evaluations.get(chosen, {}).get("transfer_test", {}).get("raw", {})
        transfer_rtn = evaluations.get(run["matched_rtn"], {}).get("transfer_test", {}).get("raw", {}) if run["matched_rtn"] else {}
        if transfer and transfer_rtn and transfer["accuracy"] < transfer_rtn["accuracy"]:
            transfer_losses.append(f"{run['model']} ({(transfer['accuracy']-transfer_rtn['accuracy'])*100:+.2f} pp)")
        candidates = run["selection"].get("candidates", [])
        if any(candidate["profile"].get("fisher") for candidate in candidates):
            fisher_available.append(run["model"])
            if run["selection"]["selected"].get("fisher"):
                fisher_chosen.append(run["model"])
        baseline_dev = run.get("baseline_development", {}).get("raw", {})
        for candidate in candidates:
            if candidate["profile"].get("activation_bits") == 4 and baseline_dev and candidate["development_accuracy"] < baseline_dev["accuracy"]:
                a4_losses.append(run["model"])
    sentences = []
    if intervals:
        span_zero = sum(ci["lower_95"] <= 0 <= ci["upper_95"] for ci in intervals)
        sentences.append(f"{span_zero} of {len(intervals)} latest main S1Q−matched-RTN intervals include zero; the inspected search does not support a universal improvement claim.")
    if native_losses:
        sentences.append("Selected main accuracy loses more than one percentage point against the unquantized checkpoint for " + ", ".join(native_losses) + ".")
    if nll_losses:
        sentences.append("Raw main NLL is worse than matched RTN for " + ", ".join(nll_losses) + "; better decision fidelity or accuracy does not guarantee better raw probability metrics.")
    if transfer_losses:
        sentences.append("Source-transfer point accuracy is lower than matched RTN for " + ", ".join(transfer_losses) + "; intervals remain visible in the transfer table.")
    if fisher_available:
        sentences.append(f"The development objective selects a Fisher profile for {len(fisher_chosen)} of {len(fisher_available)} models with completed Fisher searches; this is not evidence of a general Fisher benefit.")
    if a4_losses:
        sentences.append("Simulated 4-bit activation candidates lose development accuracy against native for " + ", ".join(a4_losses) + ".")
    return [" ".join(sentences), ""] if sentences else []


def external_interpretation(runs: list[dict]) -> list[str]:
    intervals = [run["paired_comparisons"].get("external_test", {}).get("selected_minus_matched_rtn", {}) for run in runs]
    intervals = [value for value in intervals if value.get("status") == "computed"]
    if not intervals:
        return []
    span_zero = sum(value["lower_95"] <= 0 <= value["upper_95"] for value in intervals)
    text = (f"{span_zero} of {len(intervals)} external S1Q−matched-RTN accuracy intervals include zero. "
            "The point estimates include gains, ties and losses; this small authored cohort does not establish a consistent improvement across models.")
    nano = next((run for run in runs if run["model"] == "nanojev"), None)
    if nano:
        selected = nano["selection"]["selected"]["name"]
        native = nano["evaluations"].get("native", {}).get("external_test", {}).get("raw", {})
        quantized = nano["evaluations"].get(selected, {}).get("external_test", {}).get("raw", {})
        if native and quantized:
            text += (f" NanoJev's unquantized game checkpoint already scores {native['accuracy']*100:.2f}% on these general authored decisions; "
                     f"selected S1Q scores {quantized['accuracy']*100:.2f}%, a {(quantized['accuracy']-native['accuracy'])*100:+.2f} pp change. "
                     "The low absolute baseline is consistent with a domain-transfer limitation before quantization; the paired change measures the additional quantizer effect.")
    return [text, ""]


def figure_links(report: dict) -> list[str]:
    """Embed only the figure snapshot bound to this exact aggregate payload."""
    path = ROOT / "docs/assets/figure-manifest.json"
    if not path.exists():
        return []
    manifest = read_json(path)
    payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if manifest.get("aggregate_sha256") != hashlib.sha256(payload).hexdigest():
        return ["## Paper figures", "", "Regenerate the paper exports with `python scripts/plot_results.py`, then regenerate this report to link the matching figure snapshot.", ""]
    available = {figure["name"] for figure in manifest.get("figures", [])}
    rows = []
    for name, label in (("paired_accuracy", "Main and transfer/game OOD paired accuracy differences"),
                        ("external_accuracy", "External authored-cohort paired accuracy differences"),
                        ("parameter_storage", "Complete parameter storage estimates")):
        if name in available:
            rows.append(f"- {label}: [SVG](assets/{name}.svg) · [PDF](assets/{name}.pdf)")
    if not rows:
        return []
    result = ["## Paper figures", "", *rows, ""]
    if "paired_accuracy" in available:
        result.extend(["![Selected S1Q minus matched RTN accuracy, with paired 95% cluster intervals](assets/paired_accuracy.png)", ""])
    result.extend(["Exports and plotted values are bound to this report's aggregate by [the figure manifest](assets/figure-manifest.json). "
                   "See [all figure previews and reproduction instructions](assets/README.md).", ""])
    return result


def render_markdown(report: dict) -> str:
    runs = report["runs"]
    ordinary = [run for run in runs if not run["external"]]
    latest = {}
    for run in ordinary:
        if run["model"] not in latest or latest_rank(run) > latest_rank(latest[run["model"]]):
            latest[run["model"]] = run
    current = sorted(latest.values(), key=model_sort)
    lines = ["# S1Q experimental results", "",
             "Generated from completed, hashed run summaries and saved paired predictions by `scripts/summarize_results.py`. "
             "Only `status=complete` runs are included. Values below describe this research snapshot; pending runs are not counted.", "",
             "Pilot results were inspected before the expanded Fisher search. Pilot and expanded validation results are exploratory, "
             "including repeated evaluation on overlapping Kev suites. A larger rerun does not create a new independent held-out set. "
             "External authored-cohort results are reported separately when available; they must use profiles frozen before external inference.", "",
             "The current implementation rounds weights and simulates activation quantization while executing floating point operations. "
             "Storage estimates include retained embeddings, heads and other native parameters; they are not GPU memory or integer-kernel speedups. "
             "NanoJev's native cohorts in the v0.1.0 snapshot contain only shooting tasks (Basic and Predict Position); supplied `reference_argmax_compatibility` labels measure recorded reference-policy action argmax agreement, not human annotation, optimal-action gold or observed-success probability calibration. "
             "No claim of the first quantization work on these models follows from these experiments.", "",
             "## Latest completed main cohorts", "",
             "Accuracy is a percentage; differences and interval endpoints are percentage points. Each interval is a paired percentile cluster bootstrap with "
             f"{report['bootstrap_samples']:,} resamples and seed `{report['seed']}`. Intervals are descriptive, not adjusted for multiple comparisons. "
             "Matched RTN uses identical protected tensor identities, weight format/group size, activation scheme and parameter coverage. "
             "Different models/cohorts can have different labels and sample counts; do not pool them into one accuracy.", ""]
    headers = ["Model", "Run", "Questions", "Native %", "Matched RTN %", "Selected S1Q %", "S1Q−RTN pp [95% CI]", "S1Q−native pp [95% CI]"]
    lines.extend(markdown_table(headers, selected_rows(current, split="test")))
    lines.extend(observed_limitations(current))
    lines.extend(figure_links(report))
    for split, title in (("transfer_test", "Source-transfer cohorts"), ("ood", "Native game OOD cohort")):
        lines.extend([f"## {title}", ""])
        lines.extend(markdown_table(headers, selected_rows(current, split=split)))
    lines.extend(["## External authored confirmation", "",
                  "JevBench refers specifically to `fstandhartinger/jevbench` at the pinned source commit in the data configuration. "
                  "The short original+hard cohort admits 85 public tasks in 49 scenario clusters before tokenizer admission and excludes 98 hard tasks for length. "
                  "Hard labels have model-assisted authorship and cross-review. This screened public cohort is not the full official benchmark or a leaderboard rank.", ""])
    external_runs = [run for run in runs if run["external"]]
    lines.extend(markdown_table(headers, selected_rows(external_runs, split="external_test")))
    lines.extend(external_interpretation(external_runs))
    lines.extend(["## Selected profiles and parameter storage", ""])
    storage_rows = []
    for run in current:
        profile = run["selection"]["selected"]
        meta = run["methods"].get(profile["name"], {})
        storage = meta.get("storage", {})
        activation = "native" if profile.get("activation_bits") is None else f"{profile['activation_bits']}-bit QDQ"
        storage_rows.append([run["model"], profile["name"], f"W{run['bits']}, group {run['group_size']}; {activation}",
                             compact_number(storage.get("quantized_parameter_fraction"), percent=True),
                             compact_number(storage.get("native_parameters_bytes", 0) / 2**20) if storage else "—",
                             compact_number(storage.get("estimated_complete_packed_parameters_bytes", 0) / 2**20) if storage else "—",
                             compact_number(storage.get("estimated_parameter_storage_ratio")),
                             run["matched_rtn"] or "unavailable"])
    lines.extend(markdown_table(["Model", "Selected profile", "Format / activations", "Quantized params %", "Native MiB", "Estimated complete packed MiB", "Storage ratio", "Matched RTN"], storage_rows))
    lines.extend(["## Measured packed execution", "",
                  "These are completed device measurements, reported separately from the parameter estimates above. "
                  "Persistent and peak values use PyTorch allocated CUDA memory. The packed path dequantizes one layer at a time and still executes floating point GEMM. "
                  "Mean request time includes encoding; the benchmark has no paired dense latency measurement and supports no speedup claim. "
                  "Parity compares dense and packed execution reconstructed from the same quantized artifact on the same small subset.", ""])
    packed_rows = []
    for item in report.get("packed_benchmarks", []):
        data = item["measurements"]
        packed_rows.append([data["model"], item["run"], str(data.get("requests", "—")),
                            compact_number(data["persistent_dense_cuda_bytes"] / 2**20),
                            compact_number(data["persistent_packed_cuda_bytes"] / 2**20),
                            compact_number(data["peak_packed_cuda_bytes"] / 2**20),
                            compact_number(data.get("mean_request_time_ms")),
                            f"{data['paired']['js_to_reference']:.3g}" if "js_to_reference" in data.get("paired", {}) else "—",
                            compact_number(data.get("paired", {}).get("decision_flip_rate"), percent=True)])
    lines.extend(markdown_table(["Model", "Run", "Scored decisions", "Dense persistent MiB", "Packed persistent MiB", "Packed peak MiB", "Packed mean request ms", "JS to same artifact", "Decision flips %"], packed_rows))
    lines.extend(["## Raw and temperature-calibrated metrics", "",
                  "For the main experiments each method fits its own positive per-type temperature on the separate temperature-calibration partition, then uses it unchanged here. "
                  "External evaluation reuses those frozen temperatures; an unavailable matched RTN temperature is not invented or fitted externally. "
                  "Raw metrics use unscaled decision logits, normalized into the common candidate distribution (including `[false, true]` for binary questions). "
                  "Here native means the unquantized checkpoint under that evaluation convention. Published served temperatures and output rounding are not applied in the raw table. "
                  "Temperature fitting is available to native, RTN and S1Q alike; "
                  "it cannot change argmax accuracy. NLL/Brier/ECE describe the declared target labels in these datasets, without a real-world probability guarantee. "
                  "`results/metrics.csv` retains every completed method and cohort, including nonselected local ablations.", ""])
    for split, title in (("test", "Main"), ("transfer_test", "Source transfer"), ("ood", "Game OOD"), ("external_test", "External authored")):
        stage_runs = current if split != "external_test" else [run for run in runs if run["external"]]
        rows = []
        for run in sorted(stage_runs, key=model_sort):
            selected = run["selection"]["selected"]["name"]
            names = list(dict.fromkeys(["native", run["matched_rtn"], selected]))
            for name in names:
                if name is None or split not in run["evaluations"].get(name, {}):
                    continue
                value = run["evaluations"][name][split]
                for stage, label in (("raw", "raw"), ("temperature_calibrated", "temperature")):
                    metric = value.get(stage, {})
                    if not metric:
                        continue
                    rows.append([run["model"], name, label, str(metric.get("n", "—")),
                                 compact_number(metric.get("accuracy"), percent=True), compact_number(metric.get("macro_source_accuracy"), percent=True),
                                 compact_number(metric.get("nll")), compact_number(metric.get("brier")), compact_number(metric.get("ece_15"))])
        if rows:
            lines.extend([f"### {title}", ""])
            lines.extend(markdown_table(["Model", "Method", "Stage", "Questions", "Accuracy %", "Macro source %", "NLL", "Brier", "ECE15"], rows))
    lines.extend(["## Development selection and negative candidates", "",
                  "The selection objective is development boundary-weighted JS divergence to native probabilities plus 0.1 times harmful decision flips; "
                  "selection is among the declared S1Q candidates. RTN remains a comparator even when its development objective is better. "
                  "Candidate results are not filtered to favor S1Q. Fisher variants were added after pilot inspection, so their search is exploratory. "
                  "Fake 4-bit activations must be judged by these measured development results; they do not establish a deployable full W4A4 model.", ""])
    dev_rows = []
    for run in current:
        chosen = run["selection"]["selected"]["name"]
        for candidate in run["selection"].get("candidates", []):
            profile = candidate["profile"]
            dev_rows.append([run["model"], profile["name"], "yes" if profile["name"] == chosen else "no",
                             compact_number(candidate.get("development_accuracy"), percent=True), compact_number(candidate.get("objective")),
                             "native" if profile.get("activation_bits") is None else str(profile["activation_bits"]),
                             compact_number(profile.get("sensitive_fraction", 0), percent=True), "yes" if profile.get("fisher") else "no"])
    lines.extend(markdown_table(["Model", "Candidate", "Selected", "Dev accuracy %", "Dev objective ↓", "Activation bits", "Protected linear fraction %", "Fisher"], dev_rows))
    lines.extend(["## All completed run inventory", ""])
    rows = [[run["run"], run["model"], run["study_stage"], run["selection"]["selected"]["name"],
             str(run["evaluations"].get("native", {}).get("test", run["evaluations"].get("native", {}).get("external_test", {})).get("raw", {}).get("n", "—")),
             run["summary_sha256"][:16]] for run in sorted(runs, key=model_sort)]
    lines.extend(markdown_table(["Run", "Model", "Interpretation", "Selected profile", "Main/external questions", "Summary SHA256 prefix"], rows))
    unmatched = [run for run in runs if run["rtn_matching"]["status"] != "matched"]
    if unmatched:
        lines.extend(["Matched comparisons unavailable: " + "; ".join(f"`{run['run']}` ({run['rtn_matching']['status']})" for run in unmatched) + ". Plain RTN values remain in the machine-readable metric table, but are not substituted for a matched comparison.", ""])
    if report["skipped"]:
        lines.extend(["Excluded incomplete or unsupported inputs: " + "; ".join(f"`{row['path']}` ({row['reason']})" for row in report["skipped"]) + ".", ""])
    lines.extend(["## Reproduction", "", "```sh", "python scripts/summarize_results.py --results-dir results \\",
                  "  --markdown docs/results.md --json results/aggregate.json --csv results/metrics.csv", "```", "",
                  "The aggregate JSON records full summary/prediction hashes, model/source pins, profile selection, all recorded metrics and every comparison availability status. "
                  "Saved `cluster_id` values are used for current runs; earlier files without them fall back to request/record IDs and are labeled accordingly. "
                  "The reporting script does not reselect quantizers or inspect inputs to construct a more favorable cohort.", ""])
    return "\n".join(lines)


def write_csv(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["run", "model", "study_stage", "split", "method", "selected", "matched_rtn", "stage", "label_kind", *METRIC_KEYS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            selected = run["selection"]["selected"]["name"]
            for method, splits in run["evaluations"].items():
                for split, evaluation in splits.items():
                    strata = {"all": evaluation, **evaluation.get("by_label_kind", {})}
                    for label_kind, value in strata.items():
                        for stage in ("raw", "temperature_calibrated"):
                            metric = value.get(stage, {})
                            if metric:
                                writer.writerow({"run": run["run"], "model": run["model"], "study_stage": run["study_stage"],
                                                 "split": split, "method": method, "selected": method == selected,
                                                 "matched_rtn": method == run["matched_rtn"], "stage": stage, "label_kind": label_kind,
                                                 **{key: metric[key] for key in METRIC_KEYS if key in metric}})


def aggregate_results(results_dir: str | Path, *, seed: int = 20261001, samples: int = 2000) -> dict:
    results = Path(results_dir)
    report = {"schema": "s1q-results-aggregate.v1", "seed": seed, "bootstrap_samples": samples,
              "reporter_sha256": sha256(Path(__file__)), "metrics_source_sha256": sha256(ROOT / "src/s1q/metrics.py"),
              "runs": [], "packed_benchmarks": [], "skipped": []}
    inputs = sorted(set(results.rglob("summary.json")) | set(results.rglob("external-summary.json")))
    for path in inputs:
        try:
            run = load_run(path, results_dir=results, seed=seed, samples=samples, external=path.name == "external-summary.json")
            report["runs"].append(run)
        except (ValueError, KeyError, TypeError) as error:
            report["skipped"].append({"path": relative(path, ROOT), "reason": str(error)})
    for path in sorted(results.rglob("packed-benchmark.json")):
        try:
            data = read_json(path)
            for key in ("model", "persistent_dense_cuda_bytes", "persistent_packed_cuda_bytes", "peak_packed_cuda_bytes", "paired", "mean_request_time_ms"):
                if key not in data:
                    raise ValueError("incomplete_packed_benchmark")
            run_name = relative(path.parent, results)
            run = next((item for item in report["runs"] if item["run"] == run_name), None)
            if run is None:
                raise ValueError("packed_benchmark_run_not_complete")
            selected_name = run["selection"]["selected"]["name"]
            expected = run["methods"].get(selected_name, {}).get("artifact", {}).get("sha256")
            if expected is not None and expected != data.get("artifact_sha256"):
                raise ValueError("packed_benchmark_selected_artifact_mismatch")
            report["packed_benchmarks"].append({"run": relative(path.parent, results),
                                                "source_path": relative(path, ROOT), "sha256": sha256(path),
                                                "selected_artifact_sha256_verified": expected is not None,
                                                "measurements": data})
        except (ValueError, KeyError, TypeError) as error:
            report["skipped"].append({"path": relative(path, ROOT), "reason": str(error)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/results.md")
    parser.add_argument("--json", type=Path, default=ROOT / "results/aggregate.json")
    parser.add_argument("--csv", type=Path, default=ROOT / "results/metrics.csv")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    report = aggregate_results(args.results_dir, seed=args.seed, samples=args.bootstrap_samples)
    for path in (args.markdown, args.json):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    write_csv(args.csv, report["runs"])
    print(json.dumps({"completed_runs": len(report["runs"]), "packed_benchmarks": len(report["packed_benchmarks"]), "skipped": report["skipped"],
                      "markdown": str(args.markdown), "json": str(args.json), "csv": str(args.csv)}, indent=2))


if __name__ == "__main__":
    main()
