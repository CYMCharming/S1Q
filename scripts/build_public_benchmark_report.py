"""Export aggregate-only benchmark evidence and reproducible scientific figures.

Example:
  python scripts/build_public_benchmark_report.py \
    --input optimization-20261004=research/results/optimization-20261004 \
    --input decision-extension-20261006=research/results/decision-extension-20261006/full-only-audit \
    --dataset-list work/benchmark-expansion-20261007/ranking-scopes.json \
    --output-dir work/benchmark-expansion-20261007/public-preview

An input directory contains metrics.csv and optionally an audited summary.json.
No source questions, individual predictions, paper files, or absolute host paths
are exported. Scopes explicitly pin batches, precision, and the required dataset
list. Missing method/model/dataset cells exclude that model from that scope;
missing values are never replaced by zero. Historical and expanded protocols
are never merged unless the caller explicitly declares that scope.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "rtn", "s1q-local", "s1q2-beta05", "awq-adapted", "gptq-blockdiag-adapted",
    "spinquant-nohad-adapted", "spinquant-had-adapted", "s1q-joint", "s1q-ac",
    "s1q-margin", "s1q",
)
S1Q_METHODS = ("s1q-local", "s1q2-beta05", "s1q-joint", "s1q-ac", "s1q-margin", "s1q")
LABELS = {
    "native": "Native", "rtn": "RTN", "s1q-local": "S1Qv1", "s1q2-beta05": "S1Qv2",
    "smoothquant-adapted": "SmoothQuant*",
    "awq-adapted": "AWQ*", "gptq-blockdiag-adapted": "GPTQ-block*",
    "spinquant-nohad-adapted": "SpinQuant no-Had*", "spinquant-had-adapted": "SpinQuant Had*",
    "s1q-joint": "S1Q-Joint", "s1q-ac": "S1Q-AC", "s1q-margin": "S1Q-Margin", "s1q": "S1Q (current)",
}
NUMERIC_FIELDS = (
    "accuracy", "macro_source_accuracy", "nll", "brier", "ece_15", "mean_confidence",
    "score_mae", "js_to_native", "teacher_flip_rate", "harmful_teacher_flip_rate",
    "beneficial_teacher_flip_rate", "coverage_at_risk_0_01", "coverage_at_risk_0_05",
    "coverage_at_risk_0_10", "quantization_seconds",
)
METRIC_NAMES = ("accuracy", "nll", "brier", "ece_15")
COHORT_LABELS = {"development": "Mixed Dev (exploratory)", "wanli": "WANLI subset", "mmlu-pro": "MMLU-Pro short subset"}
SOURCE_INTERPRETATIONS = {
    "semif": {"name": "SemIf", "source_category": "authored_rule_suite", "figure_note": "SemIf: authored rules"},
    "jevbench": {"name": "JevBench", "source_category": "authored_scenario_suite", "figure_note": "JevBench: authored scenarios"},
    "toolace": {"name": "ToolACE", "source_category": "domain_decision_suite", "figure_note": "ToolACE: tool choice"},
    "wildjailbreak": {"name": "WildJailBreak", "source_category": "domain_decision_suite", "figure_note": "WildJailBreak: harmful/benign decisions"},
}
BLUE, GOLD, INK, GREY = "#285F85", "#AD762B", "#27323D", "#C7CDD2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalize_method(method: str) -> str:
    return "s1q" if method == "s1q-mac" else method


def family(model: str) -> str:
    if model.startswith("intern-decision-"):
        return "Intern-Decision"
    if model.startswith("startlux-decision-"):
        return "StartLux-Decision"
    if model.startswith("kev-"):
        return "Kev"
    return "NanoJev" if model.lower() == "nanojev" else "Laya" if model.lower() == "laya" else model


def numeric(value) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite metric {value!r}")
    return number


def metric_identity(row: dict) -> tuple:
    return tuple(str(row.get(field, "")) for field in ("run", "model", "precision", "cohort", "method"))


def audited_input(batch: str, directory: Path) -> tuple[list[dict], dict]:
    metrics_path = directory / "metrics.csv"
    rows = read_csv(metrics_path)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else None
    if summary is not None:
        if summary.get("audit_errors"):
            raise ValueError(f"Input {batch} has audit errors")
        expected = {metric_identity(row): row for row in summary["metrics"]}
        actual = {metric_identity(row): row for row in rows}
        if len(actual) != len(rows) or len(expected) != len(summary["metrics"]) or actual.keys() != expected.keys():
            raise ValueError(f"Input {batch}: CSV and summary identities differ or duplicate")
        for key, row in actual.items():
            other = expected[key]
            if row["status"] != other["status"]:
                raise ValueError(f"Input {batch}: CSV and summary statuses differ")
            for field in ("n", "n_clusters", *NUMERIC_FIELDS):
                a, b = numeric(row.get(field)), numeric(other.get(field))
                if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)):
                    raise ValueError(f"Input {batch}: CSV and summary differ on {key}/{field}")
    else:
        if len({metric_identity(row) for row in rows}) != len(rows):
            raise ValueError(f"Duplicate input identities in {batch}")
    sanitized = []
    for row in rows:
        item = {
            "experiment_batch_id": batch, "run_id": row["run"], "model": row["model"],
            "family": family(row["model"]), "precision": row["precision"],
            "dataset_id": row.get("dataset_id") or row["cohort"], "cohort": row["cohort"],
            "method": normalize_method(row["method"]), "original_method_id": row["method"],
            "method_label": LABELS.get(normalize_method(row["method"]), row["method"]),
            "status": row["status"], "study_cohort": row.get("study_cohort", ""),
            "metric_target": "reference_action_compatibility" if family(row["model"]) == "NanoJev" else "labeled_decision_accuracy",
            "execution": "floating_point_native" if row["method"] == "native" else "floating_point_QDQ", "n": int(float(row["n"])) if row.get("n") else None,
            "n_clusters": int(float(row["n_clusters"])) if row.get("n_clusters") else None,
        }
        item.update({field: numeric(row.get(field)) for field in NUMERIC_FIELDS})
        if item["status"] == "complete":
            if item["n"] is None or item["n"] <= 0 or item["accuracy"] is None or not 0 <= item["accuracy"] <= 1:
                raise ValueError(f"Invalid complete input row in {batch}: {metric_identity(row)}")
        sanitized.append(item)
    provenance = {
        "experiment_batch_id": batch, "metrics_sha256": sha256(metrics_path), "metric_rows": len(rows),
        "verification": "CSV_matches_audited_summary" if summary is not None else "aggregate_input_only",
        "summary_sha256": sha256(summary_path) if summary is not None else None,
        "protocol_cohorts": sorted({row.get("study_cohort", "") for row in rows}),
        "complete_rows": sum(row["status"] == "complete" for row in rows),
        "model_names": sorted({row["model"] for row in rows}),
    }
    if summary:
        reports = []
        for run in summary.get("runs", []):
            identity = run.get("identity", {})
            model = identity.get("model", {})
            reports.append({
                "run_id": run["run"], "model": run["model"], "precision": run["precision"],
                "dataset_id": run.get("dataset_id") or run["cohort"], "report_sha256": run.get("report_sha256"),
                "checkpoint_repo": model.get("repo_id", model.get("repo")), "checkpoint_revision": model.get("revision"),
                **{key: identity.get(key) for key in ("seed", "bits", "activation_bits", "group_size", "requested_calibration", "accepted_calibration", "reservoir_size", "selected_linear_count", "selected_weight_parameters", "implementation_sha256")},
            })
        provenance["reports"] = reports
    return sanitized, provenance


def default_scopes(rows: list[dict]) -> dict:
    scopes = []
    for batch, precision in sorted({(row["experiment_batch_id"], row["precision"]) for row in rows}):
        datasets = sorted({row["dataset_id"] for row in rows if row["experiment_batch_id"] == batch and row["precision"] == precision and row["metric_target"] == "labeled_decision_accuracy"})
        scopes.append({"suite_id": f"{batch}-{precision.lower()}", "experiment_batch_ids": [batch], "precision": precision, "dataset_ids": datasets})
    return {"ranking_suites": scopes, "plot_suite_id": scopes[-1]["suite_id"] if scopes else None}


def rank_values(values: dict[str, float], *, ascending: bool = False) -> dict[str, int]:
    ordered = sorted(values, key=lambda method: (values[method] if ascending else -values[method], method))
    ranks = {}
    previous = None
    rank = 0
    for position, method in enumerate(ordered, 1):
        value = values[method]
        if previous is None or not math.isclose(value, previous, rel_tol=0, abs_tol=1e-12):
            rank = position
        ranks[method] = rank
        previous = value
    return ranks


def build_rankings(rows: list[dict], settings: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rankings, scopes, model_scores = [], [], []
    batch_ids = {row["experiment_batch_id"] for row in rows}
    suite_ids = set()
    for suite in settings["ranking_suites"]:
        suite_id = suite["suite_id"]
        if suite_id in suite_ids:
            raise ValueError(f"Duplicate ranking suite {suite_id}")
        suite_ids.add(suite_id)
        batches = suite["experiment_batch_ids"]
        dataset_ids = suite["dataset_ids"]
        precision = suite["precision"]
        if not batches or not set(batches) <= batch_ids or not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError(f"Invalid batch or dataset scope {suite_id}")
        selected = [row for row in rows if row["experiment_batch_id"] in batches and row["precision"] == precision and row["dataset_id"] in dataset_ids and row["metric_target"] == "labeled_decision_accuracy" and (not suite.get("model_names") or row["model"] in suite["model_names"])]
        index = {}
        for row in selected:
            key = (row["model"], row["dataset_id"], row["method"])
            if key in index:
                raise ValueError(f"Ambiguous duplicate model/dataset/method across batches in {suite_id}: {key}")
            if row["status"] == "complete":
                index[key] = row
        models = sorted({row["model"] for row in selected})
        primary_group = suite.get("comparison_group", "all11")
        primary_methods = tuple(suite.get("methods", METHODS))
        if not isinstance(primary_group, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", primary_group) or primary_group == "s1q6":
            raise ValueError(f"Invalid or reserved primary comparison group in {suite_id}")
        if not primary_methods or len(set(primary_methods)) != len(primary_methods) or "native" in primary_methods:
            raise ValueError(f"Primary comparison methods must be unique quantization methods in {suite_id}")
        numbered_group = re.fullmatch(r"all([0-9]+)", primary_group)
        if numbered_group and int(numbered_group.group(1)) != len(primary_methods):
            raise ValueError(f"Comparison group name and method count differ in {suite_id}")
        for group, methods in ((primary_group, primary_methods), ("s1q6", S1Q_METHODS)):
            missing = {
                model: [{"dataset_id": dataset, "method": method} for dataset in dataset_ids for method in methods if (model, dataset, method) not in index]
                for model in models
            }
            complete = [model for model in models if not missing[model]]
            for model in complete:
                for dataset in dataset_ids:
                    observed = [index[(model, dataset, method)] for method in methods]
                    if len({(row["n"], row["n_clusters"]) for row in observed}) != 1:
                        raise ValueError(f"Unequal method denominators in {suite_id}/{model}/{dataset}")
            details = {
                "suite_id": suite_id, "comparison_group": group, "experiment_batch_ids": batches,
                "is_primary_comparison": group == primary_group,
                "precision": precision, "dataset_ids": dataset_ids, "method_ids": list(methods),
                "included_models": complete, "excluded_models": {model: gaps for model, gaps in missing.items() if gaps},
                "n_models": len(complete), "n_datasets": len(dataset_ids), "n_cells": len(complete) * len(dataset_ids),
                "aggregation": "equal_dataset_within_model_then_equal_model; unrounded inputs; not pooled decision accuracy",
                "uncertainty": "Descriptive point ranks; no aggregate confidence intervals or significance claim",
                "note": suite.get("note", ""),
                "dataset_metadata": {dataset: suite.get("dataset_metadata", {}).get(dataset, SOURCE_INTERPRETATIONS.get(dataset, {"name": COHORT_LABELS.get(dataset, dataset), "source_category": "evaluated_dataset_or_cohort"})) for dataset in dataset_ids},
                "figure_scope_note": suite.get("figure_scope_note", ""),
                "accuracy_table_enabled": suite.get("include_accuracy_table", bool(suite.get("dataset_metadata"))),
                "sample_counts": [{"model": model, "dataset_id": dataset, "n": index[(model, dataset, methods[0])]["n"], "n_clusters": index[(model, dataset, methods[0])]["n_clusters"]} for model in complete for dataset in dataset_ids],
            }
            scopes.append(details)
            if not complete:
                continue
            scores = {}
            for model in complete:
                for method in methods:
                    values = {metric: [index[(model, dataset, method)].get(metric) for dataset in dataset_ids] for metric in METRIC_NAMES}
                    scores[(model, method)] = {metric: statistics.mean(numbers) if all(number is not None for number in numbers) else None for metric, numbers in values.items()}
                    model_scores.append({"suite_id": suite_id, "comparison_group": group, "model": model, "family": family(model), "precision": precision, "method": method, "method_label": LABELS.get(method, method), "n_datasets": len(dataset_ids), **scores[(model, method)]})
            for family_name in ("All", *sorted({family(model) for model in complete})):
                family_models = complete if family_name == "All" else [model for model in complete if family(model) == family_name]
                means = {method: {metric: statistics.mean(scores[(model, method)][metric] for model in family_models) if all(scores[(model, method)][metric] is not None for model in family_models) else None for metric in METRIC_NAMES} for method in methods}
                metric_ranks = {metric: rank_values({method: means[method][metric] for method in methods if means[method][metric] is not None}, ascending=metric != "accuracy") for metric in METRIC_NAMES}
                native_values = [index[(model, dataset, "native")]["accuracy"] for model in family_models for dataset in dataset_ids if (model, dataset, "native") in index]
                native_mean = statistics.mean(native_values) if len(native_values) == len(family_models) * len(dataset_ids) else None
                rtn_mean = means.get("rtn", {}).get("accuracy")
                for method in methods:
                    wins = sum(rank_values({other: scores[(model, other)]["accuracy"] for other in methods})[method] == 1 for model in family_models)
                    rankings.append({
                        "suite_id": suite_id, "comparison_group": group, "family": family_name, "precision": precision,
                        "method": method, "method_label": LABELS.get(method, method), "rank_accuracy": metric_ranks["accuracy"][method],
                        "rank_nll": metric_ranks["nll"].get(method), "rank_brier": metric_ranks["brier"].get(method), "rank_ece_15": metric_ranks["ece_15"].get(method),
                        **means[method], "accuracy_percent": 100 * means[method]["accuracy"],
                        "delta_rtn_pp": 100 * (means[method]["accuracy"] - rtn_mean) if rtn_mean is not None else None,
                        "native_accuracy_percent": 100 * native_mean if native_mean is not None else None,
                        "n_models": len(family_models), "n_datasets": len(dataset_ids), "n_cells": len(family_models) * len(dataset_ids),
                        "model_avg_accuracy_wins_including_ties": wins,
                    })
    rankings.sort(key=lambda row: (row["suite_id"], row["comparison_group"], row["family"], row["rank_accuracy"], row["method"]))
    return rankings, scopes, model_scores


def figure_scope_notes(scope: dict) -> list[str]:
    """Describe this scope's source roles without inheriting a historic caption."""
    import textwrap
    metadata = scope.get("dataset_metadata", {})
    names = [metadata.get(dataset, {}).get("name", COHORT_LABELS.get(dataset, dataset)) for dataset in scope["dataset_ids"]]
    if scope.get("figure_scope_note"):
        intro = scope["figure_scope_note"]
    elif scope["n_datasets"] > 8:
        intro = f"Scope: {scope['n_datasets']} frozen source suites; exact source list and per-cell sample sizes are in ranking_scopes.json."
    else:
        intro = "Scope: " + ", ".join(names) + "."
    notes = textwrap.wrap(intro, width=156)
    interpretations = [metadata.get(dataset, {}).get("figure_note") or SOURCE_INTERPRETATIONS.get(dataset, {}).get("figure_note") for dataset in scope["dataset_ids"]]
    interpretations = [note for note in interpretations if note]
    if interpretations:
        notes.extend(textwrap.wrap("Source roles: " + "; ".join(interpretations) + ".", width=156))
    return notes


def plot_rankings(output: Path, rankings: list[dict], scopes: list[dict], suite_id: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    scope = next((scope for scope in scopes if scope["suite_id"] == suite_id and scope["is_primary_comparison"]), None)
    comparison_group = scope["comparison_group"] if scope else None
    points = [row for row in rankings if row["suite_id"] == suite_id and row["comparison_group"] == comparison_group and row["family"] == "All"]
    if not points or scope is None:
        return {"status": "unavailable_no_complete_common_coverage", "suite_id": suite_id}
    points.sort(key=lambda row: (row["rank_accuracy"], row["method"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#A9B0B7", "axes.linewidth": 0.7, "svg.fonttype": "none", "svg.hashsalt": "s1q-public-benchmark-v1", "savefig.facecolor": "white"})
    figure = plt.figure(figsize=(15.8, 10.3), facecolor="white")
    grid = figure.add_gridspec(2, 6, height_ratios=(2.4, 1), left=.145, right=.965, top=.83, bottom=.155, hspace=.48, wspace=1.1)
    bars = figure.add_subplot(grid[0, :3])
    heatmap = figure.add_subplot(grid[0, 3:])
    figure.text(.025, .955, "S1Q  |  QUANTIZATION BENCHMARK", fontsize=21, weight="bold", va="top", color=INK)
    figure.text(.025, .909, f"{scope['precision']}  /  {scope['n_models']} common models  /  {scope['n_datasets']} evaluated datasets or cohorts", fontsize=13, color=BLUE)
    note = scope.get("note") or suite_id
    figure.text(.025, .874, note, fontsize=10.5, color="#616B75")
    positions = np.arange(len(points))
    colors = [GOLD if row["method"] == "s1q" else BLUE if row["method"] in S1Q_METHODS else GREY for row in points]
    bars.barh(positions, [row["accuracy_percent"] for row in points], height=.63, color=colors)
    bars.set_yticks(positions, [row["method_label"] for row in points])
    bars.invert_yaxis()
    bars.set_xlim(0, max(row["accuracy_percent"] for row in points) * 1.18)
    bars.set_xlabel("Mean decision accuracy (%)  ↑")
    bars.set_title("A   Common-coverage accuracy", loc="left", pad=14)
    bars.grid(axis="x", color="#E8ECEF", linewidth=.65)
    bars.set_axisbelow(True)
    bars.spines[["top", "right"]].set_visible(False)
    bars.tick_params(axis="y", length=0)
    for i, row in enumerate(points):
        bars.text(row["accuracy_percent"] + .55, i, f"{row['accuracy_percent']:.2f}", va="center", fontsize=10.5, weight="bold" if row["rank_accuracy"] == 1 else "normal")
    families = sorted({row["family"] for row in rankings if row["suite_id"] == suite_id and row["comparison_group"] == comparison_group and row["family"] != "All"})
    cells = {(row["family"], row["method"]): row for row in rankings if row["suite_id"] == suite_id and row["comparison_group"] == comparison_group}
    matrix = np.asarray([[cells[(name, row["method"])]["rank_accuracy"] for name in families] for row in points])
    cmap = LinearSegmentedColormap.from_list("s1q_rank", ("#285F85", "#EAF0F4"))
    heatmap.imshow(matrix, cmap=cmap, vmin=1, vmax=max(len(points), 2), aspect="auto")
    heatmap.set_xticks(np.arange(len(families)), [f"{name}\n{cells[(name, points[0]['method'])]['n_models']} model(s)" for name in families])
    heatmap.set_yticks(positions, [row["method_label"] for row in points])
    heatmap.tick_params(axis="both", length=0, pad=7)
    heatmap.set_title("B   Accuracy rank within each family", loc="left", pad=14)
    heatmap.spines[:].set_visible(False)
    for (y, x), rank in np.ndenumerate(matrix):
        heatmap.text(x, y, str(rank), ha="center", va="center", color="white" if rank <= 4 else INK, fontsize=11)
    heatmap.set_xticks(np.arange(-.5, len(families), 1), minor=True)
    heatmap.set_yticks(np.arange(-.5, len(points), 1), minor=True)
    heatmap.grid(which="minor", color="white", linewidth=2)
    heatmap.tick_params(which="minor", bottom=False, left=False)
    candidates = [row for row in points if row["method"] in ("s1q-margin", "s1q", "s1q-ac", "s1q-joint", "rtn")]
    for column, metric, title in ((0, "nll", "C   NLL  ↓"), (2, "brier", "D   Brier score  ↓"), (4, "ece_15", "E   Calibration error (ECE15)  ↓")):
        ax = figure.add_subplot(grid[1, column:column+2])
        values = [row[metric] for row in candidates]
        if any(value is None for value in values):
            ax.text(.5, .5, "Metric unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for i, (row, value) in enumerate(zip(candidates, values)):
            color = GOLD if row["method"] == "s1q" else BLUE if row["method"] in S1Q_METHODS else GREY
            ax.barh(i, value, color=color, height=.6)
            ax.text(value + max(values) * .025, i, f"{value:.3f}", va="center", fontsize=9.5)
        ax.set_yticks(range(len(candidates)), [row["method_label"].replace(" (current)", "") for row in candidates], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_title(title, loc="left", pad=10, fontsize=11.5)
        ax.grid(axis="x", color="#E8ECEF", linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    wrapped = figure_scope_notes(scope)
    wrapped.extend([
        "Equal dataset averages within each model, then equal model averages. Complete common coverage only; no missing values imputed.",
        "* Repository baseline adaptations, not official reproductions. Floating-point QDQ accuracy experiments; no integer-kernel speedup claim.",
        "Ranks are descriptive point estimates. Subset sizes and excluded models are in ranking_scopes.json; these comparisons do not establish statistical superiority.",
    ])
    footer_top = .028 + .019 * (len(wrapped) - 1)
    grid.update(bottom=max(.155, footer_top + .065))
    for i, line in enumerate(wrapped):
        figure.text(.025, footer_top - i * .019, line, fontsize=9, color="#616B75")
    outputs = []
    for suffix in ("png", "svg"):
        target = output / f"benchmark_ranking.{suffix}"
        figure.savefig(target, dpi=220, metadata={"Software": "S1Q aggregate benchmark exporter"} if suffix == "png" else {"Date": None, "Creator": "S1Q aggregate benchmark exporter"})
        outputs.append({"file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size})
    plt.close(figure)
    return {"status": "generated", "suite_id": suite_id, "comparison_group": comparison_group, "n_models": scope["n_models"], "n_datasets": scope["n_datasets"], "method_order": [row["method"] for row in points], "outputs": outputs}


def report_text(rankings: list[dict], scopes: list[dict], plot: dict) -> str:
    text = ["# Completed S1Q benchmark evidence", "", "This directory contains aggregate metrics only. It contains no questions, individual predictions, checkpoints, or private paper files.", "", "## Reading the results", "", "Each ranking fixes the experiment batches, precision, dataset list, method group and complete common model coverage in `ranking_scopes.json`. Accuracy is the mean of dataset accuracy within a model, then the mean across models. It is not pooled accuracy across decisions, and no missing cells are imputed. Old and expanded protocols have separate scope IDs.", "", "The current public method is **S1Q**; the internal historical ID `s1q-mac` is normalized to `s1q`. `s1q-local` and `s1q2-beta05` are S1Qv1 and S1Qv2. Additional enhancement / repair methods are preserved in `metrics.csv` but are not substituted for the fixed main comparison group.", "", "All resource-independent accuracy rows here use floating-point quantize/dequantize execution. SmoothQuant*, AWQ*, GPTQ-block*, and SpinQuant* denote repository adaptations, not official method reproductions. A method appears in a comparison only when explicitly declared by that scope and completely measured. This evidence establishes neither integer-kernel acceleration nor state-of-the-art superiority. NanoJev measures reference-action compatibility and is excluded from text decision rankings.", "", "Point rankings do not establish statistical superiority; no aggregate confidence interval is fabricated. Earlier exploratory Dev and external sub-cohort reuse is identified in the input protocol. See `provenance.json` for input hashes and audited-summary agreement."]
    text.extend(["", "Bold values mark the best quantized displayed value, including ties after rounding; accuracy is higher-is-better and NLL/Brier/ECE15 are lower-is-better. Exact ranks use unrounded values."])
    for scope in scopes:
        if not scope["is_primary_comparison"]:
            continue
        text.extend(["", f"## {scope['suite_id']}", "", f"Precision: {scope['precision']}; comparison group: {scope['comparison_group']} ({len(scope['method_ids'])} methods); {scope['n_models']} complete common models; {scope['n_datasets']} required datasets/cohorts; {scope['n_cells']} model × dataset cells.", "", "Required datasets: " + ", ".join(scope["dataset_ids"]) + ".", "", "| Rank | Method | Accuracy % | NLL | Brier | ECE15 |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
        primary_rows = [row for row in rankings if row["suite_id"] == scope["suite_id"] and row["comparison_group"] == scope["comparison_group"] and row["family"] == "All"]
        best = {metric: best_display([row.get(metric) for row in primary_rows], decimals=2 if metric == "accuracy_percent" else 3, higher=metric == "accuracy_percent") for metric in ("accuracy_percent", "nll", "brier", "ece_15")}
        for row in primary_rows:
            values = [display_best(row.get(metric), best[metric], decimals=2 if metric == "accuracy_percent" else 3) for metric in ("accuracy_percent", "nll", "brier", "ece_15")]
            text.append(f"| {row['rank_accuracy']} | {row['method_label']} | " + " | ".join(values) + " |")
        if scope["excluded_models"]:
            text.extend(["", "Excluded incomplete models: " + ", ".join(scope["excluded_models"]) + ". Missing cells are recorded explicitly in `ranking_scopes.json`."])
    if plot.get("status") == "generated":
        text.extend(["", "## Visualization", "", f"The figure displays `{plot['suite_id']}` only.", "", "![Common-coverage accuracy, family rankings and probability metrics](benchmark_ranking.png)", "", "[Vector export](benchmark_ranking.svg)"])
    if any(scope["accuracy_table_enabled"] and scope["is_primary_comparison"] for scope in scopes):
        text.extend(["", "## Accuracy by model and dataset", "", "[Complete accuracy tables](accuracy_table.md) · [Aggregate cells CSV](accuracy_by_dataset.csv) · [Table provenance](accuracy_table_manifest.json). Tables contain every declared source, Native reference rows and the same complete common method/model coverage as the primary scope."])
    text.extend(["", "## Files", "", "- `metrics.csv` and `metrics.json`: all supplied aggregate rows, including legacy enhancement variants and separate protocols.", "- `rankings.csv`: exact equal-weight rankings with coverage, probability metrics and family slices.", "- `model_scores.csv`: each model's equally weighted dataset averages for every declared comparison group.", "- `ranking_scopes.json`: exact scope, required datasets, per-cell sample sizes and exclusions.", "- `provenance.json`: input hashes, row counts and sanitized checkpoint / quantization identities.", "- `figure_manifest.json`: figure-to-scope binding and output hashes.", ""])
    return "\n".join(text)


def best_display(values: list[float | None], *, decimals: int, higher: bool) -> str | None:
    rendered = [f"{value:.{decimals}f}" for value in values if value is not None]
    return (max if higher else min)(rendered, key=float) if rendered else None


def display_best(value: float | None, best: str | None, *, decimals: int, reference: bool = False) -> str:
    if value is None:
        return "—"
    rendered = f"{value:.{decimals}f}"
    return f"**{rendered}**" if not reference and rendered == best else rendered


def build_accuracy_tables(rows: list[dict], scopes: list[dict]) -> tuple[list[dict], str, list[dict]]:
    """Keep all source columns; split them into consecutive readable parts."""
    table_rows, manifests = [], []
    text = ["# Accuracy by model and source", "", "These are public benchmark tables generated from aggregate metrics, not manuscript files. Each section fixes one primary comparison scope and precision. **Bold marks the best quantized displayed accuracy, including rounding ties; Native reference rows are plain.** Exact best flags and unrounded accuracy are in the CSV.", "", "All declared source suites are retained in their frozen order. Wide tables are split into consecutive parts of at most eight sources; each part repeats **Avg**, which is the equally weighted mean over the entire declared source list, not that part's columns. Missing cells are not filled. Source roles and actual admitted denominators are recorded below and in the CSV."]
    for scope in scopes:
        if not scope["is_primary_comparison"] or not scope["accuracy_table_enabled"]:
            continue
        datasets, methods = scope["dataset_ids"], scope["method_ids"]
        source_rows = [row for row in rows if row["experiment_batch_id"] in scope["experiment_batch_ids"] and row["precision"] == scope["precision"] and row["model"] in scope["included_models"] and row["dataset_id"] in datasets and row["method"] in ("native", *methods) and row["status"] == "complete"]
        index = {}
        for row in source_rows:
            identity = (row["model"], row["dataset_id"], row["method"])
            if identity in index:
                raise ValueError(f"Duplicate accuracy table cell in {scope['suite_id']}: {identity}")
            index[identity] = row
        text.extend(["", f"## {scope['suite_id']} · {scope['precision']}", "", f"{scope['n_models']} complete common models; {len(methods)} quantization methods plus Native; Avg = equal mean of all {len(datasets)} source accuracies. {scope.get('note', '')}", "", "| Source | Role |", "| --- | --- |"])
        for dataset in datasets:
            meta = scope["dataset_metadata"][dataset]
            name = meta.get("name", dataset).replace("|", "\\|")
            category = meta.get("source_category", "evaluated_dataset_or_cohort")
            role = meta.get("figure_note") or category.replace("_", " ")
            role = role.replace("|", "\\|")
            text.append(f"| {name} (`{dataset}`) | {role} |")
        if not scope["included_models"]:
            text.extend(["", "No complete common model coverage is available for this scope; no accuracy values are reported."])
        for model in scope["included_models"]:
            complete_methods = ("native", *methods)
            for dataset in datasets:
                needed = [(model, dataset, method) for method in complete_methods]
                if any(identity not in index for identity in needed):
                    raise ValueError(f"Missing Native or quantized accuracy table cell in {scope['suite_id']}/{model}/{dataset}")
                if len({(index[identity]["n"], index[identity]["n_clusters"]) for identity in needed}) != 1:
                    raise ValueError(f"Native/quantized table denominators differ in {scope['suite_id']}/{model}/{dataset}")
            averages = {method: statistics.mean(index[(model, dataset, method)]["accuracy"] for dataset in datasets) for method in complete_methods}
            best_avg = max(averages[method] for method in methods)
            best_avg_display = best_display([100 * averages[method] for method in methods], decimals=2, higher=True)
            best_exact = {dataset: max(index[(model, dataset, method)]["accuracy"] for method in methods) for dataset in datasets}
            best_rendered = {dataset: best_display([100 * index[(model, dataset, method)]["accuracy"] for method in methods], decimals=2, higher=True) for dataset in datasets}
            for method in complete_methods:
                for dataset in datasets:
                    row = index[(model, dataset, method)]
                    meta = scope["dataset_metadata"][dataset]
                    accuracy = row["accuracy"]
                    table_rows.append({
                        "suite_id": scope["suite_id"], "comparison_group": scope["comparison_group"],
                        "model": model, "family": family(model), "precision": scope["precision"],
                        "experiment_batch_id": row["experiment_batch_id"], "run_id": row.get("run_id", ""),
                        "method": method, "method_label": LABELS.get(method, method), "dataset_id": dataset,
                        "dataset_label": meta.get("name", dataset), "source_category": meta.get("source_category", ""),
                        "label_kind": meta.get("label_kind", ""), "n": row["n"], "n_clusters": row["n_clusters"],
                        "accuracy": accuracy, "accuracy_percent": 100 * accuracy, "accuracy_display_percent": f"{100 * accuracy:.2f}",
                        "best_quantized_accuracy_exact": method != "native" and math.isclose(accuracy, best_exact[dataset], rel_tol=0, abs_tol=1e-12),
                        "best_quantized_accuracy_displayed": method != "native" and f"{100 * accuracy:.2f}" == best_rendered[dataset],
                        "avg_accuracy": averages[method], "avg_accuracy_percent": 100 * averages[method],
                        "avg_accuracy_display_percent": f"{100 * averages[method]:.2f}",
                        "best_quantized_avg_exact": method != "native" and math.isclose(averages[method], best_avg, rel_tol=0, abs_tol=1e-12),
                        "best_quantized_avg_displayed": method != "native" and f"{100 * averages[method]:.2f}" == best_avg_display,
                    })
            text.extend(["", f"### {model} · {scope['precision']}", "", "Native-admitted decisions per source: " + "; ".join(f"{dataset}={index[(model, dataset, 'native')]['n']}" for dataset in datasets) + "."])
            blocks = [datasets[start:start+8] for start in range(0, len(datasets), 8)]
            for part, block in enumerate(blocks, 1):
                text.extend(["", f"Part {part}/{len(blocks)} — sources {datasets.index(block[0])+1}–{datasets.index(block[-1])+1} of {len(datasets)}. Avg uses all {len(datasets)} sources.", "", "| Method | " + " | ".join(block) + f" | Avg ({len(datasets)} sources) |", "| --- | " + " | ".join("---:" for _ in block) + " | ---: |"])
                for method in complete_methods:
                    values = [display_best(100 * index[(model, dataset, method)]["accuracy"], best_rendered[dataset], decimals=2, reference=method == "native") for dataset in block]
                    values.append(display_best(100 * averages[method], best_avg_display, decimals=2, reference=method == "native"))
                    text.append("| " + LABELS.get(method, method) + " | " + " | ".join(values) + " |")
        manifests.append({"suite_id": scope["suite_id"], "comparison_group": scope["comparison_group"], "precision": scope["precision"], "included_models": scope["included_models"], "dataset_ids": datasets, "method_ids": methods, "native_reference": True, "n_rows": len(scope["included_models"]) * (len(methods) + 1) * len(datasets)})
    return table_rows, "\n".join(text) + "\n", manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="BATCH_ID=SUMMARY_DIR")
    parser.add_argument("--dataset-list", type=Path, help="JSON ranking_suites with explicit batch IDs, precision, and required dataset_ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    rows, sources, batches = [], [], set()
    for argument in args.input:
        batch, separator, directory = argument.partition("=")
        if not separator or not re.fullmatch(r"[a-zA-Z0-9_.-]+", batch) or batch in batches:
            raise ValueError("Every input requires a unique explicit BATCH_ID=SUMMARY_DIR")
        batches.add(batch)
        source_rows, provenance = audited_input(batch, Path(directory))
        rows.extend(source_rows)
        sources.append(provenance)
    settings = json.loads(args.dataset_list.read_text(encoding="utf-8-sig")) if args.dataset_list else default_scopes(rows)
    rankings, scopes, model_scores = build_rankings(rows, settings)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "metrics.json", {"schema": "s1q.public-aggregate-metrics.v1", "rows": rows})
    write_csv(output / "rankings.csv", rankings)
    write_csv(output / "model_scores.csv", model_scores)
    write_json(output / "ranking_scopes.json", {"schema": "s1q.public-ranking-scopes.v1", "declared_settings": settings, "computed_scopes": scopes})
    write_json(output / "provenance.json", {"schema": "s1q.public-aggregate-provenance.v1", "script_sha256": sha256(Path(__file__)), "input_sources": sources, "aggregate_rows": len(rows), "privacy": "aggregate_only; no source questions or individual predictions", "execution": "floating_point_QDQ", "methods_are_repository_adaptations": True})
    accuracy_rows, accuracy_text, accuracy_scopes = build_accuracy_tables(rows, scopes)
    if accuracy_scopes:
        write_csv(output / "accuracy_by_dataset.csv", accuracy_rows)
        (output / "accuracy_table.md").write_text(accuracy_text, encoding="utf-8")
        write_json(output / "accuracy_table_manifest.json", {"schema": "s1q.public-accuracy-tables.v1", "scopes": accuracy_scopes, "aggregation": "equal_source_mean_over_all_declared_sources; split_parts_do_not_change_Avg", "bold_rule": "best_quantized_displayed_value_including_rounding_ties;_native_plain", "accuracy_csv_sha256": sha256(output / "accuracy_by_dataset.csv"), "accuracy_table_sha256": sha256(output / "accuracy_table.md"), "ranking_scopes_sha256": sha256(output / "ranking_scopes.json"), "metrics_sha256": sha256(output / "metrics.csv")})
    plot = {"status": "not_requested"} if args.no_plot else plot_rankings(output, rankings, scopes, settings.get("plot_suite_id") or settings["ranking_suites"][0]["suite_id"])
    plot["ranking_scopes_sha256"] = sha256(output / "ranking_scopes.json")
    plot["rankings_sha256"] = sha256(output / "rankings.csv")
    write_json(output / "figure_manifest.json", plot)
    (output / "README.md").write_text(report_text(rankings, scopes, plot), encoding="utf-8")
    print(json.dumps({"aggregate_rows": len(rows), "ranking_rows": len(rankings), "scope_count": len(scopes), "plot": plot["status"], "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
