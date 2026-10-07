"""Conditional uncertainty for a fixed public accuracy comparison scope.

Within each source, draw whole saved clusters with replacement. The same draws
are used for every model and method. Compute a ratio of correct decisions to
admitted decisions per source/model, then average sources and models equally.
Models and sources themselves are fixed, not bootstrapped. Only aggregate
intervals, counts and input hashes are written; predictions and bootstrap arrays
remain private in memory. NumPy is the only non-standard-library dependency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


AGGREGATION = "equal_dataset_within_model_then_equal_model; unrounded inputs; not pooled decision accuracy"
PAIR_FIELDS = ("key", "record_id", "qid", "type", "source", "cluster_id", "label", "label_kind")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON constant {x}")))


def unique_strings(values, context):
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"{context} requires a nonempty string list")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {context}")
    return tuple(values)


def select_scope(path: Path, suite_id: str, comparison_group: str) -> dict:
    document = read_json(path)
    if document.get("schema") != "s1q.public-ranking-scopes.v1":
        raise ValueError("A computed public ranking_scopes.json is required")
    matches = [scope for scope in document.get("computed_scopes", [])
               if scope.get("suite_id") == suite_id and scope.get("comparison_group") == comparison_group]
    if len(matches) != 1:
        raise ValueError("Exactly one computed suite/comparison scope is required")
    scope = matches[0]
    models = unique_strings(scope.get("included_models"), "included models")
    sources = unique_strings(scope.get("dataset_ids"), "source suites")
    methods = unique_strings(scope.get("method_ids"), "method IDs")
    if "native" in methods or (comparison_group == "all12" and len(methods) != 12):
        raise ValueError("Native is not a quantizer rank; all12 requires twelve methods")
    for field, expected in (("n_models", len(models)), ("n_datasets", len(sources)), ("n_cells", len(models) * len(sources))):
        if scope.get(field) != expected:
            raise ValueError(f"Scope {field} is inconsistent")
    if scope.get("aggregation") != AGGREGATION:
        raise ValueError("Scope uses a different accuracy estimator")
    counts = scope.get("sample_counts", [])
    count_cells = {(row.get("model"), row.get("dataset_id")): row for row in counts}
    if len(count_cells) != len(counts) or set(count_cells) != {(m, s) for m in models for s in sources}:
        raise ValueError("Scope sample counts must cover each fixed model/source once")
    for row in counts:
        if any(type(row.get(key)) is not int or row[key] <= 0 for key in ("n", "n_clusters")):
            raise ValueError("Scope sample counts must be positive integers")
    return scope


def predictions(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON {x}")))
            for field in PAIR_FIELDS:
                if field not in row:
                    raise ValueError(f"Missing prediction {field}")
            if any(not isinstance(row[field], str) or not row[field] for field in PAIR_FIELDS if field != "label"):
                raise ValueError("Prediction identities and metadata must be nonempty strings")
            key = row["key"]
            if key in rows or key != f"{row['record_id']}::{row['qid']}":
                raise ValueError("Duplicate or inconsistent decision key")
            logits = row.get("logits")
            if not isinstance(logits, list) or len(logits) < 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits):
                raise ValueError("Invalid option logits")
            if type(row["label"]) is not int or not 0 <= row["label"] < len(logits):
                raise ValueError("Invalid gold option index")
            if "options" in row and (not isinstance(row["options"], list) or len(row["options"]) != len(logits)):
                raise ValueError("Invalid explicit options")
            rows[key] = row
    if not rows:
        raise ValueError("Empty completed prediction file")
    return rows


def check_paired(left: dict, right: dict, *, exact_keys: bool = True):
    if exact_keys and left.keys() != right.keys():
        raise ValueError("Exact paired prediction keys are required")
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if any(a[field] != b[field] or type(a[field]) is not type(b[field]) for field in PAIR_FIELDS):
            raise ValueError(f"Paired gold/cluster/metadata differ for {key}")
        if len(a["logits"]) != len(b["logits"]):
            raise ValueError(f"Paired option count differs for {key}")
        if "options" in a or "options" in b:
            if a.get("options") != b.get("options"):
                raise ValueError(f"Paired option order differs for {key}")


@dataclass
class SourceCounts:
    """Private sufficient statistics; group identities are never serialized."""
    source: str
    clusters: tuple[str, ...]
    counts: np.ndarray  # model, cluster
    correct: np.ndarray  # model, method (including native), cluster


def count_source(source: str, models: tuple[str, ...], methods: tuple[str, ...], rows_by_model: dict) -> SourceCounts:
    native = {model: rows_by_model[model]["native"] for model in models}
    for index, model in enumerate(models):
        for method in methods:
            check_paired(native[model], rows_by_model[model][method])
        for other in models[:index]:
            check_paired(native[model], native[other], exact_keys=False)
    clusters = tuple(sorted({row["cluster_id"] for rows in native.values() for row in rows.values()}))
    position = {cluster: index for index, cluster in enumerate(clusters)}
    counts = np.zeros((len(models), len(clusters)), dtype=np.int64)
    correct = np.zeros((len(models), len(methods) + 1, len(clusters)), dtype=np.int64)
    for model_index, model in enumerate(models):
        for key, row in native[model].items():
            cluster_index = position[row["cluster_id"]]
            counts[model_index, cluster_index] += 1
            for method_index, method in enumerate(("native", *methods)):
                candidate = rows_by_model[model][method][key]
                correct[model_index, method_index, cluster_index] += int(np.argmax(candidate["logits"]) == candidate["label"])
    return SourceCounts(source, clusters, counts, correct)


def bootstrap(sources: list[SourceCounts], *, samples: int, seed: int) -> dict:
    """Return private replicate arrays for tests/aggregation, never publication."""
    if samples <= 0 or not sources:
        raise ValueError("Positive samples and nonempty sources are required")
    shape = sources[0].correct.shape[:2]
    model_points = np.zeros(shape, dtype=np.float64)
    model_replicates = np.zeros((samples, *shape), dtype=np.float64)
    rng = np.random.default_rng(seed)
    zero_denominators = 0
    for source in sources:
        if source.correct.shape[:2] != shape or source.counts.shape != (shape[0], len(source.clusters)):
            raise ValueError("Incompatible source sufficient statistics")
        if np.any(source.counts < 0) or np.any(source.correct < 0) or np.any(source.correct > source.counts[:, None, :]):
            raise ValueError("Invalid source sufficient statistics")
        totals = source.counts.sum(axis=1)
        if np.any(totals == 0):
            raise ValueError("A fixed model/source cell has no admitted decisions")
        model_points += source.correct.sum(axis=2) / totals[:, None] / len(sources)
        groups = len(source.clusters)
        counts_float = source.counts.T.astype(np.float64)
        correct_float = source.correct.reshape(-1, groups).T.astype(np.float64)
        # Whole-group multiplicities are reused for every model and method.
        # Chunking does not change NumPy's flat integer RNG draw sequence.
        chunk = max(1, min(samples, 128, 1_000_000 // groups))
        for start in range(0, samples, chunk):
            length = min(chunk, samples - start)
            indices = rng.integers(0, groups, size=(length, groups))
            weights = np.zeros((length, groups), dtype=np.float64)
            np.add.at(weights, (np.arange(length)[:, None], indices), 1)
            denominator = weights @ counts_float
            bad = denominator == 0
            zero_denominators += int(bad.sum())
            numerator = (weights @ correct_float).reshape(length, *shape)
            ratio = np.divide(numerator, denominator[:, :, None],
                              out=np.zeros_like(numerator), where=~bad[:, :, None])
            model_replicates[start:start + length] += ratio / len(sources)
    if zero_denominators:
        raise ValueError(f"Bootstrap has {zero_denominators} zero admitted-decision denominators; no draws were replaced")
    return {"model_points": model_points, "points": model_points.mean(axis=0),
            "model_replicates": model_replicates, "replicates": model_replicates.mean(axis=1),
            "zero_denominator_model_source_draws": zero_denominators}


def model_name(identity):
    model = identity.get("model")
    return model.get("name") if isinstance(model, dict) else model


def find_runs(root: Path, models: tuple[str, ...]) -> dict:
    found = {}
    for path in sorted(root.rglob("report.json")):
        report = read_json(path)
        name = model_name(report.get("identity", {}))
        if name not in models:
            continue
        if name in found:
            raise ValueError(f"Ambiguous multiple runs for fixed model {name}")
        found[name] = (path.parent, report)
    if set(found) != set(models):
        raise ValueError("Missing a fixed included model run")
    return found


def evaluation(report, method, source):
    if method == "native":
        return report.get("native_evaluations", {}).get(source, report.get(f"native_{source}", {}))
    item = report["methods"][method]
    return item.get("evaluations", {}).get(source, item.get(source, {}))


def audit_quantization(report, methods, precision):
    identity = report["identity"]
    actual = f"W{identity.get('bits')}A{identity.get('activation_bits')}"
    if actual != precision:
        raise ValueError("Run precision differs from the fixed scope")
    selected = set(identity.get("selected_linear_names", []))
    if not selected:
        raise ValueError("Missing frozen quantization layer selection")
    reference = None
    for method in methods:
        item = report.get("methods", {}).get(method, {})
        if item.get("status") != "complete":
            raise ValueError(f"A fixed method did not complete: {method}")
        quantization = item.get("quantization", {})
        layers = quantization.get("layers", {})
        if set(layers) != selected or quantization.get("activation_bits") != identity.get("activation_bits"):
            raise ValueError("Method quantization scope differs from the frozen layer selection")
        signature = {name: (layer.get("bits"), layer.get("group_size"), layer.get("shape")) for name, layer in layers.items()}
        if any(bits != identity.get("bits") or group != identity.get("group_size") for bits, group, _ in signature.values()):
            raise ValueError("Method weight precision or group size differs")
        if reference is not None and signature != reference:
            raise ValueError("Methods used different weight layer shapes/scopes")
        reference = signature


def load_counts(runs_root: Path, scope: dict):
    models, sources, methods = tuple(scope["included_models"]), tuple(scope["dataset_ids"]), tuple(scope["method_ids"])
    runs = find_runs(runs_root, models)
    scope_counts = {(row["model"], row["dataset_id"]): row for row in scope["sample_counts"]}
    input_hashes, source_statistics, sources_hashes, derivations = [], [], {}, {}
    cells = []
    for model, (folder, report) in runs.items():
        identity = report["identity"]
        if read_json(folder / "identity.json") != identity:
            raise ValueError("Standalone identity differs from report")
        marker = read_json(folder / "expanded-evaluation-complete.json")
        if marker.get("status") != "complete" or any(marker.get("methods", {}).get(method) != "complete" for method in methods):
            raise ValueError("Missing complete evaluation marker")
        if identity.get("final_test_read") is not True:
            raise ValueError("Run must explicitly declare evaluated upstream held-out requests")
        if not set(sources).issubset(identity.get("expanded_ranking_dataset_ids", [])):
            raise ValueError("Requested sources are outside the frozen rank-eligible scope")
        if marker.get("ranking_dataset_ids") != identity.get("expanded_ranking_dataset_ids"):
            raise ValueError("Completion marker and identity disagree about the frozen ranking sources")
        if marker.get("benchmark_manifest_sha256") != identity.get("benchmark_manifest_sha256"):
            raise ValueError("Completion marker and identity disagree about the data manifest")
        audit_quantization(report, methods, scope["precision"])
        for filename in ("report.json", "identity.json", "expanded-evaluation-complete.json"):
            input_hashes.append({"model": model, "role": filename, "sha256": digest(folder / filename)})
        derivation_path = folder / "derivation-manifest.json"
        if derivation_path.exists():
            derivation = read_json(derivation_path)
            if derivation.get("status") != "complete" or derivation.get("model") != model:
                raise ValueError("Incomplete or misidentified derived evidence")
            if derivation.get("benchmark_manifest_sha256") != identity.get("benchmark_manifest_sha256"):
                raise ValueError("Derived evidence uses a different frozen data manifest")
            for filename in ("report.json", "identity.json", "expanded-evaluation-complete.json"):
                if derivation.get("files", {}).get(filename, {}).get("sha256") != digest(folder / filename):
                    raise ValueError("Derived run metadata differs from its recorded hash")
            derivations[model] = derivation
            input_hashes.append({"model": model, "role": "derivation-manifest.json", "sha256": digest(derivation_path)})
        for source in sources:
            source_hash = identity.get("source_sha256", {}).get(source)
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError("Missing frozen source hash")
            if source in sources_hashes and sources_hashes[source] != source_hash:
                raise ValueError("Models did not evaluate the same frozen source requests")
            sources_hashes[source] = source_hash
    for source in sources:
        rows_by_model = {}
        for model in models:
            folder, report = runs[model]
            identity = report["identity"]
            rows_by_model[model] = {}
            expected_count = scope_counts[model, source]
            accepted = identity.get("accepted_evaluation_ids", {}).get(source)
            if not isinstance(accepted, list) or len(set(accepted)) != len(accepted) or not accepted:
                raise ValueError("Missing or duplicated frozen accepted request IDs")
            for method in ("native", *methods):
                path = folder / f"{method}-{source}.jsonl"
                rows = predictions(path)
                if {row["record_id"] for row in rows.values()} != set(accepted):
                    raise ValueError("Prediction request IDs differ from frozen admission")
                count = len(rows)
                clusters = len({row["cluster_id"] for row in rows.values()})
                if count != expected_count["n"] or clusters != expected_count["n_clusters"]:
                    raise ValueError("Prediction counts differ from the computed public scope")
                accuracy = sum(int(np.argmax(row["logits"]) == row["label"]) for row in rows.values()) / count
                recorded = evaluation(report, method, source)
                raw = recorded.get("raw", recorded)
                if raw.get("n") != count or not math.isclose(raw.get("accuracy", -1), accuracy, rel_tol=0, abs_tol=1e-10):
                    raise ValueError("Saved evaluation accuracy/count differs from predictions")
                if model in derivations:
                    entry = derivations[model].get("files", {}).get(path.name, {})
                    if entry.get("sha256") != digest(path):
                        raise ValueError("A derived prediction payload differs from its recorded hash")
                rows_by_model[model][method] = rows
                input_hashes.append({"model": model, "dataset_id": source, "method": method, "sha256": digest(path)})
            cells.append({"model": model, "dataset_id": source, "n_decisions": expected_count["n"],
                          "n_clusters": expected_count["n_clusters"], "n_requests": len(accepted)})
        statistic = count_source(source, models, methods, rows_by_model)
        common = set(rows_by_model[models[0]]["native"])
        for model in models[1:]:
            common.intersection_update(rows_by_model[model]["native"])
        source_statistics.append(statistic)
        for cell in cells:
            if cell["dataset_id"] == source:
                cell.update(shared_union_n_clusters=len(statistic.clusters), common_admitted_n_decisions=len(common))
    return source_statistics, input_hashes, cells, sources_hashes


def check_public_points(path: Path, scope: dict, methods, points) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("suite_id") == scope["suite_id"] and row.get("comparison_group") == scope["comparison_group"]
                and row.get("family", "All") == "All"]
    indexed = {row.get("method"): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(methods):
        raise ValueError("Public rankings.csv must contain each fixed method exactly once")
    maximum = 0.0
    for method, point in zip(methods, points):
        row = indexed[method]
        if row.get("precision") != scope["precision"] or any(int(row.get(field, -1)) != scope[field] for field in ("n_models", "n_datasets", "n_cells")):
            raise ValueError("Public point ranking scope differs")
        difference = abs(float(row["accuracy"]) - float(point))
        if not math.isfinite(difference) or difference > 1e-10:
            raise ValueError(f"Public point accuracy differs by more than 1e-10: {method}")
        maximum = max(maximum, difference)
    return {"provided": True, "sha256": digest(path), "maximum_absolute_difference": maximum, "tolerance": 1e-10}


def summarize(runs_root: Path, ranking_scopes: Path, *, suite_id: str, comparison_group: str,
              samples: int, seed: int, rankings_csv: Path | None = None) -> dict:
    scope = select_scope(ranking_scopes, suite_id, comparison_group)
    sources, inputs, cells, source_hashes = load_counts(runs_root, scope)
    boot = bootstrap(sources, samples=samples, seed=seed)
    names = ("native", *scope["method_ids"])
    method_rows = []
    for index, method in enumerate(names):
        low, high = np.quantile(boot["replicates"][:, index], [.025, .975])
        point = float(boot["points"][index])
        method_rows.append({"method": method, "ranked": method != "native", "accuracy": point,
                            "lower_95": float(low), "upper_95": float(high), "accuracy_percent": point * 100,
                            "lower_95_percent": float(low) * 100, "upper_95_percent": float(high) * 100})
    pair_rows = []
    for left in range(1, len(names)):
        for right in range(left + 1, len(names)):
            differences = boot["replicates"][:, left] - boot["replicates"][:, right]
            low, high = np.quantile(differences, [.025, .975])
            point = float(boot["points"][left] - boot["points"][right])
            pair_rows.append({"method": names[left], "reference": names[right], "difference": point,
                              "lower_95": float(low), "upper_95": float(high), "difference_pp": point * 100,
                              "lower_95_pp": float(low) * 100, "upper_95_pp": float(high) * 100})
    public_check = {"provided": False, "tolerance": 1e-10}
    if rankings_csv:
        public_check = check_public_points(rankings_csv, scope, scope["method_ids"], boot["points"][1:])
    return {"schema": "s1q.conditional-macro-accuracy-uncertainty.v1", "status": "passed",
            "suite_id": suite_id, "comparison_group": comparison_group, "precision": scope["precision"],
            "included_models": scope["included_models"], "dataset_ids": scope["dataset_ids"],
            "method_ids": scope["method_ids"], "n_models": scope["n_models"], "n_datasets": scope["n_datasets"],
            "n_cells": scope["n_cells"], "aggregation": AGGREGATION,
            "bootstrap": {"seed": seed, "samples": samples, "interval": "percentile 95% (2.5%, 97.5%)",
                          "resampling_unit": "whole saved cluster_id within each source",
                          "shared_draws": "same source-union cluster multiplicities across all models and methods",
                          "estimator": "model/source admitted correct-decision sum divided by admitted decision sum, then sources and models equally weighted",
                          "models_resampled": False, "sources_resampled": False,
                          "zero_denominator_model_source_draws": 0, "zero_denominator_policy": "fail; never silently replace a draw"},
            "method_accuracy": method_rows, "pairwise_differences": pair_rows, "sample_counts": cells,
            "public_point_check": public_check,
            "provenance": {"script_sha256": digest(Path(__file__)), "ranking_scopes_sha256": digest(ranking_scopes),
                           "frozen_source_sha256": source_hashes, "input_file_hashes": inputs,
                           "numpy_version": np.__version__, "input_paths_omitted": True},
            "interpretation": ["Conditional uncertainty on evaluated samples for these fixed admitted models and source suites; no universal winner claim.",
                               "Source/model cells have equal weight; larger suites and multi-question groups do not gain source weight.",
                               "Different admission subsets are retained exactly as declared; cross-model overlap counts are reported. Intervals do not establish a common population across different admission rules.",
                               "Paired differences use shared bootstrap replicates; intervals are not adjusted for multiple comparisons.",
                               "Native is a reference and receives no quantizer rank; point ranks alone do not establish a significant difference."]}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method", "reference"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--ranking-scopes", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--comparison-group", default="all12")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--rankings-csv", type=Path, help="Optional unrounded public point ranking to check within 1e-10")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; prior reports are never overwritten")
    if output.is_relative_to(args.runs_root.resolve()):
        raise ValueError("Output must be outside the immutable run directory")
    result = summarize(args.runs_root, args.ranking_scopes, suite_id=args.suite_id,
                       comparison_group=args.comparison_group, samples=args.samples, seed=args.seed,
                       rankings_csv=args.rankings_csv)
    output.mkdir(parents=True, exist_ok=False)
    (output / "accuracy_uncertainty.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "method-accuracy-ci.csv", result["method_accuracy"])
    write_csv(output / "pairwise-accuracy-ci.csv", result["pairwise_differences"])
    lines = ["# Conditional accuracy uncertainty", "", f"Fixed scope: {args.suite_id}; {result['n_models']} models × {result['n_datasets']} source suites.", "",
             "Whole groups are resampled within sources, with shared draws across methods and models. Sources and models retain equal weight. These intervals describe the evaluated samples; they do not establish a universal winner.", "",
             "| Method | Accuracy % | Conditional 95% interval % |", "| --- | ---: | ---: |"]
    lines.extend(f"| {row['method']} | {row['accuracy_percent']:.3f} | [{row['lower_95_percent']:.3f}, {row['upper_95_percent']:.3f}] |" for row in result["method_accuracy"])
    (output / "accuracy_uncertainty.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "n_models": result["n_models"], "n_sources": result["n_datasets"],
                      "method_intervals": len(result["method_accuracy"]), "paired_intervals": len(result["pairwise_differences"]),
                      "public_point_check": result["public_point_check"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
