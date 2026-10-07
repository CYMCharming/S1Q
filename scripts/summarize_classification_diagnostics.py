"""Aggregate class diagnostics on audited semantic-class sources.

This supplements the frozen accuracy ranking; it never changes its estimator,
source/model weights, data, labels, admission, or method ranking. Source bytes
and saved prediction keys/gold/candidate dimensions are checked before counting.
Candidate-to-class mapping follows the frozen source and evaluator ordering
contract. Saved option keys, when present, are also checked; logits without
saved option keys cannot independently establish observed semantic ordering.
Only aggregate counts, metrics, source schemas and hashes are published. This
module deliberately imports no project runtime, Torch, or model dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


METHODS = (
    "rtn", "s1q-local", "s1q2-beta05", "smoothquant-adapted", "awq-adapted",
    "gptq-blockdiag-adapted", "spinquant-nohad-adapted", "spinquant-had-adapted",
    "s1q-joint", "s1q-ac", "s1q-margin", "s1q",
)
ALL_METHODS = ("native", *METHODS)
PAIR_FIELDS = ("key", "record_id", "qid", "type", "source", "cluster_id", "label", "label_kind")
ACCURACY_AGGREGATION = "equal_dataset_within_model_then_equal_model; unrounded inputs; not pooled decision accuracy"
SUBSET_CAVEAT = (
    "Class distributions describe the screened frozen subset and each model's declared admitted subset; "
    "they do not estimate class prevalence or performance on the full upstream dataset."
)


@dataclass(frozen=True)
class ClassSchema:
    question_type: str
    candidate_keys: tuple[str, ...]
    class_names: tuple[str, ...]
    raw_sources: tuple[str, ...]
    order_policy: str = "fixed"


# Explicit semantic allowlist. Stable answer-position names such as a/b/c/d or
# opt_1/opt_2 are never enough to qualify a source for class diagnostics.
SCHEMAS = {
    "wildjailbreak": ClassSchema("choice", ("harmful", "benign"), ("harmful", "benign"), ("wildjailbreak",)),
    "tweet-offensive": ClassSchema("noul", ("false", "true"), ("not_offensive", "offensive"), ("tweet_offensive",)),
    "mnli": ClassSchema("choice", ("entailment", "neutral", "contradiction"),
                        ("entailment", "neutral", "contradiction"), ("mnli",)),
    "wanli": ClassSchema("choice", ("supported", "insufficient", "contradicted"),
                         ("supported", "insufficient", "contradicted"), ("wanli_nli",), "source-question permutation"),
}
SKIP_REASONS = {
    "arc": "Answer positions are question-specific, not semantic classes.",
    "csqa": "Answer positions are question-specific, not semantic classes.",
    "mmlu": "Answer positions are question-specific, not semantic classes.",
    "openbookqa": "Answer positions are question-specific, not semantic classes.",
    "sciq": "Answer positions are question-specific, not semantic classes.",
    "agnews": "Mixed multiclass and boolean questions; no single source-wide class schema in this diagnostic release.",
    "yelp": "Mixed rating and boolean questions; no single source-wide class schema in this diagnostic release.",
    "jevbench": "Multiple question-specific task schemas; no single source-wide class schema.",
    "semif": "Question-specific action classes; no single source-wide class schema.",
    "toolace": "Question-specific tool candidates; no single source-wide class schema.",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def decode(value):
    return json.loads(value, parse_constant=_constant, object_pairs_hook=_object)


def read_json(path: Path):
    return decode(path.read_text(encoding="utf-8-sig"))


def jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = decode(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be objects")
                yield row


def unique_strings(values, context):
    if not isinstance(values, (list, tuple)) or not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"{context} requires nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {context}")
    return tuple(values)


def _hash(value, context):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid {context} SHA-256")
    return value


def candidate_keys(question, schema):
    if question.get("type") != schema.question_type:
        raise ValueError("Source question type differs from its fixed class schema")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != set(schema.candidate_keys):
        raise ValueError("Source option keys differ from its fixed semantic class schema")
    # Native source criteria may be strings, nested descriptors, or null (MNLI
    # uses all three). Semantic classes are the audited keys, not the rendering
    # of their descriptions. The frozen byte hash binds those descriptions.
    if schema.question_type == "noul":
        # The saved evaluator's boolean candidate order is false/true, even
        # where the source criteria dictionary is written true/false.
        keys = schema.candidate_keys
    else:
        keys = tuple(criteria)
    if schema.order_policy == "fixed" and keys != schema.candidate_keys:
        raise ValueError("Source option order differs from its fixed class schema")
    return keys


def gold_index(question, keys):
    label = question.get("label")
    if question["type"] == "noul":
        if type(label) is not bool:
            raise ValueError("Boolean source gold must be a boolean")
        return int(label)
    if not isinstance(label, str) or label not in keys:
        raise ValueError("Source gold is outside its semantic class schema")
    return keys.index(label)


def screening_counts(entry):
    """Publish only documented numeric screening counts, never source rows."""
    screening = entry.get("screening", {})
    if not isinstance(screening, dict):
        raise ValueError("Invalid source screening metadata")
    result = {}
    for field in ("raw_candidates", "unique_eligible_remaining", "selection_limit"):
        if field in screening:
            value = screening[field]
            if type(value) is not int or value < 0:
                raise ValueError("Screening counts must be nonnegative integers")
            result[field] = value
    excluded = screening.get("excluded", {})
    if not isinstance(excluded, dict) or any(not re.fullmatch(r"[a-z0-9_]+", key) or type(value) is not int or value < 0
                                             for key, value in excluded.items()):
        raise ValueError("Screening exclusions require aggregate count metadata")
    result["excluded"] = excluded
    return result


@dataclass
class FrozenSource:
    """Private source identities are used for validation, never serialized."""
    dataset_id: str
    entry: dict
    schema: ClassSchema
    rows: dict
    request_ids: tuple[str, ...]
    summary: dict


def load_source(root, entry, schema):
    source = entry["dataset_id"]
    relative = entry.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Source path must stay inside the manifest directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escaped the manifest directory")
    expected_hash = _hash(entry.get("sha256"), "source")
    if digest(path) != expected_hash:
        raise ValueError("Frozen source SHA-256 mismatch")
    if entry.get("ranking_eligible") is not True or entry.get("label_kind") not in ("upstream_dataset_gold", "native_annotation"):
        raise ValueError("Class diagnostics require independent, rank-eligible source labels")
    rows, requests, groups, orders = {}, [], set(), set()
    counts = Counter({name: 0 for name in schema.class_names})
    for record in jsonl(path):
        meta = record.get("_meta", {})
        rid = record.get("id", meta.get("id", record.get("request_id")))
        if not isinstance(rid, str) or not rid or rid in requests:
            raise ValueError("Missing or duplicate frozen source request ID")
        cluster = meta.get("group_id", rid)
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("Missing frozen source cluster ID")
        questions = record.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Source requires a nonempty questions mapping")
        requests.append(rid)
        groups.add(cluster)
        for qid, question in questions.items():
            if not isinstance(qid, str) or not qid or not isinstance(question, dict):
                raise ValueError("Invalid frozen source question")
            keys = candidate_keys(question, schema)
            orders.add(keys)
            label = gold_index(question, keys)
            raw_source = question.get("src", record.get("source", meta.get("source", "unknown")))
            if raw_source not in schema.raw_sources:
                raise ValueError("Source question provenance differs from its fixed class schema")
            kind = question.get("label_kind", question.get("gold_label_kind", "dataset_gold"))
            if kind not in ("dataset_gold", "native_annotation", "upstream_dataset_gold"):
                raise ValueError("Source question has an unsupported gold label kind")
            key = f"{rid}::{qid}"
            if key in rows:
                raise ValueError("Duplicate frozen source decision key")
            row = {"key": key, "record_id": rid, "qid": qid, "type": question["type"],
                   "source": raw_source, "cluster_id": cluster, "label": label, "label_kind": kind,
                   "candidate_keys": keys}
            rows[key] = row
            counts[schema.class_names[schema.candidate_keys.index(keys[label])]] += 1
    for field, value in (("n_requests", len(requests)), ("n_decisions", len(rows)), ("n_groups", len(groups))):
        if type(entry.get(field)) is not int or entry[field] != value or value <= 0:
            raise ValueError(f"Frozen source {field} differs from manifest")
    if entry.get("selected_ids") != requests:
        raise ValueError("Frozen source request order differs from manifest selected IDs")
    summary = {"dataset_id": source, "name": entry.get("name", source), "source_sha256": expected_hash,
               "class_names": list(schema.class_names), "canonical_candidate_keys": list(schema.candidate_keys),
               "question_type": schema.question_type, "candidate_order_policy": schema.order_policy,
               "source_candidate_order_count": len(orders), "n_requests": len(requests),
               "n_decisions": len(rows), "n_clusters": len(groups), "gold_counts": dict(counts),
               "gold_class_proportions": {name: counts[name] / len(rows) for name in schema.class_names},
               "majority_class_accuracy": max(counts.values()) / len(rows),
               "frozen_subset_screening": screening_counts(entry),
               "subset_caveat": SUBSET_CAVEAT}
    return FrozenSource(source, entry, schema, rows, tuple(requests), summary)


def load_manifest(path):
    document = read_json(path)
    if document.get("schema") != "s1q.expanded-benchmarks.v1":
        raise ValueError("Expected an expanded benchmark manifest")
    entries = document.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest requires datasets")
    ids = unique_strings([e.get("dataset_id") for e in entries], "manifest dataset IDs")
    if "wildjailbreak" not in ids:
        raise ValueError("WildJailBreak is required for classification diagnostics")
    sources, registry = [], []
    for entry in entries:
        source = entry["dataset_id"]
        if source in SCHEMAS:
            frozen = load_source(path.parent, entry, SCHEMAS[source])
            sources.append(frozen)
            registry.append({"dataset_id": source, "status": "included", "reason": "Explicit fixed semantic class allowlist; every frozen source question validated.",
                             "class_names": list(frozen.schema.class_names), "source_sha256": frozen.summary["source_sha256"]})
        else:
            registry.append({"dataset_id": source, "status": "excluded", "reason": SKIP_REASONS.get(source, "Outside this diagnostic release's explicit semantic class allowlist.")})
    return document, sources, registry


def predictions(path, expected, schema):
    """Check keys, source-derived gold indices and width, plus optional options.

    Without saved option keys, semantic order relies on the frozen source and
    evaluator contract; it is not independently observed from saved logits.
    """
    rows = {}
    for row in jsonl(path):
        if any(field not in row for field in PAIR_FIELDS):
            raise ValueError("Missing required prediction identity/gold field")
        key = row["key"]
        if not isinstance(key, str) or key in rows or key not in expected:
            raise ValueError("Prediction keys are duplicate or outside the frozen admitted source")
        target = expected[key]
        for field in PAIR_FIELDS:
            if row[field] != target[field] or type(row[field]) is not type(target[field]):
                raise ValueError(f"Prediction {field} differs from frozen source gold/metadata")
        logits = row.get("logits")
        if not isinstance(logits, list) or len(logits) != len(target["candidate_keys"]):
            raise ValueError("Prediction candidate dimension differs from fixed source schema")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in logits):
            raise ValueError("Prediction logits must be finite numbers")
        if "options" in row and row["options"] != list(target["candidate_keys"]):
            raise ValueError("Prediction option order differs from its frozen source question")
        if set(target["candidate_keys"]) != set(schema.candidate_keys):
            raise ValueError("Prediction options are outside fixed semantic class schema")
        rows[key] = row
    if rows.keys() != expected.keys():
        raise ValueError("Exact matched prediction keys are required")
    return rows


def count_predictions(rows, expected, schema):
    matrix = [[0 for _ in schema.class_names] for _ in schema.class_names]
    for key, row in rows.items():
        target = expected[key]
        # Python max, like NumPy argmax in the ranking, resolves ties to the
        # first saved logit position. Map positions using the source/evaluator
        # ordering contract; saved logits alone do not capture semantic order.
        predicted = max(range(len(row["logits"])), key=row["logits"].__getitem__)
        gold_class = schema.candidate_keys.index(target["candidate_keys"][row["label"]])
        pred_class = schema.candidate_keys.index(target["candidate_keys"][predicted])
        matrix[gold_class][pred_class] += 1
    return matrix_metrics(matrix, schema.class_names)


def matrix_metrics(matrix, class_names):
    classes = len(class_names)
    if not classes or len(matrix) != classes or any(len(row) != classes for row in matrix):
        raise ValueError("Confusion matrix must cover the fixed class schema")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("Confusion counts must be nonnegative integers")
    gold = [sum(row) for row in matrix]
    pred = [sum(matrix[i][j] for i in range(classes)) for j in range(classes)]
    n = sum(gold)
    if n == 0:
        raise ValueError("Completed classification cells require decisions")
    per_class = []
    for index, name in enumerate(class_names):
        true = matrix[index][index]
        denominator = gold[index] + pred[index]
        per_class.append({"class": name, "gold_count": gold[index], "pred_count": pred[index],
                          "recall": true / gold[index] if gold[index] else None,
                          "precision": true / pred[index] if pred[index] else None,
                          "f1": 2 * true / denominator if denominator else 0.0})
    recalls = [row["recall"] for row in per_class if row["recall"] is not None]
    return {"n_decisions": n, "accuracy": sum(matrix[i][i] for i in range(classes)) / n,
            "balanced_accuracy": sum(recalls) / len(recalls),
            "macro_f1": sum(row["f1"] for row in per_class) / classes,
            "n_gold_present_classes": len(recalls), "gold_counts": dict(zip(class_names, gold)),
            "pred_counts": dict(zip(class_names, pred)), "confusion_matrix": matrix,
            "confusion_matrix_order": list(class_names), "per_class": per_class,
            "majority_class_accuracy": max(gold) / n}


def model_name(identity):
    model = identity.get("model")
    return model.get("name") if isinstance(model, dict) else model


def find_runs(root, models):
    found = {}
    for path in sorted(root.rglob("report.json")):
        report = read_json(path)
        name = model_name(report.get("identity", {}))
        if name not in models:
            continue
        if name in found:
            raise ValueError("Ambiguous multiple reports for one fixed model")
        found[name] = (path.parent, report)
    return found


def evaluation(report, method, source):
    item = report if method == "native" else report.get("methods", {}).get(method, {})
    evaluations = item.get("native_evaluations", {}) if method == "native" else item.get("evaluations", {})
    return evaluations.get(source, item.get(f"native_{source}" if method == "native" else source))


def audit_quantization(report):
    identity = report["identity"]
    if identity.get("bits") != 4 or identity.get("activation_bits") != 4:
        raise ValueError("Classification comparison requires frozen W4A4 precision")
    selected = unique_strings(identity.get("selected_linear_names"), "frozen selected layers")
    reference = None
    for method in METHODS:
        item = report["methods"][method]
        quantization = item.get("quantization", {})
        layers = quantization.get("layers", {})
        if set(layers) != set(selected) or quantization.get("activation_bits") != 4:
            raise ValueError("Method quantization layer/activation scope differs")
        signature = {name: (layer.get("bits"), layer.get("group_size"), layer.get("shape")) for name, layer in layers.items()}
        if any(bits != 4 or group != identity.get("group_size") for bits, group, _ in signature.values()):
            raise ValueError("Method weight precision/group size differs")
        if reference is not None and signature != reference:
            raise ValueError("Method weight layer shape/scope differs")
        reference = signature


def pending_reasons(folder, report, sources):
    reasons = []
    for filename in ("identity.json", "expanded-evaluation-complete.json"):
        if not (folder / filename).is_file():
            reasons.append(f"missing {filename}")
    marker_path = folder / "expanded-evaluation-complete.json"
    if marker_path.is_file():
        marker = read_json(marker_path)
        if marker.get("status") != "complete":
            reasons.append("evaluation marker is incomplete")
        if any(marker.get("methods", {}).get(method) != "complete" for method in METHODS):
            reasons.append("evaluation marker lacks all twelve completed methods")
    if any(report.get("methods", {}).get(method, {}).get("status") != "complete" for method in METHODS):
        reasons.append("report lacks all twelve completed methods")
    for frozen in sources:
        for method in ALL_METHODS:
            if not (folder / f"{method}-{frozen.dataset_id}.jsonl").is_file():
                reasons.append(f"missing {method}/{frozen.dataset_id} prediction file")
            if evaluation(report, method, frozen.dataset_id) is None:
                reasons.append(f"missing {method}/{frozen.dataset_id} evaluation")
    return reasons


def summarize(runs_root, manifest_path, *, models):
    models = unique_strings(models, "fixed model IDs")
    manifest, sources, registry = load_manifest(manifest_path)
    manifest_hash = digest(manifest_path)
    found = find_runs(runs_root, models)
    included, pending, cells, inputs = [], [], [], []
    ranking_ids = [e["dataset_id"] for e in manifest["datasets"] if e.get("ranking_eligible") is True]
    for model in models:
        if model not in found:
            pending.append({"model": model, "status": "pending", "reasons": ["missing report.json"]})
            continue
        folder, report = found[model]
        reasons = pending_reasons(folder, report, sources)
        if reasons:
            pending.append({"model": model, "status": "pending", "reasons": reasons})
            continue
        identity = report["identity"]
        if read_json(folder / "identity.json") != identity:
            raise ValueError("Standalone run identity differs from report")
        if identity.get("benchmark_manifest_sha256") != manifest_hash:
            raise ValueError("Run uses a different frozen benchmark manifest")
        if identity.get("final_test_read") is not True or identity.get("expanded_ranking_dataset_ids") != ranking_ids:
            raise ValueError("Run does not declare the fixed final evaluation source scope")
        marker = read_json(folder / "expanded-evaluation-complete.json")
        if marker.get("benchmark_manifest_sha256") != manifest_hash or marker.get("ranking_dataset_ids") != ranking_ids:
            raise ValueError("Completion marker differs from frozen manifest/source scope")
        audit_quantization(report)
        derivation_path = folder / "derivation-manifest.json"
        derivation = read_json(derivation_path) if derivation_path.is_file() else None
        if derivation is not None and (derivation.get("status") != "complete" or derivation.get("model") != model
                                       or derivation.get("benchmark_manifest_sha256") != manifest_hash):
            raise ValueError("Derived evidence identity/manifest differs")
        for filename in ("report.json", "identity.json", "expanded-evaluation-complete.json"):
            value = digest(folder / filename)
            if derivation is not None and derivation.get("files", {}).get(filename, {}).get("sha256") != value:
                raise ValueError("Derived run metadata SHA-256 mismatch")
            inputs.append({"model": model, "role": filename, "sha256": value})
        if derivation is not None:
            inputs.append({"model": model, "role": "derivation-manifest.json", "sha256": digest(derivation_path)})
        model_cells = []
        for frozen in sources:
            source = frozen.dataset_id
            if identity.get("source_sha256", {}).get(source) != frozen.entry["sha256"]:
                raise ValueError("Run source SHA-256 differs from manifest")
            accepted = unique_strings(identity.get("accepted_evaluation_ids", {}).get(source), "frozen admitted request IDs")
            if not set(accepted).issubset(frozen.request_ids):
                raise ValueError("Run admitted request IDs are outside the frozen source")
            expected = {key: row for key, row in frozen.rows.items() if row["record_id"] in set(accepted)}
            for method in ALL_METHODS:
                filename = f"{method}-{source}.jsonl"
                path = folder / filename
                value = digest(path)
                if derivation is not None and derivation.get("files", {}).get(filename, {}).get("sha256") != value:
                    raise ValueError("Derived prediction SHA-256 mismatch")
                rows = predictions(path, expected, frozen.schema)
                metrics = count_predictions(rows, expected, frozen.schema)
                recorded = evaluation(report, method, source)
                raw = recorded.get("raw", recorded)
                accuracy = raw.get("accuracy")
                if type(raw.get("n")) is not int or raw["n"] != len(rows) or type(accuracy) not in (int, float) \
                        or not math.isfinite(accuracy) or not math.isclose(accuracy, metrics["accuracy"], rel_tol=0, abs_tol=1e-10):
                    raise ValueError("Saved evaluation count/accuracy differs from raw predictions")
                model_cells.append({"model": model, "dataset_id": source, "method": method, "status": "complete",
                                    "n_requests": len(accepted), "n_clusters": len({r["cluster_id"] for r in expected.values()}),
                                    "source_n_requests": len(frozen.request_ids), "source_n_decisions": len(frozen.rows),
                                    "admission_equals_frozen_source": len(accepted) == len(frozen.request_ids), **metrics})
                inputs.append({"model": model, "dataset_id": source, "method": method, "sha256": value})
        included.append(model)
        cells.extend(model_cells)
    return {"schema": "s1q.classification-diagnostics.v1", "status": "complete" if not pending else "partial",
            "benchmark_id": manifest.get("benchmark_id"), "stage_id": manifest.get("stage_id"), "precision": "W4A4",
            "fixed_models": list(models), "included_models": included, "pending_models": pending,
            "method_ids": list(METHODS), "reference_method": "native", "n_included_models": len(included),
            "n_pending_models": len(pending), "n_sources": len(sources), "n_method_cells": len(cells),
            "source_registry": registry, "source_shapes": [source.summary for source in sources], "cells": cells,
            "frozen_eligibility": {key: manifest.get("eligibility", {}).get(key) for key in ("max_choices", "max_chars", "clean_only", "truncation")},
            "metric_definitions": {
                "accuracy": "Correct argmax decisions / admitted decisions; ties select the first saved logit position.",
                "candidate_order_evidence": "Candidate-to-class mapping follows the frozen source and evaluator contract: choice criteria insertion order; Boolean false/true. Source-derived gold indices, prediction keys and logit widths are checked. Optional saved options are checked when present. Rows without saved option keys do not independently capture observed semantic ordering; logits alone cannot establish it.",
                "balanced_accuracy": "Mean recall over gold-present semantic classes; gold-absent class recall is null and n_gold_present_classes is reported.",
                "macro_f1": "Arithmetic mean of per-class 2TP/(gold_count+pred_count) over the fixed source semantic classes; zero denominator contributes 0.",
                "precision": "TP/pred_count; null if the class is never predicted.",
                "confusion_matrix": "Rows are gold classes, columns are predicted classes, in confusion_matrix_order.",
                "main_accuracy_ranking": "Unchanged; " + ACCURACY_AGGREGATION,
                "diagnostic_aggregation": "Per model/source/method only; no overall class-F1 ranking or changed source/model weights.",
                "missing_models": "Pending models produce no numeric diagnostic cells and never contribute zeros.",
            },
            "provenance": {"script_sha256": digest(Path(__file__)), "benchmark_manifest_sha256": manifest_hash,
                           "frozen_source_sha256": {s.dataset_id: s.entry["sha256"] for s in sources},
                           "input_file_hashes": inputs, "input_paths_omitted": True, "raw_predictions_omitted": True},
            "interpretation": [SUBSET_CAVEAT,
                               "These are supplemental source-shape diagnostics, not post-hoc changes to the frozen main accuracy leaderboard or a redefinition of method winners.",
                               "Only fixed semantic class schemas are admitted. Stable multiple-choice answer positions do not define semantic classes.",
                               "All twelve W4A4 methods and Native must be complete for an included model, with exact source-bound gold/keys/candidate dimensions and one frozen admission set per source.",
                               "Candidate-to-class mapping follows the frozen source and evaluator ordering contract; optional saved options are checked when present. Rows without saved option keys do not independently capture observed semantic ordering. WANLI source permutations are mapped to semantic classes under that contract."]}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], separators=(",", ":"), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fields})


def write_outputs(output, result):
    output.mkdir(parents=True, exist_ok=False)
    (output / "classification_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["model", "dataset_id", "method", "status", "n_requests", "n_decisions", "n_clusters", "accuracy",
              "balanced_accuracy", "macro_f1", "majority_class_accuracy", "n_gold_present_classes", "gold_counts", "pred_counts",
              "confusion_matrix_order", "confusion_matrix", "admission_equals_frozen_source"]
    write_csv(output / "classification-diagnostics.csv", result["cells"], fields)
    class_rows = [{"model": cell["model"], "dataset_id": cell["dataset_id"], "method": cell["method"], **row}
                  for cell in result["cells"] for row in cell["per_class"]]
    write_csv(output / "classification-per-class.csv", class_rows,
              ["model", "dataset_id", "method", "class", "gold_count", "pred_count", "recall", "precision", "f1"])
    write_csv(output / "classification-source-shapes.csv", result["source_shapes"],
              ["dataset_id", "name", "class_names", "n_requests", "n_decisions", "n_clusters", "gold_counts", "gold_class_proportions",
               "majority_class_accuracy", "candidate_order_policy", "source_candidate_order_count", "frozen_subset_screening", "source_sha256", "subset_caveat"])
    lines = ["# Supplemental classification diagnostics", "", SUBSET_CAVEAT, "",
             "The frozen main accuracy ranking and its equal source/model weights remain unchanged. No combined F1 ranking is constructed.", "",
             f"W4A4: {result['n_included_models']} complete matched models; {result['n_pending_models']} pending models; Native + 12 methods.", "",
             "Confusion matrices use gold rows and predicted columns. Candidate-to-class mapping follows the frozen source and evaluator ordering contract (choice criteria insertion order; Boolean false/true). Prediction keys, source-derived gold indices and logit widths are checked; optional saved options are checked when present. Rows without saved option keys do not independently capture observed semantic ordering, so logits alone cannot establish it.", "",
             "| Source | Semantic classes | Frozen requests / decisions | Gold counts | Majority-class accuracy % |",
             "| --- | --- | ---: | --- | ---: |"]
    for source in result["source_shapes"]:
        counts = ", ".join(f"{name}: {n}" for name, n in source["gold_counts"].items())
        lines.append(f"| {source['dataset_id']} | {', '.join(source['class_names'])} | {source['n_requests']} / {source['n_decisions']} | {counts} | {source['majority_class_accuracy'] * 100:.3f} |")
    lines += ["", "| Model | Source | Method | Accuracy % | Balanced accuracy % | Macro-F1 % | Predicted counts | Per-class recall | Confusion matrix |",
              "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for cell in result["cells"]:
        recalls = ", ".join(f"{row['class']}: {row['recall']:.4f}" if row["recall"] is not None else f"{row['class']}: undefined (gold absent)" for row in cell["per_class"])
        counts = ", ".join(f"{name}: {n}" for name, n in cell["pred_counts"].items())
        lines.append(f"| {cell['model']} | {cell['dataset_id']} | {cell['method']} | {cell['accuracy'] * 100:.3f} | {cell['balanced_accuracy'] * 100:.3f} | {cell['macro_f1'] * 100:.3f} | {counts} | {recalls} | {json.dumps(cell['confusion_matrix'])} |")
    if result["pending_models"]:
        lines += ["", "Pending models have no numeric cells:", ""]
        lines.extend(f"- {row['model']}: {'; '.join(row['reasons'])}." for row in result["pending_models"])
    lines += ["", "Macro-F1 averages the fixed classes with zero F1 for an empty gold/prediction class. Balanced accuracy averages gold-present class recalls; undefined recalls remain explicit. All metrics in CSV/JSON retain unrounded inputs."]
    (output / "classification_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="Fixed model names; absent/incomplete runs are reported as pending.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; prior diagnostics are never overwritten")
    if output.is_relative_to(args.runs_root.resolve()) or output.is_relative_to(args.manifest.parent.resolve()):
        raise ValueError("Output must be outside immutable run and source directories")
    result = summarize(args.runs_root, args.manifest, models=args.models)
    write_outputs(output, result)
    print(json.dumps({key: result[key] for key in ("status", "included_models", "n_pending_models", "n_sources", "n_method_cells")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
