"""Source-bound class diagnostics and negative controls; no model runtime."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_classification_diagnostics import (
    ALL_METHODS, METHODS, SCHEMAS, candidate_keys, count_predictions, digest,
    load_manifest, main, matrix_metrics, predictions, summarize, write_outputs,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def source_row(source, index, label, keys=None):
    schema = SCHEMAS[source]
    keys = keys or schema.candidate_keys
    return {"state": "PRIVATE QUESTION TEXT NEVER PUBLISHED", "questions": {
        "q": {"type": schema.question_type, "instructions": "PRIVATE INSTRUCTIONS",
              "criteria": dict.fromkeys(keys, "class criterion"), "label": label,
              "src": schema.raw_sources[0], "label_kind": "native_annotation" if source == "wildjailbreak" else "dataset_gold"}},
        "_meta": {"id": f"private-{source}-{index}", "group_id": f"private-cluster-{source}-{index}"}}


@pytest.fixture
def complete_runs(tmp_path):
    manifest_path = tmp_path / "sources" / "manifest.json"
    entries = []
    for source in ("wildjailbreak", "tweet-offensive", "mnli", "wanli"):
        if source == "wildjailbreak":
            rows = [source_row(source, 0, "harmful"), source_row(source, 1, "benign")]
        elif source == "tweet-offensive":
            # Boolean dictionary insertion order does not dictate logits order.
            rows = [source_row(source, 0, False, ("true", "false")), source_row(source, 1, True, ("true", "false"))]
        elif source == "wanli":
            rows = [source_row(source, 0, "supported", ("contradicted", "supported", "insufficient")),
                    source_row(source, 1, "insufficient", ("insufficient", "contradicted", "supported")),
                    source_row(source, 2, "contradicted", ("supported", "insufficient", "contradicted"))]
        else:
            rows = [source_row(source, i, name) for i, name in enumerate(SCHEMAS[source].candidate_keys)]
        source_path = manifest_path.parent / "datasets" / f"{source}.jsonl"
        write_rows(source_path, rows)
        entries.append({"dataset_id": source, "name": source, "path": f"datasets/{source}.jsonl", "sha256": digest(source_path),
                        "n_requests": len(rows), "n_decisions": len(rows), "n_groups": len(rows), "ranking_eligible": True,
                        "label_kind": "native_annotation" if source == "wildjailbreak" else "upstream_dataset_gold",
                        "screening": {"raw_candidates": len(rows) + 1, "unique_eligible_remaining": len(rows), "selection_limit": 128,
                                      "excluded": {"request_too_long_chars": 1}},
                        "selected_ids": [row["_meta"]["id"] for row in rows]})
    # Even stable a/b/c/d insertion order never makes MMLU answer positions
    # semantic classes. Non-allowlisted source bytes need not be read.
    entries.append({"dataset_id": "mmlu", "ranking_eligible": True})
    write(manifest_path, {"schema": "s1q.expanded-benchmarks.v1", "benchmark_id": "toy", "stage_id": "toy", "datasets": entries})
    _, sources, _ = load_manifest(manifest_path)
    runs = tmp_path / "runs"
    for model in ("m1", "m2"):
        folder = runs / model
        native_evaluations, method_evaluations, accepted = {}, {}, {}
        for frozen in sources:
            rows = []
            for target in frozen.rows.values():
                row = {key: value for key, value in target.items() if key != "candidate_keys"}
                row["logits"] = [float(i == row["label"]) for i in range(len(target["candidate_keys"]))]
                rows.append(row)
            for method in ALL_METHODS:
                write_rows(folder / f"{method}-{frozen.dataset_id}.jsonl", rows)
            evaluation = {"raw": {"n": len(rows), "accuracy": 1.0}}
            native_evaluations[frozen.dataset_id] = evaluation
            method_evaluations[frozen.dataset_id] = evaluation
            accepted[frozen.dataset_id] = list(frozen.request_ids)
        identity = {"model": {"name": model}, "bits": 4, "activation_bits": 4, "group_size": 128,
                    "selected_linear_names": ["linear"], "final_test_read": True,
                    "expanded_ranking_dataset_ids": [entry["dataset_id"] for entry in entries],
                    "benchmark_manifest_sha256": digest(manifest_path), "accepted_evaluation_ids": accepted,
                    "source_sha256": {entry["dataset_id"]: entry["sha256"] for entry in entries if "sha256" in entry}}
        report = {"identity": identity, "native_evaluations": native_evaluations,
                  "methods": {method: {"status": "complete", "evaluations": deepcopy(method_evaluations),
                              "quantization": {"activation_bits": 4, "layers": {"linear": {"bits": 4, "group_size": 128, "shape": [2, 3]}}}}
                              for method in METHODS}}
        write(folder / "report.json", report)
        write(folder / "identity.json", identity)
        write(folder / "expanded-evaluation-complete.json", {"status": "complete", "benchmark_manifest_sha256": digest(manifest_path),
              "ranking_dataset_ids": identity["expanded_ranking_dataset_ids"], "methods": dict.fromkeys(METHODS, "complete")})
    return tmp_path, runs, manifest_path


def test_imbalanced_119_9_majority_accuracy_cannot_hide_zero_minority_recall():
    expected, predicted = {}, {}
    for index in range(128):
        key = f"toy-{index}::q"
        label = int(index >= 119)
        expected[key] = {"candidate_keys": ("harmful", "benign")}
        predicted[key] = {"label": label, "logits": [1.0, 0.0]}
    metrics = count_predictions(predicted, expected, SCHEMAS["wildjailbreak"])
    assert metrics["confusion_matrix"] == [[119, 0], [9, 0]]
    assert metrics["accuracy"] == 119 / 128  # 92.96875%, despite never predicting benign.
    assert metrics["balanced_accuracy"] == .5
    assert metrics["macro_f1"] == (238 / 247) / 2
    assert metrics["pred_counts"] == {"harmful": 128, "benign": 0}
    assert [row["recall"] for row in metrics["per_class"]] == [1.0, 0.0]
    assert metrics["per_class"][1]["precision"] is None


def test_wanli_permutations_map_to_semantic_classes_and_preserve_argmax_tie_order(complete_runs):
    _, runs, manifest = complete_runs
    _, sources, _ = load_manifest(manifest)
    frozen = next(source for source in sources if source.dataset_id == "wanli")
    rows = predictions(runs / "m1" / "native-wanli.jsonl", frozen.rows, frozen.schema)
    metrics = count_predictions(rows, frozen.rows, frozen.schema)
    assert metrics["confusion_matrix_order"] == ["supported", "insufficient", "contradicted"]
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["balanced_accuracy"] == metrics["macro_f1"] == 1
    # All scores tied => choose source-question first candidate, then map.
    for row in rows.values():
        row["logits"] = [0, 0, 0]
    metrics = count_predictions(rows, frozen.rows, frozen.schema)
    assert metrics["confusion_matrix"] == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    assert metrics["accuracy"] == 1 / 3


def test_boolean_false_true_candidate_order_is_not_criteria_insertion_order(complete_runs):
    _, runs, manifest = complete_runs
    _, sources, _ = load_manifest(manifest)
    frozen = next(source for source in sources if source.dataset_id == "tweet-offensive")
    assert all(row["candidate_keys"] == ("false", "true") for row in frozen.rows.values())
    rows = predictions(runs / "m1" / "native-tweet-offensive.jsonl", frozen.rows, frozen.schema)
    assert count_predictions(rows, frozen.rows, frozen.schema)["confusion_matrix"] == [[1, 0], [0, 1]]


def test_native_null_and_nested_class_descriptors_preserve_semantic_keys():
    question = source_row("mnli", 0, "entailment")["questions"]["q"]
    question["criteria"] = {"entailment": {"description": "entails"}, "neutral": None, "contradiction": "contradicts"}
    assert candidate_keys(question, SCHEMAS["mnli"]) == ("entailment", "neutral", "contradiction")


@pytest.mark.parametrize("change,match", [
    ("missing_key", "Exact matched prediction keys"),
    ("extra_key", "outside the frozen admitted source"),
    ("gold", "label differs"),
    ("bool_gold", "label differs"),
    ("dimension", "candidate dimension"),
    ("options", "option order"),
    ("cluster", "cluster_id differs"),
    ("source", "source differs"),
    ("duplicate", "duplicate"),
    ("missing_field", "Missing required"),
    ("nonfinite", "finite numbers"),
])
def test_negative_controls_reject_source_or_pairing_mismatch(complete_runs, change, match):
    _, runs, manifest = complete_runs
    _, sources, _ = load_manifest(manifest)
    frozen = next(source for source in sources if source.dataset_id == "wildjailbreak")
    path = runs / "m1" / "rtn-wildjailbreak.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "missing_key":
        rows.pop()
    elif change == "extra_key":
        rows[0]["key"] = "unknown::q"
    elif change == "gold":
        rows[0]["label"] = 1
    elif change == "bool_gold":
        rows[0]["label"] = False
    elif change == "dimension":
        rows[0]["logits"].append(0)
    elif change == "options":
        rows[0]["options"] = ["benign", "harmful"]
    elif change == "cluster":
        rows[0]["cluster_id"] = "other"
    elif change == "source":
        rows[0]["source"] = "other"
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "missing_field":
        del rows[0]["key"]
    elif change == "nonfinite":
        rows[0]["logits"][0] = "nan"
    write_rows(path, rows)
    with pytest.raises(ValueError, match=match):
        summarize(runs, manifest, models=["m1", "m2"])


def test_gold_absent_recall_stays_undefined_with_fixed_class_f1_denominator():
    metrics = matrix_metrics([[3, 0, 0], [0, 0, 0], [0, 0, 0]], ("a", "b", "c"))
    assert metrics["balanced_accuracy"] == 1
    assert metrics["n_gold_present_classes"] == 1
    assert metrics["macro_f1"] == 1 / 3
    assert [row["recall"] for row in metrics["per_class"]] == [1, None, None]


@pytest.mark.parametrize("kind", ["absent_report", "missing_native", "incomplete_method", "missing_evaluation"])
def test_missing_or_incomplete_model_is_pending_and_never_a_zero_cell(complete_runs, kind):
    _, runs, manifest = complete_runs
    folder = runs / "m2"
    if kind == "absent_report":
        (folder / "report.json").unlink()
    elif kind == "missing_native":
        (folder / "native-wildjailbreak.jsonl").unlink()
    else:
        report = json.loads((folder / "report.json").read_text())
        if kind == "incomplete_method":
            report["methods"]["rtn"]["status"] = "running"
        else:
            del report["methods"]["rtn"]["evaluations"]["wildjailbreak"]
        write(folder / "report.json", report)
    result = summarize(runs, manifest, models=["m1", "m2", "absent-model"])
    assert result["status"] == "partial"
    assert result["included_models"] == ["m1"]
    assert [row["model"] for row in result["pending_models"]] == ["m2", "absent-model"]
    assert len(result["cells"]) == 4 * 13
    assert all(row["model"] == "m1" and row["accuracy"] == 1 for row in result["cells"])


def test_fixed_answer_position_source_is_registered_excluded(complete_runs):
    _, _, manifest = complete_runs
    _, sources, registry = load_manifest(manifest)
    assert {source.dataset_id for source in sources} == {"wildjailbreak", "tweet-offensive", "mnli", "wanli"}
    excluded = next(row for row in registry if row["dataset_id"] == "mmlu")
    assert excluded["status"] == "excluded" and "positions" in excluded["reason"]


@pytest.mark.parametrize("change", ["wrong_keys", "wrong_order", "wrong_type"])
def test_source_semantic_schema_is_strict_even_if_gold_is_valid(change):
    question = source_row("wildjailbreak", 0, "harmful")["questions"]["q"]
    if change == "wrong_keys":
        question["criteria"] = {"a": "harmful", "b": "benign"}
    elif change == "wrong_order":
        question["criteria"] = {"benign": "criterion", "harmful": "criterion"}
    else:
        question["type"] = "noul"
    with pytest.raises(ValueError, match="class schema"):
        candidate_keys(question, SCHEMAS["wildjailbreak"])


def test_source_hash_mismatch_rejected(complete_runs):
    _, runs, manifest = complete_runs
    source_path = manifest.parent / "datasets" / "wildjailbreak.jsonl"
    source_path.write_text(source_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        summarize(runs, manifest, models=["m1"])


def test_run_manifest_hash_mismatch_rejected(complete_runs):
    _, runs, manifest = complete_runs
    document = json.loads(manifest.read_text())
    document["seed"] = 123  # Source bytes stay fixed, but the freeze identity changed.
    write(manifest, document)
    with pytest.raises(ValueError, match="different frozen benchmark manifest"):
        summarize(runs, manifest, models=["m1"])


@pytest.mark.parametrize("field", ["bits", "activation_bits"])
def test_other_precision_cannot_enter_w4a4_diagnostics(complete_runs, field):
    _, runs, manifest = complete_runs
    folder = runs / "m1"
    report = json.loads((folder / "report.json").read_text())
    report["identity"][field] = 8
    write(folder / "report.json", report)
    write(folder / "identity.json", report["identity"])
    with pytest.raises(ValueError, match="W4A4 precision"):
        summarize(runs, manifest, models=["m1"])


def test_different_weight_layer_scope_cannot_enter_comparison(complete_runs):
    _, runs, manifest = complete_runs
    folder = runs / "m1"
    report = json.loads((folder / "report.json").read_text())
    report["methods"]["rtn"]["quantization"]["layers"]["linear"]["shape"] = [3, 2]
    write(folder / "report.json", report)
    with pytest.raises(ValueError, match="shape/scope differs"):
        summarize(runs, manifest, models=["m1"])


def test_saved_accuracy_mismatch_rejected(complete_runs):
    _, runs, manifest = complete_runs
    path = runs / "m1" / "report.json"
    report = json.loads(path.read_text())
    report["native_evaluations"]["wildjailbreak"]["raw"]["accuracy"] = .5
    write(path, report)
    with pytest.raises(ValueError, match="count/accuracy differs"):
        summarize(runs, manifest, models=["m1"])


def test_exact_frozen_admission_subset_is_allowed_without_silent_class_rebalancing(complete_runs):
    _, runs, manifest = complete_runs
    folder = runs / "m2"
    identity = json.loads((folder / "identity.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    accepted = identity["accepted_evaluation_ids"]["wildjailbreak"][:1]
    identity["accepted_evaluation_ids"]["wildjailbreak"] = accepted
    for method in ALL_METHODS:
        path = folder / f"{method}-wildjailbreak.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        write_rows(path, rows[:1])
        item = report["native_evaluations"] if method == "native" else report["methods"][method]["evaluations"]
        item["wildjailbreak"]["raw"]["n"] = 1
    report["identity"] = identity
    write(folder / "identity.json", identity)
    write(folder / "report.json", report)
    result = summarize(runs, manifest, models=["m1", "m2"])
    cell = next(row for row in result["cells"] if row["model"] == "m2" and row["dataset_id"] == "wildjailbreak")
    assert cell["n_requests"] == 1 and cell["source_n_requests"] == 2
    assert cell["gold_counts"] == {"harmful": 1, "benign": 0}
    assert cell["admission_equals_frozen_source"] is False
    assert cell["per_class"][1]["recall"] is None


def test_output_is_aggregate_only_and_cli_refuses_overwriting(complete_runs):
    tmp_path, runs, manifest = complete_runs
    output = tmp_path / "diagnostics"
    assert main(["--runs-root", str(runs), "--manifest", str(manifest), "--models", "m1", "m2", "absent",
                 "--output-dir", str(output)]) == 0
    result = json.loads((output / "classification_diagnostics.json").read_text())
    assert result["n_method_cells"] == 2 * 4 * 13
    assert result["source_shapes"][0]["frozen_subset_screening"]["excluded"] == {"request_too_long_chars": 1}
    assert "do not independently capture observed semantic ordering" in result["metric_definitions"]["candidate_order_evidence"]
    assert "optional saved options are checked when present" in (output / "classification_diagnostics.md").read_text()
    for path in output.iterdir():
        encoded = path.read_text(encoding="utf-8")
        for private in (str(tmp_path), "private-cluster", "private-wildjailbreak", "PRIVATE QUESTION", "PRIVATE INSTRUCTIONS",
                        '"logits"', '"record_id"', '"qid"', '"key"', '"label"', '"selected_ids"'):
            assert private not in encoded
    with (output / "classification-per-class.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 13 * (2 + 2 + 3 + 3)
    with pytest.raises(FileExistsError, match="fresh output"):
        main(["--runs-root", str(runs), "--manifest", str(manifest), "--models", "m1", "--output-dir", str(output)])


def test_cli_rejects_output_under_immutable_runs(complete_runs):
    _, runs, manifest = complete_runs
    with pytest.raises(ValueError, match="outside immutable"):
        main(["--runs-root", str(runs), "--manifest", str(manifest), "--models", "m1", "--output-dir", str(runs / "bad")])


def test_tool_has_only_standard_library_dependencies():
    import ast
    from scripts import summarize_classification_diagnostics as module
    tree = ast.parse(Path(module.__file__).read_text())
    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imports |= {name.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
    assert not imports & {"torch", "s1q", "numpy", "sklearn", "pandas"}
