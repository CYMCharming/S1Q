"""Estimator and pairing controls for conditional macro-accuracy intervals."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest

from scripts.summarize_accuracy_uncertainty import (
    AGGREGATION, SourceCounts, bootstrap, check_paired, check_public_points,
    count_source, select_scope, summarize,
)


def row(record, question, cluster, correct):
    return {"key": f"{record}::{question}", "record_id": record, "qid": question,
            "cluster_id": cluster, "type": "choice", "source": "toy",
            "label": 1, "label_kind": "dataset_gold", "logits": [0, 1] if correct else [1, 0]}


def keyed(rows):
    return {r["key"]: r for r in rows}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_multi_question_groups_keep_decisions_together_and_ratio_estimator():
    # One correct singleton and three incorrect decisions in one cluster:
    # point is 1/4, not the incorrect equal-cluster estimate of 1/2.
    rows = keyed([row("one", "q", "singleton", True),
                  *[row("three", str(i), "multi-question", False) for i in range(3)]])
    source = count_source("toy", ("m",), ("method",), {"m": {"native": rows, "method": rows}})
    result = bootstrap([source], samples=1000, seed=4)
    assert result["points"].tolist() == [.25, .25]
    assert sorted(source.counts[0].tolist()) == [1, 3]
    # Whole-group draws can produce only 0, 1/4, 1; splitting decisions would
    # additionally produce fractions such as 1/2 or 3/4.
    assert set(result["replicates"][:, 0]) == {0.0, .25, 1.0}


def test_shared_source_draws_preserve_cross_model_correlation_and_identical_method_delta():
    rows = keyed([row(str(i), "q", str(i), i % 2 == 0) for i in range(4)])
    data = {model: {"native": deepcopy(rows), "a": deepcopy(rows), "b": deepcopy(rows)} for model in ("m1", "m2")}
    source = count_source("toy", ("m1", "m2"), ("a", "b"), data)
    result = bootstrap([source], samples=500, seed=3)
    assert np.array_equal(result["model_replicates"][:, 0], result["model_replicates"][:, 1])
    assert np.array_equal(result["replicates"][:, 1] - result["replicates"][:, 2], np.zeros(500))
    assert np.ptp(result["replicates"][:, 1]) > 0  # Truly resampled, not frozen points.


def test_missing_keys_rejected_even_when_counts_would_look_plausible():
    rows = keyed([row("one", "q", "one", True), row("two", "q", "two", False)])
    missing = deepcopy(rows)
    del missing["two::q"]
    with pytest.raises(ValueError, match="Exact paired prediction keys"):
        count_source("toy", ("m",), ("method",), {"m": {"native": rows, "method": missing}})


@pytest.mark.parametrize("change", ["gold", "options", "cluster"])
def test_gold_option_count_and_cluster_mismatch_rejected(change):
    rows = keyed([row("one", "q", "one", True)])
    changed = deepcopy(rows)
    if change == "gold":
        changed["one::q"]["label"] = 0
    elif change == "options":
        changed["one::q"]["logits"].append(2)
    else:
        changed["one::q"]["cluster_id"] = "another"
    with pytest.raises(ValueError, match="Paired"):
        check_paired(rows, changed)


def test_unequal_dataset_sizes_do_not_change_source_weights():
    many = SourceCounts("large", ("g",), np.array([[100]]), np.array([[[100], [100]]]))
    tiny = SourceCounts("tiny", ("g",), np.array([[1]]), np.array([[[0], [0]]]))
    result = bootstrap([many, tiny], samples=25, seed=0)
    assert result["points"].tolist() == [.5, .5]
    assert np.all(result["replicates"] == .5)
    assert result["points"][1] != 100 / 101  # Not pooled decision accuracy.


def test_unequal_model_counts_do_not_change_model_weights():
    source = SourceCounts("toy", ("g",), np.array([[100], [1]]),
                          np.array([[[100], [100]], [[0], [0]]]))
    result = bootstrap([source], samples=25, seed=0)
    assert result["model_points"].tolist() == [[1, 1], [0, 0]]
    assert result["points"].tolist() == [.5, .5]
    assert np.all(result["replicates"] == .5)


def test_unequal_admission_uses_model_specific_denominators_and_fails_zero_draws():
    source = SourceCounts("toy", ("a", "b"), np.array([[1, 3], [1, 0]]),
                          np.array([[[1, 0], [1, 0]], [[1, 0], [1, 0]]]))
    # A union draw of only b cannot evaluate the second model. Reject rather
    # than treating this as zero accuracy or quietly drawing replacement rows.
    with pytest.raises(ValueError, match=r"\d+ zero admitted-decision denominators; no draws were replaced"):
        bootstrap([source], samples=100, seed=1)


@pytest.fixture
def complete_runs(tmp_path):
    methods = ["rtn", "s1q-local", "s1q2-beta05", "smoothquant-adapted", "awq-adapted", "gptq-blockdiag-adapted",
               "spinquant-nohad-adapted", "spinquant-had-adapted", "s1q-joint", "s1q-ac", "s1q-margin", "s1q"]
    models, sources = ["m1", "m2"], ["large", "tiny"]
    scope = {"suite_id": "toy-conditional", "comparison_group": "all12", "precision": "W4A4",
             "included_models": models, "dataset_ids": sources, "method_ids": methods,
             "n_models": 2, "n_datasets": 2, "n_cells": 4, "aggregation": AGGREGATION,
             "sample_counts": [{"model": m, "dataset_id": s, "n": 4 if s == "large" else 1,
                                "n_clusters": 2 if s == "large" else 1} for m in models for s in sources]}
    scope_path = tmp_path / "ranking_scopes.json"
    write(scope_path, {"schema": "s1q.public-ranking-scopes.v1", "computed_scopes": [scope]})
    for model in models:
        folder = tmp_path / "runs" / model
        identity = {"model": {"name": model}, "bits": 4, "activation_bits": 4, "group_size": 128,
                    "selected_linear_names": ["linear"], "final_test_read": True,
                    "expanded_ranking_dataset_ids": sources, "benchmark_manifest_sha256": "b" * 64,
                    "source_sha256": {s: "a" * 64 for s in sources},
                    "accepted_evaluation_ids": {"large": ["one", "three"], "tiny": ["tiny"]}}
        native_evaluations, method_evaluations = {}, {}
        for source in sources:
            rows = [row("one", "q", "one", True), *[row("three", str(i), "three", False) for i in range(3)]] if source == "large" else [row("tiny", "q", "tiny", True)]
            evaluation = {"raw": {"n": len(rows), "accuracy": .25 if source == "large" else 1.0}}
            native_evaluations[source] = evaluation
            method_evaluations[source] = evaluation
            for method in ("native", *methods):
                path = folder / f"{method}-{source}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        report = {"identity": identity, "native_evaluations": native_evaluations,
                  "methods": {method: {"status": "complete", "quantization": {"activation_bits": 4,
                              "layers": {"linear": {"bits": 4, "group_size": 128, "shape": [2, 3]}}},
                              "evaluations": method_evaluations} for method in methods}}
        write(folder / "identity.json", identity)
        write(folder / "report.json", report)
        write(folder / "expanded-evaluation-complete.json", {"status": "complete", "benchmark_manifest_sha256": "b" * 64, "ranking_dataset_ids": sources,
                                                               "methods": {m: "complete" for m in methods}})
    return tmp_path, scope_path, scope


def test_end_to_end_scope_and_point_csv_check_are_exact_and_output_contains_no_predictions(complete_runs):
    tmp_path, scope_path, scope = complete_runs
    ranking = tmp_path / "rankings.csv"
    with ranking.open("w", encoding="utf-8", newline="") as handle:
        fields = ["suite_id", "comparison_group", "family", "precision", "method", "n_models", "n_datasets", "n_cells", "accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in scope["method_ids"]:
            writer.writerow({**{k: scope[k] for k in ("suite_id", "comparison_group", "precision", "n_models", "n_datasets", "n_cells")},
                             "family": "All", "method": method, "accuracy": .625})
    result = summarize(tmp_path / "runs", scope_path, suite_id=scope["suite_id"], comparison_group="all12",
                       samples=100, seed=20261007, rankings_csv=ranking)
    assert len(result["method_accuracy"]) == 13
    assert len(result["pairwise_differences"]) == 66
    assert result["public_point_check"]["maximum_absolute_difference"] == 0
    assert all(pair["lower_95"] == pair["upper_95"] == pair["difference"] == 0 for pair in result["pairwise_differences"])
    encoded = json.dumps(result)
    assert str(tmp_path) not in encoded
    assert '"logits"' not in encoded and '"label"' not in encoded and '"model_replicates"' not in encoded
    for method in result["method_accuracy"]:
        assert method["accuracy"] == .625


def test_scope_missing_cell_counts_fails_before_bootstrap(complete_runs):
    _, scope_path, scope = complete_runs
    scope["sample_counts"].pop()
    write(scope_path, {"schema": "s1q.public-ranking-scopes.v1", "computed_scopes": [scope]})
    with pytest.raises(ValueError, match="each fixed model/source once"):
        select_scope(scope_path, scope["suite_id"], "all12")


def test_point_accuracy_check_does_not_accept_rounded_or_different_values(tmp_path):
    scope = {"suite_id": "s", "comparison_group": "toy", "precision": "W4A4", "n_models": 1, "n_datasets": 1, "n_cells": 1}
    ranking = tmp_path / "rankings.csv"
    ranking.write_text("suite_id,comparison_group,family,precision,method,n_models,n_datasets,n_cells,accuracy\ns,toy,All,W4A4,a,1,1,1,0.3333\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more than 1e-10"):
        check_public_points(ranking, scope, ("a",), (1 / 3,))
