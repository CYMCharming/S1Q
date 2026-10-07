"""Comparison scope regressions using synthetic aggregate fixtures only.

No fixture is an actual S1Q result or intended for publication.
"""
from __future__ import annotations

import importlib.util
from copy import deepcopy
import json
from pathlib import Path

import pytest


@pytest.fixture
def report_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_public_benchmark_report.py"
    spec = importlib.util.spec_from_file_location("public_benchmark_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_rows(builder, models=("kev-fixture-a",), *, smoothquant=True):
    methods = ["native", *builder.METHODS]
    if smoothquant:
        methods.append("smoothquant-adapted")
    return [{
        "experiment_batch_id": "synthetic_fixture_only", "model": model,
        "precision": "W4A4", "dataset_id": "fixture-source", "family": "Kev",
        "method": method, "status": "complete", "metric_target": "labeled_decision_accuracy",
        "n": 4, "n_clusters": 4, "accuracy": .75 if method == "native" else .5,
        "nll": 1., "brier": .7, "ece_15": .2,
    } for model in models for method in methods]


def fixture_settings(builder, *, expanded=False):
    suite = {
        "suite_id": "synthetic_fixture_only", "experiment_batch_ids": ["synthetic_fixture_only"],
        "precision": "W4A4", "dataset_ids": ["fixture-source"],
        "note": "SYNTHETIC TEST FIXTURE ONLY; no measured model results.",
    }
    if expanded:
        suite.update({"comparison_group": "all12", "methods": [*builder.METHODS, "smoothquant-adapted"]})
    return {"ranking_suites": [suite]}


def test_historical_defaults_remain_eleven_methods(report_builder):
    rows, scopes, _ = report_builder.build_rankings(fixture_rows(report_builder), fixture_settings(report_builder))
    primary = next(scope for scope in scopes if scope["is_primary_comparison"])
    assert primary["comparison_group"] == "all11"
    assert len(primary["method_ids"]) == 11
    assert "smoothquant-adapted" not in primary["method_ids"]
    assert len([row for row in rows if row["family"] == "All" and row["comparison_group"] == "all11"]) == 11


def test_expanded_group_has_smoothquant_in_table_and_manifest(report_builder, tmp_path):
    rankings, scopes, _ = report_builder.build_rankings(fixture_rows(report_builder), fixture_settings(report_builder, expanded=True))
    primary = next(scope for scope in scopes if scope["is_primary_comparison"])
    assert primary["comparison_group"] == "all12"
    assert len(primary["method_ids"]) == 12
    assert "smoothquant-adapted" in primary["method_ids"]
    table = report_builder.report_text(rankings, scopes, {"status": "not_requested"})
    assert "all12 (12 methods)" in table and "SmoothQuant*" in table
    figure = report_builder.plot_rankings(tmp_path, rankings, scopes, "synthetic_fixture_only")
    assert figure["comparison_group"] == "all12"
    assert len(figure["method_order"]) == 12
    assert "smoothquant-adapted" in figure["method_order"]
    assert {item["file"] for item in figure["outputs"]} == {"benchmark_ranking.png", "benchmark_ranking.svg"}


def test_missing_smoothquant_excludes_model_only_from_twelve_method_scope(report_builder):
    rows = fixture_rows(report_builder, ("kev-fixture-a", "kev-fixture-b"))
    rows = [row for row in rows if not (row["model"] == "kev-fixture-b" and row["method"] == "smoothquant-adapted")]
    _, scopes, _ = report_builder.build_rankings(rows, fixture_settings(report_builder, expanded=True))
    by_group = {scope["comparison_group"]: scope for scope in scopes}
    assert by_group["all12"]["included_models"] == ["kev-fixture-a"]
    assert by_group["all12"]["excluded_models"]["kev-fixture-b"] == [{"dataset_id": "fixture-source", "method": "smoothquant-adapted"}]
    assert by_group["s1q6"]["included_models"] == ["kev-fixture-a", "kev-fixture-b"]


def test_group_name_cannot_claim_wrong_method_count(report_builder):
    settings = fixture_settings(report_builder, expanded=True)
    settings["ranking_suites"][0]["methods"].remove("smoothquant-adapted")
    with pytest.raises(ValueError, match="name and method count"):
        report_builder.build_rankings(fixture_rows(report_builder), settings)


def test_main_ranking_bolds_all_quantized_ties(report_builder):
    rankings, scopes, _ = report_builder.build_rankings(fixture_rows(report_builder), fixture_settings(report_builder, expanded=True))
    table = report_builder.report_text(rankings, scopes, {"status": "not_requested"})
    assert "| 1 | SmoothQuant* | **50.00** | **1.000** | **0.700** | **0.200** |" in table
    assert "ties after rounding" in table


def test_complete_twenty_two_source_table_keeps_native_plain_and_avg_global(report_builder):
    settings = fixture_settings(report_builder, expanded=True)
    suite = settings["ranking_suites"][0]
    suite["dataset_ids"] = [f"fixture-source-{index:02}" for index in range(22)]
    suite["dataset_metadata"] = {dataset: {"name": dataset, "source_category": "synthetic_fixture_only"} for dataset in suite["dataset_ids"]}
    rows = []
    for index, dataset in enumerate(suite["dataset_ids"]):
        for row in fixture_rows(report_builder):
            rows.append({**row, "dataset_id": dataset, "accuracy": (.9 if index < 8 else .1) if row["method"] == "s1q" else row["accuracy"]})
    _, scopes, _ = report_builder.build_rankings(rows, settings)
    cells, table, manifests = report_builder.build_accuracy_tables(rows, scopes)
    assert len(cells) == 13 * 22 and len(manifests) == 1
    assert manifests[0]["dataset_ids"] == suite["dataset_ids"]
    assert table.count("Part ") == 3
    assert table.count("Avg uses all 22 sources.") == 3
    native_lines = [line for line in table.splitlines() if line.startswith("| Native |")]
    assert len(native_lines) == 3 and all("**" not in line for line in native_lines)
    s1q_lines = [line for line in table.splitlines() if line.startswith("| S1Q (current) |")]
    assert len(s1q_lines) == 3 and all(line.endswith("| 39.09 |") for line in s1q_lines)
    current = [row for row in cells if row["method"] == "s1q"]
    assert all(row["avg_accuracy"] == pytest.approx((8 * .9 + 14 * .1) / 22) for row in current)
    assert sum(row["best_quantized_accuracy_exact"] for row in current) == 8
    assert not any(row["best_quantized_accuracy_exact"] for row in cells if row["method"] == "native")
    assert len({row["dataset_id"] for row in cells}) == 22


def test_accuracy_table_rejects_missing_native_cell_instead_of_filling(report_builder):
    settings = fixture_settings(report_builder, expanded=True)
    settings["ranking_suites"][0]["dataset_metadata"] = {"fixture-source": {"name": "Synthetic fixture"}}
    rows = [row for row in fixture_rows(report_builder) if row["method"] != "native"]
    _, scopes, _ = report_builder.build_rankings(rows, settings)
    with pytest.raises(ValueError, match="Missing Native"):
        report_builder.build_accuracy_tables(rows, scopes)


def test_historical_scopes_do_not_add_expanded_accuracy_tables(report_builder):
    rows = fixture_rows(report_builder)
    _, scopes, _ = report_builder.build_rankings(rows, fixture_settings(report_builder))
    cells, _, manifests = report_builder.build_accuracy_tables(rows, scopes)
    assert cells == [] and manifests == []


def uncertainty_fixture(builder, tmp_path):
    settings = fixture_settings(builder, expanded=True)
    rankings, scopes, _ = builder.build_rankings(fixture_rows(builder), settings)
    scope_document = {"schema": "s1q.public-ranking-scopes.v1", "declared_settings": settings, "computed_scopes": scopes}
    builder.write_json(tmp_path / "ranking_scopes.json", scope_document)
    builder.write_csv(tmp_path / "rankings.csv", rankings)
    scope = next(item for item in scopes if item["is_primary_comparison"])
    fields = ("suite_id", "comparison_group", "precision", "included_models", "dataset_ids", "method_ids", "n_models", "n_datasets", "n_cells", "aggregation")
    document = {"schema": "s1q.conditional-macro-accuracy-uncertainty.v1", "status": "passed", **{field: deepcopy(scope[field]) for field in fields},
        "bootstrap": {"seed": 7, "samples": 10, "interval": "percentile 95% (2.5%, 97.5%)",
            "resampling_unit": "whole saved cluster_id within each source",
            "shared_draws": "same source-union cluster multiplicities across all models and methods",
            "models_resampled": False, "sources_resampled": False, "zero_denominator_model_source_draws": 0,
            "zero_denominator_policy": "fail; never silently replace a draw"},
        "provenance": {"ranking_scopes_sha256": builder.sha256(tmp_path / "ranking_scopes.json")},
        "public_point_check": {"provided": True, "sha256": builder.sha256(tmp_path / "rankings.csv"), "maximum_absolute_difference": 0., "tolerance": 1e-10},
        "sample_counts": [{"model": row["model"], "dataset_id": row["dataset_id"], "n_decisions": row["n"], "n_clusters": row["n_clusters"]} for row in scope["sample_counts"]],
        "method_accuracy": [{"method": method, "ranked": method != "native", "accuracy": .75 if method == "native" else .5,
            "lower_95": .6 if method == "native" else .3, "upper_95": .9 if method == "native" else .7,
            "accuracy_percent": 75. if method == "native" else 50., "lower_95_percent": 60. if method == "native" else 30.,
            "upper_95_percent": 90. if method == "native" else 70.} for method in ("native", *scope["method_ids"])]}
    path = tmp_path / "accuracy_uncertainty.json"
    builder.write_json(path, document)
    return path, document, rankings, scopes, scope_document


def test_ci_serialization_matches_unchanged_public_files(report_builder, tmp_path):
    path, _, rankings, scopes, document = uncertainty_fixture(report_builder, tmp_path)
    assert report_builder.serialized_json(document) == (tmp_path / "ranking_scopes.json").read_bytes()
    assert report_builder.serialized_csv(rankings) == (tmp_path / "rankings.csv").read_bytes()
    checked = report_builder.validate_accuracy_uncertainty(path, tmp_path, rankings, scopes, document, "synthetic_fixture_only")
    assert len(checked["method_intervals"]) == 12 and "native" not in checked["method_intervals"]
    assert checked["manifest"]["source_file_sha256"] == report_builder.sha256(path)


@pytest.mark.parametrize("field,value", [
    ("suite_id", "wrong"), ("comparison_group", "all11"), ("precision", "W3A4"),
    ("included_models", ["wrong-model"]), ("dataset_ids", ["wrong-source"]), ("method_ids", ["rtn"]),
    ("n_models", 2), ("n_datasets", 2), ("n_cells", 2), ("aggregation", "pooled"),
    ("n_models", True), ("n_datasets", 1.),
])
def test_ci_rejects_scope_mismatch(report_builder, tmp_path, field, value):
    path, ci, rankings, scopes, document = uncertainty_fixture(report_builder, tmp_path)
    ci[field] = value
    report_builder.write_json(path, ci)
    with pytest.raises(ValueError, match="scope mismatch"):
        report_builder.validate_accuracy_uncertainty(path, tmp_path, rankings, scopes, document, "synthetic_fixture_only")


@pytest.mark.parametrize("target", ["scope_sha", "ranking_sha", "missing_point_check", "prior_scope_bytes", "prior_ranking_bytes"])
def test_ci_rejects_stale_hashes_and_changed_prior_export(report_builder, tmp_path, target):
    path, ci, rankings, scopes, document = uncertainty_fixture(report_builder, tmp_path)
    if target == "scope_sha":
        ci["provenance"]["ranking_scopes_sha256"] = "0" * 64
    elif target == "ranking_sha":
        ci["public_point_check"]["sha256"] = "0" * 64
    elif target == "missing_point_check":
        ci["public_point_check"]["provided"] = False
    else:
        changed = tmp_path / ("ranking_scopes.json" if target == "prior_scope_bytes" else "rankings.csv")
        changed.write_bytes(changed.read_bytes() + b"\n")
    report_builder.write_json(path, ci)
    before = {name: (tmp_path / name).read_bytes() for name in ("rankings.csv", "ranking_scopes.json")}
    with pytest.raises(ValueError, match="SHA|prior public export"):
        report_builder.validate_accuracy_uncertainty(path, tmp_path, rankings, scopes, document, "synthetic_fixture_only")
    assert before == {name: (tmp_path / name).read_bytes() for name in before}


def test_ci_rejects_different_unrounded_point_even_with_same_display(report_builder, tmp_path):
    path, ci, rankings, scopes, document = uncertainty_fixture(report_builder, tmp_path)
    row = ci["method_accuracy"][1]
    row["accuracy"] += 1e-7
    row["accuracy_percent"] = 100 * row["accuracy"]
    assert f"{row['accuracy_percent']:.2f}" == "50.00"
    report_builder.write_json(path, ci)
    with pytest.raises(ValueError, match="unrounded point differs"):
        report_builder.validate_accuracy_uncertainty(path, tmp_path, rankings, scopes, document, "synthetic_fixture_only")


@pytest.mark.parametrize("target", ["duplicate", "missing", "reversed", "percent", "native_ranked", "resample_models", "independent_draws", "denominator"])
def test_ci_rejects_invalid_intervals_or_resampling(report_builder, tmp_path, target):
    path, ci, rankings, scopes, document = uncertainty_fixture(report_builder, tmp_path)
    if target == "duplicate":
        ci["method_accuracy"].append(deepcopy(ci["method_accuracy"][1]))
    elif target == "missing":
        ci["method_accuracy"].pop()
    elif target == "reversed":
        ci["method_accuracy"][1].update(lower_95=.8, lower_95_percent=80.)
    elif target == "percent":
        ci["method_accuracy"][1]["upper_95_percent"] = 7.
    elif target == "native_ranked":
        ci["method_accuracy"][0]["ranked"] = True
    elif target == "resample_models":
        ci["bootstrap"]["models_resampled"] = True
    elif target == "independent_draws":
        ci["bootstrap"]["shared_draws"] = "independent draws"
    else:
        ci["sample_counts"][0]["n_decisions"] += 1
    report_builder.write_json(path, ci)
    with pytest.raises(ValueError):
        report_builder.validate_accuracy_uncertainty(path, tmp_path, rankings, scopes, document, "synthetic_fixture_only")


def test_ci_plot_has_only_aggregate_provenance_and_conditional_caption(report_builder, tmp_path):
    path, _, rankings, scopes, document = uncertainty_fixture(report_builder, tmp_path)
    checked = report_builder.validate_accuracy_uncertainty(path, tmp_path, rankings, scopes, document, "synthetic_fixture_only")
    figure = report_builder.plot_rankings(tmp_path, rankings, scopes, "synthetic_fixture_only", checked)
    assert figure["accuracy_uncertainty"]["source_file_sha256"] == report_builder.sha256(path)
    assert "native" not in figure["method_order"]
    assert "replicates" not in json.dumps(figure) and "input_file_hashes" not in json.dumps(figure)
    svg = (tmp_path / "benchmark_ranking.svg").read_text(encoding="utf-8")
    assert "conditional 95% evaluated-sample" in svg and "shared within-source whole-cluster draws" in svg
    assert "No multiple-comparisons adjustment or universal winner claim" in svg
    text = report_builder.report_text(rankings, scopes, figure)
    assert "marginal bar intervals must not be independently subtracted" in text


def test_ci_cli_reexport_keeps_point_outputs_byte_identical(report_builder, tmp_path, monkeypatch):
    import sys
    settings = fixture_settings(report_builder, expanded=True)
    settings_path = tmp_path / "scope-settings.json"
    report_builder.write_json(settings_path, settings)
    monkeypatch.setattr(report_builder, "audited_input", lambda *args: (fixture_rows(report_builder), {"fixture": "synthetic_only"}))
    monkeypatch.setattr(sys, "argv", ["report", "--input", "synthetic_fixture_only=synthetic", "--dataset-list", str(settings_path), "--output-dir", str(tmp_path), "--no-plot"])
    report_builder.main()
    path, _, _, _, _ = uncertainty_fixture(report_builder, tmp_path)
    unchanged = ("metrics.csv", "rankings.csv", "ranking_scopes.json", "model_scores.csv")
    before = {name: (tmp_path / name).read_bytes() for name in unchanged}
    monkeypatch.setattr(sys, "argv", ["report", "--input", "synthetic_fixture_only=synthetic", "--dataset-list", str(settings_path), "--output-dir", str(tmp_path), "--accuracy-uncertainty", str(path)])
    report_builder.main()
    assert before == {name: (tmp_path / name).read_bytes() for name in unchanged}
    manifest = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert manifest["accuracy_uncertainty"]["source_file_sha256"] == report_builder.sha256(path)
