"""Comparison scope regressions using synthetic aggregate fixtures only.

No fixture is an actual S1Q result or intended for publication.
"""
from __future__ import annotations

import importlib.util
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
