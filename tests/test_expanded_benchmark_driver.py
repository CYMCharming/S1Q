"""Manifest admission and publication metadata, without models, Torch or GPUs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def driver(monkeypatch):
    optimization = types.ModuleType("s1q.optimization")

    def add_arguments(parser, *, default_methods):
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--extra-evaluation", action="append", default=[])
        parser.add_argument("--methods", default=default_methods)
        return parser

    optimization.add_arguments = add_arguments
    optimization.run = lambda args: (_ for _ in ()).throw(AssertionError("Loader tests must not load a model"))
    monkeypatch.setitem(sys.modules, "s1q.optimization", optimization)
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_expanded_benchmarks.py"
    spec = importlib.util.spec_from_file_location("expanded_benchmark_driver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry(root, name="agnews", **overrides):
    path = root / f"{name}.jsonl"
    path.write_text('{"id":"fixture"}\n', encoding="utf-8")
    result = {
        "dataset_id": name, "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ranking_eligible": True, "label_kind": "upstream_dataset_gold",
    }
    result.update(overrides)
    return result


def manifest(root, entries):
    path = root / "manifest.json"
    path.write_text(json.dumps({
        "schema": "s1q.expanded-benchmarks.v1", "benchmark_id": "fixture-frozen-evaluation",
        "datasets": entries,
    }), encoding="utf-8")
    return path


def test_checked_dataset_preserves_declared_names_order_and_hashes(driver, tmp_path):
    path = manifest(tmp_path, [entry(tmp_path, "agnews"), entry(tmp_path, "tweet-offensive")])
    meta, selected = driver.checked_datasets(path)
    assert meta["benchmark_id"] == "fixture-frozen-evaluation"
    assert selected == [("agnews", (tmp_path / "agnews.jsonl").resolve()),
                        ("tweet-offensive", (tmp_path / "tweet-offensive.jsonl").resolve())]


def test_hash_mismatch_rejected_before_any_model_loading(driver, tmp_path):
    path = manifest(tmp_path, [entry(tmp_path, sha256="0" * 64)])
    with pytest.raises(ValueError, match="hash mismatch"):
        driver.checked_datasets(path)


def test_path_traversal_rejected_even_when_file_hash_is_correct(driver, tmp_path):
    inner = tmp_path / "bench"
    inner.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside\n", encoding="utf-8")
    malicious = entry(inner, path="../outside.jsonl", sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="escaped"):
        driver.checked_datasets(manifest(inner, [malicious]))


def test_absolute_outside_path_rejected(driver, tmp_path):
    inner = tmp_path / "bench"
    inner.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside\n", encoding="utf-8")
    malicious = entry(inner, path=str(outside), sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="escaped"):
        driver.checked_datasets(manifest(inner, [malicious]))


def test_teacher_diagnostic_excluded_without_reading_its_path(driver, tmp_path):
    gold = entry(tmp_path)
    diagnostic = {
        "dataset_id": "teacher-only", "path": "does-not-exist.jsonl", "sha256": "invalid",
        "ranking_eligible": False, "label_kind": "reference_policy_argmax_compatibility",
    }
    _, selected = driver.checked_datasets(manifest(tmp_path, [gold, diagnostic]))
    assert [name for name, _ in selected] == ["agnews"]


def test_only_teacher_diagnostics_fail_instead_of_empty_success(driver, tmp_path):
    diagnostic = entry(tmp_path, "teacher-only", ranking_eligible=False)
    with pytest.raises(ValueError, match="No frozen rank-eligible"):
        driver.checked_datasets(manifest(tmp_path, [diagnostic]))


def test_teacher_diagnostic_cannot_be_ranked_even_if_mistagged_eligible(driver, tmp_path):
    diagnostic = entry(tmp_path, "teacher-only", label_kind="reference_policy_argmax_compatibility")
    with pytest.raises(ValueError, match="(?i)(label|teacher|gold|eligible)"):
        driver.checked_datasets(manifest(tmp_path, [diagnostic]))


def test_string_false_cannot_be_treated_as_true_eligibility(driver, tmp_path):
    diagnostic = entry(tmp_path, "teacher-only", ranking_eligible="false")
    with pytest.raises(ValueError, match="(?i)(bool|eligible|eligibility)"):
        driver.checked_datasets(manifest(tmp_path, [diagnostic]))


def test_wrong_manifest_schema_cannot_be_admitted(driver, tmp_path):
    path = manifest(tmp_path, [entry(tmp_path)])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "unrelated.teacher-benchmark.v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(schema|manifest)"):
        driver.checked_datasets(path)


def test_duplicate_dataset_name_rejected(driver, tmp_path):
    value = entry(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        driver.checked_datasets(manifest(tmp_path, [value, value.copy()]))


@pytest.mark.parametrize("name", ["development", "", "../agnews", "a=b", "has space"])
def test_old_dev_or_unsafe_dataset_name_rejected(driver, tmp_path, name):
    value = entry(tmp_path)
    value["dataset_id"] = name
    with pytest.raises(ValueError, match="Invalid"):
        driver.checked_datasets(manifest(tmp_path, [value]))


def test_main_freezes_manifest_before_run_and_writes_truthful_final_metadata(driver, tmp_path, monkeypatch):
    path = manifest(tmp_path, [entry(tmp_path)])
    output = tmp_path / "run"
    captured = {}

    def fake_run(args):
        # The real runner persists vars(args) through a strict JSON writer.
        json.dumps(vars(args), allow_nan=False)
        captured.update(vars(args))
        assert args.upstream_test_records_intentionally_evaluated is True
        assert args.recipe_selected_before_expanded_evaluation is True
        assert args.benchmark_manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert args.extra_evaluation == [f"agnews={(tmp_path / 'agnews.jsonl').resolve()}"]
        output.mkdir()
        return {"identity": {"final_test_read": False,
                              "accepted_evaluation_ids": {"development": ["old-dev"], "agnews": ["new-gold"]}},
                "methods": {"s1q": {"status": "complete"}, "rtn": {"status": "complete"}}}

    monkeypatch.setattr(driver, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_expanded_benchmarks.py", "--benchmark-manifest", str(path), "--output-dir", str(output)])
    driver.main()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    marker = json.loads((output / "expanded-evaluation-complete.json").read_text(encoding="utf-8"))
    assert report["identity"] == identity
    assert identity["final_test_read"] is True
    assert identity["old_development_excluded_from_expanded_ranking"] is True
    assert identity["expanded_ranking_dataset_ids"] == ["agnews"]
    assert identity["benchmark_id"] == "fixture-frozen-evaluation"
    assert marker["ranking_dataset_ids"] == ["agnews"]
    assert marker["status"] == "complete"
    assert "development" not in marker["ranking_dataset_ids"]


def test_main_refuses_unpinned_extra_dataset(driver, tmp_path, monkeypatch):
    path = manifest(tmp_path, [entry(tmp_path)])
    monkeypatch.setattr(sys, "argv", ["run_expanded_benchmarks.py", "--benchmark-manifest", str(path),
                                    "--output-dir", str(tmp_path / "result"), "--extra-evaluation", "unlisted=unlisted.jsonl"])
    with pytest.raises(ValueError, match="pinned benchmark manifest"):
        driver.main()


def test_main_marks_method_failure_without_claiming_complete(driver, tmp_path, monkeypatch):
    path = manifest(tmp_path, [entry(tmp_path)])
    output = tmp_path / "result"

    def fake_run(args):
        output.mkdir()
        return {"identity": {"accepted_evaluation_ids": {"development": [], "agnews": ["new-gold"]}},
                "methods": {"s1q": {"status": "failed"}, "rtn": {"status": "complete"}}}

    monkeypatch.setattr(driver, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_expanded_benchmarks.py", "--benchmark-manifest", str(path), "--output-dir", str(output)])
    driver.main()
    marker = json.loads((output / "expanded-evaluation-complete.json").read_text(encoding="utf-8"))
    assert marker["status"] == "method_failure"


def test_main_rejects_empty_admission_and_does_not_write_completion_marker(driver, tmp_path, monkeypatch):
    path = manifest(tmp_path, [entry(tmp_path)])
    output = tmp_path / "result"

    def fake_run(args):
        output.mkdir()
        return {"identity": {"accepted_evaluation_ids": {"development": ["old-dev"], "agnews": []}},
                "methods": {"s1q": {"status": "complete"}}}

    monkeypatch.setattr(driver, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_expanded_benchmarks.py", "--benchmark-manifest", str(path), "--output-dir", str(output)])
    with pytest.raises(ValueError, match="Entire dataset rejected"):
        driver.main()
    assert not (output / "expanded-evaluation-complete.json").exists()
