"""Negative controls for the read-only native-equivalence merge gate."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_smoothquant_merge import audit_pair, MAIN_METHODS, CONTROL, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    manifest_path = tmp_path / "benchmarks/manifest.json"
    entries, source_hashes = [], {}
    for name in ["development", *[f"source-{i}" for i in range(22)]]:
        record = {"state": f"Evidence for {name}", "questions": {"decision": {
            "type": "choice", "instructions": "Select the correct alternative.",
            "criteria": {"A": "First alternative", "B": "Second alternative"}, "label": "B", "src": name}},
            "_meta": {"id": name + "/0", "group_id": name + "/0"}}
        path = tmp_path / "benchmarks/datasets" / f"{name}.jsonl"
        write(path, record)
        source_hashes[name] = sha(path)
        if name != "development":
            entries.append({"dataset_id": name, "path": f"datasets/{name}.jsonl", "sha256": sha(path),
                            "ranking_eligible": True, "label_kind": "upstream_dataset_gold"})
    write(manifest_path, {"schema": "s1q.expanded-benchmarks.v1", "benchmark_id": "fixture", "datasets": entries})
    source_hashes["calibration"] = "c" * 64
    names = [entry["dataset_id"] for entry in entries]
    identity = {"model": {"name": "fixture", "dtype": "float32", "revision": "a" * 40},
                "seed": 20261004, "bits": 4, "activation_bits": 4, "group_size": 128,
                "requested_calibration": 128, "accepted_calibration": 128,
                "accepted_calibration_ids": ["cal/0"], "accepted_evaluation_ids": {name: [name + "/0"] for name in source_hashes if name != "calibration"},
                "calibration_exclusions": [], "evaluation_exclusions": {name: [] for name in ["development", *names]},
                "selected_linear_names": ["layer"], "selected_linear_count": 1, "selected_weight_parameters": 6,
                "reservoir_size": 128, "source_sha256": source_hashes, "implementation_sha256": {"models.py": "a" * 64},
                "benchmark_manifest_sha256": sha(manifest_path), "expanded_ranking_dataset_ids": names, "final_test_read": True}
    config = {"model": "fixture", "bits": 4, "activation_bits": 4, "group_size": 128,
              "calibration_count": 128, "development_count": 256, "reservoir_size": 128,
              "seed": 20261004, "dtype": "bf16", "benchmark_manifest_sha256": sha(manifest_path)}
    registration = {"method": CONTROL, "official_reproduction": False,
                    "quantization_math_changed": False, "frozen_main_method_matrix_changed": False}
    for role, methods in (("main", MAIN_METHODS), ("control", (CONTROL,))):
        root = tmp_path / role
        ident, frozen = deepcopy(identity), deepcopy(config)
        if role == "control":
            ident.update(additional_control_only=True, additional_control_registration=registration)
            frozen.update(additional_control_only=True, additional_control_registration=registration)
        report = {"identity": ident, "methodology": {"uses_calibration_gold_labels_for_quantization": False},
                  "methods": {method: {"status": "complete", "quantization": {"activation_bits": 4,
                                     "layers": {"layer": {"bits": 4, "group_size": 128, "shape": [2, 3]}}}} for method in methods}}
        write(root / "report.json", report)
        write(root / "identity.json", ident)
        write(root / "frozen-config-before-evaluation.json", frozen)
        write(root / "expanded-evaluation-complete.json", {"status": "complete", "methods": {method: "complete" for method in methods},
                                                           "benchmark_manifest_sha256": sha(manifest_path), "ranking_dataset_ids": names})
        for name in ["development", *names]:
            row = {"key": name + "/0::decision", "record_id": name + "/0", "qid": "decision", "type": "choice",
                   "source": name, "cluster_id": name + "/0", "label": 1, "label_kind": "dataset_gold", "logits": [0.0, 1.0]}
            write(root / f"native-{name}.jsonl", row)
            for method in methods:
                write(root / f"{method}-{name}.jsonl", row)
    return tmp_path / "main", tmp_path / "control", manifest_path, tmp_path / "benchmarks/datasets/development.jsonl"


def test_identical_native_and_full_method_coverage_admits_control(fixture):
    report = audit_pair(*fixture)
    assert report["merge_admissible"] is True
    assert len(report["cohorts"]) == 23
    assert report["max_abs_native_logit_difference"] == 0
    assert report["max_abs_native_probability_difference"] == 0


def test_equal_softmax_but_shifted_native_logits_are_rejected(fixture):
    main, control, _, _ = fixture
    path = control / "native-source-0.jsonl"
    row = json.loads(path.read_text())
    row["logits"] = [0.5, 1.5]  # Softmax is identical, but native logits are not.
    write(path, row)
    report = audit_pair(*fixture)
    assert report["merge_admissible"] is False
    assert "threshold exceeded" in report["errors"][0]


def test_missing_smooth_source_prediction_is_rejected(fixture):
    main, control, _, _ = fixture
    (control / f"{CONTROL}-source-0.jsonl").write_text("", encoding="utf-8")
    report = audit_pair(*fixture)
    assert not report["merge_admissible"]
    assert "missing native decisions" in report["errors"][0]


def test_same_count_with_different_gold_is_rejected(fixture):
    main, control, _, _ = fixture
    path = control / "native-source-0.jsonl"
    row = json.loads(path.read_text()); row["label"] = 0; write(path, row)
    report = audit_pair(*fixture)
    assert not report["merge_admissible"]
    assert "mismatched label" in report["errors"][0]


def test_candidate_permutation_with_same_logits_is_rejected(fixture):
    main, control, _, _ = fixture
    path = control / "native-source-0.jsonl"
    row = json.loads(path.read_text()); row["options"] = ["B", "A"]; write(path, row)
    report = audit_pair(*fixture)
    assert not report["merge_admissible"]
    assert "option ordering" in report["errors"][0]


def test_native_equivalence_tolerance_cannot_be_relaxed(fixture):
    with pytest.raises(ValueError, match="1e-6"):
        audit_pair(*fixture, tolerance=1e-3)
