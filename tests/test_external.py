"""Frozen external evaluation: synthetic CPU fixture, no model downloads."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from s1q.data import file_sha256, write_records
from s1q.models import MODEL_REVISIONS, SOURCE_REVISIONS, UnsupportedRecord
from s1q.quantization import InputStatistics, quantize_model

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("s1q_evaluate_external", ROOT / "scripts/evaluate_external.py")
EXTERNAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTERNAL)


class NativeFixture:
    """Native request boundary and reversible linears without a pretrained model."""
    def __init__(self, *, fail_after_admission=False):
        self.model = nn.Module()
        self.model.backbone = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            self.model.backbone[0].weight.copy_(torch.tensor([[.73, .16], [-.21, .48]]))
            self.model.backbone[1].weight.copy_(torch.tensor([[.36, -.24], [.13, .81]]))
        self.backbone = self.model.backbone
        self.metadata = {"name": "laya", "dtype": "torch.float32", "device": "cpu", "raw_logits": True,
                         "revision": MODEL_REVISIONS["laya"][1], "source_revision": SOURCE_REVISIONS["laya"]}
        self.fail_after_admission = fail_after_admission
        self.requests = []

    def infer(self, record):
        assert set(record) == {"state", "questions"}
        assert all(set(question) <= {"type", "instructions", "criteria"} for question in record["questions"].values())
        self.requests.append(deepcopy(record))
        if record["state"] == "native overflow":
            raise UnsupportedRecord("fixture native budget")
        if self.fail_after_admission and any(hasattr(m, "_s1q_quantization_active") for m in self.backbone.modules()):
            raise ValueError("fixture post-admission failure")
        z = self.backbone(torch.tensor([[1., 2.]])).reshape(-1)
        return [z for _ in record["questions"]]


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def locked_run(tmp_path):
    run, data = tmp_path / "run", tmp_path / "data"
    artifact_path = tmp_path / "selected.pt"
    adapter = NativeFixture()
    stats = {name: InputStatistics(2, torch.ones(2), torch.ones(2))
             for name, module in adapter.backbone.named_modules() if isinstance(module, nn.Linear)}
    profile = {"name": "s1q-protected", "method": "s1q", "activation_bits": None, "sensitive_fraction": .5}
    with quantize_model(adapter.backbone, method="s1q", statistics=stats, bits=4, group_size=2,
                        sensitive_fraction=.5) as session:
        session.export(artifact_path)
        artifact = session.artifact()
        selected_quantization = session.report()
    with quantize_model(adapter.backbone, method="rtn", bits=4, group_size=2,
                        include_prefixes=tuple(artifact["layers"]), exclude_patterns=()) as session:
        rtn_quantization = session.report()
    temperatures = {"noul": 2., "choice": 3.}
    environment = {"model": adapter.metadata, "packages": {}, "seed": 11}
    selection = {"selected": profile, "bits": 4, "group_size": 2,
                 "selected_before_quantized_test": True, "objective": "development only"}
    summary = {"status": "complete", "model": "laya", "environment": environment, "selection": selection,
               "baseline": {"test": {"temperatures": temperatures}}, "quantized": {
                   profile["name"]: {"profile": profile, "quantization": selected_quantization,
                                     "artifact": {"sha256": file_sha256(artifact_path), "bytes": artifact_path.stat().st_size},
                                     "splits": {"test": {"temperatures": temperatures}}},
                   "rtn-matched": {"profile": {"name": "rtn-matched", "method": "rtn"},
                                   "quantization": rtn_quantization, "splits": {"test": {"temperatures": temperatures}}}}}
    json_file(run / "summary.json", summary)
    json_file(run / "environment.json", environment)
    json_file(run / "selection.json", selection)
    records = [{"state": "Public observed facts.", "questions": {
        "choice": {"type": "choice", "instructions": "Choose the appropriate handler.",
                   "criteria": {"b": "Second", "a": "First"}, "label": "a", "src": "jevbench/hard/router",
                   "label_kind": "authored_scenario_gold"},
        "boolean": {"type": "noul", "instructions": "Apply explicit conditions.",
                    "criteria": {"false": "Absent", "true": "Present"}, "label": True,
                    "label_kind": "authored_probability_mode", "target_distribution": {"false": .25, "true": .75},
                    "target_distribution_kind": "state_derivable_probability"}},
        "_meta": {"id": "jevbench/first", "group_id": "shared-cluster", "family": "router", "tier": "hard",
                  "provenance": {"private_rationale": "This private gold text must never reach infer"}}},
        {"state": "native overflow", "questions": {"decision": {"type": "noul", "instructions": "Read condition.",
         "criteria": {"false": "No", "true": "Yes"}, "label": False}},
         "_meta": {"id": "jevbench/overflow", "group_id": "overflow", "family": "policy", "tier": "original"}}]
    write_records(data / "external_test.jsonl", records)
    pinned = EXTERNAL.read_json(ROOT / "configs/jevbench-data-checksums.json")
    manifest = {"schema": "s1q.jevbench-external.v1", "role": "external_test_only", "selection_uses_predictions": False,
                "source_repo": pinned["repo"], "source_revision": pinned["revision"],
                "source_files": {name: pinned["files"][name] for name in
                                 ("datasets/public/original.jsonl", "datasets/public/hard.jsonl")},
                "output": {"external_test.jsonl": {"sha256": file_sha256(data / "external_test.jsonl")}},
                "accepted_records": 2, "accepted_questions": 3}
    json_file(data / "manifest.json", manifest)
    return {"run": run, "data": data, "artifact_path": artifact_path, "artifact": artifact,
            "summary": summary, "records": records, "output": tmp_path / "out"}


def run_fixture(locked, monkeypatch, adapter=None):
    adapter = adapter or NativeFixture()
    def load(*args, **kwargs):
        freeze = EXTERNAL.read_json(locked["output"] / "freeze.json")
        assert freeze["policy"]["external_fitting"] is False
        assert not (locked["output"] / "baseline.jsonl").exists()
        return adapter
    monkeypatch.setattr(EXTERNAL, "load_model", load)
    result = EXTERNAL.run_external("laya", locked["run"], locked["artifact_path"], locked["data"], locked["output"],
                                   device="cpu", dtype="fp32", seed=11, bootstrap_samples=40)
    return result, adapter


def test_complete_external_run_freezes_first_strips_gold_and_matches_rtn_mask(locked_run, monkeypatch):
    adapter = NativeFixture()
    original = [p.detach().clone() for p in adapter.model.parameters()]
    result, adapter = run_fixture(locked_run, monkeypatch, adapter)
    assert result["status"] == "complete"
    assert (result["prepared_records"], result["accepted_records"], result["accepted_questions"]) == (2, 1, 2)
    assert result["excluded_records"] == 1
    assert result["rtn_temperature_status"] == "frozen_from_scope_matched_run"
    assert set(result["quantization"]["rtn_matched"]["layers"]) == set(locked_run["artifact"]["layers"])
    assert result["quantization"]["rtn_matched"]["matched_protected_layers"] == locked_run["artifact"]["preserved_layers"]
    assert result["selected"]["raw_soft_targets"]["n"] == 1
    assert result["selected"]["questions"] == 2
    assert result["paired_accuracy_ci"]["selected_vs_rtn"]["n_clusters"] == 1
    paired = EXTERNAL.load_records(locked_run["output"] / "paired_predictions.jsonl")
    assert paired[0]["candidate_keys"] == ["b", "a"]
    assert paired[0]["label"] == 1
    assert paired[1]["candidate_keys"] == ["false", "true"]
    assert len(adapter.requests) == 4  # two baseline records + one admitted request per candidate
    assert "private_rationale" not in json.dumps(adapter.requests)
    for before, after in zip(original, adapter.model.parameters()):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert all(not module._forward_pre_hooks for module in adapter.backbone.modules())
    with pytest.raises(FileExistsError, match="frozen run"):
        run_fixture(locked_run, monkeypatch)


def test_artifact_tamper_aborts_before_native_model_load(locked_run, monkeypatch):
    with locked_run["artifact_path"].open("ab") as handle:
        handle.write(b"tamper")
    monkeypatch.setattr(EXTERNAL, "load_model", lambda *a, **k: pytest.fail("must not load native model"))
    with pytest.raises(ValueError, match="integrity mismatch"):
        EXTERNAL.run_external("laya", locked_run["run"], locked_run["artifact_path"], locked_run["data"], locked_run["output"], dtype="fp32")
    assert not locked_run["output"].exists()


def test_external_dataset_tamper_and_selection_drift_fail_closed(locked_run):
    original_data = (locked_run["data"] / "external_test.jsonl").read_bytes()
    (locked_run["data"] / "external_test.jsonl").write_bytes(original_data + b"\n")
    with pytest.raises(ValueError, match="data hash"):
        EXTERNAL.lock_inputs("laya", locked_run["run"], locked_run["artifact_path"], locked_run["data"],
                             dtype="fp32", seed=11, bootstrap_samples=40)
    (locked_run["data"] / "external_test.jsonl").write_bytes(original_data)
    selection = EXTERNAL.read_json(locked_run["run"] / "selection.json")
    selection["selected"]["sensitive_fraction"] = 0
    json_file(locked_run["run"] / "selection.json", selection)
    with pytest.raises(ValueError, match="disagree"):
        EXTERNAL.lock_inputs("laya", locked_run["run"], locked_run["artifact_path"], locked_run["data"],
                             dtype="fp32", seed=11, bootstrap_samples=40)


def test_post_admission_failure_aborts_instead_of_dropping_quantized_inputs(locked_run, monkeypatch):
    adapter = NativeFixture(fail_after_admission=True)
    original = [p.detach().clone() for p in adapter.model.parameters()]
    with pytest.raises(ValueError, match="post-admission"):
        run_fixture(locked_run, monkeypatch, adapter)
    assert not (locked_run["output"] / "summary.json").exists()
    failure = EXTERNAL.read_json(locked_run["output"] / "failure.json")
    assert failure["status"] == "failed"
    assert EXTERNAL.read_json(locked_run["output"] / "eligibility.json")["accepted_questions"] == 2
    for before, after in zip(original, adapter.model.parameters()):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_unavailable_matching_rtn_temperatures_are_explicit_not_fitted(locked_run, monkeypatch):
    del locked_run["summary"]["quantized"]["rtn-matched"]
    json_file(locked_run["run"] / "summary.json", locked_run["summary"])
    result, _ = run_fixture(locked_run, monkeypatch)
    assert result["rtn_temperature_status"] == "unavailable_no_scope_matched_fitted_run"
    assert result["rtn_matched"]["temperature_calibrated"] is None
    assert "temperatures" not in result["rtn_matched"]


def test_soft_target_metrics_preserve_named_candidate_order():
    row = {"type": "choice", "candidate_keys": ["b", "a"], "target_distribution": {"a": .75, "b": .25},
           "logits": np.log([.25, .75]).tolist()}
    result = EXTERNAL.soft_target_metrics([row])
    assert result["kl_target_to_prediction"] == pytest.approx(0, abs=1e-12)
    assert result["squared_probability_error"] == pytest.approx(0, abs=1e-12)
    assert result["cross_entropy"] == pytest.approx(-.25 * np.log(.25) - .75 * np.log(.75))


def test_no_inference_metadata_is_allowed_to_change(locked_run, monkeypatch):
    adapter = NativeFixture()
    adapter.metadata["revision"] = "wrong"
    with pytest.raises(ValueError, match="loaded metadata"):
        run_fixture(locked_run, monkeypatch, adapter)
    assert adapter.requests == []


def test_frozen_files_cannot_change_between_evaluation_stages(locked_run):
    freeze, _, _, _ = EXTERNAL.lock_inputs("laya", locked_run["run"], locked_run["artifact_path"], locked_run["data"],
                                         dtype="fp32", seed=11, bootstrap_samples=40)
    EXTERNAL.verify_frozen_files(freeze)
    with (locked_run["run"] / "environment.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        EXTERNAL.verify_frozen_files(freeze)
