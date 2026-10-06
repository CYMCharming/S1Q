"""CPU end-to-end experiment checks using real quantization and a toy adapter.

The adapter's model receives public request state only. Gold targets are fixed
fixture data, independent of teacher predictions. No model downloads or GPUs are
needed; only the external model loader is substituted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from s1q import experiment
from s1q.data import load_records, model_request, write_records
from s1q.metrics import compare
from s1q.models import UnsupportedRecord
from s1q.packed import PackedLinear, apply_packed, packed_report
from s1q.quantization import load_quantized_artifact


class ToyDecisionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 4))
        self.pointer_head = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.pointer_head(self.backbone(inputs))


class ToyAdapter:
    """Tiny native adapter with the same raw-logit/insertion-order contract."""

    def __init__(self, *, metadata=None, trace=None, selection_path=None):
        # A pinned toy checkpoint must not change with the experiment RNG seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            self.model = ToyDecisionNetwork().eval()
        self.device = torch.device("cpu")
        self.metadata = {"name": "toy", "revision": "toy-checkpoint-v1", "dtype": "fp32",
                         "source_revision": "toy-adapter-v1"}
        self.metadata.update(metadata or {})
        self.trace = trace if trace is not None else []
        self.selection_path = selection_path

    @property
    def backbone(self):
        return self.model.backbone

    def infer(self, record):
        active = any(getattr(module, "_s1q_quantization_active", False)
                     for module in self.backbone.modules())
        split = record.get("_meta", {}).get("split")
        self.trace.append({"id": record["id"], "split": split, "quantized": active})
        if active and split in ("test", "transfer_test", "ood") and self.selection_path is not None:
            assert self.selection_path.exists(), "A quantized final test ran before selection was frozen"
        if record["state"].get("reject"):
            raise UnsupportedRecord("Toy native format rejects this fixture request")
        public = model_request(record)
        assert set(public) == {"state", "questions"}
        assert all("label" not in question and "label_kind" not in question
                   for question in public["questions"].values())
        vector = torch.tensor(public["state"]["vector"], dtype=torch.float32).unsqueeze(0)
        logits = self.model(vector)[0]
        return [logits[:2] if question["type"] == "noul" else logits
                for question in public["questions"].values()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_dataset(directory: Path, *, flip_test_labels=False):
    directory.mkdir(parents=True, exist_ok=True)
    split_names = ("calibration", "temperature_calibration", "development", "test", "transfer_test", "ood")
    for offset, split in enumerate(split_names):
        records = []
        for index in range(4):
            choice = index % 3
            noul = index % 2 == 0
            score = (index + 1) % 3
            if flip_test_labels and split == "test":
                choice, noul, score = (choice + 1) % 3, not noul, (score + 1) % 3
            records.append({
                "id": f"{split}-{index}",
                "state": {"vector": [0.02 * (index + 1), math.sin(index + offset),
                                       3. * (index % 3 - 1), 0.4 * (index + offset),
                                       (-1.) ** index * (index + 1)]},
                "questions": {
                    "action": {"type": "choice", "instructions": "Choose an action",
                               "criteria": {"left": "Move left", "stay": "Remain", "right": "Move right"},
                               "label": ("left", "stay", "right")[choice], "label_kind": "dataset_gold"},
                    "event": {"type": "noul", "instructions": "Will the event occur?",
                              "label": noul, "label_kind": "observed_outcome"},
                    "score": {"type": "score", "instructions": "Rate the state",
                              "criteria": ["low", "medium", "high"], "label": score,
                              "label_kind": "dataset_gold"},
                },
                "_meta": {"split": split, "group_id": f"{split}-episode-{index // 2}",
                          "source": "toy-independent-gold"},
            })
        excluded = copy.deepcopy(records[0])
        excluded["id"] = f"{split}-rejected"
        excluded["state"]["reject"] = True
        excluded["_meta"]["group_id"] = f"{split}-rejected-episode"
        records.append(excluded)
        write_records(directory / f"{split}.jsonl", records)
    (directory / "manifest.json").write_text(
        json.dumps({"fixture": "independently-labelled-toy-v1", "records_per_split": 5,
                    "splits": {path.stem: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "records": 5}
                               for path in sorted(directory.glob("*.jsonl"))}}), encoding="utf-8")


@pytest.fixture
def toy_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    make_dataset(data)
    output = tmp_path / "results" / "original"
    adapters = []

    def loader(name, **kwargs):
        assert name == "toy"
        adapter = ToyAdapter(selection_path=output / "selection.json")
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(experiment, "load_model", loader)
    summary = experiment.run_experiment("toy", data, output, device="cpu", dtype="fp32",
                                        group_size=3, activation_search=True, export=True)
    return {"data": data, "output": output, "summary": summary, "adapter": adapters[0],
            "tmp": tmp_path, "monkeypatch": monkeypatch}


def test_full_experiment_quantizes_pairs_clusters_and_reloads_export(toy_run):
    output, summary = toy_run["output"], toy_run["summary"]
    assert summary["status"] == "complete"
    assert read_json(output / "selection.json")["selected_before_quantized_test"] is True
    candidates = list((output / "candidates").glob("*-development.jsonl"))
    assert len(candidates) == 5
    assert all(row["record_id"].startswith("development-")
               for path in candidates for row in load_records(path))
    eligibility = read_json(output / "eligibility.json")
    assert len(eligibility["accepted"]["test"]) == 4
    assert len(eligibility["excluded"]["test"]) == 1
    baseline = load_records(output / "baseline" / "test.jsonl")
    expected_keys = {f"test-{index}::{qid}" for index in range(4)
                     for qid in ("action", "event", "score")}
    assert {row["key"] for row in baseline} == expected_keys
    assert {row["cluster_id"] for row in baseline} == {"test-episode-0", "test-episode-1"}
    selected = summary["selection"]["selected"]["name"]
    selected_rows = load_records(output / selected / "test.jsonl")
    for profile, result in summary["quantized"].items():
        rows = load_records(output / profile / "test.jsonl")
        assert {row["key"] for row in rows} == expected_keys
        assert result["splits"]["test"]["paired"]["n"] == 12
        assert result["splits"]["test"]["paired_accuracy_ci"]["n_clusters"] == 2
        assert result["splits"]["test"]["raw"]["n"] == 12
        assert set(result["splits"]["test"]["by_label_kind"]) == {"dataset_gold", "observed_outcome"}
        assert result["quantization"]["quantized_linear_count"] > 0
        assert result["storage"]["quantized_parameter_fraction"] < 1
    # Every context-managed candidate/final restored the actual model exactly.
    fresh = ToyAdapter()
    for name, value in toy_run["adapter"].model.state_dict().items():
        assert torch.equal(value, fresh.model.state_dict()[name]), name
    info = summary["quantized"][selected]["artifact"]
    path = Path(info["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
    assert info["bytes"] == path.stat().st_size
    records = [row for row in load_records(toy_run["data"] / "test.jsonl") if row["id"] in eligibility["accepted"]["test"]]
    with load_quantized_artifact(fresh.backbone, path):
        replay, *_ = experiment.predict(fresh, records)
    assert [row["logits"] for row in replay] == [row["logits"] for row in selected_rows]
    packed = ToyAdapter()
    head = packed.model.pointer_head
    packed.model.backbone = apply_packed(packed.backbone, path)
    assert packed.model.pointer_head is head
    assert isinstance(packed.backbone[0], PackedLinear)
    packed_rows, *_ = experiment.predict(packed, records)
    assert [row["logits"] for row in packed_rows] == [row["logits"] for row in selected_rows]
    assert packed_report(packed.model)["native_integer_gemm"] is False
    assert any(compare(baseline, load_records(output / name / "test.jsonl"))["js_to_reference"] > 0
               for name in summary["quantized"])


def test_changing_only_test_targets_never_changes_development_selection(toy_run):
    other_data = toy_run["tmp"] / "different_test_targets"
    make_dataset(other_data, flip_test_labels=True)
    output = toy_run["tmp"] / "results" / "changed_test"
    adapter = ToyAdapter(selection_path=output / "selection.json")
    toy_run["monkeypatch"].setattr(experiment, "load_model", lambda *args, **kwargs: adapter)
    other = experiment.run_experiment("toy", other_data, output, device="cpu", dtype="fp32",
                                      group_size=3, activation_search=True, export=False)
    assert other["selection"]["selected"] == toy_run["summary"]["selection"]["selected"]
    assert other["selection"]["candidates"] == toy_run["summary"]["selection"]["candidates"]
    assert any(row["quantized"] and row["split"] == "test" for row in adapter.trace)


def test_baseline_reuse_uses_cached_logits_and_restores_episode_clusters(toy_run):
    original = toy_run["output"]
    # Backward compatibility: earlier cached predictions lacked group metadata.
    for path in (original / "baseline").glob("*.jsonl"):
        rows = load_records(path)
        for row in rows:
            row.pop("cluster_id", None)
        write_records(path, rows)
    output = toy_run["tmp"] / "results" / "reuse"
    adapter = ToyAdapter(selection_path=output / "selection.json")
    toy_run["monkeypatch"].setattr(experiment, "load_model", lambda *args, **kwargs: adapter)
    reused = experiment.run_experiment("toy", toy_run["data"], output, device="cpu", dtype="fp32",
                                       group_size=3, activation_search=False, export=False,
                                       reuse_baseline=original)
    assert reused["baseline"]["reused_baseline"] == str(original)
    assert not any(row["split"] in ("test", "transfer_test", "ood") and not row["quantized"]
                   for row in adapter.trace)
    rows = load_records(output / "baseline" / "test.jsonl")
    assert {row["cluster_id"] for row in rows} == {"test-episode-0", "test-episode-1"}
    for profile in reused["quantized"].values():
        assert profile["splits"]["test"]["paired_accuracy_ci"]["n_clusters"] == 2
    assert [row["logits"] for row in rows] == [row["logits"] for row in load_records(original / "baseline" / "test.jsonl")]


@pytest.mark.parametrize("mismatch", ["revision", "dtype", "manifest", "source_revision"])
def test_baseline_reuse_rejects_changed_checkpoint_dtype_or_manifest(toy_run, mismatch):
    metadata = {"revision": "different-checkpoint"} if mismatch == "revision" else {}
    if mismatch == "dtype":
        metadata["dtype"] = "bf16"
    if mismatch == "source_revision":
        metadata["source_revision"] = "different-inference-source"
    if mismatch == "manifest":
        manifest = read_json(toy_run["data"] / "manifest.json")
        manifest["fixture"] = "a-different-manifest"
        (toy_run["data"] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    toy_run["monkeypatch"].setattr(experiment, "load_model", lambda *args, **kwargs: ToyAdapter(metadata=metadata))
    with pytest.raises(ValueError, match="Cached baseline"):
        experiment.run_experiment("toy", toy_run["data"], toy_run["tmp"] / "mismatch",
                                  device="cpu", dtype="fp32", group_size=3,
                                  activation_search=False, export=False,
                                  reuse_baseline=toy_run["output"])


def test_completed_output_directory_cannot_be_overwritten(toy_run):
    def forbidden_loader(*args, **kwargs):
        raise AssertionError("A completed run should be refused before loading another model")
    toy_run["monkeypatch"].setattr(experiment, "load_model", forbidden_loader)
    with pytest.raises(FileExistsError):
        experiment.run_experiment("toy", toy_run["data"], toy_run["output"], device="cpu")


def test_paired_comparison_rejects_missing_or_duplicate_decisions(toy_run):
    baseline = load_records(toy_run["output"] / "baseline" / "test.jsonl")
    selected = toy_run["summary"]["selection"]["selected"]["name"]
    quantized = load_records(toy_run["output"] / selected / "test.jsonl")
    with pytest.raises(ValueError, match="unique identical"):
        compare(baseline, quantized[:-1])
    with pytest.raises(ValueError, match="unique identical"):
        compare(baseline, [*quantized, quantized[0]])


def test_baseline_reuse_rejects_changed_split_bytes_even_with_unchanged_manifest(toy_run):
    path = toy_run["data"] / "test.jsonl"
    records = load_records(path)
    records[0]["state"]["vector"][0] += 0.125
    write_records(path, records)
    toy_run["monkeypatch"].setattr(experiment, "load_model", lambda *args, **kwargs: ToyAdapter())
    with pytest.raises(ValueError, match="(?i)(baseline|fingerprint|split|data)"):
        experiment.run_experiment("toy", toy_run["data"], toy_run["tmp"] / "changed_split",
                                  device="cpu", dtype="fp32", group_size=3,
                                  activation_search=False, export=False,
                                  reuse_baseline=toy_run["output"])


@pytest.mark.parametrize("leakage", ["episode", "public_request"])
def test_experiment_refuses_record_or_episode_overlap_between_splits(tmp_path, monkeypatch, leakage):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    make_dataset(data)
    calibration = load_records(data / "calibration.jsonl")
    development = load_records(data / "development.jsonl")
    if leakage == "episode":
        development[0]["_meta"]["group_id"] = calibration[0]["_meta"]["group_id"]
    else:
        # Different IDs and episodes must not hide exact public-request reuse.
        development[0]["state"] = copy.deepcopy(calibration[0]["state"])
    write_records(data / "development.jsonl", development)
    manifest = read_json(data / "manifest.json")
    manifest["splits"]["development"]["sha256"] = hashlib.sha256((data / "development.jsonl").read_bytes()).hexdigest()
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(experiment, "load_model", lambda *args, **kwargs: ToyAdapter())
    with pytest.raises(ValueError, match="(?i)(disjoint|overlap|cluster|group|leakage|share)"):
        experiment.run_experiment("toy", data, tmp_path / "overlap", device="cpu", dtype="fp32",
                                  group_size=3, activation_search=False, export=False)


@pytest.mark.parametrize("arity", ["missing", "excess"])
def test_prediction_arity_mismatch_never_silently_truncates_questions(tmp_path, arity):
    data = tmp_path / "data"
    make_dataset(data)
    record = load_records(data / "test.jsonl")[0]
    adapter = ToyAdapter()
    infer = adapter.infer
    adapter.infer = (lambda request: infer(request)[:-1]) if arity == "missing" else (lambda request: [*infer(request), torch.zeros(2)])
    with pytest.raises(ValueError, match="(?i)(output|question|count|arity)"):
        experiment.predict(adapter, [record])


def test_exports_from_equal_basename_run_directories_never_overwrite_each_other(toy_run):
    # Two versioned run directories may intentionally share a model basename.
    monkeypatch = toy_run["monkeypatch"]
    summaries = []
    for version, bits in (("v1", 4), ("v2", 8)):
        output = toy_run["tmp"] / version / "toy"
        adapter = ToyAdapter(selection_path=output / "selection.json")
        monkeypatch.setattr(experiment, "load_model", lambda *args, _adapter=adapter, **kwargs: _adapter)
        summaries.append(experiment.run_experiment("toy", toy_run["data"], output, device="cpu", dtype="fp32",
                                                   bits=bits, group_size=3, activation_search=False, export=True,
                                                   reuse_baseline=toy_run["output"]))
    artifacts = [summary["quantized"][summary["selection"]["selected"]["name"]]["artifact"]
                 for summary in summaries]
    assert Path(artifacts[0]["path"]).resolve() != Path(artifacts[1]["path"]).resolve()
    for artifact in artifacts:
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_legacy_baseline_without_explicit_fingerprints_uses_verified_manifest(toy_run):
    environment_path = toy_run["output"] / "environment.json"
    legacy_environment = read_json(environment_path)
    legacy_environment.pop("split_files_sha256")
    environment_path.write_text(json.dumps(legacy_environment), encoding="utf-8")
    output = toy_run["tmp"] / "results" / "legacy_reuse"
    adapter = ToyAdapter(selection_path=output / "selection.json")
    toy_run["monkeypatch"].setattr(experiment, "load_model", lambda *args, **kwargs: adapter)
    summary = experiment.run_experiment("toy", toy_run["data"], output, device="cpu", dtype="fp32",
                                        group_size=3, activation_search=False, export=False,
                                        reuse_baseline=toy_run["output"])
    assert summary["status"] == "complete"
    environment = read_json(output / "environment.json")
    assert set(environment["split_files_sha256"]) == {path.stem for path in toy_run["data"].glob("*.jsonl")}
    assert not any(row["split"] == "test" and not row["quantized"] for row in adapter.trace)


def test_legacy_baseline_with_no_binding_manifest_is_rejected(toy_run):
    environment_path = toy_run["output"] / "environment.json"
    legacy_environment = read_json(environment_path)
    legacy_environment.pop("split_files_sha256")
    legacy_environment["data_manifest_sha256"] = None
    environment_path.write_text(json.dumps(legacy_environment), encoding="utf-8")
    (toy_run["data"] / "manifest.json").unlink()
    toy_run["monkeypatch"].setattr(experiment, "load_model", lambda *args, **kwargs: ToyAdapter())
    with pytest.raises(ValueError, match="(?i)(baseline|manifest|binding)"):
        experiment.run_experiment("toy", toy_run["data"], toy_run["tmp"] / "unbound_legacy",
                                  device="cpu", dtype="fp32", group_size=3,
                                  activation_search=False, export=False,
                                  reuse_baseline=toy_run["output"])


def test_rtn_fallback_uses_the_same_development_objective_and_storage_tiebreak():
    def candidate(name, method, objective, bytes_):
        return {"profile": {"name": name, "method": method},
                "paired": {"selection_objective": objective},
                "storage": {"estimated_complete_packed_parameters_bytes": bytes_}}

    cases = [candidate("rtn", "rtn", 0.01, 100),
             candidate("local", "s1q", 0.02, 90),
             candidate("v2", "s1q2", 0.02, 80)]
    assert experiment.select_candidate(cases)["profile"]["name"] == "v2"
    assert experiment.select_candidate(cases, include_rtn_candidate=True)["profile"]["name"] == "rtn"
    cases[0]["paired"]["selection_objective"] = 0.02
    assert experiment.select_candidate(cases, include_rtn_candidate=True)["profile"]["name"] == "v2"


def test_accuracy_first_selection_can_fall_back_to_rtn_and_uses_probability_tiebreaks():
    def candidate(name, method, accuracy, nll, brier):
        return {"profile": {"name": name, "method": method},
                "development": {"raw": {"accuracy": accuracy, "nll": nll, "brier": brier}},
                "paired": {"selection_objective": 0.01},
                "storage": {"estimated_complete_packed_parameters_bytes": 100}}

    cases = [candidate("rtn", "rtn", 0.75, 0.7, 0.4),
             candidate("local", "s1q", 0.7, 0.6, 0.3),
             candidate("v2", "s1q2", 0.7, 0.5, 0.35)]
    assert experiment.select_candidate(cases, include_rtn_candidate=True,
                                       policy="accuracy_first")["profile"]["name"] == "rtn"
    assert experiment.select_candidate(cases, policy="accuracy_first")["profile"]["name"] == "v2"
    cases[2]["development"]["raw"]["accuracy"] = 0.8
    assert experiment.select_candidate(cases, include_rtn_candidate=True,
                                       policy="accuracy_first")["profile"]["name"] == "v2"
    with pytest.raises(ValueError, match="selection policy"):
        experiment.select_candidate(cases, policy="unknown")
