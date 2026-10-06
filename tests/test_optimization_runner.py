"""Runner evidence contracts: fixed admission, multiple cohorts, no test access."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch import nn

import s1q.optimization as runner
from s1q.models import UnsupportedRecord


class TinyAdapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.metadata = {"name": "fixture-decision", "revision": "fixture", "dtype": "fp32"}
        self.model = nn.Module()
        self.model.backbone = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
        self.model.decision_head = nn.Linear(3, 2)
        self.backbone = self.model.backbone
        self.in_ordinary_calibration = False
        self.calibration_ids = []

    def infer(self, record):
        if record["state"].get("reject"):
            raise UnsupportedRecord("fixture native rejection")
        if self.in_ordinary_calibration:
            assert all("label" not in question and "target" not in question
                       for question in record["questions"].values())
            self.calibration_ids.append(record["id"])
        features = torch.tensor(record["state"]["features"], dtype=torch.float32)[None]
        return [self.model.decision_head(self.model.backbone(features))[0]]


def records(kind, count, offset, *, rejection=True):
    return [{"id": f"{kind}-{index}",
             "_meta": {"group_id": f"{kind}-group-{index}", "source": kind},
             "state": {"features": [offset + index / 8, (-1) ** index, index / 5, .25],
                       "reject": rejection and index == count - 1},
             "questions": {"decision": {"type": "choice", "criteria": {"a": "A", "b": "B"},
                 "label": "a" if index % 2 == 0 else "b", "target": "must be stripped"}}}
            for index in range(count)]


def save_records(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


def fixture_arguments(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    save_records(data / "calibration.jsonl", records("calibration", 6, 1))
    save_records(data / "development.jsonl", records("development", 4, 10))
    extra = tmp_path / "extra.jsonl"
    save_records(extra, records("extra", 3, 20))
    (data / "test.jsonl").write_text("POISON: final test must never be parsed", encoding="utf-8")
    return Namespace(output_dir=str(tmp_path / "result"), methods="rtn,s1q-mac,gptq-mac,s1q-mac-repair",
        model="fixture", data_dir=str(data), device="cpu", dtype="fp32", source_dir=None,
        checkpoint_dir=None, checkpoint_manifest="fixture-manifest.json", calibration_count=6,
        development_count=4, reservoir_size=8, seed=20261004, bits=4, activation_bits=4,
        group_size=2, extra_evaluation=[f"extra={extra}"], cohort="fixture_exploratory",
        repair_records=3, repair_epochs=1)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_named_extra_admission_label_stripping_and_poison_final_test(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path)
    adapter = TinyAdapter()
    output = Path(args.output_dir)
    margin_ids = []
    original_calibration = runner.CalibrationCollector
    original_margin = runner.DecisionMarginCollector
    original_load = runner.load_records
    loaded_paths = []

    class OrdinarySpy(original_calibration):
        def start(self):
            adapter.in_ordinary_calibration = True
            return super().start()

        def close(self):
            super().close()
            adapter.in_ordinary_calibration = False

    class MarginSpy(original_margin):
        def collect(self, record):
            assert all("label" not in question and "target" not in question
                       for question in record["questions"].values())
            margin_ids.append(record["id"])
            return super().collect(record)

    def model_loader(*args, **kwargs):
        # Configuration is persisted before even native evaluation starts.
        frozen = json.loads((output / "frozen-config-before-evaluation.json").read_text(encoding="utf-8"))
        assert frozen["methods"] == "rtn,s1q-mac,gptq-mac,s1q-mac-repair"
        assert kwargs["checkpoint_manifest"] == "fixture-manifest.json"
        return adapter

    def tracked_load(path):
        loaded_paths.append(Path(path).resolve())
        assert Path(path).name != "test.jsonl"
        return original_load(path)

    monkeypatch.setattr(runner, "CalibrationCollector", OrdinarySpy)
    monkeypatch.setattr(runner, "DecisionMarginCollector", MarginSpy)
    monkeypatch.setattr(runner, "load_model", model_loader)
    monkeypatch.setattr(runner, "load_records", tracked_load)
    original_weights = {name: value.detach().clone() for name, value in adapter.model.state_dict().items()}
    report = runner.run(args)
    expected_calibration = {f"calibration-{index}" for index in range(5)}
    assert set(adapter.calibration_ids) == set(margin_ids) == expected_calibration
    assert set(report["identity"]["accepted_calibration_ids"]) == expected_calibration
    assert len(report["identity"]["calibration_exclusions"]) == 1
    assert report["identity"]["final_test_read"] is False
    assert report["native_development"]["raw"]["n"] == 3
    assert report["native_evaluations"]["extra"]["raw"]["n"] == 2
    assert len(loaded_paths) == 3
    assert report["methodology"]["evaluation_used_for_configuration_selection"] is False
    assert report["decision_margin"]["uses_gold_labels"] is False
    assert report["methods"]["s1q-mac-repair"]["optimization"]["uses_gold_labels"] is False
    for method in args.methods.split(","):
        result = report["methods"][method]
        assert result["status"] == "complete", result.get("error")
        assert result["quantization"]["quantized_linear_count"] == report["identity"]["selected_linear_count"] == 2
        assert result["development"]["raw"]["n"] == 3
        assert result["evaluations"]["extra"]["raw"]["n"] == 2
        for cohort in ("development", "extra"):
            native = read_rows(output / f"native-{cohort}.jsonl")
            quantized = read_rows(output / f"{method}-{cohort}.jsonl")
            assert {row["key"] for row in native} == {row["key"] for row in quantized}
            assert {row["record_id"] for row in quantized} == set(report["identity"]["accepted_evaluation_ids"][cohort])
    for name, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)


def test_source_group_reuse_in_extra_cohort_rejected_before_quantized_eval(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path)
    reused = records("calibration", 1, 100, rejection=False)
    reused[0]["id"] = "extra-unique-record-id"
    save_records(tmp_path / "extra.jsonl", reused)
    monkeypatch.setattr(runner, "load_model", lambda *a, **k: TinyAdapter())
    with pytest.raises(ValueError, match="Dataset leakage"):
        runner.run(args)
    output = Path(args.output_dir)
    assert (output / "frozen-config-before-evaluation.json").exists()
    assert not list(output.glob("native-*.jsonl"))
    assert not list(output.glob("s1q-mac-*.jsonl"))
