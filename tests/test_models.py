"""Adapter contract tests: no weights, downloads or GPU are used.

Native schema tests run when pinned upstream sources have been acquired with
scripts/setup_sources.py. They exercise the real upstream token builders and
mock only the expensive neural forward pass.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from s1q.models import (DecisionAdapter, KevAdapter, LayaAdapter, NanoJevAdapter,
                        SOURCE_REVISIONS, UnsupportedRecord, _import_source,
                        _module_file, _source, public_request, question_keys,
                        verify_source_revision)


def request():
    return {"id": "r1", "state": "refund requested", "gold": {"flag": True},
            "questions": {
                "flag": {"type": "noul", "instructions": "Refund requested?",
                         "criteria": {"true": "A refund is requested", "false": "No refund is requested"},
                         "label": True, "target": {"true": 1.0}, "src": "secret-label-source"},
                "route": {"type": "choice", "instructions": "Choose a team",
                          "criteria": {"B": "billing", "A": "delivery"}, "label": "A"},
                "urgency": {"type": "score", "instructions": "Rate urgency",
                            "criteria": ["low", "high"], "label": 1}}}


class CaptureAdapter(DecisionAdapter):
    def __init__(self, output=None):
        super().__init__("fixture", "cpu", "fp32")
        self.output = output

    def _infer(self, record):
        self.seen = record
        return self.output or [torch.tensor([2.0, -1.0]) for _ in record["questions"]]


def test_strip_all_labels_preserve_question_and_option_order():
    original = request()
    snapshot = copy.deepcopy(original)
    adapter = CaptureAdapter()
    adapter.infer(original)
    assert original == snapshot
    assert set(adapter.seen) == {"state", "questions"}
    assert list(adapter.seen["questions"]) == ["flag", "route", "urgency"]
    assert list(adapter.seen["questions"]["route"]["criteria"]) == ["B", "A"]
    for q in adapter.seen["questions"].values():
        assert not {"label", "target", "src"}.intersection(q)
    assert adapter.stats == {"attempted": 1, "accepted": 1, "rejected": 0}


def test_boolean_alias_and_canonical_false_true_order():
    record = request()
    record["questions"]["flag"]["type"] = "boolean"
    q = public_request(record)["questions"]["flag"]
    assert q["type"] == "noul"
    # The criteria mapping is deliberately supplied true before false.
    assert question_keys(q) == ["false", "true"]
    assert question_keys(record["questions"]["route"]) == ["B", "A"]
    assert question_keys(record["questions"]["urgency"]) == ["0", "1"]


@pytest.mark.parametrize("values", [[torch.tensor([float("nan"), 0.0])],
                                   [torch.tensor([1.0])], []])
def test_invalid_native_outputs_count_as_rejected(values):
    record = request()
    record["questions"] = {"flag": record["questions"]["flag"]}
    adapter = CaptureAdapter()
    adapter._infer = lambda _: values
    with pytest.raises(RuntimeError):
        adapter.infer(record)
    assert adapter.stats == {"attempted": 1, "accepted": 0, "rejected": 1}


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_no_invalid_temperature(temperature):
    adapter = CaptureAdapter()
    with pytest.raises(ValueError, match="finite and positive"):
        adapter.probs(request(), temperature)


def test_archive_source_provenance_is_required_and_exact(tmp_path):
    with pytest.raises(RuntimeError, match="provenance missing"):
        verify_source_revision(tmp_path, "kev")
    marker = tmp_path / "SOURCE_REVISION.txt"
    marker.write_text("0" * 40)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        verify_source_revision(tmp_path, "kev")
    marker.write_text(SOURCE_REVISIONS["kev"] + "\n")
    assert verify_source_revision(tmp_path, "kev") == SOURCE_REVISIONS["kev"]


def test_git_source_drift_is_rejected_before_import(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def git_result(command, **kwargs):
        stdout = SOURCE_REVISIONS["kev"] if "rev-parse" in command else " M kev/api.py"
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr("s1q.models.subprocess.run", git_result)
    with pytest.raises(RuntimeError, match="tracked modifications"):
        verify_source_revision(tmp_path, "kev")


@pytest.fixture
def native_sources():
    root = Path(__file__).resolve().parents[1]
    result = {}
    for family in SOURCE_REVISIONS:
        candidates = [root / "work/upstream" / family, root / "work/model-audit" / family]
        path = next((p for p in candidates if p.is_dir()), None)
        if path is None:
            pytest.skip("Native adapter tests need python scripts/setup_sources.py")
        result[family] = _source(path, family)
    return result


class TinyTokenizer:
    """Character tokenizer makes native budget/collision tests deterministic."""
    cls_token_id, sep_token_id, eos_token_id = 1, 2, 2
    pad_token_id, mask_token_id, mask_token = 0, 3, "[MASK]"

    def encode(self, text, **kwargs):
        ids = [ord(char) + 10 for char in text]
        return ids[:kwargs["max_length"]] if kwargs.get("truncation") else ids

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text, **kwargs)}


def test_kev_native_options_and_gold_independence(native_sources):
    pytest.importorskip("pydantic")
    api = _module_file(native_sources["kev"] / "kev/api.py", "s1q_test_kev_api")
    adapter = KevAdapter.__new__(KevAdapter)
    DecisionAdapter.__init__(adapter, "kev-0.8b", "cpu", "fp32")
    adapter._request_class, adapter._to_record = api.SystemOneRequest, api.to_record
    adapter.context = {"max_state": 384, "max_branch": 1024, "max_packed": 2048}
    adapter.tokenizer = None

    class NativeForward:
        def encode(self, tokenizer, rec, **kwargs):
            self.record, self.kwargs = rec, kwargs
            return {"ids": [1]}

        def forward(self, enc):
            return [torch.tensor([2.0, -1.0]) for _ in self.record["questions"]]

    adapter.model = NativeForward()
    out = adapter.infer(request())
    rec = adapter.model.record
    assert rec["questions"][0]["options"] == ["no: No refund is requested", "yes: A refund is requested"]
    assert rec["questions"][1]["options"] == ["B: billing", "A: delivery"]
    assert rec["questions"][2]["options"] == ["low", "high"]
    assert adapter.model.kwargs["strict"] is True
    swapped_gold = request()
    swapped_gold["questions"]["flag"]["label"] = False
    swapped_gold["questions"]["route"]["label"] = "B"
    changed = adapter.infer(swapped_gold)
    assert adapter.model.record == rec
    assert all(torch.equal(a, b) for a, b in zip(out, changed))
    assert torch.softmax(out[0], -1)[0] > torch.softmax(out[0], -1)[1]


def make_laya(native_sources, *, max_len=1024, head_max_len=192):
    _import_source(native_sources["laya"])
    from laya.agent import Agent
    from laya.common import collate_items
    adapter = LayaAdapter.__new__(LayaAdapter)
    DecisionAdapter.__init__(adapter, "laya", "cpu", "fp32")
    agent = Agent.__new__(Agent)
    agent.tok = TinyTokenizer()
    agent.cfg = {}
    adapter.agent, adapter.tokenizer = agent, agent.tok
    adapter.max_len, adapter.head_max_len, adapter.reject_truncation = max_len, head_max_len, True
    adapter._collate_items = collate_items

    def raw_forward(*args):
        adapter.seen_batch = args
        width = args[2].shape[1]
        values = torch.full((args[0].shape[0], width), -10.0)
        values[:, 0], values[:, 1] = 2.0, -1.0
        return values, torch.zeros(args[0].shape[0], 2)

    adapter.model = raw_forward
    return adapter


def test_laya_native_boolean_order_and_raw_logits(native_sources):
    adapter = make_laya(native_sources)
    from laya.common import render_options
    q = adapter.agent._to_internal(public_request(request())["questions"]["flag"])
    assert render_options(q) == ["false: No refund is requested", "true: A refund is requested"]
    choice = adapter.agent._to_internal(public_request(request())["questions"]["route"])
    assert render_options(choice) == ["B: billing", "A: delivery"]
    logits = adapter.infer(request())
    assert len(logits) == 3
    assert logits[0].tolist() == [2.0, -1.0]
    # Native qtype order is noul, choice, score, matching request insertion order.
    assert adapter.seen_batch[-1].tolist() == [2, 0, 1]


def test_laya_rejects_state_truncation_and_candidate_collision(native_sources):
    adapter = make_laya(native_sources, max_len=128)
    record = request()
    record["state"] = "x" * 4096
    with pytest.raises(UnsupportedRecord, match="truncates"):
        adapter.infer(record)
    adapter = make_laya(native_sources, max_len=128, head_max_len=16)
    record = {"state": "s", "questions": {"pick": {"type": "choice", "instructions": "pick",
              "criteria": {"same-prefix-A": "one", "same-prefix-B": "two"}}}}
    with pytest.raises(UnsupportedRecord, match="collapses"):
        adapter.infer(record)


def make_nano(native_sources, max_len=1024):
    predictor = _module_file(native_sources["NanoJev"] / "scripts/predict_toy_decisions.py",
                             "s1q_test_nano_predictor")
    adapter = NanoJevAdapter.__new__(NanoJevAdapter)
    DecisionAdapter.__init__(adapter, "nanojev", "cpu", "fp32")
    adapter._prepare, adapter.tokenizer, adapter.max_len = predictor.prepare_examples, TinyTokenizer(), max_len

    def raw_forward(examples, pad):
        adapter.seen_examples = examples
        # Native Boolean's reference negative logit is exactly zero.
        return torch.tensor([[0.0, 1.5] if x["type"] == "boolean" else [2.0, -1.0]
                             for x in examples]), None

    adapter.model = raw_forward
    return adapter


def test_nano_native_boolean_reference_and_candidate_order(native_sources):
    adapter = make_nano(native_sources)
    logits = adapter.infer(request())
    examples = adapter.seen_examples
    assert [x["qid"] for x in examples] == ["flag", "route", "urgency"]
    assert examples[0]["candidate_ids"] == ["false", "true"]
    assert len(examples[0]["leaf_tokens"]) == 1
    assert examples[1]["candidate_ids"] == ["B", "A"]
    assert examples[1]["candidate_texts"] == ["B: billing", "A: delivery"]
    assert examples[2]["candidate_ids"] == ["0", "1"]
    assert logits[0].tolist() == [0.0, 1.5]
    assert all(not {"label", "gold", "target"}.intersection(x) for x in examples)


def test_nano_actual_tiny_forward_uses_zero_negative_logit(native_sources):
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    predictor = _module_file(native_sources["NanoJev"] / "scripts/predict_toy_decisions.py",
                             "s1q_test_nano_tiny_predictor")

    class TinyBackbone(torch.nn.Module):
        config = SimpleNamespace(hidden_size=4)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 4))

    model = predictor.load_decision_model_class()(TinyBackbone(), "none").eval()
    with torch.no_grad():
        model.scalar.weight.zero_()
        model.scalar.bias.fill_(1.5)
    record = public_request(request())
    record["questions"] = {"flag": {**record["questions"]["flag"], "type": "boolean"}}
    payload = {"states": [{"id": "fixture", **record}]}
    examples = predictor.prepare_examples(payload, TinyTokenizer(), 1024)
    with torch.inference_mode():
        logits, valid = model(examples, 0)
    assert valid.tolist() == [[True, True]]
    assert logits.tolist() == [[0.0, 1.5]]
    assert torch.softmax(logits, -1)[0, 1] == pytest.approx(torch.sigmoid(torch.tensor(1.5)).item())


def test_nano_never_silently_truncates(native_sources):
    adapter = make_nano(native_sources, max_len=32)
    with pytest.raises(ValueError):
        adapter.infer(request())
    assert not hasattr(adapter, "seen_examples")
    assert adapter.stats["rejected"] == 1


def test_nano_rejects_null_descriptions_instead_of_rewriting(native_sources):
    adapter = make_nano(native_sources)
    record = request()
    record["questions"]["route"]["criteria"]["B"] = None
    with pytest.raises(ValueError):
        adapter.infer(record)
