"""Candidate mapping, autograd and provenance gates for symbol-readout models."""
from types import SimpleNamespace
import json
import hashlib

import pytest
import torch
from torch import nn

from s1q.models import (DecisionAdapter, InternDecisionAdapter, StartLuxDecisionAdapter,
                        MODEL_REVISIONS, UnsupportedRecord, _decision_checkpoint,
                        _intern_native_module)


def record():
    return {"state": "A refund is required", "questions": {
        "route": {"type": "choice", "instructions": "Pick", "criteria": {"Z": "billing", "A": "other"}},
        "yes": {"type": "noul", "instructions": "Refund?"},
        "rating": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]}}}


def test_startlux_undecorated_core_preserves_grad_and_canonical_noul_order():
    adapter = StartLuxDecisionAdapter.__new__(StartLuxDecisionAdapter)
    DecisionAdapter.__init__(adapter, "startlux-fixture", "cpu", "fp32")
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2., 0.], [0., -1.]]))
    calls = []

    class Engine:
        @torch.no_grad()
        def _logits(self, rows):
            calls.append(torch.is_grad_enabled())
            return [layer(torch.tensor([1., 2.])) for _ in rows], 10

    def make_row(state, question, qid):
        typ = question["type"]
        keys = ["true", "false"] if typ == "noul" else list(question["criteria"]) if typ == "choice" else ["0", "1"]
        return {"type": typ, "options": [{"id": key} for key in keys]}

    adapter.native = SimpleNamespace(from_systemone=make_row,
                        validate=lambda row: [option["id"] for option in row["options"]])
    adapter.engine = Engine()
    logits = adapter.infer(record())
    assert calls == [True]
    torch.testing.assert_close(logits[0], torch.tensor([2., -2.]))
    torch.testing.assert_close(logits[1], torch.tensor([-2., 2.]))
    logits[1][1].backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_intern_reads_pre_marker_positions_not_last_token_and_preserves_field_order():
    adapter = InternDecisionAdapter.__new__(InternDecisionAdapter)
    DecisionAdapter.__init__(adapter, "intern-fixture", "cpu", "fp32")
    layer = nn.Linear(2, 4, bias=False)
    observed = {}

    class Batch(dict):
        def to(self, device):
            return self

    class Model(nn.Module):
        def forward(self, input_ids, use_cache, logits_to_keep):
            observed["positions"] = logits_to_keep.tolist()
            states = torch.stack((input_ids[0].float(), input_ids[0].float() + 1), -1)
            return SimpleNamespace(logits=layer(states[logits_to_keep])[None])

    adapter.model = Model()
    adapter.native = SimpleNamespace(validate_request=lambda request: request,
        _options=lambda q: [(key, key) for key in q["criteria"]] if q["type"] == "choice"
                           else [("no", "no"), ("yes", "yes")] if q["type"] == "noul"
                           else [("0", "low"), ("1", "high")])
    compiled = SimpleNamespace(fields=("route", "yes", "rating"),
                    symbols={"route": ("A", "B"), "yes": ("A", "B"), "rating": ("A", "B")})
    batch = Batch(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]]))
    adapter.engine = SimpleNamespace(backend=SimpleNamespace(encode=lambda row: (compiled, batch, torch.tensor([1, 3, 5]))),
                    tokenizer=SimpleNamespace(encode=lambda symbol, add_special_tokens: [0 if symbol == "A" else 2]))
    logits = adapter.infer(record())
    assert observed["positions"] == [1, 3, 5]
    assert all(z.shape == (2,) and z.requires_grad for z in logits)
    logits[2][0].backward()
    assert layer.weight.grad.abs().sum() > 0


def test_extension_requires_native_source_covered_by_matching_manifest(tmp_path):
    key = "intern-decision-0.8b"
    repo, revision = MODEL_REVISIONS[key]
    path = tmp_path / "inference.py"
    path.write_text("# authenticated native compiler\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "checkpoint-manifest.json"
    manifest.write_text(json.dumps({"repo_id": repo, "revision": revision, "files": {}}))
    with pytest.raises(ValueError, match="native inference source"):
        _decision_checkpoint(key, checkpoint_dir=tmp_path)
    manifest.write_text(json.dumps({"repo_id": repo, "revision": revision, "files": {"inference.py": sha}}))
    _, actual_repo, actual_revision, provenance = _decision_checkpoint(key, checkpoint_dir=tmp_path)
    assert (actual_repo, actual_revision) == (repo, revision)
    assert provenance["sha256_verified_files"] == {"inference.py": sha}
    path.write_text("# changed source\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _decision_checkpoint(key, checkpoint_dir=tmp_path)


def test_extension_manifest_cannot_claim_other_model_or_unpinned_revision(tmp_path):
    key = "intern-decision-0.8b"
    manifest = tmp_path / "checkpoint-manifest.json"
    manifest.write_text(json.dumps({"repo_id": "other/model", "revision": MODEL_REVISIONS[key][1], "files": {}}))
    with pytest.raises(ValueError, match="repo/revision"):
        _decision_checkpoint(key, checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match="remain pinned"):
        _decision_checkpoint(key, revision="main")


def test_intern_quote_compatibility_keeps_literal_values_and_source_bytes(tmp_path):
    path = tmp_path / "native.py"
    source = """row={'id': 'r1'}
field='route'; field_name='route'; question={'instructions': 'Choose team'}
outputs=[f'{row.get('id', '<missing id>')}: questions must be a non-empty object',
 f'{row.get('id', '<missing id>')}: question {field!r} is not an object',
 f'{row.get('id', '<missing id>')}: question {field!r} has no options',
 f'{field_name}: {question.get('instructions', '')}']
"""
    path.write_text(source, encoding="utf-8")
    before = path.read_bytes()
    module = _intern_native_module(path, "fixture-quote-compat")
    assert module.outputs == ["r1: questions must be a non-empty object", "r1: question 'route' is not an object",
                              "r1: question 'route' has no options", "route: Choose team"]
    assert path.read_bytes() == before
    assert not module.__s1q_compat__["source_file_modified"]
    assert module.__s1q_compat__["publisher_source_sha256"] == hashlib.sha256(before).hexdigest()
