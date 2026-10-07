"""Tests for benchmark isolation and explicit native target conversion."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_expanded_benchmarks", ROOT / "scripts/prepare_expanded_benchmarks.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def record(identity="sample/0", group="sample/group", label=True, instructions="Is this allowed?"):
    return {"state": "A source scenario.", "questions": {"decision": {
        "type": "noul", "instructions": instructions, "label": label}},
        "_meta": {"id": identity, "group_id": group, "source": "boolq", "variant": "clean",
                  "benchmark_upstream_partition": "test", "benchmark_upstream_file": "test.jsonl"}}


def test_gold_changes_cannot_evade_request_quarantine():
    blocked = PREP.registry([record(label=True)])
    variant = record(identity="changed", group="changed", label=False)
    assert PREP.digest(PREP.model_request(variant)) in blocked["requests"]
    assert PREP.overlaps(variant, blocked) == "states"


def test_shared_state_and_source_group_block_question_variants():
    blocked = PREP.registry([record()])
    assert PREP.overlaps(record(identity="variant", instructions="Alternative wording"), blocked) == "groups"
    variant = record(identity="variant", group="different", instructions="Alternative wording")
    assert PREP.overlaps(variant, blocked) == "states"


def test_typed_teacher_label_stays_auxiliary_and_is_not_model_input():
    source = {"id": "typed0", "state": {"task": "Review an agent trace."},
              "questions": {"check": {"type": "noul", "instructions": "Review needed?"}},
              "targets": {"check": {"label": "no", "label_origin": "teacher_agreement",
                                      "probabilities": {"yes": 0.1, "no": 0.9}}}, "images": []}
    converted = PREP.convert_intern_record(source, "typed-agreement", "typed_decisions/test.jsonl")
    assert converted["questions"]["check"]["label"] is False
    assert converted["questions"]["check"]["label_kind"] == "teacher_agreement"
    assert converted["_meta"]["ranking_eligible"] is False
    assert PREP.model_request(converted) == {"state": source["state"], "questions": source["questions"]}


def test_unrecognized_boolean_label_is_rejected():
    source = {"id": "bad", "state": "State", "questions": {"decision": {"type": "noul"}},
              "targets": {"decision": {"label": "maybe", "label_origin": "native_annotation"}}}
    with pytest.raises(ValueError, match="Unsupported Boolean"):
        PREP.convert_intern_record(source, "wildjailbreak", "wildjailbreak/test.jsonl")


def test_whole_group_quarantine_removes_all_questions(tmp_path):
    blocked = PREP.registry([record(identity="old", group="other")])
    linked = record(identity="new", group="linked")
    # The second request differs in evidence, but belongs to the same source group.
    other = record(identity="new2", group="linked")
    other["state"] = "Different evidence in the same scenario."
    manifest = PREP.freeze(tmp_path / "freeze", [linked, other], blocked, [], [],
                           seed=20261007, count=128, benchmark_id="test")
    assert manifest["datasets"] == []
    assert manifest["empty_datasets"][0]["excluded"] == {"overlap_with_reserved_states": 2}


def test_frozen_cohort_cannot_be_overwritten(tmp_path):
    output = tmp_path / "frozen"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        PREP.freeze(output, [], PREP.registry([]), [], [], seed=1, count=1, benchmark_id="test")
