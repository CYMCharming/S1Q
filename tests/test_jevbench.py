"""Offline gold-semantics and source-integrity tests for external JevBench data."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from s1q.data import load_records, model_request, write_records


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_jevbench.py"
SPEC = importlib.util.spec_from_file_location("s1q_prepare_jevbench", SCRIPT)
JEV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JEV)
REVISION = "a" * 40


def task(identity="scenario-1", *, kind="noul", expected="yes", family="policy"):
    criteria, labels = {"false": "Condition absent", "true": "Condition met"}, ["no", "yes"]
    if kind == "choice":
        criteria, labels = {"b": "Second handler", "a": "First handler"}, ["a", "b"]
    elif kind == "score":
        criteria, labels = ["No impairment", "Workaround", "Core outage"], ["0", "1", "2"]
    return {"id": identity, "family": family, "state": "Public observed facts.",
            "question": {"type": kind, "instructions": "Apply the stated rubric.", "criteria": criteria},
            "labels": labels, "expected": expected, "split": "public", "group": identity,
            "provenance": {"license": "MIT", "label_basis": "Reviewed before inference",
                           "rationale": "Private gold explanation", "exclude_reason": None}}


class JevBenchConversionTests(unittest.TestCase):
    def test_boolean_preserves_request_and_removes_rationale(self):
        row = task(expected="no")
        out = JEV.convert_task(row, tier="original", revision=REVISION)
        self.assertIs(out["questions"]["decision"]["label"], False)
        self.assertEqual(out["_meta"]["source_original_id"], row["id"])
        request = model_request(out)
        self.assertEqual(request["state"], row["state"])
        self.assertEqual(request["questions"]["decision"], row["question"])
        self.assertNotIn("Private gold explanation", json.dumps(request))
        self.assertNotIn("expected", json.dumps(request))

    def test_choice_order_follows_public_criteria_not_label_metadata(self):
        row = task(kind="choice", expected="a")
        out = JEV.convert_task(row, tier="original", revision=REVISION)
        self.assertEqual(list(out["questions"]["decision"]["criteria"]), ["b", "a"])
        self.assertEqual(out["questions"]["decision"]["label"], "a")
        self.assertEqual(out["_meta"]["source_labels"], ["a", "b"])

    def test_ordinal_target_is_an_integer_level_and_not_a_probability(self):
        row = task(kind="score", expected=2, family="ordinal")
        out = JEV.convert_task(row, tier="original", revision=REVISION)
        self.assertEqual(out["questions"]["decision"]["label"], 2)
        row["expected"] = 2.0
        with self.assertRaisesRegex(ValueError, "unrecognized_ordinal_target"):
            JEV.convert_task(row, tier="original", revision=REVISION)

    def test_probability_gold_is_separate_from_modal_label(self):
        row = task(family="probability")
        row["provenance"]["gold_probs"] = {"no": 0.3125, "yes": 0.6875}
        out = JEV.convert_task(row, tier="hard", revision=REVISION)
        q = out["questions"]["decision"]
        self.assertIs(q["label"], True)
        self.assertEqual(q["label_kind"], "authored_probability_mode")
        self.assertEqual(q["target_distribution"], {"false": 0.3125, "true": 0.6875})
        self.assertNotIn("target_distribution", model_request(out)["questions"]["decision"])
        row["expected"] = "no"
        with self.assertRaisesRegex(ValueError, "probability_mode_expected_mismatch"):
            JEV.convert_task(row, tier="hard", revision=REVISION)

    def test_unknown_or_unlabelled_tasks_are_rejected_without_pseudo_gold(self):
        mutations = [({"expected": None}, "missing_explicit_expected"),
                     ({"split": "private"}, "not_public"),
                     ({"question": {"type": "sampler", "instructions": "Sample an outcome"}}, "unsupported_question_type")]
        for mutation, reason in mutations:
            row = task()
            row.update(mutation)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                JEV.convert_task(row, tier="original", revision=REVISION)
        row = task()
        row["provenance"]["license"] = None
        with self.assertRaisesRegex(ValueError, "license_not_verified_mit"):
            JEV.convert_task(row, tier="original", revision=REVISION)

    def test_preparation_is_external_only_and_exclusions_are_auditable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "source", root / "prepared"
            first, paraphrase = task("pair-0"), task("pair-1")
            first["group"] = paraphrase["group"] = "pair"
            paraphrase["state"] = "Same facts expressed differently."
            hard = task("hard-short", kind="choice", expected="a")
            long = deepcopy(hard)
            long.update(id="hard-long", group="hard-long", state="length" * 1000)
            config = {"repo": "fstandhartinger/jevbench", "revision": REVISION,
                      "source_license": "MIT", "files": {}, "label_provenance": {},
                      "upstream_scoring_contract": {}}
            for name, rows in zip(JEV.DATA_FILES, ([first, paraphrase], [hard, long])):
                payload = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                config["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest(),
                                         "bytes": len(payload), "records": len(rows)}
            payload = b"MIT fixture notice\n"
            (source / "LICENSE").write_bytes(payload)
            config["files"]["LICENSE"] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = JEV.prepare_jevbench(output, source_dir=source, checksum_config=config_path)
            self.assertEqual(manifest["role"], "external_test_only")
            self.assertEqual(manifest["accepted_records"], 3)
            self.assertEqual(manifest["accepted_groups"], 2)
            self.assertEqual(manifest["rejected"][0]["id"], "hard-long")
            self.assertEqual(manifest["rejection_counts"], {"request_too_long_chars": 1})
            self.assertEqual({path.name for path in output.glob("*.jsonl")}, {"external_test.jsonl"})
            self.assertEqual((output / "UPSTREAM_LICENSE.txt").read_bytes(), payload)
            overlap_path = root / "calibration.jsonl"
            overlap = load_records(output / "external_test.jsonl")[0]
            overlap["_meta"]["group_id"] = "other-episode"
            overlap["_meta"]["id"] = "other-id"
            write_records(overlap_path, [overlap])
            with self.assertRaisesRegex(ValueError, "Dataset leakage"):
                JEV.prepare_jevbench(root / "failed", source_dir=source, checksum_config=config_path,
                                     disjoint_with=[overlap_path])
            (source / JEV.DATA_FILES[0]).write_bytes(b"corrupted\n")
            with self.assertRaisesRegex(ValueError, "source integrity failure"):
                JEV.prepare_jevbench(root / "corrupted", source_dir=source, checksum_config=config_path)


if __name__ == "__main__":
    unittest.main()
