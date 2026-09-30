"""Offline data integrity, split isolation and inference-leakage checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from s1q.data import (
    KEV_REVISION,
    _coalesce_request_groups,
    _reserve_nonoverlapping_groups,
    assert_disjoint,
    convert_nano_record,
    eligible_records,
    eligibility_reason,
    file_sha256,
    group_id,
    load_records,
    model_request,
    prepare_kev_data,
    prepare_nano_data,
    question_keys,
    sample_groups,
    validate_record,
    write_records,
)


FIXTURE_REVISION = "a" * 40


def record(identity="item", group=None, source="source", label="left"):
    return {
        "state": f"Visible state for {identity}",
        "questions": {"action": {"type": "choice", "instructions": "Choose an action.",
                                 "criteria": {"left": "Go left", "right": "Go right"},
                                 "label": label, "src": source}},
        "_meta": {"id": identity, "group_id": group or identity,
                  "source": source, "variant": "clean"},
    }


def make_kev_suite(root, suite, counts):
    directory = root / suite
    directory.mkdir(parents=True)
    files = {}
    for split, count in counts.items():
        rows = [record(f"{suite}/{split}/{i}", source=f"source{i % 2}") for i in range(count)]
        path = directory / f"{split}.jsonl"
        write_records(path, rows)
        files[path.name] = {"sha256": file_sha256(path), "records": count,
                            "questions": count}
    (directory / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def nano_record(identity="nano", split="test", kind="reference_argmax_compatibility"):
    return {
        "id": identity, "state_id": f"state/{identity}", "family_id": "unified_shooting",
        "state": f"Public observation {identity}", "split": split,
        "questions": {"action": {"type": "choice", "instructions": "Select the next action.",
                                 "criteria": {"left": "Left action", "right": "Right action"}}},
        "gold": {"action": "left"}, "gold_label_kind": {"action": kind},
        "metadata": {"episode_id": f"episode/{identity}",
                     "source_group_id": f"group/{identity}", "episode_success": True},
    }


class DataTests(unittest.TestCase):
    def test_jsonl_roundtrip_preserves_unicode_and_candidate_order(self):
        row = record()
        row["state"] = "中文状态 € 😀"
        row["questions"]["action"]["criteria"] = {"right": "Right", "left": "Left"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            write_records(path, [row])
            loaded = load_records(path)
        self.assertEqual(loaded, [row])
        self.assertEqual(question_keys(loaded[0]["questions"]["action"]), ["right", "left"])

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.jsonl"
            path.write_text('{"state":"one","state":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                load_records(path)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nonfinite.jsonl"
            for literal in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(literal=literal):
                    path.write_text('{"state":' + literal + '}\n', encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Non-finite"):
                        load_records(path)

    def test_duplicate_record_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate-id.jsonl"
            write_records(path, [record("same"), record("same")])
            with self.assertRaisesRegex(ValueError, "duplicate record ID"):
                load_records(path)

    def test_input_allowlist_excludes_every_target_and_metadata_field(self):
        row = record()
        row.update(gold={"action": "left"}, gold_probs={"action": {"left": 1.0, "right": 0.0}},
                   teacher={"native_probs": {"action": {"left": 0.9, "right": 0.1}}},
                   rationale="The expected answer is left", metadata={"episode_success": True})
        q = row["questions"]["action"]
        q.update(label_kind="deterministic_truth", target_distribution={"left": 1.0, "right": 0.0},
                 src="source", rationale="Secret target", label="left")
        request = model_request(row)
        self.assertEqual(set(request), {"state", "questions"})
        self.assertEqual(set(request["questions"]["action"]), {"type", "instructions", "criteria"})
        request["questions"]["action"]["criteria"]["left"] = "changed"
        self.assertEqual(row["questions"]["action"]["criteria"]["left"], "Go left")

    def test_label_types_and_candidate_membership_are_strict(self):
        invalid = [
            {"type": "noul", "instructions": "Is it true?", "label": 1},
            {"type": "score", "instructions": "Rate it", "criteria": ["low", "high"], "label": True},
            {"type": "score", "instructions": "Rate it", "criteria": ["low", "high"], "label": 2},
            {"type": "choice", "instructions": "Pick", "criteria": {"left": "Left"}, "label": "absent"},
        ]
        for question in invalid:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    validate_record({"state": "s", "questions": {"q": question}})
        self.assertEqual(question_keys({"type": "boolean"}), ["false", "true"])
        self.assertEqual(question_keys({"type": "score", "criteria": ["bad", "fair", "good"]}),
                         ["0", "1", "2"])

    def test_eligibility_rejects_whole_requests_without_truncation(self):
        row = record()
        original = copy.deepcopy(row)
        self.assertEqual(eligibility_reason(row, max_choices=1), "too_many_choices")
        self.assertEqual(eligibility_reason(row, max_chars=3), "request_too_long_chars")
        self.assertEqual(row, original)
        row["_meta"]["variant"] = "permutation"
        self.assertEqual(eligibility_reason(row), "not_clean")

    def test_tokenizer_predicate_only_receives_public_request(self):
        seen = []
        def predicate(request):
            seen.append(request)
            return False
        accepted, excluded = eligible_records([record()], predicate=predicate)
        self.assertEqual(accepted, [])
        self.assertEqual(excluded, {"tokenizer_admission": 1})
        self.assertNotIn("_meta", seen[0])
        self.assertNotIn("label", seen[0]["questions"]["action"])

    def test_group_sampling_is_label_independent_and_membership_order_independent(self):
        rows = [record(f"i{i}", group=f"g{i // 2}", source=f"s{i // 2 % 2}") for i in range(20)]
        selected = sample_groups(rows, 8, seed=42)
        altered = copy.deepcopy(rows)
        for row in altered:
            row["questions"]["action"]["label"] = "right"
        selected_altered = sample_groups(list(reversed(altered)), 8, seed=42)
        self.assertEqual({r["_meta"]["id"] for r in selected},
                         {r["_meta"]["id"] for r in selected_altered})
        for group in {group_id(r) for r in selected}:
            self.assertEqual(sum(group_id(r) == group for r in selected), 2)

    def test_groups_larger_than_budget_are_not_partially_sampled(self):
        rows = [record(f"big{i}", group="big") for i in range(5)] + [record("small")]
        selected = sample_groups(rows, 3)
        self.assertEqual([r["_meta"]["id"] for r in selected], ["small"])

    def test_group_leakage_and_exact_request_leakage_are_rejected(self):
        one, two = record("one", group="shared"), record("two", group="shared")
        with self.assertRaisesRegex(ValueError, "Dataset leakage"):
            assert_disjoint({"calibration": [one], "test": [two]})
        two = copy.deepcopy(one)
        two["_meta"] = {"id": "other", "group_id": "other", "source": "other"}
        with self.assertRaisesRegex(ValueError, "Dataset leakage"):
            assert_disjoint({"calibration": [one], "test": [two]})

    def test_prepare_kev_partitions_calibration_and_keeps_transfer_unfitted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_kev_suite(root / "source", "evals/v7/decision-v7",
                           {"calibration": 40, "development": 20, "test": 20})
            make_kev_suite(root / "source", "evals/v4/transfer-v4",
                           {"development": 20, "test": 20})
            manifest = prepare_kev_data(root / "prepared", source_root=root / "source",
                                        revision=FIXTURE_REVISION, calibration=4,
                                        temperature_calibration=4, development=5, test=6)
            splits = {name: load_records(info["path"]) for name, info in manifest["splits"].items()}
            assert_disjoint(splits)
            self.assertEqual(set(splits), {"calibration", "temperature_calibration", "development", "test",
                                          "transfer_development", "transfer_test"})
            self.assertEqual(len(splits["calibration"]), 4)
            self.assertEqual(len(splits["temperature_calibration"]), 4)
            for name, info in manifest["splits"].items():
                self.assertEqual(file_sha256(info["path"]), info["sha256"])
            repeated = prepare_kev_data(root / "repeat", source_root=root / "source",
                                        revision=FIXTURE_REVISION, calibration=4,
                                        temperature_calibration=4, development=5, test=6)
            self.assertEqual({n: s["ids"] for n, s in manifest["splits"].items()},
                             {n: s["ids"] for n, s in repeated["splits"].items()})

    def test_corrupted_kev_file_fails_checksum_before_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_kev_suite(root, "evals/v7/decision-v7", {"calibration": 3, "development": 3, "test": 3})
            (root / "evals/v7/decision-v7/test.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_kev_data(root / "prepared", source_root=root, revision=FIXTURE_REVISION,
                                 include_transfer=False)

    def test_pinned_kev_revision_rejects_unrelated_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_kev_suite(root, "evals/v7/decision-v7", {"calibration": 3, "development": 3, "test": 3})
            with self.assertRaisesRegex(ValueError, "Manifest bytes do not match"):
                prepare_kev_data(root / "prepared", source_root=root, revision=KEV_REVISION,
                                 include_transfer=False)

    def test_mutable_revisions_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            for revision in ("main", "unified-games-v1", "abcdef", "../other"):
                with self.subTest(revision=revision), self.assertRaises(ValueError):
                    prepare_kev_data(folder, revision=revision)

    def test_nano_teacher_argmax_is_never_promoted_to_gold(self):
        row = nano_record()
        row["gold"] = {}
        row["teacher"] = {"native_probs": {"action": {"left": 1.0, "right": 0.0}}}
        converted, excluded = convert_nano_record(row)
        self.assertIsNone(converted)
        self.assertEqual(excluded["question_without_independent_hard_gold"], 1)

    def test_nano_preserves_event_vs_policy_label_semantics(self):
        row = nano_record()
        row["questions"]["survive"] = {"type": "boolean", "instructions": "Will it survive?"}
        row["gold"]["survive"] = True
        row["gold_label_kind"]["survive"] = "observed_outcome"
        converted, excluded = convert_nano_record(row)
        self.assertEqual(excluded, {})
        self.assertEqual(converted["questions"]["action"]["label_kind"], "reference_argmax_compatibility")
        self.assertEqual(converted["questions"]["survive"]["label_kind"], "observed_outcome")
        self.assertEqual(converted["questions"]["survive"]["type"], "noul")
        self.assertEqual(group_id(converted), row["metadata"]["source_group_id"])
        self.assertNotIn("episode_success", json.dumps(model_request(converted)))

    def test_nano_undeclared_or_unobserved_gold_is_quarantined(self):
        for kind in ("unobserved", "hard_gold_unspecified", "unspecified_compatibility_label"):
            with self.subTest(kind=kind):
                converted, excluded = convert_nano_record(nano_record(kind=kind))
                self.assertIsNone(converted)
                self.assertEqual(excluded[f"question_label_kind_{kind}"], 1)

    def test_identical_observations_join_episode_groups_transitively(self):
        rows = [record(f"i{i}", group=f"episode{i // 2}") for i in range(6)]
        rows[0]["state"] = rows[2]["state"] = "shared observation A"
        rows[3]["state"] = rows[4]["state"] = "shared observation B"
        coalesced, audit = _coalesce_request_groups(rows)
        self.assertEqual(len({group_id(row) for row in coalesced}), 1)
        self.assertEqual(audit["upstream_groups"], 3)
        self.assertEqual(audit["components"], 1)
        self.assertEqual(audit["groups_joined"], 2)
        self.assertEqual(audit["deduplicated_rows"], 0)
        self.assertEqual(len(coalesced), 6)
        self.assertTrue(all(row["_meta"]["upstream_group_ids"] == ["episode0", "episode1", "episode2"]
                            for row in coalesced))
        reversed_rows, _ = _coalesce_request_groups(list(reversed(rows)))
        self.assertEqual({group_id(row) for row in reversed_rows}, {group_id(row) for row in coalesced})

    def test_heldout_priority_removes_complete_lower_priority_components(self):
        test = [record("test0", group="test_episode")]
        ood = [record("ood0", group="ood_episode"), record("ood1", group="ood_episode")]
        dev = [record("dev0", group="dev_episode"), record("dev1", group="dev_episode")]
        cal = [record("cal0", group="cal_episode"), record("cal1", group="cal_episode")]
        for row in (test[0], ood[0], dev[0], cal[0]):
            row["state"] = "same scene across different episodes"
        coalesced = {name: _coalesce_request_groups(rows)[0]
                     for name, rows in {"test": test, "ood": ood, "dev": dev, "calibration": cal}.items()}
        kept, audit = _reserve_nonoverlapping_groups(coalesced, ("test", "ood", "dev", "calibration"))
        self.assertEqual(len(kept["test"]), 1)
        for name in ("ood", "dev", "calibration"):
            self.assertEqual(kept[name], [])
            self.assertEqual(audit[name]["excluded_records"], 2)
            self.assertEqual(audit[name]["excluded_components"], 1)
            self.assertEqual(audit[name]["rejected_components"][0]["overlap_with"], ["test"])
            self.assertEqual(audit[name]["rejected_components"][0]["overlap_key_counts"]["request"], 1)

    def test_nano_preparation_handles_realistic_cross_episode_input_repeats(self):
        with tempfile.TemporaryDirectory() as folder:
            root, files = Path(folder), []
            source_rows = {split: [nano_record(f"{split}/{i}", split=split) for i in range(30)]
                           for split in ("calibration", "dev", "test", "ood")}
            # All splits repeat one visible scene, but source episode IDs differ.
            for split, rows in source_rows.items():
                rows[0]["state"] = "same reserved test scene"
                rows[0]["metadata"]["source_group_id"] = f"{split}/paired_episode"
                rows[1]["metadata"]["source_group_id"] = f"{split}/paired_episode"
            # Two whole calibration episodes link through a repeated observation.
            cal = source_rows["calibration"]
            cal[2]["state"] = cal[4]["state"] = "same calibration-only scene"
            for index in (2, 3):
                cal[index]["metadata"]["source_group_id"] = "calibration/episodeA"
            for index in (4, 5):
                cal[index]["metadata"]["source_group_id"] = "calibration/episodeB"
            for split, rows in source_rows.items():
                path = root / "source" / "unified/hard" / f"{split}.jsonl"
                write_records(path, rows)
                files.append({"path": f"unified/hard/{split}.jsonl", "sha256": file_sha256(path),
                              "bytes": path.stat().st_size})
            (root / "source/SHA256_MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
            manifest = prepare_nano_data(root / "prepared", source_root=root / "source",
                                         revision=FIXTURE_REVISION, calibration=None,
                                         temperature_calibration=None, development=None, test=None)
            splits = {name: load_records(info["path"]) for name, info in manifest["splits"].items()}
            assert_disjoint(splits)
            self.assertEqual(len(splits["test"]), 30)
            self.assertEqual(len(splits["ood"]), 28)
            self.assertEqual(len(splits["development"]), 28)
            ownership = {name for name in ("calibration", "temperature_calibration")
                         if any(row["_meta"]["id"] == "calibration/2" for row in splits[name])}
            self.assertEqual(len(ownership), 1)
            bucket = splits[next(iter(ownership))]
            self.assertTrue({"calibration/2", "calibration/3", "calibration/4", "calibration/5"}
                            <= {row["_meta"]["id"] for row in bucket})
            self.assertEqual(manifest["group_coalescing"]["calibration"]["groups_joined"], 1)
            self.assertEqual(manifest["source_overlaps"]["calibration"]["excluded_records"], 2)
            self.assertFalse(manifest["overlap_policy"]["heldout_resplit"])
            self.assertFalse(manifest["overlap_policy"]["silent_deduplication"])

    def test_prepare_nano_hard_suite_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root, files = Path(folder), []
            for split in ("calibration", "dev", "test", "ood"):
                path = root / "source" / "unified/hard" / f"{split}.jsonl"
                rows = [nano_record(f"{split}/{i}", split=split) for i in range(30)]
                teacher_only = nano_record(f"{split}/teacher", split=split)
                teacher_only["gold"] = {}
                teacher_only["teacher"] = {"native_probs": {"action": {"left": 0.7, "right": 0.3}}}
                write_records(path, rows + [teacher_only])
                files.append({"path": f"unified/hard/{split}.jsonl", "sha256": file_sha256(path),
                              "bytes": path.stat().st_size})
            (root / "source/SHA256_MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
            manifest = prepare_nano_data(root / "prepared", source_root=root / "source",
                                         revision=FIXTURE_REVISION, calibration=4,
                                         temperature_calibration=4, development=5, test=6)
            splits = {name: load_records(info["path"]) for name, info in manifest["splits"].items()}
            assert_disjoint(splits)
            self.assertEqual(set(splits), {"calibration", "temperature_calibration", "development", "test", "ood"})
            self.assertEqual(manifest["exclusions"]["test"]["question_without_independent_hard_gold"], 1)
            self.assertEqual(manifest["upstream_files"]["test"]["raw_records"], 31)
            self.assertEqual(manifest["upstream_files"]["test"]["eligible_hard_gold_records"], 30)
            self.assertTrue(all(r["_meta"]["revision"] == FIXTURE_REVISION for r in splits["test"]))


if __name__ == "__main__":
    unittest.main()
