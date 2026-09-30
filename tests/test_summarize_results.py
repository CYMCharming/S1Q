"""Offline reporting regressions for matched budgets, completed runs and freezes."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("s1q_results_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def quantized(name, *, method="s1q", activation_bits=None, protected=None):
    return {"profile": {"name": name, "method": method, "activation_bits": activation_bits,
                        "sensitive_fraction": 0.05 if protected else 0},
            "quantization": {"activation_bits": activation_bits, "activation_scheme": "per_token" if activation_bits else None,
                             "preserved_layers": protected or {},
                             "layers": {"encoder.block": {"shape": [4, 4], "bits": 4, "group_size": 128}}},
            "storage": {"quantized_parameter_count": 16, "retained_native_parameters_bytes": 8},
            "splits": {}}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def predictions(path, *, cluster="episode", label=0):
    rows = [{"key": "id::decision", "record_id": "id", "cluster_id": cluster,
             "label": label, "logits": [1.0, 0.0]}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class ResultReportingTests(unittest.TestCase):
    def test_same_fraction_with_different_protected_tensors_is_not_matched(self):
        selected = quantized("s1q-protected", protected={"first": "reason"})
        rtn = quantized("rtn-matched", method="rtn", protected={"other": "reason"})
        name, audit = REPORT.matched_rtn({"s1q-protected": selected, "rtn-matched": rtn}, "s1q-protected")
        self.assertIsNone(name)
        self.assertIn("preserved_layers", audit["rejected_candidates"]["rtn-matched"])

    def test_activation_and_weight_budgets_must_match(self):
        selected, rtn = quantized("selected", activation_bits=8), quantized("rtn", method="rtn")
        self.assertIsNone(REPORT.matched_rtn({"selected": selected, "rtn": rtn}, "selected")[0])
        rtn = quantized("rtn", method="rtn", activation_bits=8)
        self.assertEqual(REPORT.matched_rtn({"selected": selected, "rtn": rtn}, "selected")[0], "rtn")
        rtn["quantization"]["layers"]["encoder.block"]["bits"] = 8
        self.assertIsNone(REPORT.matched_rtn({"selected": selected, "rtn": rtn}, "selected")[0])

    def test_paired_interval_rejects_changed_cluster_or_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root / "a.jsonl", root / "b.jsonl"
            predictions(a)
            predictions(b)
            self.assertEqual(REPORT.comparison(a, b, seed=7, samples=10)["n_clusters"], 1)
            predictions(b, cluster="different_episode")
            self.assertEqual(REPORT.comparison(a, b, seed=7, samples=10)["status"], "target_options_or_clusters_not_paired")
            predictions(b, label=1)
            self.assertEqual(REPORT.comparison(a, b, seed=7, samples=10)["status"], "target_options_or_clusters_not_paired")

    def test_incomplete_runs_are_not_counted_as_experimental_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "running/summary.json", {"status": "running"})
            result = REPORT.aggregate_results(root, samples=10)
            self.assertEqual(result["runs"], [])
            self.assertEqual(result["skipped"][0]["reason"], "incomplete_status")

    def test_external_report_requires_intact_no_fitting_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = quantized("s1q-local")
            metric = {"raw": {"n": 1, "accuracy": 1.0, "nll": 0.3, "brier": 0.1, "ece_15": 0.2},
                      "temperature_calibrated": None}
            freeze = {"selection": {"selected": selected["profile"], "bits": 4, "group_size": 128},
                      "policy": {"external_fitting": False, "external_selection": False},
                      "rtn_scope": {"layers": ["encoder.block"]}}
            freeze_path = root / "freeze.json"
            write_json(freeze_path, freeze)
            summary = {"status": "complete", "model": "fixture", "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                       "baseline": metric, "selected": metric, "rtn_matched": metric,
                       "quantization": {"selected": selected["quantization"], "rtn_matched": selected["quantization"]},
                       "inference_metadata": {}}
            path = root / "summary.json"
            write_json(path, summary)
            for name in ("baseline", "selected", "rtn_matched"):
                predictions(root / f"{name}.jsonl")
            run = REPORT.load_run(path, results_dir=root, seed=7, samples=10)
            self.assertTrue(run["external"])
            self.assertEqual(run["matched_rtn"], "rtn-matched")
            self.assertEqual(run["evaluations"]["native"]["external_test"]["temperature_calibrated"], {})
            self.assertEqual(run["paired_comparisons"]["external_test"]["selected_minus_matched_rtn"]["status"], "computed")
            freeze["policy"]["external_fitting"] = True
            write_json(freeze_path, freeze)
            summary["freeze_sha256"] = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, "policy_not_confirmatory"):
                REPORT.load_run(path, results_dir=root, seed=7, samples=10)


if __name__ == "__main__":
    unittest.main()
