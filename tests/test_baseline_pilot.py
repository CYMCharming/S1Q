"""End-to-end pilot contract: fixed development records, no test access."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from scripts.compare_baselines import run_pilot
from s1q.data import file_sha256


class _Adapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.metadata = {"name": "tiny-decision", "revision": "local", "dtype": "fp32"}
        self.model = nn.Module()
        self.model.backbone = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
        self.model.decision_head = nn.Linear(3, 2)
        self.backbone = self.model.backbone

    def infer(self, record):
        inputs = torch.tensor(record["state"]["features"], dtype=torch.float32)[None]
        return [self.model.decision_head(self.model.backbone(inputs))[0]]


def _records(kind, count):
    return [{"id": f"{kind}-{index}",
             "state": {"features": [1 + index / 8, (-1) ** index, index / 5, 0.25]},
             "questions": {"decision": {"type": "choice", "criteria": {"a": "A", "b": "B"},
                                        "label": "a" if index % 2 == 0 else "b"}}}
            for index in range(count)]


class PilotTests(unittest.TestCase):
    def test_weight_only_w3_includes_smooth_and_spin_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for kind in ("calibration", "development"):
                (data / f"{kind}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in _records(kind, 5)),
                    encoding="utf-8")
            methods = ("rtn", "smoothquant-adapted", "spinquant-nohad-adapted",
                       "spinquant-had-adapted")
            with patch("scripts.compare_baselines.load_model", return_value=_Adapter()):
                report = run_pilot(model_name="tiny", data_dir=data,
                                   output_dir=root / "w3", methods=methods,
                                   bits=3, group_size=2, activation_bits=None,
                                   calibration_count=4, development_count=3,
                                   reservoir_size=4, spin_steps=2,
                                   device="cpu", dtype="fp32")
            self.assertIsNone(report["identity"]["activation_bits"])
            self.assertEqual({key: value["status"] for key, value in report["methods"].items()},
                             {key: "complete" for key in methods})
            self.assertTrue(report["methods"]["smoothquant-adapted"]["profile"]
                            ["weight_only_channel_balancing_ablation"])

    def test_fixed_development_pilot_does_not_read_final_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for kind, count in (("calibration", 6), ("development", 5)):
                (data / f"{kind}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in _records(kind, count)), encoding="utf-8")
            (data / "test.jsonl").write_text("POISON: final test must never be parsed", encoding="utf-8")
            core_methods = ("rtn", "s1q-local", "s1q2-beta0", "s1q2-beta05",
                            "gptq-blockdiag-adapted", "awq-adapted", "smoothquant-adapted")
            for bits, activation_bits in ((4, 4), (4, 8), (3, 4), (2, 4)):
                methods = core_methods + (("s1q-fisher-boundary", "s1q2-fisher-teacher",
                                           "s1q2-fisher-boundary") if (bits, activation_bits) == (4, 4) else ())
                if bits == 4:
                    methods += ("s1q3-joint",)
                if (bits, activation_bits) == (4, 4):
                    methods += ("s1q3-noclip", "s1q3-beta1-noclip", "s1q2-fine-noclip",
                                "spinquant-nohad-adapted", "spinquant-had-adapted")
                output = root / f"pilot-w{bits}a{activation_bits}"
                with patch("scripts.compare_baselines.load_model", return_value=_Adapter()) as loader:
                    report = run_pilot(model_name="tiny", data_dir=data, output_dir=output,
                                       methods=methods, bits=bits, group_size=2,
                                       activation_bits=activation_bits,
                                       calibration_count=4, development_count=3, reservoir_size=4,
                                       max_reservoir_rows=4,
                                       spin_steps=2,
                                       device="cpu", dtype="fp32",
                                       checkpoint_manifest="checksums.json")
                self.assertEqual(loader.call_args.kwargs["checkpoint_manifest"], "checksums.json")
                self.assertFalse(report["identity"]["final_test_read"])
                self.assertEqual(report["identity"]["bits"], bits)
                self.assertEqual(report["identity"]["activation_bits"], activation_bits)
                self.assertEqual(report["identity"]["accepted_development"], 3)
                project = Path(__file__).resolve().parents[1]
                expected_sources = {
                    "quantization.py": project / "src/s1q/quantization.py",
                    "spinquant_proxy.py": project / "src/s1q/spinquant_proxy.py",
                    "models.py": project / "src/s1q/models.py",
                    "experiment.py": project / "src/s1q/experiment.py",
                    "compare_baselines.py": project / "scripts/compare_baselines.py",
                }
                self.assertEqual(report["identity"]["implementation_sha256"],
                                 {name: file_sha256(path) for name, path in expected_sources.items()})
                self.assertEqual(set(report["methods"]), set(methods))
                expected_keys = {row["key"] for row in _read_rows(output / "native-development.jsonl")}
                for method in methods:
                    result = report["methods"][method]
                    self.assertEqual(result["status"], "complete", result.get("error"))
                    self.assertEqual(result["quantized_linear_count"], report["identity"]["selected_linear_count"])
                    self.assertIn("paired_vs_rtn", result)
                    keys = {row["key"] for row in _read_rows(output / f"{method}-development.jsonl")}
                    self.assertEqual(keys, expected_keys)
                if (bits, activation_bits) == (4, 4):
                    self.assertEqual(set(report["decision_fisher"]), {"teacher", "correct_boundary"})
                    self.assertTrue(report["decision_fisher"]["correct_boundary"]["uses_gold_labels"])


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
