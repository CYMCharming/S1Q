"""Read-only gate for adding a separate SmoothQuant control to a frozen matrix.

No model, Torch, GPU, or existing result is loaded or modified. Native logits and
derived softmax probabilities must agree within 1e-6 for every decision in all
22 source suites AND Mixed Dev. Options are checked against the same hash-pinned
native requests, since historical prediction JSONL omits candidate names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

MAIN_METHODS = (
    "rtn", "s1q-local", "s1q2-beta05", "awq-adapted", "gptq-blockdiag-adapted",
    "spinquant-nohad-adapted", "spinquant-had-adapted", "s1q-joint", "s1q-ac",
    "s1q-margin", "s1q",
)
CONTROL = "smoothquant-adapted"
ROW_FIELDS = ("key", "record_id", "qid", "type", "source", "cluster_id", "label", "label_kind")
MATCH_IDENTITY = (
    "model", "seed", "bits", "activation_bits", "group_size", "requested_calibration",
    "accepted_calibration", "accepted_calibration_ids", "accepted_evaluation_ids",
    "calibration_exclusions", "evaluation_exclusions", "selected_linear_names",
    "selected_linear_count", "selected_weight_parameters", "reservoir_size",
    "source_sha256", "implementation_sha256",
)
MATCH_CONFIG = (
    "model", "bits", "activation_bits", "group_size", "calibration_count",
    "development_count", "reservoir_size", "seed", "dtype", "benchmark_manifest_sha256",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON {x}")))


def read_rows(path):
    result = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON {x}"))))
    return result


def softmax(values):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Empty, non-numeric or non-finite logits")
    maximum = max(values)
    exponents = [math.exp(v - maximum) for v in values]
    denominator = sum(exponents)
    return [v / denominator for v in exponents]


def expected_decisions(records, accepted_ids):
    by_id = {}
    for index, record in enumerate(records):
        identity = str(record.get("id", record.get("_meta", {}).get("id", record.get("request_id", index))))
        if identity in by_id:
            raise ValueError(f"Duplicated frozen request identity {identity}")
        by_id[identity] = record
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("Duplicate native accepted request identities")
    result = {}
    for identity in accepted_ids:
        if identity not in by_id:
            raise ValueError(f"Unknown accepted request {identity}")
        record = by_id[identity]
        for qid, question in record["questions"].items():
            kind = question["type"]
            if kind in ("noul", "boolean"):
                if type(question["label"]) is not bool:
                    raise ValueError("Frozen Boolean gold is not Boolean")
                options, gold = ["false", "true"], int(question["label"])
            elif kind == "choice":
                options = list(question["criteria"])
                gold = options.index(question["label"])
            elif kind == "score":
                options, gold = [str(i) for i in range(len(question["criteria"]))], question["label"]
            else:
                raise ValueError(f"Unsupported native question type {kind}")
            key = f"{identity}::{qid}"
            result[key] = {
                "key": key, "record_id": identity, "qid": qid, "type": kind,
                "source": question.get("src", record.get("source", record.get("_meta", {}).get("source", "unknown"))),
                "cluster_id": record.get("_meta", {}).get("group_id", identity), "label": gold,
                "label_kind": question.get("label_kind", question.get("gold_label_kind", "dataset_gold")),
                "options": options,
            }
    return result


def validate_predictions(rows, expected, context):
    by_key = {}
    for row in rows:
        key = row.get("key")
        if key in by_key:
            raise ValueError(f"{context}: duplicate decision key {key}")
        if key not in expected:
            raise ValueError(f"{context}: unexpected decision key {key}")
        target = expected[key]
        for field in ROW_FIELDS:
            if row.get(field) != target[field]:
                raise ValueError(f"{context}: mismatched {field} for {key}")
        if type(row.get("label")) is not type(target["label"]):
            raise ValueError(f"{context}: gold-index type mismatch for {key}")
        if len(row["logits"]) != len(target["options"]):
            raise ValueError(f"{context}: option count mismatch for {key}")
        if "options" in row and row["options"] != target["options"]:
            raise ValueError(f"{context}: option ordering mismatch for {key}")
        softmax(row["logits"])
        by_key[key] = row
    if set(by_key) != set(expected):
        raise ValueError(f"{context}: missing native decisions")
    return by_key


def audit_pair(main_run, control_run, manifest_path, development_path, tolerance=1e-6):
    if not 0 <= tolerance <= 1e-6:
        raise ValueError("Native merge tolerance must be between zero and 1e-6")
    result = {"status": "not_admissible", "merge_admissible": False,
              "tolerance": tolerance, "expected_main_methods": list(MAIN_METHODS),
              "additional_method": CONTROL, "errors": [], "cohorts": []}
    try:
        manifest = read_json(manifest_path)
        if manifest.get("schema") != "s1q.expanded-benchmarks.v1":
            raise ValueError("Unrecognized frozen benchmark manifest")
        datasets = [d for d in manifest["datasets"] if d.get("ranking_eligible") is True]
        if len(datasets) != 22:
            raise ValueError("This gate requires all 22 frozen rank-eligible source suites")
        names = [d["dataset_id"] for d in datasets]
        if len(set(names)) != len(names) or "development" in names:
            raise ValueError("Invalid duplicate source-suite names")
        manifest_hash = sha(manifest_path)
        sources = {"development": development_path}
        source_hashes = {"development": sha(development_path)}
        for entry in datasets:
            path = (manifest_path.parent / entry["path"]).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError("Frozen source path escaped the manifest directory")
            if sha(path) != entry["sha256"]:
                raise ValueError(f"Frozen source hash mismatch: {entry['dataset_id']}")
            sources[entry["dataset_id"]] = path
            source_hashes[entry["dataset_id"]] = entry["sha256"]
        main_report = read_json(main_run / "report.json")
        control_report = read_json(control_run / "report.json")
        main_identity, control_identity = main_report["identity"], control_report["identity"]
        main_config = read_json(main_run / "frozen-config-before-evaluation.json")
        control_config = read_json(control_run / "frozen-config-before-evaluation.json")
        for root, report, identity, expected_methods in (
            (main_run, main_report, main_identity, MAIN_METHODS),
            (control_run, control_report, control_identity, (CONTROL,)),
        ):
            marker = read_json(root / "expanded-evaluation-complete.json")
            if marker.get("status") != "complete" or set(marker.get("methods", {})) != set(expected_methods):
                raise ValueError("Missing or incomplete frozen main/control completion marker")
            if set(report["methods"]) != set(expected_methods):
                raise ValueError("Unexpected main/control method set")
            if any(report["methods"][method].get("status") != "complete" for method in expected_methods):
                raise ValueError("A main/control method did not complete")
            if marker.get("benchmark_manifest_sha256") != manifest_hash or identity.get("benchmark_manifest_sha256") != manifest_hash:
                raise ValueError("Run/marker does not identify the exact frozen dataset manifest")
            if marker.get("ranking_dataset_ids") != names or identity.get("expanded_ranking_dataset_ids") != names:
                raise ValueError("Run did not evaluate all frozen source suites in the declared order")
            if set(identity.get("accepted_evaluation_ids", {})) != set(sources):
                raise ValueError("Native accepted cohorts differ from the required 22 suites plus development")
            if identity.get("final_test_read") is not True:
                raise ValueError("Run did not explicitly declare upstream held-out evaluation")
            if read_json(root / "identity.json") != identity:
                raise ValueError("Standalone identity differs from report")
            for cohort, expected_hash in source_hashes.items():
                if identity.get("source_sha256", {}).get(cohort) != expected_hash:
                    raise ValueError(f"Run source hash differs for {cohort}")
            if report.get("methodology", {}).get("uses_calibration_gold_labels_for_quantization") is not False:
                raise ValueError("Control/main must explicitly use label-free calibration")
        for key in MATCH_IDENTITY:
            if key not in main_identity or key not in control_identity or main_identity[key] != control_identity[key]:
                raise ValueError(f"Main and SmoothQuant identity mismatch: {key}")
        for key in MATCH_CONFIG:
            if key not in main_config or key not in control_config or main_config[key] != control_config[key]:
                raise ValueError(f"Main and SmoothQuant frozen configuration mismatch: {key}")
        registration = control_identity.get("additional_control_registration")
        if not isinstance(registration, dict) or registration.get("method") != CONTROL or registration.get("official_reproduction") is not False:
            raise ValueError("SmoothQuant control needs an explicit repository-adaptation registration")
        if control_identity.get("additional_control_only") is not True or control_config.get("additional_control_only") is not True:
            raise ValueError("SmoothQuant must remain a separate additional control")
        if control_config.get("additional_control_registration") != registration:
            raise ValueError("SmoothQuant registration changed after the before-evaluation freeze")
        if registration.get("quantization_math_changed") is not False or registration.get("frozen_main_method_matrix_changed") is not False:
            raise ValueError("Additional control changed frozen main or quantization math")
        for method, report in [(m, main_report) for m in MAIN_METHODS] + [(CONTROL, control_report)]:
            quantization = report["methods"][method]["quantization"]
            layers = quantization["layers"]
            if set(layers) != set(main_identity["selected_linear_names"]):
                raise ValueError(f"Quantization layer mask mismatch: {method}")
            if quantization.get("activation_bits") != main_identity["activation_bits"]:
                raise ValueError(f"Activation precision mismatch: {method}")
            for name, layer in layers.items():
                if layer.get("bits") != main_identity["bits"] or layer.get("group_size") != main_identity["group_size"]:
                    raise ValueError(f"Layer precision or group-size mismatch: {method}/{name}")
                if layer.get("shape") != main_report["methods"]["rtn"]["quantization"]["layers"][name].get("shape"):
                    raise ValueError(f"Layer shape mismatch: {method}/{name}")
        max_logits = max_probabilities = 0.0
        input_files = {}
        for cohort, path in sources.items():
            accepted = main_identity["accepted_evaluation_ids"][cohort]
            if not accepted:
                raise ValueError(f"Entire native suite rejected: {cohort}")
            expected = expected_decisions(read_rows(path), accepted)
            left_path, right_path = main_run / f"native-{cohort}.jsonl", control_run / f"native-{cohort}.jsonl"
            left = validate_predictions(read_rows(left_path), expected, f"main native/{cohort}")
            right = validate_predictions(read_rows(right_path), expected, f"control native/{cohort}")
            logit_difference = probability_difference = 0.0
            for key, row in left.items():
                logit_difference = max(logit_difference, max(abs(a - b) for a, b in zip(row["logits"], right[key]["logits"])))
                probability_difference = max(probability_difference, max(abs(a - b) for a, b in zip(softmax(row["logits"]), softmax(right[key]["logits"]))))
            max_logits, max_probabilities = max(max_logits, logit_difference), max(max_probabilities, probability_difference)
            if logit_difference > tolerance or probability_difference > tolerance:
                raise ValueError(f"Native equivalence threshold exceeded for {cohort}: logits={logit_difference}, probabilities={probability_difference}")
            for method, folder in [(m, main_run) for m in MAIN_METHODS] + [(CONTROL, control_run)]:
                prediction_path = folder / f"{method}-{cohort}.jsonl"
                validate_predictions(read_rows(prediction_path), expected, f"{method}/{cohort}")
            input_files[cohort] = {"dataset_sha256": source_hashes[cohort],
                                   "main_native_sha256": sha(left_path), "control_native_sha256": sha(right_path)}
            result["cohorts"].append({"dataset_id": cohort, "n_requests": len(accepted), "n_decisions": len(expected),
                                      "same_keys_labels_options_and_metadata": True,
                                      "max_abs_native_logit_difference": logit_difference,
                                      "max_abs_native_probability_difference": probability_difference,
                                      "twelve_method_prediction_coverage_verified": True})
        result.update({"status": "passed", "merge_admissible": True, "model": main_config["model"],
                       "benchmark_manifest_sha256": manifest_hash, "benchmark_id": manifest.get("benchmark_id"),
                       "n_source_suites": 22, "development_also_verified": True,
                       "max_abs_native_logit_difference": max_logits,
                       "max_abs_native_probability_difference": max_probabilities,
                       "source_and_native_file_hashes": input_files,
                       "run_hashes": {"main_report_sha256": sha(main_run / "report.json"),
                                      "main_identity_sha256": sha(main_run / "identity.json"),
                                      "main_frozen_config_sha256": sha(main_run / "frozen-config-before-evaluation.json"),
                                      "control_report_sha256": sha(control_run / "report.json"),
                                      "control_identity_sha256": sha(control_run / "identity.json"),
                                      "control_frozen_config_sha256": sha(control_run / "frozen-config-before-evaluation.json")},
                       "additional_control_registration": registration,
                       "main_implementation_sha256": main_identity["implementation_sha256"],
                       "option_order_evidence": "Both runs hash-identify the same unmodified native request JSONL; native row dimensions and gold indices agree with each request's original option order.",
                       "interpretation": "Admission verifies comparability only. It does not establish SmoothQuant accuracy, superiority or integer-kernel acceleration."})
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path)
    parser.add_argument("--control-run", type=Path)
    parser.add_argument("--pairs-json", type=Path, help="Optional array of {model, main_run, control_run}")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.pairs_json:
        if args.main_run or args.control_run:
            parser.error("Use either --pairs-json or a single main/control pair")
        pairs = read_json(args.pairs_json)
    elif args.main_run and args.control_run:
        pairs = [{"main_run": str(args.main_run), "control_run": str(args.control_run)}]
    else:
        parser.error("A main/control pair or --pairs-json is required")
    report = {"schema": "s1q.smoothquant-merge-audit.v1", "audit_script_sha256": sha(Path(__file__)),
              "read_only_existing_runs": True, "models": [], "all_models_admissible": False}
    for pair in pairs:
        item = audit_pair(Path(pair["main_run"]), Path(pair["control_run"]), args.manifest,
                          args.development_data, args.tolerance)
        if pair.get("model") and item.get("model") and pair["model"] != item["model"]:
            item["status"], item["merge_admissible"] = "not_admissible", False
            item["errors"].append("Requested pair model differs from the run")
        report["models"].append(item)
    report["all_models_admissible"] = bool(report["models"]) and all(m["merge_admissible"] for m in report["models"])
    resolved_output = args.output.resolve()
    for pair in pairs:
        if any(resolved_output.is_relative_to(Path(pair[name]).resolve()) for name in ("main_run", "control_run")):
            parser.error("Audit output must be outside existing main/control run directories")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_models_admissible": report["all_models_admissible"],
                      "models": [{"model": item.get("model"), "status": item["status"], "errors": item["errors"]} for item in report["models"]]}))
    return 0 if report["all_models_admissible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
