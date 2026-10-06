"""Matched exploratory optimization with label-free calibration and saved logits.

No final-test file is read. Hyperparameters/methods are fixed before evaluation.
All methods use the same admitted calibration requests, quantized Linear scope,
precision and hardware; extra named evaluation cohorts never enter calibration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from s1q.baselines import quantize_baseline_model
from s1q.data import assert_disjoint, file_sha256, load_records
from s1q.decision_margin import DecisionMarginCollector
from s1q.experiment import evaluate, predict, storage_report, write_json, write_rows
from s1q.metrics import compare
from s1q.models import load_model
from s1q.optimized_quantization import quantize_optimized_model
from s1q.quantization import CalibrationCollector, quantize_model, selected_linear_modules

PACKAGE = Path(__file__).resolve().parent
BASES = ("rtn", "s1q-local", "s1q2-beta05", "awq-adapted", "gptq-blockdiag-adapted",
         "spinquant-nohad-adapted", "spinquant-had-adapted")
NEW = ("s1q-joint", "s1q-ac", "s1q-margin", "s1q-mac", "awq-ac", "awq-mac",
       "gptq-ac", "gptq-mac", "spinquant-nohad-mac", "spinquant-had-mac")
ALL = ("s1q", "s1q-repair") + BASES + NEW + tuple(f"{base}-repair" for base in BASES + NEW if base != "rtn")


def implementation_method(method):
    """Preserve frozen experiment aliases while publishing the method as S1Q."""
    return "s1q-mac" if method == "s1q" else method


def identity(record, index):
    return str(record.get("id") or record.get("_meta", {}).get("id") or index)


def subset(records, count, seed):
    ranked = sorted(enumerate(records), key=lambda item: (
        hashlib.sha256(f"{seed}|{identity(item[1], item[0])}".encode()).hexdigest(), item[0]))
    return [record for _, record in ranked[:count]]


def unlabeled(record):
    result = copy.deepcopy(record)
    for q in result["questions"].values():
        q.pop("label", None)
        q.pop("target", None)
    return result


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    methods = tuple(args.methods.split(","))
    canonical = tuple(implementation_method(m.removesuffix("-repair")) +
                      ("-repair" if m.endswith("-repair") else "") for m in methods)
    if not methods or len(set(canonical)) != len(methods) or any(m not in ALL for m in methods):
        raise ValueError("Unknown/duplicated methods")
    if min(args.calibration_count, args.development_count, args.reservoir_size) <= 0:
        raise ValueError("Positive budgets required")
    output.mkdir(parents=True)
    write_json(output / "frozen-config-before-evaluation.json", vars(args))
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    adapter = load_model(args.model, device=args.device, dtype=args.dtype,
                         source_dir=args.source_dir, checkpoint_dir=args.checkpoint_dir,
                         checkpoint_manifest=args.checkpoint_manifest)
    adapter.model.eval()
    data = Path(args.data_dir)
    calibration_path, development_path = data / "calibration.jsonl", data / "development.jsonl"
    calibration = subset(load_records(calibration_path), args.calibration_count, args.seed)
    _, accepted_cal, cal_rejections, _ = predict(adapter, calibration, allow_rejection=True)
    stripped_cal = [unlabeled(r) for r in accepted_cal]
    source_hashes = {"calibration": file_sha256(calibration_path), "development": file_sha256(development_path)}
    cohorts = {"development": subset(load_records(development_path), args.development_count, args.seed)}
    for extra in args.extra_evaluation:
        name, path = extra.split("=", 1)
        if name in cohorts or not name.replace("-", "").isalnum():
            raise ValueError("Named cohort must be unique and filename-safe")
        cohorts[name] = load_records(Path(path))
        source_hashes[name] = file_sha256(Path(path))
    assert_disjoint({"calibration": calibration, **cohorts})
    native_rows, accepted_eval, eval_rejections = {}, {}, {}
    for name, records in cohorts.items():
        rows, accepted, excluded, _ = predict(adapter, records, allow_rejection=True)
        native_rows[name], accepted_eval[name], eval_rejections[name] = rows, accepted, excluded
        write_rows(output / f"native-{name}.jsonl", rows)
    scope = selected_linear_modules(adapter.backbone)
    implementation = {f"src/s1q/{name}": file_sha256(PACKAGE / name)
                      for name in ("optimization.py", "quantization.py", "baselines.py",
                                   "decision_margin.py", "activation_compensation.py",
                                   "optimized_quantization.py", "decision_repair.py",
                                   "models.py", "spinquant_proxy.py")}
    record_identity = {"cohort": args.cohort, "final_test_read": False,
                       "model": adapter.metadata, "seed": args.seed, "bits": args.bits,
                       "activation_bits": args.activation_bits, "group_size": args.group_size,
                       "gpu": torch.cuda.get_device_name(adapter.device) if adapter.device.type == "cuda" else None,
                       "platform": platform.platform(), "torch": torch.__version__,
                       "cuda": torch.version.cuda, "source_sha256": source_hashes,
                       "requested_calibration": args.calibration_count, "accepted_calibration": len(accepted_cal),
                       "accepted_calibration_ids": [identity(r,i) for i,r in enumerate(accepted_cal)],
                       "accepted_evaluation_ids": {n: [identity(r,i) for i,r in enumerate(rows)] for n,rows in accepted_eval.items()},
                       "calibration_exclusions": cal_rejections, "evaluation_exclusions": eval_rejections,
                       "selected_linear_names": sorted(scope), "selected_linear_count": len(scope),
                       "selected_weight_parameters": sum(m.weight.numel() for m in scope.values()),
                       "reservoir_size": args.reservoir_size, "implementation_sha256": implementation}
    report = {"identity": record_identity, "native_development": evaluate(native_rows["development"], {}),
              "native_evaluations": {n: evaluate(rows,{}) for n,rows in native_rows.items() if n != "development"},
              "methods": {}, "methodology": {"same_calibration_records_and_scope": True,
              "uses_calibration_gold_labels_for_quantization": False, "official_baseline_reproductions": False,
              "activation_execution": "floating dynamic token QDQ, no integer activation kernel",
              "quantizer_hyperparameters_fixed_before_evaluation": True,
              "evaluation_used_for_configuration_selection": False}}
    write_json(output / "identity.json", record_identity)
    write_json(output / "report.json", report)
    with torch.inference_mode(), CalibrationCollector(adapter.backbone, reservoir_size=args.reservoir_size, seed=args.seed) as collector:
        for record in stripped_cal:
            adapter.infer(record)
    ordinary = collector.statistics()
    margin = None
    if any(implementation_method(m.removesuffix("-repair")) in ("s1q-margin", "s1q-mac", "awq-mac", "gptq-mac", "spinquant-nohad-mac", "spinquant-had-mac") for m in methods):
        print(json.dumps({"stage": "collect_decision_margin", "records": len(stripped_cal)}), flush=True)
        start = time.perf_counter()
        with DecisionMarginCollector(adapter, modules=scope, reservoir_size=args.reservoir_size, seed=args.seed) as collector:
            for record in stripped_cal:
                collector.collect(record)
        margin = collector.statistics()
        report["decision_margin"] = {**collector.metadata(), "seconds": time.perf_counter() - start}
        write_json(output / "report.json", report)
    teacher = None
    if any(m.endswith("-repair") for m in methods):
        with torch.inference_mode():
            teacher = [[z.detach().float().cpu() for z in adapter.infer(r)] for r in stripped_cal[:args.repair_records]]
    all_rows = {}
    for method in methods:
        print(json.dumps({"stage": "optimization_method", "method": method}), flush=True)
        start = time.perf_counter()
        session = None
        try:
            actual = implementation_method(method.removesuffix("-repair"))
            meta = None
            if actual in ("s1q-local", "s1q2-beta05"):
                session = quantize_model(adapter.backbone, method="s1q" if actual == "s1q-local" else "s1q2",
                                         bits=args.bits, group_size=args.group_size, statistics=ordinary,
                                         activation_bits=args.activation_bits)
            elif actual in BASES:
                session = quantize_baseline_model(adapter.backbone, method=actual, bits=args.bits,
                                                 group_size=args.group_size, statistics=ordinary,
                                                 activation_bits=args.activation_bits)
            else:
                important = actual in ("s1q-margin", "s1q-mac", "awq-mac", "gptq-mac", "spinquant-nohad-mac", "spinquant-had-mac")
                session, meta = quantize_optimized_model(
                    adapter.backbone, method="s1q" if method.removesuffix("-repair") == "s1q" else actual,
                    statistics=margin if important else ordinary,
                    bits=args.bits, group_size=args.group_size, activation_bits=args.activation_bits,
                    compensation=actual not in ("s1q-joint", "s1q-margin"))
            with session:
                if set(session.layers) != set(scope):
                    raise RuntimeError("Quantization scope mismatch")
                if method.endswith("-repair"):
                    from s1q.decision_repair import repair_session
                    meta = repair_session(adapter, session, stripped_cal[:args.repair_records], teacher,
                                          seed=args.seed, epochs=args.repair_epochs)
                if adapter.device.type == "cuda":
                    torch.cuda.synchronize(adapter.device)
                quantization_seconds = time.perf_counter() - start
                rows_by_cohort = {}
                for name, records in accepted_eval.items():
                    rows, _, _, _ = predict(adapter, records)
                    rows_by_cohort[name] = rows
                    write_rows(output / f"{method}-{name}.jsonl", rows)
                storage = storage_report(adapter, session)
                if method.startswith("spinquant-"):
                    rotation = session.report()["extra_rotation_bytes_fp32"]
                    storage["rotation_bytes_fp32"] = rotation
                    storage["estimated_complete_packed_parameters_bytes"] += rotation
                    storage["estimated_parameter_storage_ratio"] = (
                        storage["estimated_complete_packed_parameters_bytes"] / storage["native_parameters_bytes"])
                result = {"status": "complete", "profile": {"method": method, "official_reproduction": False},
                          "quantization_seconds": quantization_seconds, "total_seconds": time.perf_counter()-start,
                          "development": evaluate(rows_by_cohort["development"],{}),
                          "evaluations": {n: evaluate(rows,{}) for n,rows in rows_by_cohort.items() if n != "development"},
                          "paired_vs_native": compare(native_rows["development"], rows_by_cohort["development"]),
                          "quantization": session.report(), "optimization": meta, "storage": storage}
                all_rows[method] = rows_by_cohort["development"]
        except Exception as error:
            if session is not None:
                session.restore()
            result = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                      "seconds": time.perf_counter()-start}
            print(json.dumps({"stage": "method_failed", "method": method, "error": str(error)}), flush=True)
        report["methods"][method] = result
        write_json(output / "report.json", report)
    for method, rows in all_rows.items():
        report["methods"][method]["paired_controls"] = {control: compare(reference,rows)
            for control,reference in all_rows.items() if control in BASES and control != method}
    write_json(output / "report.json", report)
    print(json.dumps({"stage": "complete", "output": str(output), "methods": {k:v["status"] for k,v in report["methods"].items()}}), flush=True)
    return report


def add_arguments(parser, *, default_methods="s1q,rtn"):
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default=default_methods,
                        help="Comma-separated methods; s1q is the current decision-margin/compensation method")
    parser.add_argument("--bits", type=int, default=4, choices=(2,3,4,8))
    parser.add_argument("--activation-bits", type=int, default=4, choices=(4,8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--development-count", type=int, default=256)
    parser.add_argument("--reservoir-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--source-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--extra-evaluation", action="append", default=[])
    parser.add_argument("--cohort", default="exploratory_development")
    parser.add_argument("--repair-records", type=int, default=32)
    parser.add_argument("--repair-epochs", type=int, default=2)
    return parser


def main(argv=None, *, default_methods="s1q,rtn"):
    parser = add_arguments(argparse.ArgumentParser(description=__doc__), default_methods=default_methods)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
