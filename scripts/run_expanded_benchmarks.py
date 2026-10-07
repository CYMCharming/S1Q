"""Evaluate frozen quantization recipes on explicitly pinned added datasets.

Upstream test records are intentionally evaluated, so this driver marks that
fact explicitly. Existing calibration and development files are reused without
using any new evaluation target for fitting or configuration selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from s1q.optimization import add_arguments, run

METHODS = "rtn,s1q-local,s1q2-beta05,awq-adapted,gptq-blockdiag-adapted,spinquant-nohad-adapted,spinquant-had-adapted,s1q-joint,s1q-ac,s1q-margin,s1q"


def checked_datasets(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "s1q.expanded-benchmarks.v1":
        raise ValueError("Unsupported expanded benchmark manifest schema")
    root = manifest_path.parent.resolve()
    selected = []
    names = set()
    for entry in manifest["datasets"]:
        if type(entry.get("ranking_eligible")) is not bool:
            raise ValueError("ranking_eligible must be an explicit boolean")
        if not entry["ranking_eligible"]:
            continue
        if entry.get("label_kind") not in ("upstream_dataset_gold", "native_annotation", "authored_reference_gold", "authored_scenario_gold"):
            raise ValueError(f"Main rankings require audited independent labels: {entry.get('dataset_id')} / {entry.get('label_kind')}")
        name = entry["dataset_id"]
        if name == "development" or not name.replace("-", "").isalnum() or name in names:
            raise ValueError(f"Invalid or duplicate evaluation dataset: {name}")
        names.add(name)
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("A dataset path escaped the manifest directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {name}")
        selected.append((name, path))
    if not selected:
        raise ValueError("No frozen rank-eligible datasets")
    return manifest, selected


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__), default_methods=METHODS)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest, datasets = checked_datasets(args.benchmark_manifest)
    if args.extra_evaluation:
        raise ValueError("Extra datasets must be recorded in the pinned benchmark manifest")
    args.extra_evaluation = [f"{name}={path}" for name, path in datasets]
    args.cohort = "frozen_expanded_evaluation_20261007"
    # Stored in the before-evaluation configuration, rather than inferred later.
    args.benchmark_manifest_sha256 = hashlib.sha256(args.benchmark_manifest.read_bytes()).hexdigest()
    args.benchmark_manifest = str(args.benchmark_manifest)
    args.upstream_test_records_intentionally_evaluated = True
    args.recipe_selected_before_expanded_evaluation = True
    report = run(args)
    identity = report["identity"]
    identity["final_test_read"] = True
    identity["evaluation_role"] = "frozen request-disjoint expanded evaluation; upstream test/development provenance in manifest"
    identity["benchmark_manifest_sha256"] = args.benchmark_manifest_sha256
    identity["expanded_ranking_dataset_ids"] = [name for name, _ in datasets]
    identity["old_development_excluded_from_expanded_ranking"] = True
    identity["benchmark_id"] = manifest.get("benchmark_id", manifest.get("stage_id"))
    for name, records in identity["accepted_evaluation_ids"].items():
        if name != "development" and not records:
            raise ValueError(f"Entire dataset rejected by native model: {name}")
    output = Path(args.output_dir)
    for filename, value in (("report.json", report), ("identity.json", identity)):
        (output / filename).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "expanded-evaluation-complete.json").write_text(json.dumps({
        "benchmark_manifest_sha256": args.benchmark_manifest_sha256,
        "ranking_dataset_ids": identity["expanded_ranking_dataset_ids"],
        "status": "complete" if all(x["status"] == "complete" for x in report["methods"].values()) else "method_failure",
        "methods": {k: v["status"] for k, v in report["methods"].items()},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
