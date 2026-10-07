"""Create a new audited 12-method derived run without changing either source.

This is private evidence assembly, not an experiment. The original main and
SmoothQuant frozen configurations are retained separately. A merged report is
compatible with summarize_optimization.py; all original eleven-method payloads
remain byte-identical and are never recalculated or overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

# Sibling CLI helper; source-tree execution needs no installation or Torch.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_smoothquant_merge import audit_pair, CONTROL, MAIN_METHODS, read_json, sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def derive(main_run, control_run, manifest, development_data, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing derived run")
    if output.is_relative_to(main_run.resolve()) or output.is_relative_to(control_run.resolve()):
        raise ValueError("Derived output must be separate from both source runs")
    audit = audit_pair(main_run, control_run, manifest, development_data)
    if not audit["merge_admissible"]:
        raise ValueError(f"SmoothQuant control did not pass the merge gate: {audit['errors']}")
    cohort_names = [row["dataset_id"] for row in audit["cohorts"]]
    source_hashes, copied = {}, []

    def copy(source, target, role):
        before = sha(source)
        shutil.copy2(source, target)
        after = sha(target)
        if before != after or sha(source) != before:
            raise ValueError(f"Source changed or byte-copy failed: {source.name}")
        source_hashes[(str(source), role)] = before
        copied.append({"file": target.name, "from": role, "source_sha256": before,
                       "derived_sha256": after, "bytes": target.stat().st_size})

    output.mkdir(parents=True, exist_ok=False)
    try:
        for cohort in cohort_names:
            copy(main_run / f"native-{cohort}.jsonl", output / f"native-{cohort}.jsonl", "main_native")
            for method in MAIN_METHODS:
                copy(main_run / f"{method}-{cohort}.jsonl", output / f"{method}-{cohort}.jsonl", "main_frozen_method")
            copy(control_run / f"{CONTROL}-{cohort}.jsonl", output / f"{CONTROL}-{cohort}.jsonl", "separate_smoothquant_control")
        for role, source_root in (("main", main_run), ("control", control_run)):
            for filename in ("report.json", "identity.json", "frozen-config-before-evaluation.json", "expanded-evaluation-complete.json"):
                copy(source_root / filename, output / f"source-{role}-{filename}", f"original_{role}_metadata")
        # Retain the original before-evaluation freeze; do not manufacture a
        # retrospective configuration pretending all twelve methods ran together.
        copy(main_run / "frozen-config-before-evaluation.json", output / "frozen-config-before-evaluation.json", "original_main_config")
        write(output / "native-equivalence-merge-audit.json", audit)
        audit_hash = sha(output / "native-equivalence-merge-audit.json")
        report = deepcopy(read_json(main_run / "report.json"))
        control = read_json(control_run / "report.json")
        report["methods"][CONTROL] = deepcopy(control["methods"][CONTROL])
        provenance = {"schema": "s1q.additional-control-derived-run.v1", "additional_method": CONTROL,
                      "official_reproduction": False, "native_equivalence_audit_sha256": audit_hash,
                      "input_hashes": audit["run_hashes"],
                      "additional_control_registration": audit["additional_control_registration"],
                      "main_method_results_unchanged": True,
                      "original_config_files": {"main": "source-main-frozen-config-before-evaluation.json",
                                                "control": "source-control-frozen-config-before-evaluation.json"},
                      "independent_evaluation_runs": True,
                      "interpretation": "Derived union after the 1e-6 native-equivalence and full coverage gate; does not claim a twelve-method experiment ran as one process."}
        report["identity"]["additional_control_provenance"] = provenance
        report["identity"]["derived_method_union"] = [*MAIN_METHODS, CONTROL]
        report["derivation"] = provenance
        if any(report["methods"][m] != read_json(main_run / "report.json")["methods"][m] for m in MAIN_METHODS):
            raise ValueError("An original method result changed in the derived report")
        write(output / "report.json", report)
        write(output / "identity.json", report["identity"])
        write(output / "expanded-evaluation-complete.json", {
            "benchmark_manifest_sha256": audit["benchmark_manifest_sha256"],
            "ranking_dataset_ids": [name for name in cohort_names if name != "development"],
            "status": "complete", "methods": {m: "complete" for m in [*MAIN_METHODS, CONTROL]},
            "derived_from_separate_controls": True, "native_equivalence_audit_sha256": audit_hash})
        for (filename, _), original_hash in source_hashes.items():
            if sha(Path(filename)) != original_hash:
                raise ValueError("An original input changed during evidence assembly")
        # Record every existing payload in the derived directory. The manifest
        # itself cannot hash its own bytes, so it is deliberately the only omission.
        files = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(output.iterdir()) if path.is_file()}
        derivation = {"schema": "s1q.derived-evidence-manifest.v1", "status": "complete",
                      "model": audit["model"], "n_methods": 12, "n_source_suites": 22,
                      "development_also_present": True, "original_runs_unchanged": True,
                      "benchmark_manifest_sha256": audit["benchmark_manifest_sha256"],
                      "native_equivalence_audit_sha256": audit_hash,
                      "derivation_script_sha256": sha(Path(__file__)),
                      "copied_payloads": copied, "files": files,
                      "input_metadata_hashes": audit["run_hashes"]}
        write(output / "derivation-manifest.json", derivation)
        return derivation
    except Exception as error:
        write(output / "derivation-failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.main_run, args.control_run, args.manifest, args.development_data, args.output)
    print(json.dumps({"status": result["status"], "model": result["model"],
                      "n_methods": result["n_methods"], "files": len(result["files"]),
                      "original_runs_unchanged": result["original_runs_unchanged"]}))


if __name__ == "__main__":
    main()
