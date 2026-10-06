"""Run the frozen new-family smoke/full protocol after a real native gate.

Each invocation handles one checkpoint in a fresh process, preserving native
package isolation. The output tree must be new. No quantizer tuning, upstream
environment installation, or final-test read is performed here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from s1q.optimization import add_arguments, run

METHODS = "rtn,s1q-local,s1q2-beta05,awq-adapted,gptq-blockdiag-adapted,spinquant-nohad-adapted,spinquant-had-adapted,s1q-joint,s1q-ac,s1q-margin,s1q"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--wanli", type=Path, required=True)
    parser.add_argument("--mmlu", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--parity-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root / args.model
    if root.exists():
        raise FileExistsError(root)
    gate = json.loads(args.parity_json.read_text())
    manifest = args.checkpoint_dir / "checkpoint-manifest.json"
    checkpoint = json.loads(manifest.read_text())
    if gate.get("status") != "passed" or not gate.get("scope_complete") or gate["max_probability_error"] > 1e-6:
        raise ValueError("Native probability/gradient gate must pass before quantization")
    if gate["model"]["revision"] != checkpoint["revision"]:
        raise ValueError("Gate is for a different checkpoint")
    energies = gate.get("per_layer_margin_gradient_energy", {})
    if len(energies) != gate["selected_linear_count"] or any(not value > 0 for value in energies.values()):
        raise ValueError("Every selected Linear must have positive native-margin gradient energy")
    root.mkdir(parents=True)
    paths = {"protocol": args.protocol, "source_zip": args.source_zip,
             "parity": args.parity_json, "checkpoint_manifest": manifest,
             "calibration": args.data_dir / "calibration.jsonl", "development": args.data_dir / "development.jsonl",
             "wanli": args.wanli, "mmlu": args.mmlu}
    launch = {"status": "running", "model": args.model, "methods": METHODS.split(","),
              "driver_sha256": sha(Path(__file__)),
              "sha256": {name: sha(path) for name, path in paths.items()},
              "paths": {name: str(path) for name, path in paths.items()},
              "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
              "shared_gpu_timing_is_not_speedup_evidence": True,
              "prelaunch_nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu", "--format=csv"], capture_output=True, text=True, check=True).stdout,
              "stages": {}}
    launch_path = root / "launch.json"
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    for name, calibration, development, reservoir, methods in (
        ("w4a4-startup-smoke", 8, 16, 16, "rtn,s1q"),
        ("w4a4-full-v1", 128, 256, 128, METHODS),
    ):
        values = ["--model", args.model, "--data-dir", str(args.data_dir),
                  "--output-dir", str(root / name), "--checkpoint-dir", str(args.checkpoint_dir),
                  "--checkpoint-manifest", str(manifest), "--methods", methods,
                  "--bits", "4", "--activation-bits", "4", "--group-size", "128",
                  "--calibration-count", str(calibration), "--development-count", str(development),
                  "--reservoir-size", str(reservoir), "--seed", "20261004", "--device", "cuda", "--dtype", "bf16",
                  "--cohort", "new_family_startup_smoke" if "smoke" in name else "new_family_fixed_recipe_extension"]
        if "full" in name:
            values += ["--extra-evaluation", "wanli=" + str(args.wanli),
                       "--extra-evaluation", "mmlu-pro=" + str(args.mmlu)]
        torch.cuda.reset_peak_memory_stats()
        result = run(add_arguments(argparse.ArgumentParser()).parse_args(values))
        stage = {"status": "complete" if all(v["status"] == "complete" for v in result["methods"].values()) else "method_failure",
                 "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "methods": {key: value["status"] for key, value in result["methods"].items()},
                 "accepted_calibration": result["identity"]["accepted_calibration"],
                 "output": str(root / name)}
        launch["stages"][name] = stage
        launch_path.write_text(json.dumps(launch, indent=2) + "\n")
        print(json.dumps({"stage": name, **stage}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if stage["status"] != "complete" and "smoke" in name:
            raise RuntimeError("A quantizer startup smoke failed; full-budget matrix is gated")
    launch["status"] = "complete" if all(s["status"] == "complete" for s in launch["stages"].values()) else "method_failure"
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")


if __name__ == "__main__":
    main()
