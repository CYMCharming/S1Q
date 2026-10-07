"""Evaluate the existing SmoothQuant adaptation as an additional frozen control.

This entry point does not edit the quantizer or the eleven-method matrix. It
registers one existing baseline in this process, delegates to the frozen expanded
driver, and restores the registration afterwards. Use a separate output folder.
The W4A4 implementation is a repository adaptation, not the official W8A8
SmoothQuant reproduction or evidence of an integer-kernel speedup.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_expanded_benchmarks as driver
from s1q import optimization
from s1q.baselines import BASELINE_METHODS

CONTROL_METHOD = "smoothquant-adapted"


def control_metadata():
    paths = ("scripts/run_expanded_smoothquant.py", "scripts/run_expanded_benchmarks.py",
             "src/s1q/optimization.py", "src/s1q/baselines.py", "src/s1q/quantization.py")
    return {
        "method": CONTROL_METHOD,
        "role": "separately evaluated additional baseline control",
        "official_reproduction": False,
        "implementation": "existing quantize_baseline_model; channel balancing with fixed smooth_alpha=0.5",
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        "registration": "process-local BASES/ALL addition; originals restored on completion or failure",
        "quantization_math_changed": False,
        "frozen_main_method_matrix_changed": False,
        "hyperparameters_selected_on_expanded_results": False,
        "scope": "Reuse the manifest, seed, calibration, module scope, precision and native admission of the main frozen experiment. Report as SmoothQuant*, a repository adaptation.",
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--methods", default=CONTROL_METHOD)
    args, forwarded = parser.parse_known_args(argv)
    if args.methods != CONTROL_METHOD:
        parser.error(f"This additional-control entry point accepts only --methods {CONTROL_METHOD}")
    if CONTROL_METHOD not in BASELINE_METHODS:
        raise RuntimeError("The existing baseline implementation does not support SmoothQuant")
    saved_bases, saved_all, saved_run = optimization.BASES, optimization.ALL, driver.run
    saved_argv = sys.argv
    metadata = control_metadata()

    def controlled_run(run_args):
        run_args.additional_control_registration = metadata
        run_args.additional_control_only = True
        report = saved_run(run_args)
        report["identity"]["additional_control_registration"] = metadata
        report["identity"]["additional_control_only"] = True
        return report

    try:
        optimization.BASES = saved_bases + (() if CONTROL_METHOD in saved_bases else (CONTROL_METHOD,))
        optimization.ALL = saved_all + (() if CONTROL_METHOD in saved_all else (CONTROL_METHOD,))
        driver.run = controlled_run
        sys.argv = [str(Path(__file__).resolve()), *forwarded, "--methods", CONTROL_METHOD]
        return driver.main()
    finally:
        optimization.BASES, optimization.ALL, driver.run = saved_bases, saved_all, saved_run
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
