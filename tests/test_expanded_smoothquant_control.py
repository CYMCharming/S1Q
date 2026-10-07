"""The additional control routes to existing math and leaves frozen code intact."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_expanded_smoothquant", ROOT / "scripts/run_expanded_smoothquant.py")
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


def test_smoothquant_routes_only_to_existing_baseline_and_restores_registry(monkeypatch):
    original_bases, original_all = CONTROL.optimization.BASES, CONTROL.optimization.ALL
    original_argv = sys.argv
    observed = {}

    def existing_run(args):
        assert CONTROL.CONTROL_METHOD in CONTROL.optimization.BASES
        assert CONTROL.CONTROL_METHOD in CONTROL.optimization.ALL
        observed.update(args.additional_control_registration)
        return {"identity": {}}

    def existing_driver_main():
        assert sys.argv[-2:] == ["--methods", "smoothquant-adapted"]
        assert "--model" in sys.argv and "kev-0.8b" in sys.argv
        return CONTROL.driver.run(SimpleNamespace())

    monkeypatch.setattr(CONTROL.driver, "run", existing_run)
    monkeypatch.setattr(CONTROL.driver, "main", existing_driver_main)
    report = CONTROL.main(["--model", "kev-0.8b"])
    assert report["identity"]["additional_control_only"] is True
    assert observed["official_reproduction"] is False
    assert observed["quantization_math_changed"] is False
    assert observed["frozen_main_method_matrix_changed"] is False
    assert CONTROL.optimization.BASES == original_bases
    assert CONTROL.optimization.ALL == original_all
    assert CONTROL.driver.run is existing_run
    assert sys.argv is original_argv


def test_failed_control_restores_registration_and_main_method_default(monkeypatch):
    original_bases, original_all, original_methods = CONTROL.optimization.BASES, CONTROL.optimization.ALL, CONTROL.driver.METHODS
    original_argv = sys.argv
    def fail():
        raise RuntimeError("controlled test failure")
    monkeypatch.setattr(CONTROL.driver, "main", fail)
    with pytest.raises(RuntimeError, match="controlled test failure"):
        CONTROL.main([])
    assert CONTROL.optimization.BASES == original_bases
    assert CONTROL.optimization.ALL == original_all
    assert CONTROL.driver.METHODS == original_methods
    assert sys.argv is original_argv


@pytest.mark.parametrize("methods", ["rtn", "smoothquant-adapted,rtn", "s1q", "smoothquant-adapted-repair"])
def test_other_methods_cannot_change_the_main_matrix(methods):
    original_bases, original_all = CONTROL.optimization.BASES, CONTROL.optimization.ALL
    with pytest.raises(SystemExit):
        CONTROL.main(["--methods", methods])
    assert CONTROL.optimization.BASES == original_bases
    assert CONTROL.optimization.ALL == original_all


def test_metadata_identifies_every_source_and_fixed_adaptation():
    metadata = CONTROL.control_metadata()
    assert metadata["method"] == "smoothquant-adapted"
    assert "smooth_alpha=0.5" in metadata["implementation"]
    assert len(metadata["source_sha256"]) == 5
    assert all(len(value) == 64 for value in metadata["source_sha256"].values())
