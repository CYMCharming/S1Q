"""Current public S1Q entry points retain the frozen MAC algorithm."""
import copy
from argparse import ArgumentParser

import pytest
import torch

from s1q.api import quantize_s1q
from s1q.optimization import add_arguments, run
from s1q.optimized_quantization import quantize_optimized_model
from test_optimization_runner import TinyAdapter, fixture_arguments, records
from test_optimized_quantization import fixture_model


def test_current_name_and_frozen_alias_have_identical_quantized_codes():
    model, _, statistics = fixture_model()
    current, _ = quantize_optimized_model(model, statistics=statistics, group_size=4)
    expected = {name: (layer.integer_weight.clone(), layer.scales.clone(), layer.input_scale.clone())
                for name, layer in current.layers.items()}
    assert all(layer.method == "s1q" for layer in current.layers.values())
    current.restore()
    legacy, _ = quantize_optimized_model(model, method="s1q-mac", statistics=statistics, group_size=4)
    try:
        for name, layer in legacy.layers.items():
            for actual, reference in zip((layer.integer_weight, layer.scales, layer.input_scale), expected[name]):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    finally:
        legacy.restore()


def test_public_api_strips_labels_without_mutating_requests_or_model():
    torch.manual_seed(16)
    adapter = TinyAdapter()
    adapter.model.eval()
    calibration = records("calibration", 8, 1, rejection=False)
    untouched = copy.deepcopy(calibration)
    original = {name: value.clone() for name, value in adapter.model.state_dict().items()}
    native_infer = adapter.infer

    def unlabeled_infer(request):
        assert all("label" not in q and "target" not in q for q in request["questions"].values())
        return native_infer(request)

    adapter.infer = unlabeled_infer
    session, metadata = quantize_s1q(adapter, calibration, group_size=2, reservoir_size=8)
    assert metadata["method"] == "S1Q"
    assert metadata["calibration"]["uses_gold_labels"] is False
    assert all(layer.method == "s1q" for layer in session.layers.values())
    session.restore()
    assert calibration == untouched
    for name, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_latest_runner_name_works_and_duplicate_aliases_fail(tmp_path, monkeypatch):
    import s1q.optimization as runner
    args = fixture_arguments(tmp_path)
    args.methods = "s1q,rtn"
    monkeypatch.setattr(runner, "load_model", lambda *a, **k: TinyAdapter())
    report = run(args)
    assert report["methods"]["s1q"]["status"] == "complete"
    assert (tmp_path / "result" / "s1q-development.jsonl").exists()
    args.output_dir = str(tmp_path / "duplicate")
    args.methods = "s1q,s1q-mac"
    with pytest.raises(ValueError, match="Unknown/duplicated"):
        run(args)
    assert not (tmp_path / "duplicate").exists()


def test_latest_cli_default_and_legacy_arguments_are_explicit():
    args = add_arguments(ArgumentParser()).parse_args([
        "--model", "kev-4b", "--data-dir", "data", "--output-dir", "result"])
    assert args.methods == "s1q,rtn"
    assert (args.bits, args.activation_bits, args.group_size, args.seed) == (4, 4, 128, 20261004)
