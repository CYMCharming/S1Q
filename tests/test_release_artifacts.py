"""Byte-exact reassembly and corrupted-download transaction checks."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fetch_artifacts", Path(__file__).resolve().parents[1] / "scripts/fetch_artifacts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
export_spec = importlib.util.spec_from_file_location("release_artifacts", Path(__file__).resolve().parents[1] / "scripts/release_artifacts.py")
export_module = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(export_module)


def fixture_entry():
    blobs = {"part001": b"abc", "part002": b"defg"}
    entry = {"filename": "evaluated.pt", "parts": [
        {"name": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in blobs.items()],
        "artifact_bytes": 7, "artifact_sha256": hashlib.sha256(b"abcdefg").hexdigest()}
    return entry, blobs


def test_release_exact_reassembly(tmp_path):
    entry, blobs = fixture_entry()
    path = module.assemble(entry, tmp_path, lambda name: io.BytesIO(blobs[name]))
    assert path.read_bytes() == b"abcdefg"


def test_release_tampering_leaves_no_model(tmp_path):
    entry, blobs = fixture_entry()
    blobs["part002"] = b"xxxx"
    with pytest.raises(ValueError, match="checksum"):
        module.assemble(entry, tmp_path, lambda name: io.BytesIO(blobs[name]))
    assert not list(tmp_path.iterdir())


def test_release_does_not_overwrite_existing(tmp_path):
    entry, blobs = fixture_entry()
    (tmp_path / "evaluated.pt").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        module.assemble(entry, tmp_path, lambda name: io.BytesIO(blobs[name]))
    assert (tmp_path / "evaluated.pt").read_bytes() == b"existing"


def test_release_preserves_existing_partial(tmp_path):
    entry, blobs = fixture_entry()
    partial = tmp_path / "evaluated.pt.partial"
    partial.write_bytes(b"previous interrupted download")
    with pytest.raises(FileExistsError):
        module.assemble(entry, tmp_path, lambda name: io.BytesIO(blobs[name]))
    assert partial.read_bytes() == b"previous interrupted download"


def release_run(tmp_path):
    data = b"the exact evaluated artifact bytes"
    source = tmp_path / "original.pt"
    source.write_bytes(data)
    run = tmp_path / "run"
    run.mkdir()
    summary = {"status": "complete", "model": "fixture", "bits": 4, "group_size": 128,
               "environment": {"model": {}}, "selection": {"selected": {"name": "s1q-local"}},
               "quantized": {"s1q-local": {"storage": {}, "artifact": {
                   "path": str(source), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "scope": "fixture"}}}}
    (run / "summary.json").write_text(json.dumps(summary))
    return run, source, data


def test_export_and_download_are_byte_exact(tmp_path):
    run, _, data = release_run(tmp_path)
    parts = tmp_path / "release"
    entry = export_module.export_run(run, parts, 7)
    assert all(p["bytes"] <= 7 for p in entry["parts"])
    path = module.assemble(entry, tmp_path / "download", lambda name: (parts / name).open("rb"))
    assert path.read_bytes() == data


def test_export_tampering_leaves_no_release_assets(tmp_path):
    run, source, data = release_run(tmp_path)
    source.write_bytes(b"x" * len(data))
    parts = tmp_path / "release"
    with pytest.raises(ValueError, match="integrity"):
        export_module.export_run(run, parts, 7)
    assert not list(parts.iterdir())


def test_export_collision_rolls_back_only_owned_assets(tmp_path):
    run, _, _ = release_run(tmp_path)
    parts = tmp_path / "release"
    parts.mkdir()
    collision = parts / "fixture-s1q-local-w4.pt.part002"
    collision.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        export_module.export_run(run, parts, 7)
    assert list(parts.iterdir()) == [collision]
    assert collision.read_bytes() == b"existing"


def test_download_commit_rejects_concurrent_target(tmp_path):
    entry, blobs = fixture_entry()

    def opener(name):
        (tmp_path / "evaluated.pt").write_bytes(b"concurrent writer")
        return io.BytesIO(blobs[name])

    with pytest.raises(FileExistsError):
        module.assemble(entry, tmp_path, opener)
    assert (tmp_path / "evaluated.pt").read_bytes() == b"concurrent writer"
    assert not (tmp_path / "evaluated.pt.partial").exists()
