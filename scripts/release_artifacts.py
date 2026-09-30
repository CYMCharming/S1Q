"""Split evaluated artifacts into release assets without changing any bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_run(run_dir: Path, destination: Path, part_bytes: int) -> dict:
    summary_path = run_dir / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if summary["status"] != "complete":
        raise ValueError("Only completed, evaluated artifacts may be released")
    profile = summary["selection"]["selected"]["name"]
    result = summary["quantized"][profile]
    artifact = result["artifact"]
    source = Path(artifact["path"])
    if not source.is_file() or source.stat().st_size != artifact["bytes"]:
        raise ValueError(f"Missing or changed evaluated artifact: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{summary['model']}-{profile}-w{summary['bits']}"
    parts = []
    total_hash = hashlib.sha256()
    total_bytes = 0
    committed = []
    # TemporaryDirectory is confined to this explicit release-output directory.
    with tempfile.TemporaryDirectory(dir=destination, prefix=".s1q-") as stage:
        with source.open("rb") as incoming:
            index = 0
            while True:
                first = incoming.read(min(part_bytes, 8 * 1024 * 1024))
                if not first:
                    break
                index += 1
                name = f"{stem}.pt.part{index:03}"
                h = hashlib.sha256()
                size = 0
                with (Path(stage) / name).open("xb") as outgoing:
                    chunk = first
                    while chunk:
                        outgoing.write(chunk); h.update(chunk); total_hash.update(chunk)
                        size += len(chunk); total_bytes += len(chunk)
                        if size >= part_bytes:
                            break
                        chunk = incoming.read(min(part_bytes - size, 8 * 1024 * 1024))
                parts.append({"name": name, "bytes": size, "sha256": h.hexdigest()})
        if total_bytes != artifact["bytes"] or total_hash.hexdigest() != artifact["sha256"]:
            raise ValueError("Evaluated artifact changed or failed integrity validation during export")
        try:
            for part in parts:
                target = destination / part["name"]
                # An exclusive hard-link commit also rejects concurrent collisions.
                os.link(Path(stage) / part["name"], target)
                committed.append(target)
        except BaseException:
            for target in committed:
                target.unlink()
            raise
    return {"model": summary["model"], "run": run_dir.name, "profile": summary["selection"]["selected"],
            "bits": summary["bits"], "group_size": summary["group_size"],
            "model_metadata": summary["environment"]["model"],
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(), "artifact_sha256": artifact["sha256"],
            "artifact_bytes": artifact["bytes"], "filename": stem + ".pt", "parts": parts,
            "scope": artifact["scope"], "storage": result["storage"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--part-bytes", type=int, default=1500 * 1024 * 1024)
    parser.add_argument("--tag", default="v0.1.0")
    args = parser.parse_args()
    if not 0 < args.part_bytes < 2 * 1024 ** 3:
        raise ValueError("Release parts must be positive and below 2 GiB")
    entries = [export_run(run, args.output_dir, args.part_bytes) for run in args.run_dir]
    catalog = {"schema": "s1q-release-v1", "tag": args.tag,
               "base_url": f"https://github.com/CYMCharming/S1Q/releases/download/{args.tag}/",
               "artifacts": entries}
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog": str(args.catalog), "models": [e["model"] for e in entries],
                      "parts": sum(len(e["parts"]) for e in entries)}))


if __name__ == "__main__":
    main()
