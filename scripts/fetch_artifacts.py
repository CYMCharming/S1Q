"""Download and verify the exact evaluated S1Q artifact from a public release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen


def assemble(entry: dict, destination: Path, opener) -> Path:
    name = entry["filename"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Unsafe artifact filename")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix(target.suffix + ".partial")
    total_hash = hashlib.sha256()
    total_bytes = 0
    created = False
    try:
        with temporary.open("xb") as output:
            created = True
            for part in entry["parts"]:
                if Path(part["name"]).name != part["name"] or "/" in part["name"] or "\\" in part["name"]:
                    raise ValueError("Unsafe part filename")
                part_hash = hashlib.sha256()
                part_bytes = 0
                with opener(part["name"]) as incoming:
                    for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                        output.write(chunk); part_hash.update(chunk); total_hash.update(chunk)
                        part_bytes += len(chunk); total_bytes += len(chunk)
                if part_bytes != part["bytes"] or part_hash.hexdigest() != part["sha256"]:
                    raise ValueError(f"Release part checksum/size mismatch: {part['name']}")
        if total_bytes != entry["artifact_bytes"] or total_hash.hexdigest() != entry["artifact_sha256"]:
            raise ValueError("Assembled artifact checksum/size mismatch")
        # Same-directory link commits atomically without overwriting an existing file.
        os.link(temporary, target)
        temporary.unlink()
    except BaseException:
        # Only remove the temporary file created by this invocation.
        if created and temporary.exists():
            temporary.unlink()
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "configs/release-artifacts.json")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/released"))
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog["schema"] != "s1q-release-v1" or not catalog["base_url"].startswith("https://github.com/CYMCharming/S1Q/releases/download/"):
        raise ValueError("Unsupported release catalog")
    matches = [e for e in catalog["artifacts"] if e["model"] == args.model]
    if len(matches) != 1:
        raise ValueError(f"Choose an available model: {[e['model'] for e in catalog['artifacts']]}")
    entry = matches[0]
    target = assemble(entry, args.output_dir, lambda name: urlopen(urljoin(catalog["base_url"], name), timeout=120))
    print(json.dumps({"artifact": str(target), "sha256": entry["artifact_sha256"],
                      "requires_native_checkpoint": entry["model_metadata"], "scope": entry["scope"]}, indent=2))


if __name__ == "__main__":
    main()
