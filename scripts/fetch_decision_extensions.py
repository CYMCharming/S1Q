"""Acquire pinned new decision checkpoints with LFS and local byte manifests.

This tool only writes newly named per-checkpoint directories. It never deletes
an existing model/cache, alters an environment, or launches evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repos", nargs="+", required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    metadata = json.loads(args.metadata.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    for repo in args.repos:
        info = metadata[repo]
        entries = [f for f in info["files"] if not f["path"].startswith("assets/")]
        folder = args.root / repo.split("/")[-1]
        folder.mkdir(exist_ok=True)
        required = sum(f["bytes"] or 0 for f in entries if not (folder / f["path"]).is_file())
        free = shutil.disk_usage(args.root).free
        if free < required + (4 << 30):
            raise RuntimeError(f"Insufficient free disk for {repo}: need {required} plus 4GiB reserve, have {free}")
        print(json.dumps({"stage": "download", "repo": repo, "revision": info["sha"],
                          "required_bytes": required, "free_bytes": free}), flush=True)
        files, sizes = {}, {}
        for entry in entries:
            relative = entry["path"]
            path = Path(hf_hub_download(repo, relative, revision=info["sha"], local_dir=folder))
            size, sha = path.stat().st_size, digest(path)
            if entry["bytes"] is not None and size != entry["bytes"]:
                raise ValueError(f"Upstream file size mismatch: {repo}/{relative}")
            expected = (entry.get("lfs") or {}).get("sha256")
            if expected and sha != expected:
                raise ValueError(f"Pinned LFS SHA-256 mismatch: {repo}/{relative}")
            files[relative], sizes[relative] = sha, size
            print(json.dumps({"stage": "file_verified", "repo": repo,
                              "path": relative, "bytes": size,
                              "upstream_lfs_sha256_verified": bool(expected)}), flush=True)
        manifest = {"repo_id": repo, "revision": info["sha"], "files": files, "bytes": sizes,
                    "weight_bytes": sum(size for name, size in sizes.items() if name.endswith(".safetensors")),
                    "pinned_lfs_verified": True}
        (folder / "checkpoint-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"stage": "checkpoint_complete", "repo": repo,
                          "directory": str(folder), "weight_bytes": manifest["weight_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
