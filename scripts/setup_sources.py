"""Acquire immutable public upstream code without model downloads.

Run from the S1Q root: python scripts/setup_sources.py
No Python ML packages are required; git and network access are required.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


REPOSITORIES = {
    "kev": "https://github.com/jaredpalmer/kev.git",
    "laya": "https://github.com/NandhaKishorM/laya.git",
    "NanoJev": "https://github.com/TianyuCodings/NanoJev.git",
    "decima": "https://github.com/amyrmahdy/decima.git",
}


def source_revisions(project_root: Path) -> dict[str, str]:
    """Read the single revision manifest without importing Torch."""
    tree = ast.parse((project_root / "src/s1q/models.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCE_REVISIONS"
                                               for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("SOURCE_REVISIONS manifest is missing")


def git(path: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(path), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def acquire(destination: Path, family: str, revision: str) -> None:
    path = destination / family
    if path.exists():
        if not (path / ".git").exists():
            raise RuntimeError(f"Refusing to overwrite non-checkout directory {path}")
        if git(path, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"Refusing to change a checkout with tracked modifications: {path}")
        actual = git(path, "rev-parse", "HEAD")
        if actual != revision:
            raise RuntimeError(f"Existing {path} is at {actual}; use a new --destination for pinned {revision}")
    else:
        path.mkdir(parents=True)
        git(path, "init")
        git(path, "remote", "add", "origin", REPOSITORIES[family])
        git(path, "fetch", "--depth", "1", "origin", revision)
        git(path, "checkout", "--detach", revision)
    if git(path, "rev-parse", "HEAD") != revision:
        raise RuntimeError(f"Could not verify pinned source {family}")
    (path / "SOURCE_REVISION.txt").write_text(revision + "\n", encoding="utf-8")
    print(f"{family}: {revision} -> {path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=project_root / "work/upstream")
    parser.add_argument("--families", nargs="+", choices=list(REPOSITORIES), default=list(REPOSITORIES))
    args = parser.parse_args()
    revisions = source_revisions(project_root)
    for family in args.families:
        acquire(args.destination.resolve(), family, revisions[family])


if __name__ == "__main__":
    main()
