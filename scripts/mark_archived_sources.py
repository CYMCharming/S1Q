"""Write archive provenance after copying a locally verified git archive.

This is not source verification by itself; only use after the archive export.
Public setup should use setup_sources.py to verify real Git commits.
"""
from pathlib import Path
from s1q.models import SOURCE_REVISIONS
for family,revision in SOURCE_REVISIONS.items():
    root=Path("work/upstream")/family
    if not root.is_dir(): raise FileNotFoundError(root)
    (root/"SOURCE_REVISION.txt").write_text(revision+"\n")
