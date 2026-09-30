"""Prepare a pinned JevBench authored cohort for external evaluation only.

This converter never creates calibration/development data and never reads model
predictions. Public raw files stay in the ignored output directory. Usage:

    python scripts/prepare_jevbench.py --source-dir work/jevbench \
        --output-dir work/data/jevbench-external

Use --max-chars 0 for a separately named native-length cohort. No request text or
candidate is truncated. Model-specific tokenizer admission must still be frozen
before evaluating a common cohort across architectures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s1q.data import (  # noqa: E402
    _coalesce_request_groups, _digest, _reject_constant, _unique_object,
    assert_disjoint, eligibility_reason, file_sha256, group_id, load_records,
    model_request, validate_record, write_records,
)

CHECKSUM_CONFIG = ROOT / "configs" / "jevbench-data-checksums.json"
DATA_FILES = ("datasets/public/original.jsonl", "datasets/public/hard.jsonl")


def source_bytes(relative: str, *, source_dir: Path | None, config: dict) -> bytes:
    """Read exact pinned Git blobs, or verify exact exported/downloaded bytes.

    Reading the Git object avoids Windows checkout CRLF conversion. Git trust is
    limited to this invocation, without changing the user's global configuration.
    """
    if source_dir is not None:
        if (source_dir / ".git").exists():
            raw = subprocess.check_output([
                "git", "-c", f"safe.directory={source_dir.resolve().as_posix()}",
                "-C", str(source_dir), "show", f"{config['revision']}:{relative}",
            ])
        else:
            raw = (source_dir / relative).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/{config['repo']}/{config['revision']}/{relative}"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
    expected = config["files"][relative]
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"Pinned JevBench source integrity failure: {relative}")
    return raw


def parse_source(raw: bytes) -> list[dict]:
    records, seen = [], set()
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Invalid JevBench source identity at line {line_no}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate JevBench source identity: {row['id']}")
        seen.add(row["id"])
        records.append(row)
    return records


def convert_task(row: dict, *, tier: str, revision: str) -> dict:
    """Map only explicit declared authored gold to the Kev labelled schema."""
    if row.get("split") != "public":
        raise ValueError("not_public")
    provenance = row.get("provenance") or {}
    if provenance.get("license") != "MIT":
        raise ValueError("license_not_verified_mit")
    if provenance.get("exclude_reason"):
        raise ValueError("upstream_excluded")
    if not provenance.get("label_basis"):
        raise ValueError("missing_label_provenance")
    if row.get("expected") is None:
        raise ValueError("missing_explicit_expected")
    state = row.get("state")
    if not isinstance(state, (str, dict)):
        raise ValueError("invalid_state_type")
    if isinstance(state, dict) and set(state) & {"expected", "label", "ground_truth", "answer_key"}:
        raise ValueError("upstream_state_gold_field")
    labels = row.get("labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(k, str) for k in labels) or len(set(labels)) != len(labels):
        raise ValueError("invalid_candidate_labels")
    original_q = row.get("question") or {}
    kind = original_q.get("type")
    instructions, criteria = original_q.get("instructions"), original_q.get("criteria")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("invalid_instructions")
    expected = row["expected"]
    if kind == "noul":
        if labels != ["no", "yes"] or expected not in ("no", "yes"):
            raise ValueError("unrecognized_noul_target")
        if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
            raise ValueError("unrecognized_noul_criteria")
        target = expected == "yes"
    elif kind == "choice":
        if not isinstance(criteria, dict) or set(criteria) != set(labels) or expected not in labels:
            raise ValueError("unrecognized_choice_target")
        target = expected
    elif kind == "score":
        if not isinstance(criteria, list) or labels != [str(i) for i in range(len(criteria))]:
            raise ValueError("unrecognized_ordinal_labels")
        if type(expected) is not int or not 0 <= expected < len(criteria):
            raise ValueError("unrecognized_ordinal_target")
        target = expected
    else:
        raise ValueError("unsupported_question_type")
    if any(not isinstance(c, str) or not c for c in (criteria.values() if isinstance(criteria, dict) else criteria)):
        raise ValueError("invalid_criterion_text")

    q = {"type": kind, "instructions": instructions, "criteria": deepcopy(criteria),
         "label": target, "label_kind": "authored_scenario_gold", "src": f"jevbench/{tier}/{row['family']}"}
    # Mathematical probability targets are not independent sampled events. Keep
    # both the modal classification target and the exact named soft target.
    if "gold_probs" in provenance:
        probs = provenance["gold_probs"]
        if row.get("family") != "probability" or not isinstance(probs, dict) or set(probs) != set(labels):
            raise ValueError("unrecognized_probability_target")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
            raise ValueError("invalid_probability_target")
        if abs(sum(probs.values()) - 1) > 1e-3:
            raise ValueError("invalid_probability_target_sum")
        modal = sorted(labels, key=lambda key: (-probs[key], key))[0]
        if modal != expected:
            raise ValueError("probability_mode_expected_mismatch")
        q["label_kind"] = "authored_probability_mode"
        q["target_distribution"] = ({"false": probs["no"], "true": probs["yes"]}
                                    if kind == "noul" else deepcopy(probs))
        q["target_distribution_kind"] = "state_derivable_probability"
    elif row.get("family") == "probability":
        raise ValueError("probability_family_missing_distribution")

    identity = row["id"]
    record = {"state": deepcopy(state), "questions": {"decision": q}, "_meta": {
        "id": f"jevbench/{identity}", "source_original_id": identity,
        "group_id": f"jevbench/{row.get('group') or identity}",
        "source_group_id": row.get("group") or identity,
        "source": f"jevbench/{tier}/{row['family']}", "family": row["family"],
        "tier": tier, "variant": "clean", "split": "external_test",
        "source_split": "public", "source_repo": "fstandhartinger/jevbench",
        "source_revision": revision, "source_license": "MIT",
        "source_expected": expected, "source_labels": deepcopy(labels),
        "provenance": deepcopy(provenance),
    }}
    validate_record(record)
    return record


def prepare_jevbench(output_dir: str | Path, *, source_dir: str | Path | None = None,
                     max_choices: int = 10, max_chars: int | None = 1000,
                     disjoint_with: list[str | Path] | None = None,
                     checksum_config: str | Path = CHECKSUM_CONFIG) -> dict:
    """Create a complete admitted external cohort and a rejection/provenance log."""
    if max_choices < 2 or (max_chars is not None and max_chars <= 0):
        raise ValueError("max_choices must be >=2 and max_chars positive or None")
    config = json.loads(Path(checksum_config).read_text(encoding="utf-8"))
    output = Path(output_dir)
    source = Path(source_dir) if source_dir is not None else None
    source_payloads = {path: source_bytes(path, source_dir=source, config=config) for path in DATA_FILES}
    license_bytes = source_bytes("LICENSE", source_dir=source, config=config)
    accepted, rejected, source_counts = [], [], {}
    seen = set()
    for path, payload in source_payloads.items():
        tier = Path(path).stem
        rows = parse_source(payload)
        if len(rows) != config["files"][path]["records"]:
            raise ValueError(f"Pinned JevBench source count mismatch: {path}")
        source_counts[tier] = len(rows)
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"Cross-file duplicate JevBench source identity: {row['id']}")
            seen.add(row["id"])
            try:
                converted = convert_task(row, tier=tier, revision=config["revision"])
                reason = eligibility_reason(converted, max_choices=max_choices, max_chars=max_chars)
            except ValueError as error:
                reason = str(error)
            if reason is None:
                accepted.append(converted)
            else:
                rejected.append({"id": row["id"], "tier": tier, "family": row.get("family"),
                                 "group": row.get("group"), "reason": reason})
    accepted, coalescing = _coalesce_request_groups(accepted)
    external_requests = {_digest(model_request(row)) for row in accepted}
    disjoint_audit = []
    for path in disjoint_with or []:
        other = load_records(path)
        assert_disjoint({"external_test": accepted, str(path): other})
        # The assert includes exact requests and bootstrap groups, before results.
        disjoint_audit.append({"path": str(path), "sha256": file_sha256(path),
                               "records": len(other), "shared_requests": 0, "shared_groups": 0})
    output.mkdir(parents=True, exist_ok=True)
    write_records(output / "external_test.jsonl", accepted)
    (output / "UPSTREAM_LICENSE.txt").write_bytes(license_bytes)
    families = Counter(row["_meta"]["source"] for row in accepted)
    types = Counter(q["type"] for row in accepted for q in row["questions"].values())
    kinds = Counter(q["label_kind"] for row in accepted for q in row["questions"].values())
    manifest = {
        "schema": "s1q.jevbench-external.v1", "role": "external_test_only",
        "source_repo": config["repo"], "source_revision": config["revision"],
        "source_license": config["source_license"], "source_files": {path: config["files"][path] for path in DATA_FILES},
        "selection_uses_predictions": False, "sampling": "All eligible original+hard public authored tasks; no random/pilot subsample.",
        "eligibility": {"max_choices": max_choices, "max_chars": max_chars, "truncation": False,
                        "tokenizer_admission": "Must be established separately on all chosen model adapters before inference."},
        "source_records": sum(source_counts.values()), "source_tier_counts": source_counts,
        "accepted_records": len(accepted), "accepted_questions": sum(len(row["questions"]) for row in accepted),
        "accepted_groups": len({group_id(row) for row in accepted}), "accepted_by_source": dict(families),
        "accepted_by_type": dict(types), "accepted_by_label_kind": dict(kinds),
        "unique_inference_requests": len(external_requests), "group_coalescing": coalescing,
        "rejected_records": len(rejected), "rejection_counts": dict(Counter(row["reason"] for row in rejected)),
        "rejected": rejected, "selected_ids": [row["_meta"]["source_original_id"] for row in accepted],
        "checked_disjoint_with": disjoint_audit,
        "output": {"external_test.jsonl": {"sha256": file_sha256(output / "external_test.jsonl")}},
        "label_semantics": config["label_provenance"], "upstream_scoring_contract": config["upstream_scoring_contract"],
        "limitations": ["Public-only screened authored cohort, not official full benchmark or rank.",
                         "Hard scenarios were authored and cross-reviewed with model assistance.",
                         "External to S1Q method selection; model pretraining contamination is unknown.",
                         "No calibration, temperature fitting, thresholds, bit selection or method tuning may use these tasks or predictions."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Local pinned Git clone or exact LF-byte export; omit to fetch verified GitHub raw files")
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    parser.add_argument("--max-choices", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=1000, help="Complete request character limit; 0 disables it for native-length analysis")
    parser.add_argument("--check-disjoint-with", type=Path, action="append", default=[], help="Prepared calibration/development/test JSONL; fail on group or exact request overlap")
    args = parser.parse_args()
    manifest = prepare_jevbench(args.output_dir, source_dir=args.source_dir,
                               max_choices=args.max_choices, max_chars=args.max_chars or None,
                               disjoint_with=args.check_disjoint_with)
    print(json.dumps({key: manifest[key] for key in (
        "source_revision", "source_records", "accepted_records", "accepted_groups",
        "accepted_by_type", "accepted_by_label_kind", "rejection_counts")}, indent=2))


if __name__ == "__main__":
    main()
