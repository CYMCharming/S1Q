"""Pinned, grouped decision evaluation data; source files remain local-only.

Every prepared row uses Kev's public labelled request schema. Labels and all
metadata must be removed by model adapters before inference (``model_request``).
No teacher prediction is ever promoted to an independent hard label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable
from urllib.request import urlopen


KEV_REVISION = "0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91"
NANO_DATA_REVISION = "7afc5257c0f3ff0ba08512729888a51d94b40e7e"
DEFAULT_SEED = 20261001
KEV_MANIFEST_HASHES = {
    "evals/v7/decision-v7": "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2",
    "evals/v4/transfer-v4": "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL without coercing labels or hiding duplicate identities."""
    records, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected an object")
            identity = (row.get("_meta") or {}).get("id") or row.get("id")
            if identity is not None:
                if identity in seen:
                    raise ValueError(f"{path}:{lineno}: duplicate record ID {identity}")
                seen.add(identity)
            records.append(row)
    return records


def write_records(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_request(record: dict) -> dict:
    """The entire inference allowlist. Gold and terminal metadata never enter it."""
    return {"state": deepcopy(record["state"]), "questions": {
        qid: {key: deepcopy(q[key]) for key in ("type", "instructions", "criteria") if key in q}
        for qid, q in record["questions"].items()
    }}


def question_keys(question: dict) -> list[str]:
    kind = question["type"]
    if kind in ("noul", "boolean"):
        return ["false", "true"]
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict):
        return list(criteria)
    if kind == "score" and isinstance(criteria, list):
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"Invalid question type or criteria: {kind}")


def validate_record(record: dict) -> None:
    if "state" not in record or not isinstance(record.get("questions"), dict) or not record["questions"]:
        raise ValueError("A record needs state and a nonempty questions mapping")
    for qid, q in record["questions"].items():
        if not isinstance(q, dict) or "label" not in q:
            raise ValueError(f"Missing independent target for question {qid}")
        keys = question_keys(q)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError(f"Empty or duplicated candidate keys: {qid}")
        y = q["label"]
        if q["type"] in ("noul", "boolean"):
            valid = type(y) is bool
        elif q["type"] == "score":
            valid = type(y) is int and 0 <= y < len(keys)
        else:
            valid = isinstance(y, str) and y in keys
        if not valid:
            raise ValueError(f"Invalid {q['type']} label for question {qid}: {y!r}")


def group_id(record: dict) -> str:
    meta = record.get("_meta") or {}
    metadata = record.get("metadata") or {}
    value = (meta.get("group_id") or metadata.get("source_group_id")
             or metadata.get("episode_id") or record.get("state_id")
             or meta.get("id") or record.get("id"))
    return str(value) if value is not None else _digest(model_request(record))


def eligibility_reason(record: dict, *, max_choices: int = 10,
                       max_chars: int | None = 1000, clean_only: bool = True) -> str | None:
    """Cheap, declared screening; tokenizer admission must still run afterwards.

    The character bound counts the complete JSON request, not only the state.
    No candidate or source text is truncated. Set max_chars=None for native suites.
    """
    meta = record.get("_meta") or {}
    if clean_only and meta.get("variant", "clean") != "clean":
        return "not_clean"
    if any(len(question_keys(q)) > max_choices for q in record["questions"].values()):
        return "too_many_choices"
    if max_chars is not None:
        chars = len(json.dumps(model_request(record), ensure_ascii=False, separators=(",", ":")))
        if chars > max_chars:
            return "request_too_long_chars"
    return None


def eligible_records(records: list[dict], *, predicate: Callable[[dict], bool] | None = None,
                     **eligibility) -> tuple[list[dict], dict]:
    accepted, excluded = [], Counter()
    for row in records:
        validate_record(row)
        reason = eligibility_reason(row, **eligibility)
        if reason is None and predicate is not None and not predicate(model_request(row)):
            reason = "tokenizer_admission"
        if reason is None:
            accepted.append(row)
        else:
            excluded[reason] += 1
    return accepted, dict(excluded)


def sample_groups(records: list[dict], limit: int | None, *, seed: int = DEFAULT_SEED,
                  salt: str = "sample") -> list[dict]:
    """Deterministic source-round-robin, whole-group sample of at most limit rows.

    Group hashing never depends on correctness, confidences or quantized results.
    A group that does not fit is skipped whole. Returned order is stable.
    """
    if limit is not None and limit < 0:
        raise ValueError("Sample limit must be nonnegative or None")
    groups = defaultdict(list)
    for row in records:
        groups[group_id(row)].append(row)
    sources = defaultdict(list)
    for gid, rows in groups.items():
        source = str((rows[0].get("_meta") or {}).get("source", "unknown"))
        sources[source].append(gid)
    score = lambda value: _digest([seed, salt, value])
    for source in sources:
        sources[source].sort(key=score)
    queues = [sources[source] for source in sorted(sources, key=score)]
    selected = []
    while any(queues):
        for queue in queues:
            if not queue:
                continue
            gid = queue.pop(0)
            rows = groups[gid]
            if limit is None or len(selected) + len(rows) <= limit:
                selected.extend(rows)
    return selected


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    """Reject group and exact complete-request reuse across S1Q partitions."""
    registries = [{}, {}]
    for split, rows in splits.items():
        for row in rows:
            for key, registry in zip((group_id(row), _digest(model_request(row))), registries):
                if key in registry and registry[key] != split:
                    raise ValueError(f"Dataset leakage: {registry[key]} and {split} share {key}")
                registry[key] = split


def _coalesce_request_groups(records: list[dict]) -> tuple[list[dict], dict]:
    """Join whole source groups linked by identical public inference requests.

    Repeated observations can connect different episodes transitively. Union the
    complete connected component before assigning any S1Q calibration bucket;
    assigning per-row hashes would split some original episodes. Source rows are
    retained, not silently deduplicated, and original identities remain auditable.
    """
    parents = {group_id(row): group_id(row) for row in records}

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            smaller, larger = sorted((left, right))
            parents[larger] = smaller

    request_groups = defaultdict(set)
    request_counts = Counter()
    for row in records:
        fingerprint = _digest(model_request(row))
        request_groups[fingerprint].add(group_id(row))
        request_counts[fingerprint] += 1
    for groups in request_groups.values():
        ordered = sorted(groups)
        for value in ordered[1:]:
            union(ordered[0], value)
    components = defaultdict(list)
    for value in parents:
        components[find(value)].append(value)
    component_ids, component_sources = {}, {}
    for values in components.values():
        values = sorted(values)
        identity = values[0] if len(values) == 1 else "s1q-component/" + _digest(values)
        for value in values:
            component_ids[value] = identity
            component_sources[value] = values
    result = []
    for row in records:
        row = deepcopy(row)
        original = group_id(row)
        meta = row.setdefault("_meta", {})
        meta["upstream_group_id"] = original
        meta["upstream_group_ids"] = component_sources[original]
        meta["group_id"] = component_ids[original]
        result.append(row)
    audit = {
        "records": len(records), "upstream_groups": len(parents),
        "components": len(components), "groups_joined": len(parents) - len(components),
        "cross_group_duplicate_request_buckets": sum(len(groups) > 1 for groups in request_groups.values()),
        "repeated_request_buckets": sum(count > 1 for count in request_counts.values()),
        "repeated_request_rows_beyond_first": sum(count - 1 for count in request_counts.values()),
        "largest_component_records": max(Counter(group_id(row) for row in result).values(), default=0),
        "deduplicated_rows": 0,
    }
    return result, audit


def _reserve_nonoverlapping_groups(splits: dict[str, list[dict]],
                                   priority: tuple[str, ...]) -> tuple[dict, dict]:
    """Keep fixed upstream heldouts; quarantine lower-priority components whole.

    Reservation uses complete admitted pools before pilot sampling, not only the
    sampled heldout rows. Match both original group identities and exact model
    requests. Metadata describes every rejected component and retained source.
    """
    if set(priority) != set(splits) or len(priority) != len(set(priority)):
        raise ValueError("Reservation priority must name every input split exactly once")
    registry, retained, audit = {}, {}, {}
    for name in priority:
        components = defaultdict(list)
        for row in splits[name]:
            components[group_id(row)].append(row)
        accepted, rejected = [], []
        for identity in sorted(components):
            rows = components[identity]
            upstream_groups, requests = set(), set()
            for row in rows:
                meta = row.get("_meta") or {}
                upstream_groups.update(meta.get("upstream_group_ids") or [meta.get("upstream_group_id", group_id(row))])
                requests.add(_digest(model_request(row)))
            keys = {("source_group", value) for value in upstream_groups} | {("request", value) for value in requests}
            overlaps = {key: registry[key] for key in keys if key in registry}
            if overlaps:
                rejected.append({
                    "component_id": identity, "records": len(rows),
                    "upstream_group_ids": sorted(upstream_groups),
                    "overlap_with": sorted(set(overlaps.values())),
                    "overlap_key_counts": dict(Counter(key[0] for key in overlaps)),
                    "overlapping_request_hashes": sorted(key[1] for key in overlaps if key[0] == "request"),
                })
                continue
            accepted.extend(rows)
            for key in keys:
                registry[key] = name
        retained[name] = accepted
        audit[name] = {
            "input_records": len(splits[name]), "retained_records": len(accepted),
            "excluded_records": sum(item["records"] for item in rejected),
            "excluded_components": len(rejected), "rejected_components": rejected,
        }
    return retained, audit


def _revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Use a full immutable 40-character commit SHA, not a branch/tag")
    return value


def _kev_file(relative: str, revision: str, cache: Path, source_root: Path | None) -> Path:
    if source_root is not None:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    path = cache / relative
    if not path.is_file():
        url = f"https://raw.githubusercontent.com/jaredpalmer/kev/{_revision(revision)}/{relative}"
        with urlopen(url, timeout=90) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def _verified_kev_splits(suite: str, revision: str, cache: Path,
                         source_root: Path | None, names: tuple[str, ...]) -> tuple[dict, dict]:
    manifest_path = _kev_file(f"{suite}/manifest.json", revision, cache, source_root)
    if revision == KEV_REVISION and file_sha256(manifest_path) != KEV_MANIFEST_HASHES[suite]:
        raise ValueError(f"Manifest bytes do not match the pinned Kev revision: {suite}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, provenance = {}, {"suite": suite, "revision": revision,
                               "manifest_sha256": file_sha256(manifest_path), "files": {}}
    for name in names:
        path = _kev_file(f"{suite}/{name}.jsonl", revision, cache, source_root)
        expected = manifest["files"][f"{name}.jsonl"]["sha256"]
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {suite}/{name}.jsonl")
        records[name] = load_records(path)
        provenance["files"][name] = {"sha256": actual, "records": len(records[name])}
    return records, provenance


def _save_prepared(output_dir: Path, splits: dict[str, list[dict]], manifest: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["splits"] = {}
    for name, rows in splits.items():
        path = output_dir / f"{name}.jsonl"
        write_records(path, rows)
        manifest["splits"][name] = {
            "path": str(path.resolve()), "sha256": file_sha256(path), "records": len(rows),
            "questions": sum(len(r["questions"]) for r in rows),
            "groups": len({group_id(r) for r in rows}),
            "sources": dict(Counter((r.get("_meta") or {}).get("source", "unknown") for r in rows)),
            "ids": [(r.get("_meta") or {}).get("id", r.get("id")) for r in rows],
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def prepare_kev_data(output_dir: str | Path, *, revision: str = KEV_REVISION,
                     source_root: str | Path | None = None, calibration: int | None = 64,
                     temperature_calibration: int | None = 64, development: int | None = 64,
                     test: int | None = 256, seed: int = DEFAULT_SEED,
                     max_choices: int = 10, max_chars: int | None = 1000,
                     clean_only: bool = True, include_transfer: bool = True,
                     predicate: Callable[[dict], bool] | None = None) -> dict:
    """Prepare pinned Kev v7 data with disjoint quantization/temperature sets.

    Calibration's groups are hash-partitioned 50/50 before sampling. Development
    chooses methods; test is confirmatory. Transfer-v4 is never used to fit scales.
    Limits count records and are pilot budgets, not evidence of final quality.
    A supplied source_root must contain bytes matching the pinned manifests.
    """
    output_dir = Path(output_dir)
    source_root = Path(source_root) if source_root is not None else None
    revision = _revision(revision)
    raw, provenance = _verified_kev_splits("evals/v7/decision-v7", revision,
        output_dir / "upstream" / revision, source_root, ("calibration", "development", "test"))
    admitted, exclusions = {}, {}
    eligibility = dict(max_choices=max_choices, max_chars=max_chars, clean_only=clean_only)
    for name, rows in raw.items():
        admitted[name], exclusions[name] = eligible_records(rows, predicate=predicate, **eligibility)
    quant_pool, temperature_pool = [], []
    for row in admitted["calibration"]:
        bucket = int(_digest([seed, "calibration_partition", group_id(row)]), 16) % 2
        (quant_pool if bucket == 0 else temperature_pool).append(row)
    splits = {
        "calibration": sample_groups(quant_pool, calibration, seed=seed, salt="quant_calibration"),
        "temperature_calibration": sample_groups(temperature_pool, temperature_calibration,
                                                  seed=seed, salt="temperature_calibration"),
        "development": sample_groups(admitted["development"], development, seed=seed, salt="development"),
        "test": sample_groups(admitted["test"], test, seed=seed, salt="test"),
    }
    assert_disjoint(splits)
    upstream = [provenance]
    if include_transfer:
        transfer, transfer_provenance = _verified_kev_splits("evals/v4/transfer-v4", revision,
            output_dir / "upstream" / revision, source_root, ("development", "test"))
        upstream.append(transfer_provenance)
        for name, rows in transfer.items():
            eligible, excluded = eligible_records(rows, predicate=predicate, **eligibility)
            exclusions[f"transfer_{name}"] = excluded
            splits[f"transfer_{name}"] = sample_groups(eligible, development if name == "development" else test,
                                                         seed=seed, salt=f"transfer_{name}")
        assert_disjoint(splits)
    manifest = {"schema": "s1q-data-v1", "dataset": "kev-frozen-suites", "seed": seed,
        "eligibility": eligibility, "tokenizer_predicate_used": predicate is not None,
        "exclusions": exclusions, "upstream": upstream,
        "calibration_partition": "SHA-256 by group; bucket 0 quantization, bucket 1 temperature",
        "license": "Mixed source licenses; see docs/evaluation-design.md. Do not redistribute source JSONL.",
        "interpretation": "decision-v7 shares training source families with Kev; transfer is source holdout for Kev only"}
    return _save_prepared(output_dir, splits, manifest)


def convert_nano_record(row: dict) -> tuple[dict | None, dict]:
    """Convert only explicit hard gold questions, preserving their target semantics.

    Teacher-only Maze/action rows have no hard gold and are counted/excluded.
    Expert argmax labels are compatibility targets, not probability calibration.
    """
    questions, excluded = {}, Counter()
    gold = row.get("gold") or {}
    kinds = row.get("gold_label_kind") or {}
    for qid, question in row.get("questions", {}).items():
        if qid not in gold:
            excluded["question_without_independent_hard_gold"] += 1
            continue
        kind = kinds.get(qid, "hard_gold_unspecified") if isinstance(kinds, dict) else kinds
        if kind in ("unobserved", "unspecified_compatibility_label", "hard_gold_unspecified"):
            excluded[f"question_label_kind_{kind}"] += 1
            continue
        q = deepcopy(question)
        if q["type"] == "boolean":
            q["type"] = "noul"
        q.update(label=deepcopy(gold[qid]), src=row.get("family_id", "nanojev-games"), label_kind=kind)
        if qid in (row.get("gold_probs") or {}):
            q["target_distribution"] = deepcopy(row["gold_probs"][qid])
            prob_kinds = row.get("gold_probs_kind") or {}
            q["target_distribution_kind"] = prob_kinds.get(qid) if isinstance(prob_kinds, dict) else prob_kinds
        questions[qid] = q
    if not questions:
        excluded["record_without_supported_hard_gold"] += 1
        return None, dict(excluded)
    metadata = row.get("metadata") or {}
    converted = {"state": deepcopy(row["state"]), "questions": questions,
                 "_meta": {"id": row["id"],
                    "group_id": str(metadata.get("source_group_id") or metadata.get("episode_id") or row["state_id"]),
                    "episode_id": metadata.get("episode_id"), "state_id": row["state_id"],
                    "source": row.get("family_id", "nanojev-games"), "split": row["split"],
                    "variant": "clean", "repo": "C-Tianyu/NanoJev-Data",
                    "revision": NANO_DATA_REVISION}}
    validate_record(converted)
    return converted, dict(excluded)


def prepare_nano_data(output_dir: str | Path, *, revision: str = NANO_DATA_REVISION,
                      source_root: str | Path | None = None, calibration: int | None = 64,
                      temperature_calibration: int | None = 64, development: int | None = 64,
                      test: int | None = 256, seed: int = DEFAULT_SEED, max_choices: int = 10,
                      max_chars: int | None = None, predicate: Callable[[dict], bool] | None = None) -> dict:
    """Prepare current NanoJev hard-target game splits; never load soft as duplicate data."""
    output_dir, revision = Path(output_dir), _revision(revision)
    if source_root is None:
        from huggingface_hub import hf_hub_download
        fetch = lambda name: Path(hf_hub_download("C-Tianyu/NanoJev-Data", name, repo_type="dataset",
            revision=revision, token=False))
    else:
        source_root = Path(source_root)
        fetch = lambda name: source_root / name
    manifest_path = fetch("SHA256_MANIFEST.json")
    upstream_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {entry["path"]: entry["sha256"] for entry in upstream_manifest["files"]}
    converted, exclusions, files = {}, {}, {}
    for name in ("calibration", "dev", "test", "ood"):
        relative = f"unified/hard/{name}.jsonl"
        path = fetch(relative)
        actual = file_sha256(path)
        if actual != hashes[relative]:
            raise ValueError(f"SHA-256 mismatch for NanoJev {relative}")
        rows, excluded = [], Counter()
        for row in load_records(path):
            item, rejected = convert_nano_record(row)
            excluded.update(rejected)
            if item is not None:
                item["_meta"]["revision"] = revision
                rows.append(item)
        eligible, rejected = eligible_records(rows, max_choices=max_choices, max_chars=max_chars,
                                               clean_only=True, predicate=predicate)
        excluded.update(rejected)
        converted[name], exclusions[name] = eligible, dict(excluded)
        files[name] = {"sha256": actual, "raw_records": len(load_records(path)),
                       "eligible_hard_gold_records": len(eligible)}
    # Some original game observations repeat across otherwise split-disjoint
    # episodes. First keep complete within-split connected components, then
    # reserve the fixed heldouts and remove overlapping lower-priority components.
    coalesced, coalescing = {}, {}
    for name, rows in converted.items():
        coalesced[name], coalescing[name] = _coalesce_request_groups(rows)
    converted, overlaps = _reserve_nonoverlapping_groups(
        coalesced, priority=("test", "ood", "dev", "calibration"))
    for name, item in overlaps.items():
        if item["excluded_records"]:
            exclusions[name]["overlap_with_reserved_higher_priority_split"] = item["excluded_records"]
    quant_pool, temperature_pool = [], []
    for row in converted["calibration"]:
        bucket = int(_digest([seed, "calibration_partition", group_id(row)]), 16) % 2
        (quant_pool if bucket == 0 else temperature_pool).append(row)
    splits = {
        "calibration": sample_groups(quant_pool, calibration, seed=seed, salt="quant_calibration"),
        "temperature_calibration": sample_groups(temperature_pool, temperature_calibration,
                                                  seed=seed, salt="temperature_calibration"),
        "development": sample_groups(converted["dev"], development, seed=seed, salt="development"),
        "test": sample_groups(converted["test"], test, seed=seed, salt="test"),
        "ood": sample_groups(converted["ood"], test, seed=seed, salt="ood"),
    }
    assert_disjoint(splits)
    manifest = {"schema": "s1q-data-v1", "dataset": "nanojev-unified-games-hard", "seed": seed,
        "repo": "C-Tianyu/NanoJev-Data", "revision": revision,
        "upstream_manifest_sha256": file_sha256(manifest_path), "upstream_files": files,
        "eligibility": {"max_choices": max_choices, "max_chars": max_chars},
        "tokenizer_predicate_used": predicate is not None, "exclusions": exclusions,
        "group_coalescing": coalescing, "source_overlaps": overlaps,
        "overlap_policy": {
            "priority": ["test", "ood", "dev", "calibration"],
            "unit": "whole connected component of original groups and identical public requests",
            "reservation_scope": "all admitted upstream records before sampling",
            "heldout_resplit": False, "silent_deduplication": False,
        },
        "license": "Dataset license not declared in inspected card; source bytes remain local-only.",
        "interpretation": "Expert action gold is compatibility agreement; observed_outcome labels alone support event calibration. Teacher-only questions excluded."}
    return _save_prepared(output_dir, splits, manifest)
