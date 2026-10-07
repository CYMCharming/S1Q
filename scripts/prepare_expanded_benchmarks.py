"""Prepare new, source-separated decision benchmarks without changing requests.

No prediction is used for sampling. Existing predictions are consulted only to
quarantine previously evaluated identities; calibration and temperature records
are quarantined by identity, source group, state and complete request. Source
JSONL is local-only. The manifest and code can be published independently.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s1q.data import (KEV_REVISION, eligibility_reason, file_sha256, group_id,
                      load_records, model_request, sample_groups, validate_record,
                      write_records)

KEV_SUITES = (
    ("evals/v7/decision-v7", ("test", "development")),
    ("evals/v4/transfer-v4", ("test", "development")),
    ("evals/public-pool-v6", ("test", "development")),
    ("evals/external/wanli-v2", ("development",)),
    ("evals/external/semif-v1", ("development",)),
)
INTERN_REVISION = "3572c8a68b5df5dafe02d0e093989ba8ec0183bc"
KEV_MANIFEST_SHA256 = {
    "evals/v7/decision-v7": "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2",
    "evals/v4/transfer-v4": "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1",
    "evals/public-pool-v6": "3fd5785634340347a0e25fe564d3d2824430b604b6f75b733d0eb50cb5969ca3",
    "evals/external/wanli-v2": "3bd7bf16d140122af3e0019f737f4f92d1353b40e66acdea6a69dd8e7a2ded36",
    "evals/external/semif-v1": "0de05eac16b0ddeeb2719c50a94a9148d6ae195f66303aec74aa103a3845ad11",
}
INTERN_FILE_SHA256 = {
    "jevbench/easy.jsonl": "3b8afa6e55e4decf1c8e116910443215690951dd51e15b7a6e4b813bc83c1eae",
    "jevbench/original.jsonl": "157c355d5d07fba47d516acdb048607a5ee4ef14347bbf87cf67dac73d1a193e",
    "jevbench/hard.jsonl": "5ed49f0a43a468993747a22b007fe65fc1187ed433d3c3a95ff9b69a293a8e34",
    "toolace/test.jsonl": "259a600d06c616c6b8bcc2138ec1e96be9eda0eb1817e871e57509c1be94ab2b",
    "typed_decisions/test.jsonl": "fa43c0389d9dbdf18cf1afe75c73f320d7620f3bb2a8be19e7cca7df5e78588f",
    "wildjailbreak/test.jsonl": "2b14e5449850463b530e92493244c4f4a888d91ff4b458130e17f661d2ce8938",
}
SOURCE_NAMES = {
    "agnews": "AG News", "amazon": "Amazon Reviews", "arc": "ARC",
    "boolq": "BoolQ", "csqa": "CommonsenseQA", "emotion": "Emotion",
    "imdb": "IMDb", "mmlu": "MMLU", "mnli": "MNLI",
    "openbookqa": "OpenBookQA", "paws": "PAWS", "qnli": "QNLI",
    "sciq": "SciQ", "sst5": "SST-5", "trec": "TREC",
    "tweet_offensive": "TweetEval Offensive", "wanli": "WANLI", "yelp": "Yelp",
    "semif_authored": "SemIf authored",
    "jevbench": "JevBench new eligible public tasks",
    "toolace": "ToolACE tool-choice classification",
    "wildjailbreak": "WildJailBreak harmful/benign classification",
    "typed-agreement": "Typed Decisions teacher agreement (auxiliary only)",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def record_id(row):
    return str((row.get("_meta") or {}).get("id") or row.get("id") or digest(model_request(row)))


def registry(rows):
    result = {key: set() for key in ("ids", "groups", "states", "requests", "upstream_texts")}
    for row in rows:
        result["ids"].add(record_id(row))
        result["groups"].add(group_id(row))
        result["states"].add(digest(row["state"]))
        result["requests"].add(digest(model_request(row)))
        meta = row.get("_meta") or {}
        if meta.get("text_sha256"):
            result["upstream_texts"].add(meta["text_sha256"])
        result["groups"].update(map(str, meta.get("upstream_group_ids", [])))
    return result


def overlaps(row, blocked):
    meta = row.get("_meta") or {}
    checks = {
        "ids": record_id(row), "groups": group_id(row),
        "states": digest(row["state"]), "requests": digest(model_request(row)),
        "upstream_texts": meta.get("text_sha256"),
    }
    return next((key for key, value in checks.items() if value is not None and value in blocked[key]), None)


def make_block_registry(root, candidate_rows):
    rows, evidence = [], []
    # A prepared test file alone is not evidence that it was evaluated. All
    # calibration and development files used locally are conservatively blocked.
    for folder in ("work/data-validation", "work/data-pilot", "work/data-decima-probe"):
        for name in ("calibration", "temperature_calibration", "development", "transfer_development"):
            path = root / folder / f"{name}.jsonl"
            if path.exists():
                values = load_records(path)
                rows.extend(values)
                evidence.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                                 "records": len(values), "reason": "calibration_or_previously_used_development"})
    # Every upstream calibration request is also reserved, even if not in the
    # current 128-request sample. Never turn unselected calibration into test.
    for suite, _ in KEV_SUITES:
        path = root / "work/model-audit/kev" / suite / "calibration.jsonl"
        if path.exists():
            values = load_records(path)
            rows.extend(values)
            evidence.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                             "records": len(values), "reason": "upstream_calibration_reserved"})
    blocked = registry(rows)
    used_ids, used_groups = set(), set()
    prediction_paths = []
    for result_root in (root / "research/results", root / "results"):
        prediction_paths.extend(result_root.rglob("*.jsonl"))
    for path in sorted(prediction_paths):
        if not any(token in path.name for token in ("native", "baseline", "predictions")):
            continue
        count = 0
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                identity = item.get("record_id") or item.get("id")
                if identity:
                    used_ids.add(str(identity))
                    count += 1
                identity = item.get("cluster_id") or item.get("group_id")
                if identity:
                    used_groups.add(str(identity))
        if count:
            evidence.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                             "decisions": count, "reason": "stored_previous_inference"})
    blocked["ids"].update(used_ids)
    blocked["groups"].update(used_groups)
    # Expand previously evaluated identities to their state/request/group hashes
    # using candidate source rows. This also quarantines question variants.
    previously_seen = [r for r in candidate_rows if record_id(r) in used_ids or group_id(r) in used_groups]
    expanded = registry(previously_seen)
    for key in blocked:
        blocked[key].update(expanded[key])
    return blocked, evidence


def read_kev_sources(source_root):
    rows, provenance = [], []
    for suite, names in KEV_SUITES:
        manifest_path = source_root / suite / "manifest.json"
        if file_sha256(manifest_path) != KEV_MANIFEST_SHA256[suite]:
            raise ValueError(f"Pinned Kev manifest mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in names:
            path = source_root / suite / f"{name}.jsonl"
            expected = manifest["files"][path.name]["sha256"]
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(f"Upstream SHA-256 mismatch: {path}")
            values = load_records(path)
            for row in values:
                row = deepcopy(row)
                row.setdefault("_meta", {})["benchmark_upstream_file"] = f"{suite}/{path.name}"
                row["_meta"]["benchmark_upstream_partition"] = name
                rows.append(row)
            provenance.append({"repo": "jaredpalmer/kev", "revision": KEV_REVISION,
                               "path": f"{suite}/{path.name}", "sha256": actual,
                               "manifest_sha256": file_sha256(manifest_path), "records": len(values)})
    return rows, provenance


def convert_intern_record(row, source, path):
    """Copy the public native request; attach explicit targets as metadata only."""
    if row.get("images"):
        raise ValueError("Text-only benchmark cannot discard image inputs")
    if source == "jevbench":
        questions = {"decision": deepcopy(row["question"])}
        targets = {"decision": {"label": row["expected"], "label_origin": "authored_scenario_gold"}}
        identity = "jevbench/" + row["id"]
        gid = "jevbench/" + str(row.get("group") or row["id"])
    else:
        questions = deepcopy(row["questions"])
        targets = row["targets"]
        identity = f"intern-benchmark/{source}/" + row["id"]
        gid = identity
    for key, question in questions.items():
        target = targets[key]
        label = target["label"]
        kind = question["type"]
        if kind in ("noul", "boolean"):
            if type(label) is bool:
                pass
            elif label in ("no", "false"):
                label = False
            elif label in ("yes", "true"):
                label = True
            else:
                raise ValueError(f"Unsupported Boolean target {label!r}")
        elif kind == "score":
            if isinstance(label, str) and label.isdigit():
                label = int(label)
        question["label"] = label
        question["label_kind"] = target["label_origin"]
        question["src"] = source
    result = {"state": deepcopy(row["state"]), "questions": questions, "_meta": {
        "id": identity, "group_id": gid, "source": source,
        "source_original_id": row["id"], "variant": "clean", "repo": "InternLM/Intern-Decision",
        "revision": INTERN_REVISION, "benchmark_upstream_file": "benchmarks/accuracy-v1/" + path,
        "benchmark_upstream_partition": "public" if source == "jevbench" else "test",
        "tier": Path(path).stem if source == "jevbench" else None,
        "ranking_eligible": source != "typed-agreement",
    }}
    validate_record(result)
    return result


def read_intern_sources(source_root):
    manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, provenance = [], []
    for path, expected in INTERN_FILE_SHA256.items():
        actual = file_sha256(source_root / path)
        if actual != expected or manifest["files"][path] != expected:
            raise ValueError(f"Pinned Intern benchmark integrity mismatch: {path}")
        source = "jevbench" if path.startswith("jevbench/") else "typed-agreement" if path.startswith("typed_decisions/") else path.split("/")[0]
        values, seen, duplicate_rows = [], {}, 0
        for line in (source_root / path).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in seen:
                if row != seen[row["id"]]:
                    raise ValueError(f"Conflicting duplicated Intern source identity: {path}:{row['id']}")
                duplicate_rows += 1
                continue
            seen[row["id"]] = row
            values.append(row)
        rows.extend(convert_intern_record(row, source, path) for row in values)
        dataset = next(item for item in manifest["datasets"].values() if item["path"] == path)
        provenance.append({"repo": "InternLM/Intern-Decision", "revision": INTERN_REVISION,
                           "path": "benchmarks/accuracy-v1/" + path, "sha256": actual,
                           "manifest_sha256": file_sha256(manifest_path), "records": len(values) + duplicate_rows,
                           "unique_records": len(values), "excluded_identical_duplicate_source_rows": duplicate_rows,
                           "upstream": dataset["upstream"], "license": dataset["upstream_license_metadata"]})
    return rows, provenance


def freeze(output, rows, blocked, evidence, provenance, *, seed, count, benchmark_id):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen cohort: {output}")
    output.mkdir(parents=True)
    sources = defaultdict(list)
    for row in rows:
        source = (row.get("_meta") or {}).get("source", "unknown")
        if source in SOURCE_NAMES:
            sources[source].append(row)
    datasets, empty_datasets = [], []
    overall_used = {key: set(value) for key, value in blocked.items()}
    for source in sorted(sources):
        rejected, accepted = Counter(), []
        seen = set()
        # Favor upstream test rows. Non-evaluated development is supplemental;
        # its original partition remains auditable, never relabeled as upstream test.
        pool = sorted(sources[source], key=lambda r: (
            (r.get("_meta") or {}).get("benchmark_upstream_partition") != "test",
            digest([seed, "candidate", record_id(r)])))
        group_reasons = {}
        for row in pool:
            validate_record(row)
            reason = eligibility_reason(row, max_choices=10, max_chars=1000, clean_only=True)
            shared = overlaps(row, overall_used)
            if shared:
                group_reasons[group_id(row)] = "overlap_with_reserved_" + shared
            if reason:
                group_reasons.setdefault(group_id(row), reason)
        for row in pool:
            reason = group_reasons.get(group_id(row))
            if reason:
                rejected[reason] += 1
                continue
            key = digest(model_request(row))
            if key in seen:
                rejected["duplicate_upstream_complete_request"] += 1
                continue
            seen.add(key)
            accepted.append(row)
        limit = 256 if source == "wanli" else count
        test_pool = [r for r in accepted if r["_meta"]["benchmark_upstream_partition"] == "test"]
        supplement = [r for r in accepted if r["_meta"]["benchmark_upstream_partition"] != "test"]
        selected = sample_groups(test_pool, limit, seed=seed, salt=f"{benchmark_id}/{source}/test")
        selected.extend(sample_groups(supplement, limit - len(selected), seed=seed,
                                     salt=f"{benchmark_id}/{source}/supplement"))
        if not selected:
            empty_datasets.append({"dataset_id": source.replace("_", "-"),
                                   "raw_candidates": len(pool), "excluded": dict(rejected),
                                   "n_requests": 0, "n_decisions": 0,
                                   "ranking_eligible": source != "typed-agreement"})
            continue
        dataset_id = source.replace("_", "-").replace("semif-authored", "semif")
        for row in selected:
            row["_meta"]["benchmark_source"] = source
            row["_meta"]["source"] = dataset_id
        path = output / "datasets" / f"{dataset_id}.jsonl"
        write_records(path, selected)
        hashes = registry(selected)
        for key in overall_used:
            overall_used[key].update(hashes[key])
        source_pins = sorted({((r.get("_meta") or {}).get("repo", "unknown"),
                              (r.get("_meta") or {}).get("revision", "unknown")) for r in selected})
        datasets.append({"dataset_id": dataset_id, "name": SOURCE_NAMES[source],
                         "path": path.relative_to(output).as_posix(), "sha256": file_sha256(path),
                         "n_requests": len(selected), "n_decisions": sum(len(r["questions"]) for r in selected),
                         "n_groups": len({group_id(r) for r in selected}),
                         "label_kind": "teacher_agreement" if source == "typed-agreement" else "authored_scenario_gold" if source in ("jevbench", "semif_authored") else "native_annotation" if source in ("toolace", "wildjailbreak") else "upstream_dataset_gold",
                         "ranking_eligible": source != "typed-agreement",
                         "source_pin": [{"repo": repo, "revision": revision} for repo, revision in source_pins],
                         "origins": dict(Counter(r["_meta"]["benchmark_upstream_partition"] for r in selected)),
                         "origin_files": dict(Counter(r["_meta"]["benchmark_upstream_file"] for r in selected)),
                         "tiers": dict(Counter(r["_meta"].get("tier") for r in selected if r["_meta"].get("tier"))),
                         "screening": {"raw_candidates": len(pool), "unique_eligible_remaining": len(accepted),
                                       "excluded": dict(rejected), "selection_limit": limit},
                         "selected_ids": [record_id(r) for r in selected]})
    block_path = output / "exclusion_registry.json"
    block_path.write_text(json.dumps({key: sorted(values) for key, values in blocked.items()}, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema": "s1q.expanded-benchmarks.v1", "benchmark_id": benchmark_id,
                "stage_id": benchmark_id, "seed": seed, "datasets": datasets,
                "empty_datasets": empty_datasets,
                "excluded_source_families": dict(Counter((r.get("_meta") or {}).get("source", "unknown") for r in rows if (r.get("_meta") or {}).get("source", "unknown") not in SOURCE_NAMES)),
                "selection_uses_predictions": False,
                "selection_policy": "Prediction-free SHA-256 sampling by complete source group; upstream test first, then unexamined development supplement; never truncate or reduce candidate sets.",
                "calibration_disjoint": True, "previously_evaluated_native_identities_disjoint": True,
                "eligibility": {"max_choices": 10, "max_chars": 1000, "clean_only": True, "truncation": False},
                "source_provenance": provenance, "exclusion_evidence": evidence,
                "exclusion_registry": {"path": block_path.name, "sha256": file_sha256(block_path),
                                       "counts": {key: len(value) for key, value in blocked.items()}},
                "scope": "New S1Q evaluation requests. Upstream development supplements are disclosed. Text-only, native typed schema, bounded request length. This is not a claim of source-family or model-pretraining decontamination. Model-tokenizer admission must be checked separately.",
                "license": "Mixed upstream licenses. Source JSONL remains local-only; publish metrics, hashes, identities and reproducible preparation code."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "work/model-audit/kev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--benchmark-id", default="kev-expanded-20261007")
    parser.add_argument("--intern-source-root", type=Path,
                        help="Optional fixed accuracy-v1 bundle; combines JevBench/ToolACE/WildJailBreak and keeps Typed teacher agreement auxiliary")
    parser.add_argument("--exclusion-registry", type=Path,
                        help="Reuse a frozen registry for reproduction instead of consulting later predictions")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    rows, provenance = read_kev_sources(args.source_root)
    if args.intern_source_root:
        intern_rows, intern_provenance = read_intern_sources(args.intern_source_root)
        rows.extend(intern_rows)
        provenance.extend(intern_provenance)
    if args.exclusion_registry:
        blocked = {key: set(values) for key, values in json.loads(args.exclusion_registry.read_text(encoding="utf-8")).items()}
        evidence = [{"path": args.exclusion_registry.name, "sha256": file_sha256(args.exclusion_registry),
                     "reason": "frozen_registry_reproduction"}]
    else:
        blocked, evidence = make_block_registry(ROOT, rows)
    manifest = freeze(args.output, rows, blocked, evidence, provenance,
                      seed=args.seed, count=args.count, benchmark_id=args.benchmark_id)
    print(json.dumps({"manifest": str(args.output / "manifest.json"),
                      "datasets": [{key: item[key] for key in ("dataset_id", "n_requests", "n_decisions", "origins")} for item in manifest["datasets"]],
                      "total_requests": sum(d["n_requests"] for d in manifest["datasets"]),
                      "total_decisions": sum(d["n_decisions"] for d in manifest["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
