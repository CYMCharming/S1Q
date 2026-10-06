"""Audit and summarize every local optimization run from saved predictions.

No model or GPU is loaded, no method is selected, and no source data is read.
Reported metrics are recomputed from raw logits. All completed, failed and
unpaired methods remain visible. Confidence intervals use the existing paired
cluster bootstrap (saved cluster_id, record_id fallback), not independent rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s1q.metrics import compare, metrics  # noqa: E402

DEFAULT_REFERENCES = (
    "native", "rtn", "s1q-local", "s1q2-beta05", "awq-adapted",
    "gptq-blockdiag-adapted", "gptq-full-adapted",
    "spinquant-nohad-adapted", "spinquant-had-adapted",
)
METRIC_FIELDS = (
    "n", "accuracy", "macro_source_accuracy", "nll", "brier", "ece_15",
    "mean_confidence", "score_mae",
)
RISK_FIELDS = {"0.01": "coverage_at_risk_0_01", "0.05": "coverage_at_risk_0_05",
               "0.1": "coverage_at_risk_0_10"}
IDENTITY_FIELDS = ("run", "model", "precision", "cohort", "method", "status")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    keys = set()
    for row in rows:
        if row["key"] in keys:
            raise ValueError(f"Duplicate decision key {row['key']!r}")
        keys.add(row["key"])
        logits = np.asarray(row["logits"], dtype=float)
        if logits.ndim != 1 or len(logits) < 2 or not np.isfinite(logits).all():
            raise ValueError(f"Invalid decision logits {row['key']!r}")
        if type(row["label"]) is not int or not 0 <= row["label"] < len(logits):
            raise ValueError(f"Invalid decision target {row['key']!r}")
    if not rows:
        raise ValueError("Empty completed prediction file")
    return rows


def check_pair(reference: list[dict], candidate: list[dict]) -> None:
    ref = {r["key"]: r for r in reference}
    other = {r["key"]: r for r in candidate}
    if len(ref) != len(reference) or len(other) != len(candidate) or ref.keys() != other.keys():
        raise ValueError("Unique identical decision keys are required")
    for key, r in ref.items():
        q = other[key]
        for field in ("label", "type", "source", "record_id", "label_kind"):
            if r.get(field) != q.get(field):
                raise ValueError(f"Paired {field} differs at {key!r}")
        if len(r["logits"]) != len(q["logits"]):
            raise ValueError(f"Paired candidate count differs at {key!r}")
        if r.get("cluster_id", r["record_id"]) != q.get("cluster_id", q["record_id"]):
            raise ValueError(f"Paired cluster differs at {key!r}")


def vectorized_paired_accuracy_ci(reference: list[dict], candidate: list[dict],
                                  *, seed: int, samples: int) -> dict:
    """Same cluster bootstrap/RNG sequence as metrics.py, evaluated in bulk.

    Chunking caps index allocation while preserving the flat RNG draw order.
    Only execution is vectorized; groups, estimator and quantiles are unchanged.
    """
    ref = {r["key"]: r for r in reference}
    if set(ref) != {r["key"] for r in candidate}:
        raise ValueError("Paired CI requires identical keys")
    groups = defaultdict(list)
    for row in candidate:
        r = ref[row["key"]]
        delta = int(np.argmax(row["logits"]) == row["label"]) - int(np.argmax(r["logits"]) == r["label"])
        groups[row.get("cluster_id", row["record_id"])].append(delta)
    sums = np.array([sum(values) for values in groups.values()])
    counts = np.array([len(values) for values in groups.values()])
    if not len(sums):
        return {"n_clusters": 0}
    rng = np.random.default_rng(seed)
    chunk_size = max(1, min(samples, 8_000_000 // len(sums)))
    pieces = []
    for start in range(0, samples, chunk_size):
        indices = rng.integers(0, len(sums), size=(min(chunk_size, samples-start), len(sums)))
        pieces.append(sums[indices].sum(axis=1) / counts[indices].sum(axis=1))
    boot = np.concatenate(pieces)
    return {"n_clusters": len(sums), "delta": float(sums.sum() / counts.sum()),
            "lower_95": float(np.quantile(boot, .025)), "upper_95": float(np.quantile(boot, .975)),
            "seed": seed, "resamples": samples}


def scalar_metrics(raw: dict) -> dict:
    return {**{key: raw[key] for key in METRIC_FIELDS if key in raw},
            **{field: raw.get("coverage_at_risk", {}).get(risk)
               for risk, field in RISK_FIELDS.items()}}


def pair(reference: list[dict], candidate: list[dict], *, seed: int, samples: int) -> dict:
    check_pair(reference, candidate)
    result = compare(reference, candidate)
    result["accuracy_ci"] = vectorized_paired_accuracy_ci(reference, candidate, seed=seed, samples=samples)
    ref = {r["key"]: r for r in reference}
    result["beneficial_flip_rate"] = float(np.mean([
        int(np.argmax(r["logits"]) == r["label"] and
            np.argmax(ref[r["key"]]["logits"]) != r["label"]) for r in candidate
    ]))
    return result


def precision(identity: dict) -> str:
    bits = identity.get("bits", identity.get("weight_bits"))
    abits = identity.get("activation_bits")
    return f"W{bits}A{abits}" if abits is not None else f"W{bits}A-native"


def comparison_row(base: dict, reference: str, source: str, result: dict) -> dict:
    interval = result["accuracy_ci"]
    return {**base, "reference": reference, "source": source,
            "n": result["n"], "n_clusters": interval["n_clusters"],
            "accuracy_delta_pp": interval["delta"] * 100,
            "lower_95_pp": interval["lower_95"] * 100,
            "upper_95_pp": interval["upper_95"] * 100,
            **{k: result[k] for k in ("js_to_reference", "decision_flip_rate",
                                      "harmful_flip_rate", "beneficial_flip_rate")},
            "bootstrap_seed": interval["seed"], "bootstrap_samples": interval["resamples"]}


def report_cohorts(report: dict) -> list[str]:
    """Discover unique saved evaluation cohorts, including the primary alias."""
    cohorts = set(report.get("native_evaluations", {}))
    if "native_development" in report:
        cohorts.add("development")
    for item in report.get("methods", {}).values():
        cohorts.update(item.get("evaluations", {}))
        if "development" in item:
            cohorts.add("development")
    return sorted(cohorts, key=lambda name: (name != "development", name))


def cohort_evaluations(report: dict, cohort: str) -> dict:
    native = report.get("native_evaluations", {}).get(cohort)
    if native is None:
        native = report.get(f"native_{cohort}", {})
    evaluations = {"native": native}
    for name, item in report.get("methods", {}).items():
        evaluation = item.get("evaluations", {}).get(cohort)
        evaluations[name] = item.get(cohort, {}) if evaluation is None else evaluation
    return evaluations


def _summarize_cohort(runs_root: Path, *, references: tuple[str, ...], seed: int,
                      samples: int, cohort: str) -> dict:
    reports = sorted(runs_root.rglob("report.json"))
    if not reports:
        raise FileNotFoundError(f"No report.json found under {runs_root}")
    runs, metric_rows, source_rows, paired_rows, errors = [], [], [], [], []
    for report_path in reports:
        run_name = relative(report_path.parent, runs_root)
        report = read_json(report_path)
        if cohort not in report_cohorts(report):
            continue
        identity = report.get("identity", {})
        metadata = identity.get("model", report.get("model", {}))
        model = metadata.get("name", "unknown") if isinstance(metadata, dict) else str(metadata)
        base = {"run": run_name, "model": model, "precision": precision(identity),
                "cohort": cohort, "study_cohort": identity.get("cohort")}
        run = {**base, "report_sha256": digest(report_path), "identity": identity,
               "methods": {}, "comparisons": {}, "audit_errors": []}
        rows_by_method = {}
        evaluations = cohort_evaluations(report, cohort)
        for method, evaluation in evaluations.items():
            item = {} if method == "native" else report["methods"][method]
            status = item.get("status", "complete")
            path = report_path.parent / f"{method}-{cohort}.jsonl"
            record = {**base, "method": method, "status": status,
                      "quantization_seconds": item.get("quantization_seconds"),
                      "error": item.get("error")}
            detail = {"status": status, "reported_evaluation": evaluation,
                      "quantization_seconds": item.get("quantization_seconds"),
                      "error": item.get("error")}
            if status not in ("complete", "completed", "success", "ok"):
                run["methods"][method] = detail
                metric_rows.append(record)
                continue
            try:
                rows = read_rows(path)
                accepted_ids = identity.get("accepted_evaluation_ids", {}).get(cohort)
                if accepted_ids is None and cohort == "development":
                    accepted_ids = identity.get("accepted_development_ids")
                if accepted_ids is not None and {r["record_id"] for r in rows} != set(accepted_ids):
                    raise ValueError("Prediction record IDs differ from frozen accepted evaluation IDs")
                raw = metrics(rows)
                expected = evaluation.get("raw", evaluation)
                for field in METRIC_FIELDS:
                    if field in expected and (field not in raw or not math.isclose(
                            float(raw[field]), float(expected[field]), rel_tol=1e-9, abs_tol=1e-9)):
                        raise ValueError(f"Reported {field} differs from recomputed logits")
                for risk, coverage in expected.get("coverage_at_risk", {}).items():
                    if risk not in raw.get("coverage_at_risk", {}) or not math.isclose(
                            float(raw["coverage_at_risk"][risk]), float(coverage), rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f"Reported coverage at risk {risk} differs from recomputed logits")
                detail.update({"raw": raw, "predictions_sha256": digest(path),
                               "n_clusters": len({r.get("cluster_id", r["record_id"]) for r in rows})})
                record.update(scalar_metrics(raw))
                record["n_clusters"] = detail["n_clusters"]
                rows_by_method[method] = rows
                groups = defaultdict(list)
                for row in rows:
                    groups[row.get("source", "unknown")].append(row)
                detail["by_source"] = {}
                for source, members in sorted(groups.items()):
                    source_raw = metrics(members)
                    detail["by_source"][source] = source_raw
                    source_rows.append({**base, "method": method, "status": status, "source": source,
                                        **scalar_metrics(source_raw),
                                        "n_clusters": len({r.get("cluster_id", r["record_id"]) for r in members})})
            except (ValueError, KeyError, TypeError, OSError) as error:
                detail["audit_error"] = str(error)
                record["status"], record["error"] = "audit_failed", str(error)
                run["audit_errors"].append({"method": method, "error": str(error)})
            run["methods"][method] = detail
            metric_rows.append(record)
        for method, rows in rows_by_method.items():
            run["comparisons"][method] = {}
            for reference in references:
                if reference == method or reference not in rows_by_method:
                    continue
                try:
                    result = pair(rows_by_method[reference], rows, seed=seed, samples=samples)
                    run["comparisons"][method][reference] = result
                    paired_rows.append(comparison_row({**base, "method": method, "status": "complete"},
                                                      reference, "ALL", result))
                    if reference == "native":
                        current = next(x for x in metric_rows if x["run"] == run_name and x["method"] == method)
                        current.update({"teacher_flip_rate": result["decision_flip_rate"],
                                        "harmful_teacher_flip_rate": result["harmful_flip_rate"],
                                        "beneficial_teacher_flip_rate": result["beneficial_flip_rate"],
                                        "js_to_native": result["js_to_reference"]})
                    for source in sorted({r.get("source", "unknown") for r in rows}):
                        selected = [r for r in rows if r.get("source", "unknown") == source]
                        ref_selected = [r for r in rows_by_method[reference] if r.get("source", "unknown") == source]
                        source_result = pair(ref_selected, selected, seed=seed, samples=samples)
                        paired_rows.append(comparison_row({**base, "method": method, "status": "complete"},
                                                          reference, source, source_result))
                except (ValueError, KeyError, TypeError) as error:
                    run["audit_errors"].append({"method": method, "reference": reference, "error": str(error)})
        errors.extend({"run": run_name, "cohort": cohort, **error} for error in run["audit_errors"])
        runs.append(run)
    return {"schema": "s1q.optimization-summary.v1", "runs_root": str(runs_root.resolve()),
            "cohort_filename_suffix": cohort, "references": list(references),
            "bootstrap": {"seed": seed, "samples": samples,
                          "basis": "Saved cluster_id (record_id fallback); unstratified paired cluster bootstrap.",
                          "execution": "Vectorized identical NumPy RNG integer draw sequence; unchanged cluster estimator."},
            "runs": runs, "metrics": metric_rows, "by_source": source_rows,
            "paired": paired_rows, "audit_errors": errors,
            "limitations": ["All runs are retained; no best-method selection is performed.",
                            "Study exposure and split independence are declared by each run identity.",
                            "Same-cell raw logits are required; historical baseline scores are never substituted."]}


def summarize(runs_root: Path, *, references: tuple[str, ...], seed: int,
              samples: int, cohort: str | None = None) -> dict:
    reports = sorted(runs_root.rglob("report.json"))
    if not reports:
        raise FileNotFoundError(f"No report.json found under {runs_root}")
    available = {name for path in reports for name in report_cohorts(read_json(path))}
    if cohort is not None and cohort not in available:
        raise ValueError(f"Requested cohort {cohort!r} absent from all reports")
    cohorts = [cohort] if cohort is not None else sorted(available, key=lambda name: (name != "development", name))
    summary = None
    for name in cohorts:
        part = _summarize_cohort(runs_root, references=references, seed=seed,
                                 samples=samples, cohort=name)
        if summary is None:
            summary = part
        else:
            for key in ("runs", "metrics", "by_source", "paired", "audit_errors"):
                summary[key].extend(part[key])
    if summary is None:
        raise ValueError("Reports contain no declared evaluation cohorts")
    summary["schema"] = "s1q.optimization-summary.v2"
    summary["cohort_filename_suffix"] = cohort
    summary["evaluation_cohorts"] = cohorts
    summary["report_files"] = len(reports)
    for key in ("runs", "metrics", "by_source", "paired", "audit_errors"):
        summary[key].sort(key=lambda row: (row["run"], cohorts.index(row["cohort"])))
    return summary


def write_csv(path: Path, rows: list[dict], prefix: tuple[str, ...]) -> None:
    fields = list(prefix) + sorted({key for row in rows for key in row} - set(prefix))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(summary: dict) -> str:
    lines = ["# Local S1Q optimization results", "",
             "Every saved run and method is listed. Scores are recomputed from raw predictions; "
             "accuracy is a fraction in CSV/JSON and a percentage below.", "",
             "| Run | Model | Precision | Cohort | Method | N | Acc. % | NLL | Brier | Teacher flip % | Status |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    def number(row: dict, field: str, *, percent: bool = False) -> str:
        value = row.get(field)
        return "—" if value is None else f"{value * (100 if percent else 1):.2f}"
    for row in summary["metrics"]:
        lines.append(f"| {row['run']} | {row['model']} | {row['precision']} | {row['cohort']} | {row['method']} | "
                     f"{row.get('n', '—')} | {number(row, 'accuracy', percent=True)} | "
                     f"{number(row, 'nll')} | {number(row, 'brier')} | "
                     f"{number(row, 'teacher_flip_rate', percent=True)} | {row['status']} |")
    lines.extend(["", "## Paired differences", "",
                  "Intervals resample saved clusters within each run. Full per-source results are in CSV.", "",
                  "| Run | Cohort | Method | Reference | Delta pp | 95% CI pp | Clusters |",
                  "| --- | --- | --- | --- | ---: | --- | ---: |"])
    for row in summary["paired"]:
        if row["source"] == "ALL":
            lines.append(f"| {row['run']} | {row['cohort']} | {row['method']} | {row['reference']} | "
                         f"{row['accuracy_delta_pp']:+.2f} | [{row['lower_95_pp']:+.2f}, "
                         f"{row['upper_95_pp']:+.2f}] | {row['n_clusters']} |")
    if summary["audit_errors"]:
        lines.extend(["", "## Audit errors", ""])
        for error in summary["audit_errors"]:
            lines.append(f"- {error['run']} / {error['cohort']} / {error['method']}: {error['error']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "research/results/optimization-20261004/runs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cohort", help="Restrict to one cohort; default audits all declared cohorts")
    parser.add_argument("--references", nargs="+", default=DEFAULT_REFERENCES)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    output = args.output_dir or args.runs_root.parent
    summary = summarize(args.runs_root, references=tuple(args.references), seed=args.seed,
                        samples=args.bootstrap_samples, cohort=args.cohort)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "metrics.csv", summary["metrics"], IDENTITY_FIELDS)
    write_csv(output / "by-source.csv", summary["by_source"], IDENTITY_FIELDS + ("source",))
    write_csv(output / "paired.csv", summary["paired"], IDENTITY_FIELDS + ("reference", "source"))
    (output / "summary.md").write_text(render(summary), encoding="utf-8")
    print(json.dumps({"report_files": summary["report_files"], "cohort_runs": len(summary["runs"]),
                      "cohorts": summary["evaluation_cohorts"], "metric_rows": len(summary["metrics"]),
                      "source_rows": len(summary["by_source"]), "paired_rows": len(summary["paired"]),
                      "audit_errors": len(summary["audit_errors"]), "output": str(output.resolve())}))
    if summary["audit_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
